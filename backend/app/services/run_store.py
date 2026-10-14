"""
Simulation run (execution) store: one immutable evidence directory per execution
"""

import os
import json
import uuid
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_run_logger

RUNS_DIRNAME = "runs"
RUN_RECORD_FILE = "run.json"
LEGACY_RUN_ID = "legacy"

# Files copied from the simulation (configuration) directory into every run.
# agent_roster.json is the population frozen at prepare (active agents with their
# simulation/evidence roles plus context-only entities); agent_selection.json and
# experiment_plan.json are the user's configuration at the time the run started.
SNAPSHOT_FILES = (
    "simulation_config.json",
    "reddit_profiles.json",
    "twitter_profiles.csv",
    "agent_roster.json",
    "agent_selection.json",
    "experiment_plan.json",
)


class RunStatus(str, Enum):
    """Persistent lifecycle status of one simulation execution"""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    REPORTING = "reporting"
    REPORT_COMPLETED = "report_completed"


# A run in one of these states already holds evidence and must never be replaced
# by a plain start request.
FINISHED_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.STOPPED,
    RunStatus.REPORTING,
    RunStatus.REPORT_COMPLETED,
}

# States from which a report can be generated from the run's evidence
REPORTABLE_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.STOPPED,
    RunStatus.FAILED,
    RunStatus.REPORTING,
    RunStatus.REPORT_COMPLETED,
}

# Map of the internal runner status (run_state.json) to the lifecycle status
_RUNNER_TO_RUN_STATUS = {
    "idle": RunStatus.NOT_STARTED,
    "starting": RunStatus.RUNNING,
    "running": RunStatus.RUNNING,
    "paused": RunStatus.RUNNING,
    "stopping": RunStatus.RUNNING,
    "stopped": RunStatus.STOPPED,
    "completed": RunStatus.COMPLETED,
    "failed": RunStatus.FAILED,
}


def run_status_from_runner(runner_status: Optional[str]) -> RunStatus:
    """Map a runner status string to a RunStatus"""
    return _RUNNER_TO_RUN_STATUS.get(runner_status or "idle", RunStatus.NOT_STARTED)


@dataclass
class RunRecord:
    """Metadata of one simulation execution (persisted as runs/<run_id>/run.json)"""

    run_id: str
    simulation_id: str
    status: RunStatus = RunStatus.NOT_STARTED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    platform: str = "parallel"
    max_rounds: Optional[int] = None
    total_rounds: int = 0
    process_pid: Optional[int] = None
    action_counts: Dict[str, int] = field(default_factory=dict)
    report_ids: List[str] = field(default_factory=list)
    latest_report_id: Optional[str] = None
    last_report_error: Optional[str] = None
    error: Optional[str] = None
    previous_run_id: Optional[str] = None
    project_id: Optional[str] = None
    graph_id: Optional[str] = None
    snapshot: Dict[str, Any] = field(default_factory=dict)
    legacy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "simulation_id": self.simulation_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "platform": self.platform,
            "max_rounds": self.max_rounds,
            "total_rounds": self.total_rounds,
            "process_pid": self.process_pid,
            "action_counts": self.action_counts,
            "report_ids": self.report_ids,
            "latest_report_id": self.latest_report_id,
            "last_report_error": self.last_report_error,
            "error": self.error,
            "previous_run_id": self.previous_run_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "snapshot": self.snapshot,
            "legacy": self.legacy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        status = data.get("status", RunStatus.NOT_STARTED.value)
        try:
            status = RunStatus(status)
        except ValueError:
            status = RunStatus.NOT_STARTED
        return cls(
            run_id=data["run_id"],
            simulation_id=data.get("simulation_id", ""),
            status=status,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            platform=data.get("platform", "parallel"),
            max_rounds=data.get("max_rounds"),
            total_rounds=data.get("total_rounds", 0),
            process_pid=data.get("process_pid"),
            action_counts=data.get("action_counts", {}) or {},
            report_ids=data.get("report_ids", []) or [],
            latest_report_id=data.get("latest_report_id"),
            last_report_error=data.get("last_report_error"),
            error=data.get("error"),
            previous_run_id=data.get("previous_run_id"),
            project_id=data.get("project_id"),
            graph_id=data.get("graph_id"),
            snapshot=data.get("snapshot", {}) or {},
            legacy=data.get("legacy", False),
        )

    @property
    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES


class RunStore:
    """
    File-based store for simulation executions

    Layout:
        uploads/simulations/<simulation_id>/              configuration (state, config, profiles)
        uploads/simulations/<simulation_id>/runs/<run_id>/  one immutable execution

    Simulations created before runs existed keep their artifacts in the simulation
    directory itself; they are exposed read-only as the virtual run "legacy".
    """

    _locks: Dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    # ---------- paths ----------

    @classmethod
    def base_dir(cls) -> str:
        return Config.OASIS_SIMULATION_DATA_DIR

    @classmethod
    def sim_dir(cls, simulation_id: str) -> str:
        return os.path.join(cls.base_dir(), simulation_id)

    @classmethod
    def runs_dir(cls, simulation_id: str) -> str:
        return os.path.join(cls.sim_dir(simulation_id), RUNS_DIRNAME)

    @classmethod
    def run_dir(cls, simulation_id: str, run_id: str) -> str:
        """Directory of a run (the simulation directory itself for the legacy run)"""
        if run_id == LEGACY_RUN_ID:
            return cls.sim_dir(simulation_id)
        return os.path.join(cls.runs_dir(simulation_id), run_id)

    @classmethod
    def lock(cls, simulation_id: str) -> threading.RLock:
        """Per-simulation lock serializing start/restart/status transitions"""
        with cls._locks_guard:
            if simulation_id not in cls._locks:
                cls._locks[simulation_id] = threading.RLock()
            return cls._locks[simulation_id]

    @staticmethod
    def new_run_id() -> str:
        """Sortable, unique run id"""
        return f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # ---------- legacy ----------

    @classmethod
    def has_legacy_evidence(cls, simulation_id: str) -> bool:
        """Whether the simulation directory holds artifacts of a pre-runs execution"""
        sim_dir = cls.sim_dir(simulation_id)
        for rel in (
            "run_state.json",
            os.path.join("twitter", "actions.jsonl"),
            os.path.join("reddit", "actions.jsonl"),
            "actions.jsonl",
        ):
            if os.path.exists(os.path.join(sim_dir, rel)):
                return True
        return False

    @classmethod
    def _legacy_record(cls, simulation_id: str) -> RunRecord:
        sim_dir = cls.sim_dir(simulation_id)
        data: Dict[str, Any] = {}
        state_file = os.path.join(sim_dir, "run_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        status = run_status_from_runner(data.get("runner_status"))
        if status == RunStatus.NOT_STARTED and cls.has_legacy_evidence(simulation_id):
            # Actions exist without a usable run_state.json
            status = RunStatus.COMPLETED
        return RunRecord(
            run_id=LEGACY_RUN_ID,
            simulation_id=simulation_id,
            status=status,
            created_at=data.get("started_at") or "",
            updated_at=data.get("updated_at") or "",
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            total_rounds=data.get("total_rounds", 0),
            process_pid=data.get("process_pid"),
            action_counts={
                "twitter": data.get("twitter_actions_count", 0),
                "reddit": data.get("reddit_actions_count", 0),
            },
            error=data.get("error"),
            legacy=True,
        )

    # ---------- read ----------

    @classmethod
    def _read_record(cls, path: str) -> Optional[RunRecord]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return RunRecord.from_dict(json.load(f))
        except Exception:
            return None

    @classmethod
    def get_run(cls, simulation_id: str, run_id: str) -> Optional[RunRecord]:
        """Load a run record (synthesized read-only for the legacy run)"""
        if not run_id:
            return None
        if run_id == LEGACY_RUN_ID:
            if cls.has_legacy_evidence(simulation_id):
                return cls._legacy_record(simulation_id)
            return None
        if os.path.basename(run_id) != run_id:
            return None
        return cls._read_record(
            os.path.join(cls.run_dir(simulation_id, run_id), RUN_RECORD_FILE)
        )

    @classmethod
    def list_runs(cls, simulation_id: str) -> List[RunRecord]:
        """All runs of a simulation, oldest first (legacy run first if present)"""
        runs: List[RunRecord] = []
        if cls.has_legacy_evidence(simulation_id):
            runs.append(cls._legacy_record(simulation_id))
        runs_dir = cls.runs_dir(simulation_id)
        records: List[RunRecord] = []
        if os.path.isdir(runs_dir):
            for name in os.listdir(runs_dir):
                record = cls._read_record(os.path.join(runs_dir, name, RUN_RECORD_FILE))
                if record:
                    records.append(record)
        # Creation time orders runs created within the same second correctly
        records.sort(key=lambda r: (r.created_at, r.run_id))
        return runs + records

    @classmethod
    def get_current_run(cls, simulation_id: str) -> Optional[RunRecord]:
        """The most recent run (the legacy run if no new-style run exists)"""
        runs = cls.list_runs(simulation_id)
        return runs[-1] if runs else None

    @classmethod
    def resolve_run_id(cls, simulation_id: str, run_id: Optional[str] = None) -> Optional[str]:
        """Return run_id if given, otherwise the current run's id"""
        if run_id:
            return run_id
        current = cls.get_current_run(simulation_id)
        return current.run_id if current else None

    @classmethod
    def get_run_dir(cls, simulation_id: str, run_id: Optional[str] = None) -> str:
        """
        Directory holding a run's artifacts

        Falls back to the simulation directory (legacy layout) when there is no run.
        """
        resolved = cls.resolve_run_id(simulation_id, run_id)
        if not resolved:
            return cls.sim_dir(simulation_id)
        return cls.run_dir(simulation_id, resolved)

    # ---------- write ----------

    @classmethod
    def save(cls, record: RunRecord):
        """Atomically persist a run record (never for the read-only legacy run)"""
        if record.legacy or record.run_id == LEGACY_RUN_ID:
            return
        record.updated_at = datetime.now().isoformat()
        run_dir = cls.run_dir(record.simulation_id, record.run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, RUN_RECORD_FILE)
        tmp_path = f"{path}.{threading.get_ident()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def update(cls, simulation_id: str, run_id: str, **fields) -> Optional[RunRecord]:
        """Read-modify-write a run record under the simulation lock"""
        with cls.lock(simulation_id):
            record = cls.get_run(simulation_id, run_id)
            if not record:
                return None
            if record.legacy:
                return record
            for key, value in fields.items():
                if key == "status" and not isinstance(value, RunStatus):
                    value = RunStatus(value)
                setattr(record, key, value)
            cls.save(record)
            return record

    @classmethod
    def set_status(cls, simulation_id: str, run_id: str, status: RunStatus, **fields) -> Optional[RunRecord]:
        """Change the lifecycle status of a run"""
        record = cls.update(simulation_id, run_id, status=status, **fields)
        if record:
            get_run_logger("mirofish.runs", simulation_id, run_id).info(
                f"Run status -> {record.status.value}"
            )
        return record

    @classmethod
    def add_report(cls, simulation_id: str, run_id: str, report_id: str) -> Optional[RunRecord]:
        """Link a report to a run"""
        with cls.lock(simulation_id):
            record = cls.get_run(simulation_id, run_id)
            if not record or record.legacy:
                return record
            if report_id not in record.report_ids:
                record.report_ids.append(report_id)
            record.latest_report_id = report_id
            cls.save(record)
            return record

    @classmethod
    def create_run(
        cls,
        simulation_id: str,
        platform: str = "parallel",
        max_rounds: Optional[int] = None,
        previous_run_id: Optional[str] = None,
    ) -> RunRecord:
        """
        Create a new run directory with a snapshot of everything the execution uses

        The snapshot (config, profiles, personas, source texts, assessment brief)
        makes the run self-contained: later edits to the project or a re-prepare of
        the simulation never change the evidence of an existing run.

        Args:
            simulation_id: Simulation ID
            platform: twitter / reddit / parallel
            max_rounds: Optional round limit
            previous_run_id: Run this one replaces (restart)

        Returns:
            The new RunRecord (status not_started)
        """
        sim_dir = cls.sim_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"simulation_config.json not found for {simulation_id}")

        run_id = cls.new_run_id()
        run_dir = cls.run_dir(simulation_id, run_id)
        while os.path.exists(run_dir):
            run_id = cls.new_run_id()
            run_dir = cls.run_dir(simulation_id, run_id)
        os.makedirs(run_dir)
        log = get_run_logger("mirofish.runs", simulation_id, run_id)

        copied = []
        for name in SNAPSHOT_FILES:
            src = os.path.join(sim_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(run_dir, name))
                copied.append(name)

        project_id, graph_id = None, None
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                project_id = state_data.get("project_id")
                graph_id = state_data.get("graph_id")
            except Exception:
                pass

        snapshot: Dict[str, Any] = {"files": copied}
        roster_path = os.path.join(run_dir, "agent_roster.json")
        if os.path.exists(roster_path):
            try:
                with open(roster_path, "r", encoding="utf-8") as f:
                    roster = json.load(f)
                snapshot["population"] = {
                    "selection_hash": roster.get("selection_hash"),
                    "active_agents": len(roster.get("active_agents", [])),
                    "context_entities": len(roster.get("context_entities", [])),
                }
            except Exception as e:
                log.warning(f"Unreadable agent roster: {e}")
        else:
            snapshot["population"] = {"legacy": True, "note": "prepared before explicit agent selection; every prepared agent is active"}
        try:
            from .run_evidence import snapshot_run_inputs

            snapshot.update(snapshot_run_inputs(run_dir, project_id))
        except Exception as e:
            # A snapshot problem must never block the simulation itself;
            # the report will fall back to the project files and flag it.
            log.warning(f"Input snapshot incomplete: {e}")
            snapshot["snapshot_error"] = str(e)

        try:
            from .research_instructions import snapshot_run_plan

            snapshot["experiment_plan"] = snapshot_run_plan(run_dir, simulation_id)
        except Exception as e:
            # Without a valid plan no private stage runs; never fall back to posting it
            log.warning(f"Experiment plan not resolved: {e}")
            snapshot["experiment_plan_error"] = str(e)

        record = RunRecord(
            run_id=run_id,
            simulation_id=simulation_id,
            status=RunStatus.NOT_STARTED,
            platform=platform,
            max_rounds=max_rounds,
            previous_run_id=previous_run_id,
            project_id=project_id,
            graph_id=graph_id,
            snapshot=snapshot,
        )
        cls.save(record)
        log.info(
            f"Run created (platform={platform}, max_rounds={max_rounds}, "
            f"previous_run={previous_run_id}, snapshot={copied})"
        )
        return record
