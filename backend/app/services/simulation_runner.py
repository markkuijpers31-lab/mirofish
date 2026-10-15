"""
OASIS Simulation Runner
Runs simulations in the background and records each Agent's actions, supporting real-time status monitoring
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger, get_run_logger
from ..utils.locale import get_locale, set_locale, t
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse
from .run_store import LEGACY_RUN_ID, RunStatus, RunStore

logger = get_logger("mirofish.simulation_runner")

# Flag to track whether cleanup handlers have been registered
_cleanup_registered = False

# Platform detection
IS_WINDOWS = sys.platform == "win32"


class RunnerStatus(str, Enum):
    """Runner status"""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """Agent action record"""

    round_num: int
    timestamp: str
    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class RoundSummary:
    """Per-round summary"""

    round_num: int
    start_time: str
    end_time: Optional[str] = None
    simulated_hour: int = 0
    twitter_actions: int = 0
    reddit_actions: int = 0
    active_agents: List[int] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_hour": self.simulated_hour,
            "twitter_actions": self.twitter_actions,
            "reddit_actions": self.reddit_actions,
            "active_agents": self.active_agents,
            "actions_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """Simulation run state (real-time)"""

    simulation_id: str
    runner_status: RunnerStatus = RunnerStatus.IDLE

    # Execution this state belongs to (None/"legacy" for pre-runs simulations)
    run_id: Optional[str] = None

    # Progress information
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0

    # Per-platform independent round counts and simulation time (for dual-platform parallel display)
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0

    # Platform status
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0

    # Platform completion status (detected via simulation_end events in actions.jsonl)
    twitter_completed: bool = False
    reddit_completed: bool = False

    # Per-round summary
    rounds: List[RoundSummary] = field(default_factory=list)

    # Recent actions (for frontend real-time display)
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50

    # Timestamp
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Error information
    error: Optional[str] = None

    # Process ID (for stopping)
    process_pid: Optional[int] = None

    def add_action(self, action: AgentAction):
        """Add an action to the recent actions list"""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[: self.max_recent_actions]

        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1

        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "run_id": self.run_id,
            "runner_status": self.runner_status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "simulated_hours": self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(
                self.current_round / max(self.total_rounds, 1) * 100, 1
            ),
            # Per-platform independent round counts and time
            "twitter_current_round": self.twitter_current_round,
            "reddit_current_round": self.reddit_current_round,
            "twitter_simulated_hours": self.twitter_simulated_hours,
            "reddit_simulated_hours": self.reddit_simulated_hours,
            "twitter_running": self.twitter_running,
            "reddit_running": self.reddit_running,
            "twitter_completed": self.twitter_completed,
            "reddit_completed": self.reddit_completed,
            "twitter_actions_count": self.twitter_actions_count,
            "reddit_actions_count": self.reddit_actions_count,
            "total_actions_count": self.twitter_actions_count
            + self.reddit_actions_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "process_pid": self.process_pid,
        }

    def to_detail_dict(self) -> Dict[str, Any]:
        """Detailed info including recent actions"""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    Simulation Runner

    Responsibilities:
    1. Run OASIS simulations in a background process
    2. Parse run logs and record each Agent's actions
    3. Provide real-time status query interface
    4. Support pause/stop/resume operations
    """

    # Run state storage directory
    RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), "../../uploads/simulations")

    # Scripts directory
    SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "../../scripts")

    # In-memory run states, keyed by "<simulation_id>:<run_id>"
    _run_states: Dict[str, SimulationRunState] = {}
    # One tracked OASIS process per simulation (the active run)
    _processes: Dict[str, subprocess.Popen] = {}
    _active_runs: Dict[str, str] = {}  # simulation_id -> run_id of the tracked process
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # stdout file handles
    _stderr_files: Dict[str, Any] = {}  # stderr file handles

    # Graph memory update config (kept for API compatibility; the local graph is
    # built from source documents only, raw action logs are the report's evidence)
    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> enabled

    @staticmethod
    def _state_key(simulation_id: str, run_id: Optional[str]) -> str:
        return f"{simulation_id}:{run_id or LEGACY_RUN_ID}"

    @classmethod
    def _run_dir(cls, simulation_id: str, run_id: Optional[str]) -> str:
        """Directory holding the artifacts of a run (legacy: the simulation dir)"""
        return RunStore.run_dir(simulation_id, run_id or LEGACY_RUN_ID)

    @classmethod
    def _log(cls, simulation_id: str, run_id: Optional[str], area: Optional[str] = None):
        return get_run_logger("mirofish.simulation_runner", simulation_id, run_id, area)

    @classmethod
    def get_run_state(
        cls, simulation_id: str, run_id: Optional[str] = None
    ) -> Optional[SimulationRunState]:
        """
        Get the run state of a run (default: the simulation's current run)

        Args:
            simulation_id: Simulation ID
            run_id: Run ID (optional)
        """
        run_id = RunStore.resolve_run_id(simulation_id, run_id)
        key = cls._state_key(simulation_id, run_id)
        if key in cls._run_states:
            return cls._run_states[key]

        # Try loading from file
        state = cls._load_run_state(simulation_id, run_id)
        if state:
            cls._run_states[key] = state
        return state

    @classmethod
    def _load_run_state(
        cls, simulation_id: str, run_id: Optional[str] = None
    ) -> Optional[SimulationRunState]:
        """Load run state from file"""
        state_file = os.path.join(cls._run_dir(simulation_id, run_id), "run_state.json")
        if not os.path.exists(state_file):
            return None

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = SimulationRunState(
                simulation_id=simulation_id,
                run_id=run_id or LEGACY_RUN_ID,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # Per-platform independent round counts and time
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )

            # Load recent actions
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(
                    AgentAction(
                        round_num=a.get("round_num", 0),
                        timestamp=a.get("timestamp", ""),
                        platform=a.get("platform", ""),
                        agent_id=a.get("agent_id", 0),
                        agent_name=a.get("agent_name", ""),
                        action_type=a.get("action_type", ""),
                        action_args=a.get("action_args", {}),
                        result=a.get("result"),
                        success=a.get("success", True),
                    )
                )

            return state
        except Exception as e:
            logger.error(f"Failed to load run state: {str(e)}")
            return None

    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Save run state to the run's own directory (never for the legacy run)"""
        if not state.run_id or state.run_id == LEGACY_RUN_ID:
            cls._run_states[cls._state_key(state.simulation_id, state.run_id)] = state
            return
        run_dir = cls._run_dir(state.simulation_id, state.run_id)
        os.makedirs(run_dir, exist_ok=True)
        state_file = os.path.join(run_dir, "run_state.json")

        data = state.to_detail_dict()

        # Unique temp name: the monitor thread and API threads may save concurrently
        tmp_file = f"{state_file}.{threading.get_ident()}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, state_file)

        cls._run_states[cls._state_key(state.simulation_id, state.run_id)] = state

    @classmethod
    def is_process_alive(cls, simulation_id: str) -> bool:
        """Whether a tracked OASIS process of this simulation is still running"""
        process = cls._processes.get(simulation_id)
        return bool(process and process.poll() is None)

    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: Optional[int] = None,  # Max simulation rounds (optional, for truncating long simulations)
        enable_graph_memory_update: bool = False,  # Kept for API compatibility
        graph_id: Optional[str] = None,  # Graph store ID
        run_id: Optional[str] = None,  # Run to execute (created if omitted)
    ) -> SimulationRunState:
        """
        Execute a run of a simulation

        Every execution runs inside its own run directory
        (uploads/simulations/<simulation_id>/runs/<run_id>/), so it can never
        append to or overwrite the evidence of another execution.

        Args:
            simulation_id: Simulation ID
            platform: Run platform (twitter/reddit/parallel)
            max_rounds: Maximum number of simulation rounds (optional)
            enable_graph_memory_update: Ignored (see class docstring)
            graph_id: Graph store ID
            run_id: Run created by RunStore.create_run (a new run is created if omitted)

        Returns:
            SimulationRunState
        """
        if cls.is_process_alive(simulation_id):
            raise ValueError(t("api.simAlreadyRunning", id=simulation_id))

        if not run_id:
            run_id = RunStore.create_run(simulation_id, platform, max_rounds).run_id
        record = RunStore.get_run(simulation_id, run_id)
        if record is None or record.legacy:
            raise ValueError(f"Run {run_id} cannot be executed")
        if record.status != RunStatus.NOT_STARTED:
            # A run is executed exactly once; re-running means creating a new run
            raise ValueError(
                f"Run {run_id} already has status '{record.status.value}' and cannot be started again"
            )

        log = cls._log(simulation_id, run_id)
        run_dir = cls._run_dir(simulation_id, run_id)
        config_path = os.path.join(run_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            raise ValueError(t("api.simConfigNotFoundPrepare"))

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Initialize run state
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)

        # A contracted run executes exactly the Run Contract's rounds, never a
        # length derived from the generated time_config
        from .run_contract import contract_round_count

        contract_rounds = contract_round_count(run_dir)
        if contract_rounds is not None:
            if max_rounds is not None and max_rounds != contract_rounds:
                raise ValueError(
                    f"max_rounds {max_rounds} differs from the Run Contract of run {run_id} ({contract_rounds})"
                )
            max_rounds = contract_rounds
            total_rounds = contract_rounds
            log.info(f"Run length locked by Run Contract: exactly {contract_rounds} rounds")
        # If max_rounds specified, truncate
        elif max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                log.info(
                    f"Rounds truncated: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})"
                )

        state = SimulationRunState(
            simulation_id=simulation_id,
            run_id=run_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )

        cls._save_run_state(state)

        if enable_graph_memory_update:
            log.info(
                "Graph memory update requested: not applied (local graph holds source "
                "documents; the report reads this run's raw action logs directly)"
            )
        cls._graph_memory_enabled[simulation_id] = False

        # Determine which script to run (scripts are in backend/scripts/)
        if platform == "twitter":
            script_name = "run_twitter_simulation.py"
            state.twitter_running = True
        elif platform == "reddit":
            script_name = "run_reddit_simulation.py"
            state.reddit_running = True
        else:
            script_name = "run_parallel_simulation.py"
            state.twitter_running = True
            state.reddit_running = True

        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)

        if not os.path.exists(script_path):
            raise ValueError(t("api.scriptNotFound", path=script_path))

        # Create action queue
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue

        # Start simulation process
        try:
            # The script derives every output path from dirname(--config), so
            # pointing it at the run's config snapshot keeps all artifacts
            # (actions.jsonl, DBs, IPC, env_status, simulation.log) in the run dir.
            cmd = [
                sys.executable,  # Python interpreter
                script_path,
                "--config",
                config_path,  # Run's config snapshot
            ]

            # If max_rounds specified, add to command line args
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])

            # Create main log file to avoid stdout/stderr pipe buffer overflow blocking the process
            main_log_path = os.path.join(run_dir, "simulation.log")
            main_log_file = open(main_log_path, "a", encoding="utf-8")

            # Set child process env vars to ensure UTF-8 encoding on Windows
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"  # Python 3.7+: makes all open() default to UTF-8
            env["PYTHONIOENCODING"] = "utf-8"  # Ensure stdout/stderr use UTF-8
            env["MIROFISH_SIMULATION_ID"] = simulation_id
            env["MIROFISH_RUN_ID"] = run_id

            # Use start_new_session=True to create a new process group, ensuring all child processes can be terminated via os.killpg
            process = subprocess.Popen(
                cmd,
                cwd=run_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # stderr also written to same file
                text=True,
                encoding="utf-8",  # Explicit encoding
                bufsize=1,
                env=env,  # Pass env with UTF-8 settings
                start_new_session=True,  # New process group, ensures all related processes are terminated on server shutdown
            )

            # Save file handles for later cleanup
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None  # No separate stderr needed

            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._active_runs[simulation_id] = run_id
            cls._save_run_state(state)
            RunStore.set_status(
                simulation_id,
                run_id,
                RunStatus.RUNNING,
                started_at=state.started_at,
                process_pid=process.pid,
                total_rounds=total_rounds,
                platform=platform,
                max_rounds=max_rounds,
            )

            # Capture locale before spawning monitor thread
            current_locale = get_locale()

            # Start monitor thread (bound to this run's process and state only)
            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id, current_locale, run_id, process, state, main_log_file),
                daemon=True,
            )
            monitor_thread.start()
            cls._monitor_threads[simulation_id] = monitor_thread

            log.info(
                f"Simulation started: pid={process.pid}, platform={platform}, total_rounds={total_rounds}"
            )

        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            RunStore.set_status(simulation_id, run_id, RunStatus.FAILED, error=str(e))
            raise

        return state

    @staticmethod
    def _pid_alive(pid: Optional[int], run_id: Optional[str] = None) -> bool:
        """
        Whether pid is alive (and, where /proc exists, really runs this run's config)

        After a container restart PIDs are reused, so on Linux the command line is
        checked for the run id instead of trusting the bare PID.
        """
        if not pid:
            return False
        cmdline_path = f"/proc/{pid}/cmdline"
        if run_id and os.path.isdir("/proc/self"):
            try:
                with open(cmdline_path, "rb") as f:
                    return run_id.encode() in f.read()
            except OSError:
                return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    @classmethod
    def reconcile_run(cls, simulation_id: str, record):
        """
        Repair the persisted status of a run whose process disappeared

        After a server/container restart a run can be persisted as 'running'
        while no process exists anymore. Such a run is marked 'completed' when
        every platform log contains simulation_end, otherwise 'failed'
        (interrupted). Evidence is never touched.

        Args:
            simulation_id: Simulation ID
            record: RunRecord

        Returns:
            The (possibly updated) RunRecord
        """
        if record is None or record.legacy or record.status != RunStatus.RUNNING:
            return record
        if cls._active_runs.get(simulation_id) == record.run_id and cls.is_process_alive(simulation_id):
            return record
        if simulation_id not in cls._active_runs and cls._pid_alive(record.process_pid, record.run_id):
            return record

        run_dir = cls._run_dir(simulation_id, record.run_id)
        logs = [
            os.path.join(run_dir, p, "actions.jsonl")
            for p in ("twitter", "reddit")
            if os.path.exists(os.path.join(run_dir, p, "actions.jsonl"))
        ]
        def _ended(path: str) -> bool:
            with open(path, "r", encoding="utf-8") as f:
                return any('"simulation_end"' in line for line in f)
        all_ended = bool(logs) and all(_ended(p) for p in logs)

        state = cls.get_run_state(simulation_id, record.run_id)
        log = cls._log(simulation_id, record.run_id)
        if state is None:
            state = SimulationRunState(simulation_id=simulation_id, run_id=record.run_id)
        state.twitter_running = False
        state.reddit_running = False
        if all_ended:
            state.runner_status = RunnerStatus.COMPLETED
            state.completed_at = state.completed_at or datetime.now().isoformat()
            log.info("Run process is gone but all platforms completed: marking completed")
            cls._save_run_state(state)
            cls._finalize_run(state, RunStatus.COMPLETED)
        else:
            state.runner_status = RunnerStatus.FAILED
            state.error = "Simulation process is no longer running (interrupted, e.g. by a server restart)"
            log.warning("Run process is gone before completion: marking failed (evidence kept)")
            cls._save_run_state(state)
            cls._finalize_run(state, RunStatus.FAILED)
        return RunStore.get_run(simulation_id, record.run_id)

    @classmethod
    def _finalize_run(cls, state: SimulationRunState, status: RunStatus):
        """
        Persist the terminal lifecycle status of a run (once) and log action counts

        Args:
            state: The run's state
            status: COMPLETED / FAILED / STOPPED
        """
        if not state.run_id or state.run_id == LEGACY_RUN_ID:
            return
        record = RunStore.get_run(state.simulation_id, state.run_id)
        if record is None:
            return
        if record.status in (RunStatus.REPORTING, RunStatus.REPORT_COMPLETED):
            return
        if record.status == RunStatus.COMPLETED and status != RunStatus.COMPLETED:
            # Completed evidence is never downgraded (e.g. env closed afterwards)
            return
        if record.status == status:
            return
        counts = {
            "twitter": state.twitter_actions_count,
            "reddit": state.reddit_actions_count,
            "total": state.twitter_actions_count + state.reddit_actions_count,
        }
        RunStore.set_status(
            state.simulation_id,
            state.run_id,
            status,
            completed_at=state.completed_at or datetime.now().isoformat(),
            action_counts=counts,
            error=state.error,
        )
        log = cls._log(state.simulation_id, state.run_id)
        log.info(
            f"Simulation {status.value}: actions twitter={counts['twitter']} "
            f"reddit={counts['reddit']} total={counts['total']}, rounds={state.current_round}/{state.total_rounds}"
        )
        if status == RunStatus.COMPLETED:
            cls._mark_simulation_state(state.simulation_id, "completed")

    @classmethod
    def _mark_simulation_state(cls, simulation_id: str, status: str):
        """Mirror a run's terminal status into the simulation's state.json"""
        state_file = os.path.join(RunStore.sim_dir(simulation_id), "state.json")
        try:
            if not os.path.exists(state_file):
                return
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            state_data["status"] = status
            state_data["updated_at"] = datetime.now().isoformat()
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[{simulation_id}] Failed to update state.json: {e}")

    @classmethod
    def _monitor_simulation(
        cls,
        simulation_id: str,
        locale: str = "en",
        run_id: Optional[str] = None,
        process: Optional[subprocess.Popen] = None,
        state: Optional[SimulationRunState] = None,
        log_file: Any = None,
    ):
        """
        Monitor one run's process and parse its action logs

        The monitor only ever touches its own run (directory, state object,
        process handle); a monitor of an older run can therefore never
        overwrite the state of a newer run.
        """
        set_locale(locale)
        run_dir = cls._run_dir(simulation_id, run_id)
        log = cls._log(simulation_id, run_id)

        # New log structure: per-platform action logs
        twitter_actions_log = os.path.join(run_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(run_dir, "reddit", "actions.jsonl")

        process = process or cls._processes.get(simulation_id)
        state = state or cls.get_run_state(simulation_id, run_id)

        if not process or not state:
            return

        twitter_position = 0
        reddit_position = 0

        try:
            while process.poll() is None:  # Process still running
                # Read Twitter action log
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )

                # Read Reddit action log
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )

                # Update state
                cls._save_run_state(state)
                if state.runner_status == RunnerStatus.COMPLETED:
                    cls._finalize_run(state, RunStatus.COMPLETED)
                time.sleep(2)

            # After process ends, read logs one final time
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(
                    twitter_actions_log, twitter_position, state, "twitter"
                )
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(
                    reddit_actions_log, reddit_position, state, "reddit"
                )

            # Process ended
            exit_code = process.returncode

            if state.runner_status in (RunnerStatus.STOPPED, RunnerStatus.STOPPING):
                state.runner_status = RunnerStatus.STOPPED
                log.info(f"Process ended after stop request (exit code {exit_code})")
            elif state.runner_status == RunnerStatus.COMPLETED:
                # Completed earlier (all platforms logged simulation_end); the
                # process only stayed alive for interviews.
                log.info(f"Simulation environment process ended (exit code {exit_code})")
            elif exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at = datetime.now().isoformat()
            else:
                state.runner_status = RunnerStatus.FAILED
                # Read error info from main log file
                main_log_path = os.path.join(run_dir, "simulation.log")
                error_info = ""
                try:
                    if os.path.exists(main_log_path):
                        with open(main_log_path, "r", encoding="utf-8") as f:
                            error_info = f.read()[-2000:]  # Last 2000 chars
                except Exception:
                    pass
                state.error = f"Process exit code: {exit_code}, error: {error_info}"
                log.error(f"Simulation failed: {state.error}")

            state.twitter_running = False
            state.reddit_running = False
            cls._save_run_state(state)
            if state.runner_status == RunnerStatus.COMPLETED:
                cls._finalize_run(state, RunStatus.COMPLETED)
            elif state.runner_status == RunnerStatus.FAILED:
                cls._finalize_run(state, RunStatus.FAILED)
            elif state.runner_status == RunnerStatus.STOPPED:
                cls._finalize_run(state, RunStatus.STOPPED)

        except Exception as e:
            log.error(f"Monitor thread exception: {str(e)}")
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            cls._finalize_run(state, RunStatus.FAILED)

        finally:
            # Only release handles that still belong to this run
            if cls._processes.get(simulation_id) is process:
                cls._processes.pop(simulation_id, None)
                cls._active_runs.pop(simulation_id, None)
                cls._action_queues.pop(simulation_id, None)
                cls._graph_memory_enabled.pop(simulation_id, None)

            handle = log_file or cls._stdout_files.get(simulation_id)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                if cls._stdout_files.get(simulation_id) is handle:
                    cls._stdout_files.pop(simulation_id, None)
                    cls._stderr_files.pop(simulation_id, None)

    @classmethod
    def _read_action_log(
        cls, log_path: str, position: int, state: SimulationRunState, platform: str
    ) -> int:
        """
        Read the action log file

        Args:
            log_path: Log file path
            position: Last read position
            state: Run state object
            platform: Platform name (twitter/reddit)

        Returns:
            New read position
        """
        log = cls._log(state.simulation_id, state.run_id, platform)
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)

                            # Never count lines written by another execution
                            line_run_id = action_data.get("run_id")
                            if line_run_id and state.run_id and line_run_id != state.run_id:
                                continue

                            # Process event-type entries
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")

                                # Detect simulation_end event, mark platform as completed
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                    log.info(
                                        f"Platform simulation completed: total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}"
                                    )

                                    # Check if all enabled platforms have completed
                                    all_completed = cls._check_all_platforms_completed(
                                        state
                                    )
                                    if all_completed:
                                        state.runner_status = RunnerStatus.COMPLETED
                                        state.completed_at = datetime.now().isoformat()
                                        log.info("All platform simulations completed")

                                # Update round info (from round_end events)
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get(
                                        "simulated_hours", 0
                                    )

                                    # Update per-platform independent round counts and time
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours

                                    # Overall round count: max of both platforms
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # Overall time: max of both platforms
                                    state.simulated_hours = max(
                                        state.twitter_simulated_hours,
                                        state.reddit_simulated_hours,
                                    )

                                continue

                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get(
                                    "timestamp", datetime.now().isoformat()
                                ),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)

                            # Update round
                            if (
                                action.round_num
                                and action.round_num > state.current_round
                            ):
                                state.current_round = action.round_num

                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            log.warning(f"Failed to read action log: {log_path}, error={e}")
            return position

    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        Check whether all enabled platforms have finished simulation

        Determines whether a platform is enabled by checking whether the corresponding actions.jsonl file exists

        Returns:
            True if all enabled platforms have completed
        """
        run_dir = cls._run_dir(state.simulation_id, state.run_id)
        twitter_log = os.path.join(run_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(run_dir, "reddit", "actions.jsonl")

        # Check which platforms are enabled (by checking if files exist)
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)

        # If a platform is enabled but not completed, return False
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False

        # At least one platform is enabled and completed
        return twitter_enabled or reddit_enabled

    @classmethod
    def _terminate_process(
        cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10
    ):
        """
        Cross-platform termination of a process and its child processes

        Args:
            process: The process to terminate
            simulation_id: Simulation ID (for logging)
            timeout: Timeout in seconds to wait for the process to exit
        """
        if IS_WINDOWS:
            # Windows: use taskkill to terminate process tree
            # /F = force terminate, /T = terminate process tree (including children)
            logger.info(
                f"Terminating process tree (Windows): simulation={simulation_id}, pid={process.pid}"
            )
            try:
                # Try graceful termination first
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T"],
                    capture_output=True,
                    timeout=5,
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Force terminate
                    logger.warning(f"Process unresponsive, force terminating: {simulation_id}")
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(process.pid), "/T"],
                        capture_output=True,
                        timeout=5,
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"taskkill failed, trying terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: use process group termination
            # Since start_new_session=True, process group ID equals main process PID
            pgid = os.getpgid(process.pid)
            logger.info(f"Terminating process group (Unix): simulation={simulation_id}, pgid={pgid}")

            # Send SIGTERM to entire process group first
            os.killpg(pgid, signal.SIGTERM)

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # If still running after timeout, force SIGKILL
                logger.warning(f"Process group unresponsive to SIGTERM, force terminating: {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)

    @classmethod
    def _kill_pid_group(cls, pid: Optional[int], simulation_id: str) -> bool:
        """Best-effort termination of an untracked process group (e.g. after a server restart)"""
        if not pid or IS_WINDOWS:
            return False
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info(f"[{simulation_id}] Sent SIGTERM to untracked process group of pid={pid}")
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    @classmethod
    def terminate_run_process(cls, simulation_id: str, run_id: Optional[str] = None):
        """
        Terminate the OASIS process of a run without changing its evidence

        Used before a restart: a completed run whose process is still waiting for
        interview commands keeps its 'completed' status.
        """
        process = cls._processes.get(simulation_id)
        active_run = cls._active_runs.get(simulation_id)
        if process and process.poll() is None and (run_id is None or active_run == run_id):
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.error(f"[{simulation_id}][{active_run}] Failed to terminate process: {e}")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
            return
        state = cls.get_run_state(simulation_id, run_id)
        # Only signal an untracked PID when it verifiably belongs to this run
        # (PIDs are reused after a container restart)
        if (
            state
            and state.process_pid
            and process is None
            and cls._pid_alive(state.process_pid, state.run_id)
        ):
            cls._kill_pid_group(state.process_pid, simulation_id)

    @classmethod
    def stop_simulation(
        cls, simulation_id: str, run_id: Optional[str] = None
    ) -> SimulationRunState:
        """Stop a running run (default: the current run). Its evidence is kept."""
        run_id = RunStore.resolve_run_id(simulation_id, run_id)
        state = cls.get_run_state(simulation_id, run_id)
        if not state:
            raise ValueError(t("api.simulationNotFound", id=simulation_id))

        if state.runner_status not in [
            RunnerStatus.RUNNING,
            RunnerStatus.PAUSED,
            RunnerStatus.STARTING,
        ]:
            raise ValueError(
                t("api.simNotRunning", id=simulation_id, status=state.runner_status)
            )

        log = cls._log(simulation_id, run_id)
        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)

        # Terminate process (tracked handle first, persisted pid as fallback)
        cls.terminate_run_process(simulation_id, run_id)

        state.runner_status = RunnerStatus.STOPPED
        state.twitter_running = False
        state.reddit_running = False
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)
        cls._finalize_run(state, RunStatus.STOPPED)

        cls._graph_memory_enabled.pop(simulation_id, None)

        log.info("Simulation stopped (evidence kept)")
        return state

    @classmethod
    def _read_actions_from_file(
        cls,
        file_path: str,
        default_platform: Optional[str] = None,
        platform_filter: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> List[AgentAction]:
        """
        Read actions from a single action file

        Args:
            file_path: Action log file path
            default_platform: Default platform (used when the action record has no platform field)
            platform_filter: Filter by platform
            agent_id: Filter by Agent ID
            round_num: Filter by round
        """
        if not os.path.exists(file_path):
            return []

        actions = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Skip non-action records (e.g. simulation_start, round_start, round_end events)
                    if "event_type" in data:
                        continue

                    # Skip records without agent_id (non-Agent actions)
                    if "agent_id" not in data:
                        continue

                    # Get platform: prefer platform in record, otherwise use default
                    record_platform = data.get("platform") or default_platform or ""

                    # Filter
                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue

                    actions.append(
                        AgentAction(
                            round_num=data.get("round", 0),
                            timestamp=data.get("timestamp", ""),
                            platform=record_platform,
                            agent_id=data.get("agent_id", 0),
                            agent_name=data.get("agent_name", ""),
                            action_type=data.get("action_type", ""),
                            action_args=data.get("action_args", {}),
                            result=data.get("result"),
                            success=data.get("success", True),
                        )
                    )

                except json.JSONDecodeError:
                    continue

        return actions

    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[AgentAction]:
        """
        Get the complete action history of one run for all platforms (no pagination limit)

        Args:
            simulation_id: Simulation ID
            platform: Filter by platform (twitter/reddit)
            agent_id: Filter by Agent
            round_num: Filter by round
            run_id: Run ID (default: current run; legacy layout as fallback)

        Returns:
            Complete action list (sorted by timestamp, newest first)
        """
        sim_dir = RunStore.get_run_dir(simulation_id, run_id)
        actions = []

        # Read Twitter action file (auto-set platform to twitter based on file path)
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        if not platform or platform == "twitter":
            actions.extend(
                cls._read_actions_from_file(
                    twitter_actions_log,
                    default_platform="twitter",  # Auto-fill platform field
                    platform_filter=platform,
                    agent_id=agent_id,
                    round_num=round_num,
                )
            )

        # Read Reddit action file (auto-set platform to reddit based on file path)
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        if not platform or platform == "reddit":
            actions.extend(
                cls._read_actions_from_file(
                    reddit_actions_log,
                    default_platform="reddit",  # Auto-fill platform field
                    platform_filter=platform,
                    agent_id=agent_id,
                    round_num=round_num,
                )
            )

        # If per-platform files don't exist, try reading old single-file format
        if not actions:
            actions_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                actions_log,
                default_platform=None,  # Old format files should have platform field
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num,
            )

        # Sort by timestamp (newest first)
        actions.sort(key=lambda x: x.timestamp, reverse=True)

        return actions

    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[AgentAction]:
        """
        Get action history (with pagination)

        Args:
            simulation_id: Simulation ID
            limit: Return count limit
            offset: Offset
            platform: Filter by platform
            agent_id: Filter by Agent
            round_num: Filter by round

        Returns:
            Action list
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num,
            run_id=run_id,
        )

        # Pagination
        return actions[offset : offset + limit]

    @classmethod
    def get_timeline(
        cls,
        simulation_id: str,
        start_round: int = 0,
        end_round: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get simulation timeline (aggregated by round)

        Args:
            simulation_id: Simulation ID
            start_round: Start round
            end_round: End round

        Returns:
            Summary info for each round
        """
        actions = cls.get_actions(simulation_id, limit=10000, run_id=run_id)

        # Group by round
        rounds: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            round_num = action.round_num

            if round_num < start_round:
                continue
            if end_round is not None and round_num > end_round:
                continue

            if round_num not in rounds:
                rounds[round_num] = {
                    "round_num": round_num,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "active_agents": set(),
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }

            r = rounds[round_num]

            if action.platform == "twitter":
                r["twitter_actions"] += 1
            else:
                r["reddit_actions"] += 1

            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = (
                r["action_types"].get(action.action_type, 0) + 1
            )
            r["last_action_time"] = action.timestamp

        # Convert to list
        result = []
        for round_num in sorted(rounds.keys()):
            r = rounds[round_num]
            result.append(
                {
                    "round_num": round_num,
                    "twitter_actions": r["twitter_actions"],
                    "reddit_actions": r["reddit_actions"],
                    "total_actions": r["twitter_actions"] + r["reddit_actions"],
                    "active_agents_count": len(r["active_agents"]),
                    "active_agents": list(r["active_agents"]),
                    "action_types": r["action_types"],
                    "first_action_time": r["first_action_time"],
                    "last_action_time": r["last_action_time"],
                }
            )

        return result

    @classmethod
    def get_agent_stats(
        cls, simulation_id: str, run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get statistics for each Agent

        Returns:
            Agent statistics list
        """
        actions = cls.get_actions(simulation_id, limit=10000, run_id=run_id)

        agent_stats: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            agent_id = action.agent_id

            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": action.agent_name,
                    "total_actions": 0,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }

            stats = agent_stats[agent_id]
            stats["total_actions"] += 1

            if action.platform == "twitter":
                stats["twitter_actions"] += 1
            else:
                stats["reddit_actions"] += 1

            stats["action_types"][action.action_type] = (
                stats["action_types"].get(action.action_type, 0) + 1
            )
            stats["last_action_time"] = action.timestamp

        # Sort by total action count
        result = sorted(
            agent_stats.values(), key=lambda x: x["total_actions"], reverse=True
        )

        return result

    # Flag to prevent duplicate cleanup
    _cleanup_done = False

    @classmethod
    def cleanup_all_simulations(cls):
        """
        Clean up all running simulation processes

        Called when the server shuts down to ensure all child processes are terminated
        """
        # Prevent duplicate cleanup
        if cls._cleanup_done:
            return
        cls._cleanup_done = True

        # Check if there is content to clean up (avoid logging for empty processes)
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)

        if not has_processes and not has_updaters:
            return  # No content to clean up, return silently

        logger.info("Cleaning up all simulation processes...")

        # Graph memory updater removed (was Zep-based)
        cls._graph_memory_enabled.clear()

        # Copy dict to avoid modification during iteration
        processes = list(cls._processes.items())

        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # Process still running
                    logger.info(f"Terminating simulation process: {simulation_id}, pid={process.pid}")

                    try:
                        # Use cross-platform process termination
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # Process may no longer exist, try direct termination
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()

                    # Update run_state.json of the active run (completed runs stay completed)
                    run_id = cls._active_runs.get(simulation_id)
                    state = cls.get_run_state(simulation_id, run_id)
                    if state and state.runner_status != RunnerStatus.COMPLETED:
                        state.runner_status = RunnerStatus.STOPPED
                        state.twitter_running = False
                        state.reddit_running = False
                        state.completed_at = datetime.now().isoformat()
                        state.error = "Server shutdown, simulation terminated"
                        cls._save_run_state(state)
                        cls._finalize_run(state, RunStatus.STOPPED)
                    elif state:
                        continue

                    # Also update state.json, set status to stopped
                    try:
                        sim_dir = RunStore.sim_dir(simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"Attempting to update state.json: {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, "r", encoding="utf-8") as f:
                                state_data = json.load(f)
                            state_data["status"] = "stopped"
                            state_data["updated_at"] = datetime.now().isoformat()
                            with open(state_file, "w", encoding="utf-8") as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(
                                f"Updated state.json status to stopped: {simulation_id}"
                            )
                        else:
                            logger.warning(f"state.json does not exist: {state_file}")
                    except Exception as state_err:
                        logger.warning(
                            f"Failed to update state.json: {simulation_id}, error={state_err}"
                        )

            except Exception as e:
                logger.error(f"Failed to clean up process: {simulation_id}, error={e}")

        # Clean up file handles
        for simulation_id, file_handle in list(cls._stdout_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stdout_files.clear()

        for simulation_id, file_handle in list(cls._stderr_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stderr_files.clear()

        # Clean up in-memory state
        cls._processes.clear()
        cls._active_runs.clear()
        cls._action_queues.clear()

        logger.info("Simulation process cleanup complete")

    @classmethod
    def register_cleanup(cls):
        """
        Register the cleanup function

        Called when the Flask app starts, to ensure all simulation processes are cleaned up when the server shuts down
        """
        global _cleanup_registered

        if _cleanup_registered:
            return

        # In Flask debug mode, only register cleanup in the reloader child process (the one actually running the app)
        # WERKZEUG_RUN_MAIN=true indicates the reloader child process
        # If not in debug mode, this env var is absent, so register anyway
        is_reloader_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        is_debug_mode = (
            os.environ.get("FLASK_DEBUG") == "1"
            or os.environ.get("WERKZEUG_RUN_MAIN") is not None
        )

        # In debug mode, only register in reloader child; in non-debug mode, always register
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # Mark as registered to prevent child process from trying again
            return

        # Save original signal handlers
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP only exists on Unix systems (macOS/Linux), not on Windows
        original_sighup = None
        has_sighup = hasattr(signal, "SIGHUP")
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)

        def cleanup_handler(signum=None, frame=None):
            """Signal handler: clean up simulation processes first, then call the original handler"""
            # Only log if there are processes to clean up
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"Received signal {signum}, starting cleanup...")
            cls.cleanup_all_simulations()

            # Call original signal handler to let Flask exit normally
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP: sent when terminal closes
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # Default behavior: normal exit
                    sys.exit(0)
            else:
                # If original handler is not callable (e.g. SIG_DFL), use default behavior
                raise KeyboardInterrupt

        # Register atexit handler (as fallback)
        atexit.register(cls.cleanup_all_simulations)

        # Register signal handlers (main thread only)
        try:
            # SIGTERM: default signal from kill command
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT: Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP: terminal close (Unix only)
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # Not in main thread, can only use atexit
            logger.warning("Cannot register signal handlers (not in main thread), using atexit only")

        _cleanup_registered = True

    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        Get a list of IDs of all currently running simulations
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running

    # ============== Interview functionality ==============

    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        Check whether the simulation environment is alive (can receive Interview commands)

        Args:
            simulation_id: Simulation ID

        Returns:
            True means the environment is alive; False means the environment has been shut down
        """
        sim_dir = RunStore.get_run_dir(simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Get detailed status info of the simulation environment

        Args:
            simulation_id: Simulation ID

        Returns:
            Status detail dict, containing status, twitter_available, reddit_available, timestamp
        """
        sim_dir = RunStore.get_run_dir(simulation_id)
        status_file = os.path.join(sim_dir, "env_status.json")

        default_status = {
            "status": "stopped",
            "twitter_available": False,
            "reddit_available": False,
            "timestamp": None,
        }

        if not os.path.exists(status_file):
            return default_status

        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
            return {
                "status": status.get("status", "stopped"),
                "twitter_available": status.get("twitter_available", False),
                "reddit_available": status.get("reddit_available", False),
                "timestamp": status.get("timestamp"),
            }
        except (json.JSONDecodeError, OSError):
            return default_status

    @classmethod
    def interview_agent(
        cls,
        simulation_id: str,
        agent_id: int,
        prompt: str,
        platform: Optional[str] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Interview a single Agent

        Args:
            simulation_id: Simulation ID
            agent_id: Agent ID
            prompt: Interview question
            platform: Specify a platform (optional)
                - "twitter": only interview the Twitter platform
                - "reddit": only interview the Reddit platform
                - None: in dual-platform simulations, interview both platforms at once and return an integrated result
            timeout: Timeout in seconds

        Returns:
            Interview result dict

        Raises:
            ValueError: Simulation does not exist or environment is not running
            TimeoutError: Timed out waiting for a response
        """
        sim_dir = RunStore.get_run_dir(simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(t("api.simulationNotFound", id=simulation_id))

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(t("api.envNotRunningInterview", id=simulation_id))

        logger.info(
            f"Sending Interview command: simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}"
        )

        response = ipc_client.send_interview(
            agent_id=agent_id, prompt=prompt, platform=platform, timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "agent_id": agent_id,
                "prompt": prompt,
                "result": response.result,
                "timestamp": response.timestamp,
            }
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "prompt": prompt,
                "error": response.error,
                "timestamp": response.timestamp,
            }

    @classmethod
    def interview_agents_batch(
        cls,
        simulation_id: str,
        interviews: List[Dict[str, Any]],
        platform: Optional[str] = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Batch interview multiple Agents

        Args:
            simulation_id: Simulation ID
            interviews: Interview list, each element contains {"agent_id": int, "prompt": str, "platform": str (optional)}
            platform: Default platform (optional, overridden by each interview item's platform)
                - "twitter": default to only interviewing the Twitter platform
                - "reddit": default to only interviewing the Reddit platform
                - None: in dual-platform simulations, each Agent is interviewed on both platforms at once
            timeout: Timeout in seconds

        Returns:
            Batch interview result dict

        Raises:
            ValueError: Simulation does not exist or environment is not running
            TimeoutError: Timed out waiting for a response
        """
        sim_dir = RunStore.get_run_dir(simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(t("api.simulationNotFound", id=simulation_id))

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(t("api.envNotRunningInterview", id=simulation_id))

        logger.info(
            f"Sending batch Interview command: simulation_id={simulation_id}, count={len(interviews)}, platform={platform}"
        )

        response = ipc_client.send_batch_interview(
            interviews=interviews, platform=platform, timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "interviews_count": len(interviews),
                "result": response.result,
                "timestamp": response.timestamp,
            }
        else:
            return {
                "success": False,
                "interviews_count": len(interviews),
                "error": response.error,
                "timestamp": response.timestamp,
            }

    @classmethod
    def interview_all_agents(
        cls,
        simulation_id: str,
        prompt: str,
        platform: Optional[str] = None,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Interview all Agents (global interview)

        Uses the same question to interview all Agents in the simulation

        Args:
            simulation_id: Simulation ID
            prompt: Interview question (all Agents use the same question)
            platform: Specify a platform (optional)
                - "twitter": only interview the Twitter platform
                - "reddit": only interview the Reddit platform
                - None: in dual-platform simulations, each Agent is interviewed on both platforms at once
            timeout: Timeout in seconds

        Returns:
            Global interview result dict
        """
        sim_dir = RunStore.get_run_dir(simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(t("api.simulationNotFound", id=simulation_id))

        # Get all Agent info from config file
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(t("api.simConfigNotFound", id=simulation_id))

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(t("api.simConfigNoAgents", id=simulation_id))

        # Build batch interview list
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({"agent_id": agent_id, "prompt": prompt})

        logger.info(
            f"Sending global Interview command: simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}"
        )

        return cls.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews,
            platform=platform,
            timeout=timeout,
        )

    @classmethod
    def close_simulation_env(
        cls, simulation_id: str, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Close the simulation environment (rather than stopping the simulation process)

        Sends a close-environment command to the simulation so that it gracefully exits command-waiting mode

        Args:
            simulation_id: Simulation ID
            timeout: Timeout in seconds

        Returns:
            Operation result dict
        """
        sim_dir = RunStore.get_run_dir(simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(t("api.simulationNotFound", id=simulation_id))

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            return {"success": True, "message": t("api.envAlreadyClosed")}

        logger.info(f"Sending close environment command: simulation_id={simulation_id}")

        try:
            response = ipc_client.send_close_env(timeout=timeout)

            return {
                "success": response.status.value == "completed",
                "message": t("api.envCloseCommandSent"),
                "result": response.result,
                "timestamp": response.timestamp,
            }
        except TimeoutError:
            # Timeout may be because environment is shutting down
            return {
                "success": True,
                "message": t("api.envCloseCommandSentTimeout"),
            }

    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get Interview history from a single database"""
        import sqlite3

        if not os.path.exists(db_path):
            return []

        results = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if agent_id is not None:
                cursor.execute(
                    """
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview' AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (agent_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview'
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            for user_id, info_json, created_at in cursor.fetchall():
                try:
                    info = json.loads(info_json) if info_json else {}
                except json.JSONDecodeError:
                    info = {"raw": info_json}

                results.append(
                    {
                        "agent_id": user_id,
                        "response": info.get("response", info),
                        "prompt": info.get("prompt", ""),
                        "timestamp": created_at,
                        "platform": platform_name,
                    }
                )

            conn.close()

        except Exception as e:
            logger.error(f"Failed to read Interview history ({platform_name}): {e}")

        return results

    @classmethod
    def get_interview_history(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get Interview history records (read from the database)

        Args:
            simulation_id: Simulation ID
            platform: Platform type (reddit/twitter/None)
                - "reddit": only get the Reddit platform history
                - "twitter": only get the Twitter platform history
                - None: get all history for both platforms
            agent_id: Specify an Agent ID (optional, only get that Agent's history)
            limit: Return count limit per platform

        Returns:
            Interview history record list
        """
        sim_dir = RunStore.get_run_dir(simulation_id)

        results = []

        # Determine which platforms to query
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # When platform not specified, query both platforms
            platforms = ["twitter", "reddit"]

        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path, platform_name=p, agent_id=agent_id, limit=limit
            )
            results.extend(platform_results)

        # Sort by time descending
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # If querying multiple platforms, limit total count
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]

        return results
