"""
Shared fixtures: isolated uploads directory, prepared simulations and fakes
"""

import csv
import json
import os
import threading

import pytest

from app.config import Config
from app.models.project import ProjectManager
from app.services import local_graph_store as graph_store_module
from app.services import simulation_runner as runner_module
from app.services.report_agent import ReportManager
from app.services.simulation_manager import SimulationManager
from app.services.simulation_runner import SimulationRunner


@pytest.fixture(autouse=True)
def offline_llm_environment(monkeypatch):
    """Tests never call a real LLM or embedding endpoint"""
    from app.services import graph_tools as graph_tools_module
    from app.services import local_graph_extractor as extractor_module
    from app.utils import embeddings as embeddings_module

    monkeypatch.setattr(Config, "LLM_API_KEY", Config.LLM_API_KEY or "test-key")
    for module in (embeddings_module, extractor_module, graph_tools_module):
        monkeypatch.setattr(module, "embed", lambda *args, **kwargs: None)


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Point every storage location at a temporary uploads directory"""
    root = tmp_path / "uploads"
    sims = root / "simulations"
    sims.mkdir(parents=True)
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(root))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(sims))
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(sims))
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(sims))
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(root / "projects"))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(root / "reports"))
    monkeypatch.setattr(graph_store_module, "GRAPH_DIR", str(root / "graphs"))
    for attr in ("_run_states", "_processes", "_active_runs", "_action_queues",
                 "_monitor_threads", "_stdout_files", "_stderr_files", "_graph_memory_enabled"):
        monkeypatch.setattr(SimulationRunner, attr, {})
    return root


@pytest.fixture
def client(uploads):
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


AGENTS = [
    # Two entities for the same consumer persona: "C01" and "C01 Dutch first-time family"
    {"agent_id": 0, "entity_name": "C01", "entity_type": "Consumer"},
    {"agent_id": 1, "entity_name": "C01 Dutch first-time family", "entity_type": "Consumer"},
    {"agent_id": 2, "entity_name": "C02 German camper couple", "entity_type": "Consumer"},
    {"agent_id": 3, "entity_name": "Campsite Owner Association", "entity_type": "Organization"},
]

BRIEF = """Assess the product test for personas C01-C03.

For each persona report:
- starting workflow
- pricing exposure
- checkout outcome

Hypotheses:
H1: Families are willing to pay for the premium plan.
H2: Couples prefer the free alternative.
"""


def make_project(project_id="proj_test", requirement=BRIEF, documents=None):
    """Create a project with per-file source texts (new layout)"""
    documents = documents or [("product.md", "# Product\nThe premium plan costs 4.99 per month.")]
    project = ProjectManager.create_project(name="Test")
    project_dir = os.path.join(ProjectManager.PROJECTS_DIR, project.project_id)
    if project_id != project.project_id:
        os.rename(project_dir, os.path.join(ProjectManager.PROJECTS_DIR, project_id))
        project.project_id = project_id
    project.simulation_requirement = requirement
    for i, (name, text) in enumerate(documents, start=1):
        text_file = ProjectManager.save_source_text(project_id, i, name, text)
        project.files.append({
            "filename": name,
            "saved_filename": None,
            "extension": os.path.splitext(name)[1],
            "size": len(text),
            "text_file": text_file,
            "text_length": len(text),
            "extraction_status": "ok",
        })
    ProjectManager.save_project(project)
    return project


def make_prepared_simulation(simulation_id="sim_test", project_id="proj_test", status="ready", agents=None):
    """Write a prepared simulation (state, config, profiles) like the prepare step does"""
    agents = agents or AGENTS
    sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({
            "simulation_id": simulation_id,
            "project_id": project_id,
            "graph_id": "graph_test",
            "status": status,
            "config_generated": True,
        }, f)
    with open(os.path.join(sim_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "simulation_id": simulation_id,
            "time_config": {"total_simulation_hours": 2, "minutes_per_round": 60},
            "agent_configs": agents,
            "event_config": {"initial_posts": []},
        }, f)
    with open(os.path.join(sim_dir, "reddit_profiles.json"), "w", encoding="utf-8") as f:
        json.dump([
            {"user_id": a["agent_id"], "name": a["entity_name"], "username": f"user{a['agent_id']}",
             "bio": "bio", "persona": f"Persona of {a['entity_name']}"}
            for a in agents
        ], f)
    with open(os.path.join(sim_dir, "twitter_profiles.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "name", "username", "user_char", "description"])
        for a in agents:
            writer.writerow([a["agent_id"], a["entity_name"], f"user{a['agent_id']}", "char", "desc"])
    return sim_dir


def select_entities(simulation_id, roles=None, active=None):
    """
    Save a confirmed agent selection (graph entities are never agents by default)

    Args:
        simulation_id: Simulation ID (its graph must exist)
        roles: Optional {entity name: (simulation_role, evidence_role)}
        active: Optional set of entity names to activate (default: every entity)
    """
    from app.services.agent_selection import AgentSelectionStore, apply_selection_update
    from app.services.entity_reader import EntityReader

    state = SimulationManager().get_simulation(simulation_id)
    entities = EntityReader().filter_defined_entities(state.graph_id, enrich_with_edges=False).entities
    updates = []
    for e in entities:
        item = {"entity_uuid": e.uuid, "active": active is None or e.name in active}
        if roles and e.name in roles:
            item["simulation_role"], item["evidence_role"] = roles[e.name]
        updates.append(item)
    document = apply_selection_update(
        entities, {"entities": updates, "confirmed": True}, AgentSelectionStore.load(simulation_id), state.graph_id
    )
    AgentSelectionStore.save(simulation_id, document)
    return document


def default_actions(platform):
    """Actions a fake simulation run writes"""
    if platform == "twitter":
        return [
            {"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST",
             "action_args": {"content": "Looking for a family campsite planner"}},
            {"round": 2, "agent_id": 1, "agent_name": "C01 Dutch first-time family", "action_type": "CREATE_POST",
             "action_args": {"content": "4.99 per month is too expensive for us"}},
        ]
    return [
        {"round": 1, "agent_id": 2, "agent_name": "C02 German camper couple", "action_type": "CREATE_COMMENT",
         "action_args": {"content": "We will stick with the free app", "post_content": "Premium plan launched"}},
    ]


def write_actions(run_dir, platform, actions, run_id=None, completed=True, start_event=True):
    """Append a full execution (start, actions, end) to a platform action log"""
    os.makedirs(os.path.join(run_dir, platform), exist_ok=True)
    path = os.path.join(run_dir, platform, "actions.jsonl")
    lines = []
    if start_event:
        lines.append({"event_type": "simulation_start", "platform": platform, "timestamp": "2026-01-01T10:00:00"})
    for i, a in enumerate(actions):
        lines.append({"timestamp": f"2026-01-01T10:00:{i:02d}", "success": True, **a})
    if completed:
        lines.append({"event_type": "simulation_end", "platform": platform, "total_rounds": 2,
                      "total_actions": len(actions), "timestamp": "2026-01-01T11:00:00"})
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            if run_id:
                line = {**line, "run_id": run_id}
            f.write(json.dumps(line) + "\n")
    return path


class FakePopen:
    """Stands in for the OASIS subprocess; writes action logs into its cwd like the real script"""

    instances = []
    finish_immediately = True
    # Optional {platform: [actions]} written instead of default_actions()
    actions = None

    def __init__(self, cmd, cwd=None, env=None, **kwargs):
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or {}
        self.pid = 4242 + len(FakePopen.instances)
        self.returncode = None
        self._done = threading.Event()
        FakePopen.instances.append(self)
        run_id = self.env.get("MIROFISH_RUN_ID")
        if FakePopen.finish_immediately:
            for platform in ("twitter", "reddit"):
                actions = (FakePopen.actions or {}).get(platform) if FakePopen.actions else default_actions(platform)
                write_actions(cwd, platform, actions or [], run_id=run_id)
            self.finish(0)

    def finish(self, code=0):
        self.returncode = code
        self._done.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.returncode

    def terminate(self):
        self.finish(-15)

    def kill(self):
        self.finish(-9)


@pytest.fixture
def fake_popen(monkeypatch):
    FakePopen.instances = []
    FakePopen.finish_immediately = True
    FakePopen.actions = None
    monkeypatch.setattr(runner_module.subprocess, "Popen", FakePopen)
    real_sleep = runner_module.time.sleep
    monkeypatch.setattr(runner_module.time, "sleep", lambda _s: real_sleep(0.01))
    monkeypatch.setattr(
        SimulationRunner, "_terminate_process",
        classmethod(lambda cls, process, simulation_id, timeout=10: process.terminate()),
    )
    return FakePopen


def wait_for_monitor(simulation_id, timeout=10):
    """Wait until the run's monitor thread has finished"""
    thread = SimulationRunner._monitor_threads.get(simulation_id)
    if thread:
        thread.join(timeout)
        assert not thread.is_alive(), "monitor thread did not finish"
