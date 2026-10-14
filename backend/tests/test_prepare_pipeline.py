import json
import os

import pytest

from conftest import make_project

from app.api.simulation import _check_simulation_prepared
from app.services import simulation_manager as manager_module
from app.services.local_graph_store import LocalGraphStore
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.run_store import RunStatus, RunStore
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.services.simulation_manager import SimulationManager, SimulationStatus

ENTITIES = [
    ("C01 Dutch first-time family", "Consumer", "Family with two young kids"),
    ("C02 German camper couple", "Consumer", "Retired couple"),
    ("Camping De Linde", "CampsiteOwner", "Family campsite"),
]


def make_graph(graph_id="graph_test", entities=ENTITIES):
    store = LocalGraphStore(graph_id)
    store.set_meta("test", "", {}, "2026-01-01")
    for name, etype, summary in entities:
        store.add_node(name=name, labels=["Entity", etype], summary=summary)
    return graph_id


@pytest.fixture
def offline_config(monkeypatch):
    """Config generation without an LLM: every LLM step uses its built-in defaults"""
    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry", lambda self, prompt, system_prompt: {})


def _prepare(manager, simulation_id):
    return manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="Assess C01-C02",
        document_text="Kamptribe product text",
        use_llm_for_profiles=False,
        parallel_profile_count=1,
    )


def _new_simulation():
    project = make_project()
    make_graph()
    manager = SimulationManager()
    state = manager.create_simulation(project_id=project.project_id, graph_id="graph_test")
    return manager, state.simulation_id


def test_prepare_generates_deterministic_profiles_and_config(uploads, offline_config):
    manager, sim_id = _new_simulation()

    state = _prepare(manager, sim_id)

    assert state.status == SimulationStatus.READY
    assert state.profiles_generated and state.config_generated
    sim_dir = RunStore.sim_dir(sim_id)
    reddit = json.load(open(os.path.join(sim_dir, "reddit_profiles.json")))
    config = json.load(open(os.path.join(sim_dir, "simulation_config.json")))
    assert [p["name"] for p in reddit] == [e[0] for e in ENTITIES]
    assert [p["user_id"] for p in reddit] == [0, 1, 2]
    assert [a["entity_name"] for a in config["agent_configs"]] == [e[0] for e in ENTITIES]
    assert [a["agent_id"] for a in config["agent_configs"]] == [0, 1, 2]
    assert all(p["country"] != "China" for p in reddit)
    assert _check_simulation_prepared(sim_id)[0] is True
    assert manager.get_profiles(sim_id, platform="twitter")[0]["name"] == ENTITIES[0][0]

    # Preparing again yields the same identities (usernames are no longer random)
    usernames = [p["username"] for p in reddit]
    _prepare(SimulationManager(), sim_id)
    again = json.load(open(os.path.join(sim_dir, "reddit_profiles.json")))
    assert [p["username"] for p in again] == usernames


def test_persona_generation_failure_leaves_no_startable_simulation(uploads, offline_config, client, monkeypatch):
    manager, sim_id = _new_simulation()

    def broken(self, **kwargs):
        raise RuntimeError("persona LLM failed")

    monkeypatch.setattr(OasisProfileGenerator, "generate_profiles_from_entities", broken)
    with pytest.raises(RuntimeError, match="persona LLM failed"):
        _prepare(manager, sim_id)

    state = json.load(open(os.path.join(RunStore.sim_dir(sim_id), "state.json")))
    assert state["status"] == "failed"
    assert state["profiles_generated"] is False and state["config_generated"] is False
    assert "persona LLM failed" in state["error"]
    assert _check_simulation_prepared(sim_id)[0] is False
    assert client.post("/api/simulation/start", json={"simulation_id": sim_id}).status_code == 400
    assert RunStore.list_runs(sim_id) == []


def test_config_failure_after_a_previous_run_keeps_the_run_and_blocks_start(uploads, offline_config, client, monkeypatch):
    manager, sim_id = _new_simulation()
    _prepare(manager, sim_id)
    record = RunStore.create_run(sim_id)
    RunStore.set_status(sim_id, record.run_id, RunStatus.COMPLETED)
    run_files = sorted(os.listdir(RunStore.run_dir(sim_id, record.run_id)))

    def broken(self, **kwargs):
        raise RuntimeError("config LLM failed")

    monkeypatch.setattr(SimulationConfigGenerator, "generate_config", broken)
    with pytest.raises(RuntimeError, match="config LLM failed"):
        _prepare(SimulationManager(), sim_id)

    state = json.load(open(os.path.join(RunStore.sim_dir(sim_id), "state.json")))
    assert state["status"] == "failed"
    assert state["config_generated"] is False
    # The stale config of the earlier preparation is still on disk but no longer startable
    assert os.path.exists(os.path.join(RunStore.sim_dir(sim_id), "simulation_config.json"))
    assert _check_simulation_prepared(sim_id)[0] is False
    realtime = client.get(f"/api/simulation/{sim_id}/config/realtime").json["data"]
    assert realtime["status"] == "failed" and realtime["generation_stage"] == "failed"
    assert realtime["config"] is None
    # The earlier run is untouched
    assert RunStore.get_run(sim_id, record.run_id).status == RunStatus.COMPLETED
    assert sorted(os.listdir(RunStore.run_dir(sim_id, record.run_id))) == run_files
    started = client.post("/api/simulation/start", json={"simulation_id": sim_id, "new_run": True})
    assert started.status_code == 400


def test_empty_graph_fails_prepare_with_a_clear_error(uploads, client, monkeypatch):
    project = make_project()
    LocalGraphStore("graph_empty").set_meta("empty", "", {}, "2026-01-01")
    manager = SimulationManager()
    sim_id = manager.create_simulation(project_id=project.project_id, graph_id="graph_empty").simulation_id

    with pytest.raises(ValueError, match="No matching entities found"):
        _prepare(manager, sim_id)

    profiles = client.get(f"/api/simulation/{sim_id}/profiles/realtime").json["data"]
    assert profiles["status"] == "failed"
    assert "No matching entities" in profiles["error"]
    assert profiles["is_generating"] is False


def test_prepare_api_marks_task_failed_for_an_empty_graph(uploads, client):
    import time

    project = make_project()
    LocalGraphStore("graph_empty").set_meta("empty", "", {}, "2026-01-01")
    sim_id = SimulationManager().create_simulation(project_id=project.project_id, graph_id="graph_empty").simulation_id

    response = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False})
    task_id = response.json["data"]["task_id"]
    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "failed"
    assert _check_simulation_prepared(sim_id)[0] is False
