import json
import os

from conftest import (
    FakePopen,
    make_prepared_simulation,
    make_project,
    wait_for_monitor,
    write_actions,
)

from app.services.run_evidence import RunEvidence
from app.services.run_store import LEGACY_RUN_ID, RunStatus, RunStore
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunState,
    SimulationRunner,
)


def _start(client, **extra):
    response = client.post("/api/simulation/start", json={"simulation_id": "sim_test", **extra})
    assert response.status_code == 200, response.json
    return response.json["data"]


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _completed_run(client):
    data = _start(client)
    assert data["started"] is True
    wait_for_monitor("sim_test")
    return data["run_id"]


def test_start_executes_inside_its_own_run_directory(uploads, client, fake_popen):
    make_project()
    make_prepared_simulation()

    data = _start(client)
    wait_for_monitor("sim_test")

    run_id = data["run_id"]
    run_dir = RunStore.run_dir("sim_test", run_id)
    assert run_dir.endswith(os.path.join("sim_test", "runs", run_id))
    assert run_dir.startswith(str(uploads))  # persistent uploads volume

    process = FakePopen.instances[0]
    assert process.cwd == run_dir
    assert process.cmd[process.cmd.index("--config") + 1] == os.path.join(run_dir, "simulation_config.json")
    assert process.env["MIROFISH_RUN_ID"] == run_id

    for name in ("run.json", "simulation_config.json", "reddit_profiles.json", "twitter_profiles.csv",
                 "personas.json", "sources/manifest.json", "assessment/brief.md",
                 "twitter/actions.jsonl", "reddit/actions.jsonl"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    # Nothing is written into the simulation (configuration) directory itself
    assert not os.path.exists(os.path.join(RunStore.sim_dir("sim_test"), "twitter"))

    record = RunStore.get_run("sim_test", run_id)
    assert record.status == RunStatus.COMPLETED
    assert record.action_counts["twitter"] == 2
    assert record.action_counts["reddit"] == 1
    with open(os.path.join(RunStore.sim_dir("sim_test"), "state.json")) as f:
        assert json.load(f)["status"] == "completed"


def test_page_reload_does_not_start_a_new_simulation(uploads, client, fake_popen):
    make_project()
    make_prepared_simulation()
    run_id = _completed_run(client)
    actions_path = os.path.join(RunStore.run_dir("sim_test", run_id), "twitter", "actions.jsonl")
    before = _read(actions_path)

    # What an old cached frontend sent on every Step 3 mount
    again = _start(client, force=True)

    assert again["started"] is False
    assert again["run_exists"] is True
    assert again["run_id"] == run_id
    assert again["run_status"] == "completed"
    assert len(FakePopen.instances) == 1
    assert len(RunStore.list_runs("sim_test")) == 1
    assert _read(actions_path) == before


def test_completed_state_survives_refresh_and_server_restart(uploads, client, fake_popen):
    make_project()
    make_prepared_simulation()
    run_id = _completed_run(client)

    # Simulate a backend restart: in-memory state is gone
    SimulationRunner._run_states.clear()
    SimulationRunner._processes.clear()
    SimulationRunner._active_runs.clear()

    status = client.get("/api/simulation/sim_test/run-status").json["data"]
    assert status["run_id"] == run_id
    assert status["run_status"] == "completed"
    assert status["runner_status"] == "completed"
    assert status["total_actions_count"] == 3


def test_start_while_running_is_idempotent(uploads, client, fake_popen):
    make_project()
    make_prepared_simulation()
    FakePopen.finish_immediately = False

    first = _start(client)
    try:
        second = _start(client)
        assert second["started"] is False
        assert second["already_running"] is True
        assert second["run_id"] == first["run_id"]
        assert len(FakePopen.instances) == 1
    finally:
        FakePopen.instances[0].finish(0)
        wait_for_monitor("sim_test")


def test_restart_creates_new_run_and_keeps_old_run(uploads, client, fake_popen):
    make_project()
    make_prepared_simulation()
    old_run = _completed_run(client)
    old_dir = RunStore.run_dir("sim_test", old_run)
    old_files = {
        name: _read(os.path.join(old_dir, name))
        for name in ("twitter/actions.jsonl", "reddit/actions.jsonl", "simulation_config.json")
    }

    restarted = _start(client, new_run=True)
    wait_for_monitor("sim_test")

    assert restarted["started"] is True
    assert restarted["run_id"] != old_run
    assert restarted["previous_run_id"] == old_run
    for name, content in old_files.items():
        assert _read(os.path.join(old_dir, name)) == content, name
    assert RunStore.get_run("sim_test", old_run).status == RunStatus.COMPLETED

    runs = client.get("/api/simulation/sim_test/runs").json["data"]
    assert [r["run_id"] for r in runs["runs"]] == [old_run, restarted["run_id"]]
    assert runs["current_run_id"] == restarted["run_id"]


def test_action_logs_of_different_runs_are_never_mixed(uploads, client, fake_popen):
    make_project()
    make_prepared_simulation()
    first = _completed_run(client)
    second = _start(client, new_run=True)["run_id"]
    wait_for_monitor("sim_test")

    for run_id in (first, second):
        run_dir = RunStore.run_dir("sim_test", run_id)
        for platform in ("twitter", "reddit"):
            with open(os.path.join(run_dir, platform, "actions.jsonl")) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            assert {line["run_id"] for line in lines} == {run_id}
            assert sum(1 for line in lines if line.get("event_type") == "simulation_start") == 1
        evidence = RunEvidence.load("sim_test", run_id)
        assert len(evidence.actions) == 3
        assert evidence.integrity["warnings"] == []


def test_monitor_ignores_lines_of_other_runs(uploads, tmp_path):
    log = tmp_path / "actions.jsonl"
    log.write_text(
        json.dumps({"agent_id": 1, "agent_name": "a", "action_type": "CREATE_POST", "run_id": "run_old"}) + "\n"
        + json.dumps({"agent_id": 1, "agent_name": "a", "action_type": "CREATE_POST", "run_id": "run_new"}) + "\n"
    )
    state = SimulationRunState(simulation_id="sim_test", run_id="run_new")
    SimulationRunner._read_action_log(str(log), 0, state, "twitter")
    assert state.twitter_actions_count == 1


def test_stale_monitor_does_not_release_the_new_runs_process(uploads):
    old_process, new_process = FakePopen.__new__(FakePopen), FakePopen.__new__(FakePopen)
    old_process.returncode = 0
    new_process.returncode = None
    SimulationRunner._processes["sim_test"] = new_process
    SimulationRunner._active_runs["sim_test"] = "run_new"
    old_state = SimulationRunState(simulation_id="sim_test", run_id="run_old", runner_status=RunnerStatus.RUNNING)

    SimulationRunner._monitor_simulation("sim_test", "en", "run_old", old_process, old_state, None)

    assert SimulationRunner._processes["sim_test"] is new_process
    assert SimulationRunner._active_runs["sim_test"] == "run_new"


def test_legacy_simulation_folder_is_read_only(uploads, client, fake_popen):
    make_project()
    sim_dir = make_prepared_simulation(status="running")
    # Old layout: artifacts in the simulation dir, two executions appended to one log
    with open(os.path.join(sim_dir, "run_state.json"), "w") as f:
        json.dump({"runner_status": "completed", "twitter_actions_count": 3}, f)
    write_actions(sim_dir, "twitter", [
        {"round": 0, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST", "action_args": {"content": "old run"}},
    ])
    log_path = write_actions(sim_dir, "twitter", [
        {"round": 0, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST", "action_args": {"content": "rerun"}},
    ])
    before = _read(log_path)

    status = client.get("/api/simulation/sim_test/run-status").json["data"]
    assert status["run_id"] == LEGACY_RUN_ID
    assert status["run_status"] == "completed"

    evidence = RunEvidence.load("sim_test")
    assert [a["content"] for a in evidence.actions] == ["rerun"]
    assert "2 executions" in evidence.integrity["warnings"][0]

    plain = _start(client)
    assert plain["started"] is False and plain["run_id"] == LEGACY_RUN_ID

    restarted = _start(client, new_run=True)
    wait_for_monitor("sim_test")
    assert restarted["run_id"] != LEGACY_RUN_ID
    assert _read(log_path) == before
    assert os.path.exists(os.path.join(sim_dir, "run_state.json"))


def test_interrupted_run_is_reconciled_without_touching_evidence(uploads, client):
    make_project()
    make_prepared_simulation()
    record = RunStore.create_run("sim_test")
    RunStore.set_status("sim_test", record.run_id, RunStatus.RUNNING)
    run_dir = RunStore.run_dir("sim_test", record.run_id)
    path = write_actions(run_dir, "twitter", [
        {"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST", "action_args": {"content": "x"}},
    ], run_id=record.run_id, completed=False)
    before = _read(path)

    status = client.get("/api/simulation/sim_test/run-status").json["data"]

    assert status["run_status"] == "failed"
    assert _read(path) == before


def test_simulation_script_logger_tags_every_line_with_run_id(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setenv("MIROFISH_RUN_ID", "run_abc")
    monkeypatch.setenv("MIROFISH_SIMULATION_ID", "sim_test")
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "action_logger.py")
    spec = importlib.util.spec_from_file_location("action_logger_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    logger = module.PlatformActionLogger("twitter", str(tmp_path))
    logger.log_simulation_start({"time_config": {}, "agent_configs": []})
    logger.log_action(1, 0, "C01", "CREATE_POST", {"content": "hi"})
    logger.log_simulation_end(1, 1)

    lines = [json.loads(line) for line in (tmp_path / "twitter" / "actions.jsonl").read_text().splitlines()]
    assert [line["run_id"] for line in lines] == ["run_abc"] * 3
