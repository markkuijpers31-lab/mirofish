"""
End-to-end validation on a synthetic product-fit project (no paid services):
16 selected agents out of 26 graph entities, private instructions, action origins
and run immutability
"""

import csv
import hashlib
import json
import os
import time

import pytest

from conftest import FakePopen, make_project, wait_for_monitor, write_actions

from app.services.assessment_spec import AssessmentSpec
from app.services.entity_reader import EntityReader
from app.services.graph_tools import GraphToolsService
from app.services.local_graph_store import LocalGraphStore
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.report_evidence_sections import StructuredReportBuilder
from app.services.run_evidence import RunEvidence
from app.services.run_store import RunStore
from app.services.simulation_config_generator import SimulationConfigGenerator

GRAPH_ID = "graph_validation"

CONSUMERS = [(f"C{i:02d}", "Consumer", f"Research consumer {i} planning a camping trip") for i in range(1, 11)]
EXISTING = [
    ("U01 Free", "ExistingUser", "Uses the free tier"),
    ("U02 Planner", "ExistingUser", "Pays for the Planner tier"),
    ("U03 Advanced", "ExistingUser", "Pays for the Advanced tier"),
    ("U04 Trip Pass", "ExistingUser", "Bought a Trip Pass"),
]
ADVOCATES = [("A01", "Advocate", "Recommends the product"), ("A02", "Advocate", "Writes camping guides")]
CONTEXT = [
    ("Google Maps", "Tool", "Navigation and map app"),
    ("Google My Maps", "Tool", "Custom map builder"),
    ("Roadtrippers", "Platform", "Road trip planner"),
    ("ChatGPT", "Tool", "AI assistant"),
    ("Reddit", "Platform", "Online forum"),
    ("YouTube", "Platform", "Video platform"),
    ("HERE", "Tool", "Map provider"),
    ("OpenStreetMap", "Platform", "Open map data"),
    ("National Park Service", "Organization", "US park authority"),
    ("Kamptribe", "Product", "Camping trip planner under test"),
]
ALL = CONSUMERS + EXISTING + ADVOCATES + CONTEXT
CONTEXT_NAMES = [n for n, _, _ in CONTEXT]

ROLES = {
    **{n: ("prospect", "target_consumer") for n, _, _ in CONSUMERS},
    "U01 Free": ("existing_free", "influencer_context"),
    "U02 Planner": ("existing_planner", "influencer_context"),
    "U03 Advanced": ("existing_advanced", "influencer_context"),
    "U04 Trip Pass": ("existing_trip_pass", "influencer_context"),
    "A01": ("brand_advocate", "influencer_context"),
    "A02": ("brand_advocate", "influencer_context"),
}
ACTIVE_NAMES = list(ROLES)

C01_INSTRUCTION = (
    "Privately assess whether a dedicated trip planner might help with a difficult travel day. "
    "Do not publish this instruction."
)
C02_INSTRUCTION = (
    "Quietly compare how long it takes you to find an open campsite with your usual tools "
    "versus a planner, zebra-violet-checkpoint."
)


@pytest.fixture
def offline_config(monkeypatch):
    prompts = []

    def fake(self, prompt, system_prompt):
        prompts.append(prompt)
        return {}

    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry", fake)
    return prompts


def _graph():
    store = LocalGraphStore(GRAPH_ID)
    store.set_meta("validation", "", {}, "2026-01-01")
    ids = {n: store.add_node(name=n, labels=["Entity", t], summary=s) for n, t, s in ALL}
    store.add_edge(name="USES", fact="C01 plans every route with Google Maps.",
                   source_node_uuid=ids["C01"], target_node_uuid=ids["Google Maps"])
    store.add_edge(name="EVALUATES", fact="U02 Planner pays for Kamptribe Planner.",
                   source_node_uuid=ids["U02 Planner"], target_node_uuid=ids["Kamptribe"])
    return ids


def _wait_prepare(client, task_id):
    deadline = time.time() + 30
    while time.time() < deadline:
        data = client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]
        if data["status"] in ("completed", "failed"):
            assert data["status"] == "completed", data
            return
        time.sleep(0.05)
    raise AssertionError("prepare timed out")


def _select(client, sim_id, active_names, roles=ROLES):
    rows = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]["entities"]
    updates = []
    for e in rows:
        item = {"entity_uuid": e["entity_uuid"], "active": e["name"] in active_names}
        if e["name"] in roles:
            item["simulation_role"], item["evidence_role"] = roles[e["name"]]
        updates.append(item)
    response = client.put(f"/api/simulation/{sim_id}/agent-selection", json={"entities": updates, "confirmed": True})
    assert response.status_code == 200, response.json
    return response.json["data"]


def _prepare(client, sim_id):
    response = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False})
    assert response.status_code == 200, response.json
    _wait_prepare(client, response.json["data"]["task_id"])


def _setup(client):
    make_project()
    ids = _graph()
    sim_id = client.post("/api/simulation/create", json={"project_id": "proj_test", "graph_id": GRAPH_ID}).json["data"]["simulation_id"]
    return sim_id, ids


def _tree_checksums(directory):
    sums = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                sums[os.path.relpath(path, directory)] = hashlib.sha256(f.read()).hexdigest()
    return sums


def _profile_names(sim_dir):
    with open(os.path.join(sim_dir, "reddit_profiles.json"), encoding="utf-8") as f:
        reddit = [p["name"] for p in json.load(f)]
    with open(os.path.join(sim_dir, "twitter_profiles.csv"), encoding="utf-8", newline="") as f:
        twitter = [row["name"] for row in csv.DictReader(f)]
    return reddit, twitter


# ---------------------------------------------------------------- step 2

def test_step2_only_selected_entities_become_agents_and_context_stays_rich(uploads, client, offline_config):
    sim_id, ids = _setup(client)
    view = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    assert view["summary"]["graph_entities"] == 26 and view["summary"]["active_agents"] == 0

    saved = _select(client, sim_id, ACTIVE_NAMES)
    assert saved["summary"]["active_agents"] == 16 and saved["summary"]["context_entities"] == 10
    _prepare(client, sim_id)

    sim_dir = RunStore.sim_dir(sim_id)
    reddit, twitter = _profile_names(sim_dir)
    assert reddit == ACTIVE_NAMES and twitter == ACTIVE_NAMES
    for platform in ("twitter", "reddit"):
        api_names = [p["name"] for p in client.get(f"/api/simulation/{sim_id}/profiles?platform={platform}").json["data"]["profiles"]]
        assert sorted(api_names) == sorted(ACTIVE_NAMES)
    with open(os.path.join(sim_dir, "simulation_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    assert [a["entity_name"] for a in config["agent_configs"]] == ACTIVE_NAMES
    for name in CONTEXT_NAMES:
        assert name not in reddit and name not in twitter
        assert all(a["entity_name"] != name for a in config["agent_configs"])
    assert all(p["poster_agent_id"] < 16 for p in config["event_config"]["initial_posts"])

    # Context entities stay retrievable through every graph/context API
    listed = client.get(f"/api/simulation/entities/{GRAPH_ID}").json["data"]
    assert set(CONTEXT_NAMES) <= {e["name"] for e in listed["entities"]} and listed["filtered_count"] == 26
    detail = client.get(f"/api/simulation/entities/{GRAPH_ID}/{ids['Google Maps']}").json["data"]
    assert detail["name"] == "Google Maps" and any("C01" in e["fact"] for e in detail["related_edges"])
    graph = client.get(f"/api/graph/data/{GRAPH_ID}").json["data"]
    assert set(CONTEXT_NAMES) <= {n["name"] for n in graph["nodes"]}
    tools = GraphToolsService(llm_client=object())
    for name in CONTEXT_NAMES:
        assert any(n.name == name for n in LocalGraphStore(GRAPH_ID).search_nodes(name)), name
    assert any("Google Maps" in f for f in tools.quick_search(GRAPH_ID, "Google Maps").facts)
    c01 = EntityReader().get_entity_with_context(GRAPH_ID, ids["C01"])
    assert "Google Maps" in OasisProfileGenerator(graph_id=GRAPH_ID)._build_entity_context(c01)
    assert any("Context-only entities (10;" in p and "National Park Service" in p for p in offline_config)
    with open(os.path.join(sim_dir, "agent_roster.json"), encoding="utf-8") as f:
        roster = json.load(f)
    assert sorted(c["name"] for c in roster["context_entities"]) == sorted(CONTEXT_NAMES)


# ---------------------------------------------------------------- step 3 (fake process)

def test_step3_private_instructions_stay_private_with_fake_actions(uploads, client, offline_config, fake_popen):
    sim_id, _ = _setup(client)
    _select(client, sim_id, ACTIVE_NAMES)
    _prepare(client, sim_id)
    plan = {"stages": [
        {"id": "c01_private", "name": "C01 difficult day", "trigger_round": 1,
         "target": {"persona_ids": ["C01"]}, "private_instruction": C01_INSTRUCTION},
        {"id": "c02_private", "name": "C02 timing", "trigger_round": 2,
         "target": {"persona_ids": ["C02"]}, "private_instruction": C02_INSTRUCTION},
    ]}
    assert client.put(f"/api/simulation/{sim_id}/experiment-plan", json=plan).status_code == 200

    FakePopen.actions = {
        "twitter": [{"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST",
                     "action_args": {"content": "Rough travel day ahead, any tips for finding a campsite late?"},
                     "origin": "agent_generated"}],
        "reddit": [{"round": 2, "agent_id": 1, "agent_name": "C02", "action_type": "CREATE_COMMENT",
                    "action_args": {"content": "I usually just check the map", "post_content": "Campsite tips?"},
                    "origin": "agent_generated"}],
    }
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    run_dir = RunStore.run_dir(sim_id, run_id)

    with open(os.path.join(run_dir, "experiment_plan.json"), encoding="utf-8") as f:
        resolved = json.load(f)
    targets = {s["id"]: s["resolved_agent_ids"] for s in resolved["stages"]}
    assert targets == {"c01_private": [0], "c02_private": [1]}

    for instruction in (C01_INSTRUCTION, C02_INSTRUCTION):
        for platform in ("twitter", "reddit"):
            with open(os.path.join(run_dir, platform, "actions.jsonl"), encoding="utf-8") as f:
                assert instruction not in f.read()
        for name in ("reddit_profiles.json", "twitter_profiles.csv"):
            with open(os.path.join(run_dir, name), encoding="utf-8") as f:
                assert instruction not in f.read()
        with open(os.path.join(run_dir, "simulation_config.json"), encoding="utf-8") as f:
            config = json.load(f)
        assert all(instruction not in p["content"] for p in config["event_config"]["initial_posts"])
        for path in (f"/api/simulation/{sim_id}/actions", f"/api/simulation/{sim_id}/timeline"):
            response = client.get(path)
            assert response.status_code == 200 and response.json["success"] is True, path
            assert instruction not in json.dumps(response.json)
    actions = client.get(f"/api/simulation/{sim_id}/actions").json["data"]
    assert "Rough travel day ahead" in json.dumps(actions)  # the search did see the run's public content


# ---------------------------------------------------------------- step 4

def test_step4_only_agent_generated_actions_count_as_behaviour(uploads, client, offline_config, fake_popen):
    sim_id, _ = _setup(client)
    _select(client, sim_id, ACTIVE_NAMES)
    _prepare(client, sim_id)
    record = RunStore.create_run(sim_id)
    run_dir = RunStore.run_dir(sim_id, record.run_id)
    write_actions(run_dir, "twitter", [
        {"round": 0, "agent_id": 14, "agent_name": "A01", "action_type": "CREATE_POST",
         "action_args": {"content": "Camping season is open"}, "origin": "system_seed"},
        {"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST",
         "action_args": {"content": "Trying a planner for tomorrow"}, "origin": "agent_generated"},
        {"round": 1, "agent_id": 1, "agent_name": "C02", "action_type": "CREATE_POST",
         "action_args": {"content": C02_INSTRUCTION}, "origin": "researcher_instruction", "origin_flag": "instruction_leak"},
        {"round": 2, "agent_id": 11, "agent_name": "U02 Planner", "action_type": "CREATE_POST",
         "action_args": {"content": "Planner tier saved me hours"}, "origin": "agent_generated"},
        {"event_type": "private_instruction", "origin": "researcher_instruction", "round": 1, "agent_id": 0,
         "stage_id": "c01_private", "instruction": C01_INSTRUCTION, "status": "delivered"},
        {"event_type": "stage_completed", "origin": "orchestration", "round": 1, "agent_id": 0, "stage_id": "c01_private"},
    ], run_id=record.run_id)

    evidence = RunEvidence.load(sim_id, record.run_id)
    assert [a["origin"] for a in evidence.actions] == ["system_seed", "agent_generated", "researcher_instruction", "agent_generated"]
    assert len(evidence.orchestration_events) == 2  # inspectable, never actions
    stats = evidence.stats()
    assert stats["behavioural_actions"] == 2 and stats["total_actions"] == 4
    assert stats["private_instructions_delivered"] == 1
    metrics = stats["consumer_metrics"]
    assert metrics["actions"] == 1 and metrics["by_persona"] == {"C01": 1}
    assert metrics["excluded_by_origin"] == {"system_seed": 1, "researcher_instruction": 1}
    assert metrics["excluded_by_evidence_role"] == {"influencer_context": 1}
    assert len(metrics["target_consumer_personas"]) == 10

    class LLM:
        def chat_json(self, messages, temperature=0.1, max_tokens=None):
            import re
            refs = re.findall(r"\[((?:TW|RD)-\d{5})\]", messages[-1]["content"])
            if "You analyse ONE persona" in messages[0]["content"]:
                return {"fields": [{"field": "outcome", "status": "observed", "summary": "s", "refs": refs}]}
            return {"verdict": "supported", "rationale": "r", "evidence_for": [{"summary": "s", "refs": refs}]}

    builder = StructuredReportBuilder(evidence, AssessmentSpec(funnel_fields=["outcome"],
                                      hypotheses=[{"id": "H1", "text": "Prospects adopt the planner"}]), LLM())
    personas = builder.analyse_personas()
    assert len(personas) == 10  # target consumers only
    behavioural_refs = {a["ref"] for a in evidence.actions if a["origin"] == "agent_generated"}
    cited = {r for p in personas for f in p["fields"] for r in f["refs"]}
    cited |= {r for h in builder.assess_hypotheses() for i in h["evidence_for"] for r in i["refs"]}
    assert cited and cited <= behavioural_refs


# ---------------------------------------------------------------- step 5

def test_step5_runs_are_immutable_across_selection_changes(uploads, client, offline_config, fake_popen):
    sim_id, _ = _setup(client)
    _select(client, sim_id, ACTIVE_NAMES)
    _prepare(client, sim_id)
    FakePopen.actions = {"twitter": [{"round": 1, "agent_id": 15, "agent_name": "A02", "action_type": "CREATE_POST",
                                      "action_args": {"content": "Run A post"}}], "reddit": []}
    run_a = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    dir_a = RunStore.run_dir(sim_id, run_a)
    sums_a = _tree_checksums(dir_a)

    # New population for run B: drop the advocates, add Kamptribe as a platform account
    new_active = [n for n in ACTIVE_NAMES if not n.startswith("A0")] + ["Kamptribe"]
    saved = _select(client, sim_id, new_active, {**ROLES, "Kamptribe": ("brand_platform", "excluded")})
    assert saved["preparation_invalidated"] is True and saved["summary"]["active_agents"] == 15
    _prepare(client, sim_id)
    FakePopen.actions = {"twitter": [{"round": 1, "agent_id": 14, "agent_name": "Kamptribe", "action_type": "CREATE_POST",
                                      "action_args": {"content": "Run B post"}}], "reddit": []}
    run_b = client.post("/api/simulation/start", json={"simulation_id": sim_id, "new_run": True}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    dir_b = RunStore.run_dir(sim_id, run_b)

    assert run_a != run_b
    assert _tree_checksums(dir_a) == sums_a
    with open(os.path.join(dir_a, "agent_roster.json"), encoding="utf-8") as f:
        assert len(json.load(f)["active_agents"]) == 16
    reddit_a, twitter_a = _profile_names(dir_a)
    assert reddit_a == twitter_a == ACTIVE_NAMES
    with open(os.path.join(dir_b, "agent_roster.json"), encoding="utf-8") as f:
        assert [a["name"] for a in json.load(f)["active_agents"]] == new_active
    reddit_b, _ = _profile_names(dir_b)
    assert reddit_b == new_active

    ev_a, ev_b = RunEvidence.load(sim_id, run_a), RunEvidence.load(sim_id, run_b)
    assert [a["content"] for a in ev_a.actions] == ["Run A post"]
    assert [a["content"] for a in ev_b.actions] == ["Run B post"]
    assert ev_a.integrity["warnings"] == [] and ev_b.integrity["warnings"] == []
    for run_id, directory in ((run_a, dir_a), (run_b, dir_b)):
        for platform in ("twitter", "reddit"):
            with open(os.path.join(directory, platform, "actions.jsonl"), encoding="utf-8") as f:
                assert {json.loads(line)["run_id"] for line in f if line.strip()} == {run_id}
    assert {a["name"] for a in ev_a.agents} == set(ACTIVE_NAMES)
