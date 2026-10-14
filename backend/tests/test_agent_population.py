"""
Agent selection: graph entities are not agents until explicitly selected; the chosen
population (with simulation and evidence roles) is frozen per preparation and per run
"""

import json
import os

import pytest

from conftest import FakePopen, make_project, select_entities, wait_for_monitor

from app.services.agent_selection import AgentSelectionStore, build_selection_view
from app.services.entity_canonicalizer import canonical_key, find_duplicates
from app.services.entity_reader import EntityReader
from app.services.local_graph_store import LocalGraphStore
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.run_evidence import RunEvidence
from app.services.run_store import RunStore
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.services.simulation_manager import SimulationManager

GRAPH_ID = "graph_population"

CONSUMERS = [(f"C{i:02d} Research consumer {i}", "Consumer", f"Research persona number {i}") for i in range(1, 11)]
EXISTING_USERS = [(f"Existing user {n}", "ExistingUser", f"Long-time user {n}") for n in "ABCDE"]
ADVOCATES = [(f"Advocate {n}", "Advocate", f"Recommends the product ({n})") for n in "XYZ"]
PLATFORMS = [
    (name, etype, f"{name} is a tool or brand mentioned in the sources")
    for name, etype in [
        ("Kamptribe", "Brand"), ("KAMPTRIBE", "Brand"), ("kamptribe.com", "Platform"),
        ("Google Maps", "Tool"), ("Maps", "Tool"), ("Google My Maps", "Tool"), ("My Maps", "Tool"),
        ("Roadtrippers", "Platform"), ("ChatGPT", "Tool"), ("HERE WeGo", "Tool"), ("Reddit", "Platform"),
        ("YouTube", "Platform"), ("Park4Night", "Platform"), ("ACSI", "Brand"), ("Campercontact", "Platform"),
        ("Booking.com", "Platform"), ("Airbnb", "Platform"), ("Pitchup", "Platform"), ("Hipcamp", "Platform"),
        ("Komoot", "Tool"),
    ]
]
ALL_ENTITIES = CONSUMERS + EXISTING_USERS + ADVOCATES + PLATFORMS

ROLES = {
    **{n: ("prospect", "target_consumer") for n, _, _ in CONSUMERS},
    **{n: ("existing_free", "influencer_context") for n, _, _ in EXISTING_USERS},
    **{n: ("brand_advocate", "influencer_context") for n, _, _ in ADVOCATES},
}


def make_graph(entities, graph_id=GRAPH_ID, edges=()):
    store = LocalGraphStore(graph_id)
    store.set_meta("population", "", {}, "2026-01-01")
    uuids = {}
    for name, etype, summary in entities:
        uuids[name] = store.add_node(name=name, labels=["Entity", etype], summary=summary)
    for source, target, fact in edges:
        store.add_edge(name="USES", fact=fact, source_node_uuid=uuids[source], target_node_uuid=uuids[target])
    return uuids


@pytest.fixture
def offline_config(monkeypatch):
    prompts = []

    def fake_llm(self, prompt, system_prompt):
        prompts.append(prompt)
        return {}

    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry", fake_llm)
    return prompts


def _simulation(entities=ALL_ENTITIES, edges=()):
    project = make_project()
    uuids = make_graph(entities, edges=edges)
    state = SimulationManager().create_simulation(project_id=project.project_id, graph_id=GRAPH_ID)
    return state.simulation_id, uuids


def _prepare(sim_id):
    return SimulationManager().prepare_simulation(
        simulation_id=sim_id,
        simulation_requirement="Product-fit test",
        document_text="Product documentation",
        use_llm_for_profiles=False,
        parallel_profile_count=1,
    )


def _read(sim_dir, name):
    with open(os.path.join(sim_dir, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1 / 2

def test_ten_consumers_and_thirty_platforms_give_exactly_ten_agents(uploads, offline_config):
    platforms = [(f"Platform {i:02d}", "Platform", "A tool") for i in range(30)]
    sim_id, _ = _simulation(CONSUMERS + platforms)
    select_entities(sim_id, roles=ROLES, active={n for n, _, _ in CONSUMERS})

    state = _prepare(sim_id)

    sim_dir = RunStore.sim_dir(sim_id)
    profiles = _read(sim_dir, "reddit_profiles.json")
    config = _read(sim_dir, "simulation_config.json")
    roster = _read(sim_dir, "agent_roster.json")
    assert state.active_agents_count == 10 and state.entities_count == 40 and state.context_entities_count == 30
    assert [p["name"] for p in profiles] == [n for n, _, _ in CONSUMERS]
    assert [a["agent_id"] for a in config["agent_configs"]] == list(range(10))
    assert {a["evidence_role"] for a in config["agent_configs"]} == {"target_consumer"}
    assert len(roster["active_agents"]) == 10 and len(roster["context_entities"]) == 30
    # Context-only entities never post and never take turns: no profile, no config, no seed
    names = {a["entity_name"] for a in config["agent_configs"]}
    assert not any(p.startswith("Platform") for p in names)
    assert all(p["poster_agent_id"] in range(10) for p in config["event_config"]["initial_posts"])


def test_explicitly_selected_platform_becomes_an_agent(uploads, offline_config):
    sim_id, _ = _simulation(CONSUMERS[:2] + PLATFORMS[:4])
    select_entities(
        sim_id,
        roles={**ROLES, "Google Maps": ("brand_platform", "excluded")},
        active={CONSUMERS[0][0], CONSUMERS[1][0], "Google Maps"},
    )

    _prepare(sim_id)

    sim_dir = RunStore.sim_dir(sim_id)
    profiles = _read(sim_dir, "reddit_profiles.json")
    agents = _read(sim_dir, "simulation_config.json")["agent_configs"]
    assert "Google Maps" in [p["name"] for p in profiles]
    maps = next(a for a in agents if a["entity_name"] == "Google Maps")
    assert maps["simulation_role"] == "brand_platform" and maps["evidence_role"] == "excluded"
    assert len(agents) == 3


def test_selection_view_defaults_to_no_agents_and_never_persists_on_read(uploads):
    sim_id, _ = _simulation(CONSUMERS[:3] + PLATFORMS[:4])
    entities = EntityReader().filter_defined_entities(GRAPH_ID, enrich_with_edges=False).entities

    view = build_selection_view(entities, None, ["C01", "C02"])

    assert view["summary"]["active_agents"] == 0 and view["summary"]["graph_entities"] == 7
    assert view["confirmed"] is False and AgentSelectionStore.load(sim_id) is None
    by_name = {e["name"]: e for e in view["entities"]}
    assert by_name[CONSUMERS[0][0]]["suggested_simulation_role"] == "prospect"
    assert by_name["Google Maps"]["suggested_simulation_role"] == "brand_platform"
    assert by_name[CONSUMERS[0][0]]["brief_persona_id"] == "C01"


# ---------------------------------------------------------------- 3

def test_inactive_entities_stay_available_as_graph_context(uploads, offline_config):
    consumer = CONSUMERS[0][0]
    sim_id, uuids = _simulation(
        CONSUMERS[:2] + PLATFORMS[:4],
        edges=[(consumer, "Google Maps", f"{consumer} plans every trip with Google Maps.")],
    )
    select_entities(sim_id, roles=ROLES, active={CONSUMERS[0][0], CONSUMERS[1][0]})

    _prepare(sim_id)

    # Still in the graph and retrievable
    assert any(n.name == "Google Maps" for n in LocalGraphStore(GRAPH_ID).search_nodes("Google Maps"))
    assert EntityReader().filter_defined_entities(GRAPH_ID).filtered_count == 6
    # Available as context of the active agent that relates to it
    entity = EntityReader().get_entity_with_context(GRAPH_ID, uuids[consumer])
    context = OasisProfileGenerator(graph_id=GRAPH_ID)._build_entity_context(entity)
    assert "Google Maps" in context
    # Listed to the config LLM as context-only entities (may be mentioned, never accounts)
    assert any("Context-only entities" in p and "Google Maps" in p for p in offline_config)
    roster = _read(RunStore.sim_dir(sim_id), "agent_roster.json")
    assert "Google Maps" in [c["name"] for c in roster["context_entities"]]


# ---------------------------------------------------------------- synthetic fixture

def test_synthetic_population_18_active_20_context(uploads, client, offline_config, fake_popen):
    sim_id, _ = _simulation()
    selection = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    assert selection["summary"]["graph_entities"] == 38 and selection["summary"]["active_agents"] == 0

    updates = [
        {"entity_uuid": e["entity_uuid"], "active": e["name"] in ROLES,
         **({"simulation_role": ROLES[e["name"]][0], "evidence_role": ROLES[e["name"]][1]} if e["name"] in ROLES else {})}
        for e in selection["entities"]
    ]
    saved = client.put(f"/api/simulation/{sim_id}/agent-selection", json={"entities": updates, "confirmed": True}).json["data"]
    summary = saved["summary"]
    assert summary["active_agents"] == 18 and summary["context_entities"] == 20
    by_role = {r["id"]: r["count"] for r in summary["by_simulation_role"]}
    assert by_role == {"prospect": 10, "existing_free": 5, "brand_advocate": 3}
    by_evidence = {r["id"]: r["count"] for r in summary["by_evidence_role"]}
    assert by_evidence == {"target_consumer": 10, "influencer_context": 8, "excluded": 0}

    task_id = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]["task_id"]
    _wait_prepare(client, task_id)
    profiles = client.get(f"/api/simulation/{sim_id}/profiles").json["data"]["profiles"]
    assert len(profiles) == 18

    # Run: a target consumer, an existing user and a seed post act
    FakePopen.actions = {
        "twitter": [
            {"round": 0, "agent_id": 12, "agent_name": "Existing user C", "action_type": "CREATE_POST",
             "action_args": {"content": "Spring camping season starts"}, "origin": "system_seed"},
            {"round": 1, "agent_id": 0, "agent_name": CONSUMERS[0][0], "action_type": "CREATE_POST",
             "action_args": {"content": "Trying the trip planner"}, "origin": "agent_generated"},
            {"round": 1, "agent_id": 10, "agent_name": "Existing user A", "action_type": "CREATE_POST",
             "action_args": {"content": "I have used it for years"}, "origin": "agent_generated"},
        ],
        "reddit": [],
    }
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    evidence = RunEvidence.load(sim_id, run_id)
    metrics = evidence.consumer_metrics()
    assert metrics["actions"] == 1 and metrics["by_persona"] == {"C01": 1}
    assert len(metrics["target_consumer_personas"]) == 10
    assert metrics["excluded_by_origin"] == {"system_seed": 1}
    assert metrics["excluded_by_evidence_role"] == {"influencer_context": 1}


def _wait_prepare(client, task_id):
    import time

    deadline = time.time() + 20
    while time.time() < deadline:
        data = client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]
        if data["status"] in ("completed", "failed"):
            assert data["status"] == "completed", data
            return data
        time.sleep(0.05)
    raise AssertionError("prepare timed out")


# ---------------------------------------------------------------- 7 / 8 / 9

def _prepared_api_simulation(client, active, roles=ROLES):
    sim_id, _ = _simulation(CONSUMERS[:3] + PLATFORMS[3:6])
    select_entities(sim_id, roles=roles, active=active)
    task_id = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]["task_id"]
    _wait_prepare(client, task_id)
    return sim_id


def test_roles_survive_prepare_run_snapshot_and_reload(uploads, client, offline_config, fake_popen):
    roles = {**ROLES, CONSUMERS[2][0]: ("existing_planner", "influencer_context")}
    sim_id = _prepared_api_simulation(client, {n for n, _, _ in CONSUMERS[:3]}, roles)
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)

    run_dir = RunStore.run_dir(sim_id, run_id)
    roster = _read(run_dir, "agent_roster.json")
    assert [(a["name"], a["simulation_role"], a["evidence_role"]) for a in roster["active_agents"]] == [
        (CONSUMERS[0][0], "prospect", "target_consumer"),
        (CONSUMERS[1][0], "prospect", "target_consumer"),
        (CONSUMERS[2][0], "existing_planner", "influencer_context"),
    ]
    assert os.path.exists(os.path.join(run_dir, "agent_selection.json"))
    assert RunStore.get_run(sim_id, run_id).snapshot["population"]["active_agents"] == 3

    # Reload from disk (fresh objects)
    SimulationManager._simulations = {}
    evidence = RunEvidence.load(sim_id, run_id)
    assert evidence.roles_source == "roster"
    assert evidence.persona_evidence_role("C03") == "influencer_context"
    assert evidence.persona_simulation_role("C01") == "prospect"
    state = SimulationManager().get_simulation(sim_id)
    assert state.active_agents_count == 3
    view = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    assert view["population"]["prepared"] is True and view["population"]["in_sync"] is True
    assert {e["name"]: e["evidence_role"] for e in view["entities"] if e["active"]}[CONSUMERS[2][0]] == "influencer_context"


def test_new_selection_never_leaks_into_an_existing_run(uploads, client, offline_config, fake_popen):
    sim_id = _prepared_api_simulation(client, {CONSUMERS[0][0], CONSUMERS[1][0]})
    first = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    first_dir = RunStore.run_dir(sim_id, first)
    snapshot = {n: open(os.path.join(first_dir, n), "rb").read()
                for n in ("agent_roster.json", "agent_selection.json", "reddit_profiles.json", "simulation_config.json")}

    # New population: add a platform account
    entities = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]["entities"]
    maps = next(e for e in entities if e["name"] == "Google Maps")
    saved = client.put(f"/api/simulation/{sim_id}/agent-selection", json={
        "entities": [{"entity_uuid": maps["entity_uuid"], "active": True}], "confirmed": True,
    }).json["data"]
    assert saved["preparation_invalidated"] is True
    # Not startable until prepared again for the new population
    assert client.post("/api/simulation/start", json={"simulation_id": sim_id, "new_run": True}).status_code == 400

    task_id = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]["task_id"]
    _wait_prepare(client, task_id)
    second = client.post("/api/simulation/start", json={"simulation_id": sim_id, "new_run": True}).json["data"]["run_id"]
    wait_for_monitor(sim_id)

    assert second != first
    for name, data in snapshot.items():
        assert open(os.path.join(first_dir, name), "rb").read() == data, name
    assert len(_read(first_dir, "agent_roster.json")["active_agents"]) == 2
    assert len(_read(RunStore.run_dir(sim_id, second), "agent_roster.json")["active_agents"]) == 3
    assert "Google Maps" not in {a["name"] for a in RunEvidence.load(sim_id, first).agents}


def test_refresh_and_navigation_never_change_the_population(uploads, client, offline_config, fake_popen):
    sim_id = _prepared_api_simulation(client, {CONSUMERS[0][0], CONSUMERS[1][0]})
    sim_dir = RunStore.sim_dir(sim_id)
    before = {n: open(os.path.join(sim_dir, n), "rb").read()
              for n in ("agent_roster.json", "agent_selection.json", "reddit_profiles.json")}

    for _ in range(3):  # reloads of Step 2 / Step 3
        assert client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]["population"]["prepared"]
        again = client.post("/api/simulation/prepare", json={"simulation_id": sim_id}).json["data"]
        assert again["already_prepared"] is True
        client.get(f"/api/simulation/{sim_id}/experiment-plan")
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    assert client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["started"] is False

    # Re-saving the identical selection does not invalidate anything
    view = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    resaved = client.put(f"/api/simulation/{sim_id}/agent-selection", json={
        "entities": [{"entity_uuid": e["entity_uuid"], "active": e["active"]} for e in view["entities"]],
        "confirmed": True,
    }).json["data"]
    assert resaved["preparation_invalidated"] is False
    for name, data in before.items():
        if name != "agent_selection.json":
            assert open(os.path.join(sim_dir, name), "rb").read() == data, name
    assert len(_read(RunStore.run_dir(sim_id, run_id), "agent_roster.json")["active_agents"]) == 2
    assert len(FakePopen.instances) == 1


def test_prepare_attaches_to_a_running_preparation(uploads, client, offline_config, monkeypatch):
    import threading

    sim_id, _ = _simulation(CONSUMERS[:2])
    select_entities(sim_id)
    release = threading.Event()
    original = SimulationManager.prepare_simulation

    def slow_prepare(self, *args, **kwargs):
        release.wait(10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SimulationManager, "prepare_simulation", slow_prepare)
    first = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]
    second = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]
    assert second["already_running"] is True and second["task_id"] == first["task_id"]
    # The population cannot change while preparing
    entity = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]["entities"][0]
    blocked = client.put(f"/api/simulation/{sim_id}/agent-selection", json={
        "entities": [{"entity_uuid": entity["entity_uuid"], "active": False}], "confirmed": True,
    })
    assert blocked.status_code == 409
    release.set()
    _wait_prepare(client, first["task_id"])


def test_invalid_selections_are_rejected(uploads, client):
    sim_id, _ = _simulation(CONSUMERS[:2])
    url = f"/api/simulation/{sim_id}/agent-selection"
    assert client.put(url, json={"confirmed": True}).status_code == 400  # nobody active
    assert client.put(url, json={"entities": [{"entity_uuid": "nope", "active": True}]}).status_code == 400
    entity = client.get(url).json["data"]["entities"][0]
    bad_role = client.put(url, json={"entities": [{"entity_uuid": entity["entity_uuid"], "simulation_role": "ceo"}]})
    assert bad_role.status_code == 400
    custom = client.put(url, json={
        "role_catalog": [{"id": "early_adopter", "label": "Early adopter", "default_evidence_role": "target_consumer"},
                         {"id": "other", "label": "Other"}],
        "entities": [{"entity_uuid": entity["entity_uuid"], "active": True, "simulation_role": "early_adopter",
                      "evidence_role": "target_consumer"},
                     {"entity_uuid": client.get(url).json["data"]["entities"][1]["entity_uuid"], "simulation_role": "other"}],
        "confirmed": True,
    })
    assert custom.status_code == 200, custom.json
    assert custom.json["data"]["summary"]["by_simulation_role"] == [{"id": "early_adopter", "label": "Early adopter", "count": 1}]


# ---------------------------------------------------------------- dedup

def test_canonicalisation_groups_high_confidence_variants_and_flags_ambiguous_ones():
    assert canonical_key("Kamptribe") == canonical_key("KAMPTRIBE") == canonical_key("kamptribe.com") == "kamptribe"
    assert canonical_key("https://www.Kamptribe.com/") == "kamptribe"
    entities = [{"uuid": str(i), "name": n, "summary": "", "order": i} for i, (n, _, _) in enumerate(PLATFORMS + CONSUMERS[:2])]
    entities.append({"uuid": "c01b", "name": "C01", "summary": "", "order": 99})
    dups = find_duplicates(entities)
    by_name = {e["name"]: dups[e["uuid"]] for e in entities}

    canonical = by_name["Kamptribe"]
    assert canonical["duplicate_of"] is None
    assert set(canonical["aliases"]) == {"KAMPTRIBE", "kamptribe.com"}
    for alias in ("KAMPTRIBE", "kamptribe.com"):
        assert by_name[alias]["duplicate_of"] == canonical["canonical_uuid"]
        assert by_name[alias]["duplicate_confidence"] == "high"
    assert by_name["C01"]["duplicate_of"] == by_name[CONSUMERS[0][0]]["canonical_uuid"]

    # Ambiguous short forms are flagged, never merged
    assert by_name["Maps"]["duplicate_of"] is None
    assert "Google Maps" in [f["name"] for f in by_name["Maps"]["possible_duplicates"]]
    assert "Google My Maps" in [f["name"] for f in by_name["My Maps"]["possible_duplicates"]]
    assert by_name["Google Maps"]["duplicate_of"] is None
    assert "Google My Maps" not in [f["name"] for f in by_name["Google Maps"]["possible_duplicates"]]
    # Product names with digits are not persona identifiers
    gpt = find_duplicates([{"uuid": "a", "name": "GPT4"}, {"uuid": "b", "name": "GPT4 Turbo"}])
    assert gpt["b"]["duplicate_of"] is None
