"""
Private research instructions never become public posts; action origins keep system
and research content out of behavioural evidence
"""

import json
import os
import sqlite3
import sys
import time

import pytest

from conftest import FakePopen, make_prepared_simulation, make_project, select_entities, wait_for_monitor, write_actions

from app.services.local_graph_store import LocalGraphStore
from app.services.report_evidence_sections import StructuredReportBuilder
from app.services.assessment_spec import AssessmentSpec
from app.services.research_instructions import (
    extract_private_instructions, is_unsafe_public_text, plan_from_brief, validate_plan,
)
from app.services.run_evidence import RunEvidence
from app.services.run_store import RunStore
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.services.simulation_manager import SimulationManager

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_orchestration as orchestration  # noqa: E402

OPENING_C03 = "Ask the community which app to use for a first camping trip in Zeeland with two toddlers."
OPENING_C01 = "Describe how you would plan a long weekend near the coast without any app."

BRIEF_DOC = f"""# Research brief

Assess personas C01-C03.

## Opening prompts
C01 - Opening prompt: {OPENING_C01}
C03 - Cold-start opening prompt: {OPENING_C03}

## Persona notes
C02 - German camper couple, retired.
"""

CONSUMERS = [
    ("C01 Dutch first-time family", "Consumer", "Family with two kids."),
    ("C02 German camper couple", "Consumer", "Retired couple."),
    # Graph extraction from the brief put the opening prompt into this entity's summary
    ("C03 Belgian student group", "Consumer", f"Students on a budget. Cold-start opening prompt: {OPENING_C03}"),
    ("Google Maps", "Tool", "Map app"),
]


class SeedLLM:
    """Config LLM stub: proposes seed posts that copy the research instructions"""

    def __init__(self):
        self.prompts = []

    def __call__(self, generator, prompt, system_prompt):
        self.prompts.append(prompt)
        if "initial_posts" in prompt:
            return {
                "hot_topics": ["camping"],
                "narrative_direction": "Seasonal planning",
                "initial_posts": [
                    {"content": f"C03 - Cold-start opening prompt: {OPENING_C03}", "poster_type": "Consumer"},
                    {"content": OPENING_C01, "poster_type": "Consumer"},
                    {"content": "Stage 2 test task: compare two planners and report back", "poster_type": "Tool"},
                    {"content": "Camping season on the coast has started, the first sites are open!", "poster_type": "Consumer"},
                ],
            }
        return {}


def _graph():
    store = LocalGraphStore("graph_private")
    store.set_meta("private", "", {}, "2026-01-01")
    for name, etype, summary in CONSUMERS:
        store.add_node(name=name, labels=["Entity", etype], summary=summary)


def _prepared(client, monkeypatch):
    llm = SeedLLM()
    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry",
                        lambda self, prompt, system_prompt: llm(self, prompt, system_prompt))
    make_project(documents=[("assessment_brief.md", BRIEF_DOC), ("product.md", "# Product\nA trip planner app.")])
    _graph()
    sim_id = SimulationManager().create_simulation(project_id="proj_test", graph_id="graph_private").simulation_id
    select_entities(sim_id, roles={"Google Maps": ("brand_platform", "excluded")})
    task_id = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]["task_id"]
    deadline = time.time() + 20
    while time.time() < deadline:
        status = client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed", status
    return sim_id, llm


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 4

def test_private_instructions_never_become_public_posts(uploads, client, fake_popen, monkeypatch):
    sim_id, llm = _prepared(client, monkeypatch)
    sim_dir = RunStore.sim_dir(sim_id)
    config = _read(os.path.join(sim_dir, "simulation_config.json"))

    posts = [p["content"] for p in config["event_config"]["initial_posts"]]
    assert posts == ["Camping season on the coast has started, the first sites are open!"]
    assert all(p["origin"] == "system_seed" for p in config["event_config"]["initial_posts"])
    rejected = [r["content"] for r in config["event_config"]["rejected_seed_posts"]]
    assert len(rejected) == 3 and any(OPENING_C03 in r for r in rejected)
    # Seed posts go to non-consumer accounts when one exists
    maps_id = next(a["agent_id"] for a in config["agent_configs"] if a["entity_name"] == "Google Maps")
    assert config["event_config"]["initial_posts"][0]["poster_agent_id"] == maps_id

    # The instructions never reached the config LLM nor the personas
    assert not any(OPENING_C03 in p or OPENING_C01 in p for p in llm.prompts)
    profiles = _read(os.path.join(sim_dir, "reddit_profiles.json"))
    assert not any(OPENING_C03 in (p["persona"] + p["bio"]) for p in profiles)
    assert OPENING_C03 in config["private_texts"]

    # The run gets them as private, targeted stages
    FakePopen.actions = {"twitter": [], "reddit": []}
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    plan = _read(os.path.join(RunStore.run_dir(sim_id, run_id), "experiment_plan.json"))
    c03_agent = next(a["agent_id"] for a in config["agent_configs"] if a["entity_name"].startswith("C03"))
    c03_stage = next(s for s in plan["stages"] if s["target"]["persona_ids"] == ["C03"])
    assert c03_stage["private_instruction"] == OPENING_C03
    assert c03_stage["resolved_agent_ids"] == [c03_agent]
    assert c03_stage["trigger_round"] == 1 and plan["source"] == "brief"

    # The runner's seed step never publishes instruction text, even from an older config
    legacy_config = {**config, "event_config": {"initial_posts": [
        {"content": f"C03 - Cold-start opening prompt: {OPENING_C03}", "poster_agent_id": c03_agent},
        {"content": "Nice weather for camping", "poster_agent_id": 0},
    ]}, "private_texts": []}
    publish, dropped = orchestration.safe_seed_posts(legacy_config, plan["private_texts"])
    assert [p["content"] for p in publish] == ["Nice weather for camping"]
    assert len(dropped) == 1


def test_leaked_instruction_in_an_agent_post_is_classified_and_excluded(uploads):
    private = [OPENING_C03]
    leaked = {"agent_id": 2, "action_type": "CREATE_POST", "action_args": {"content": f"Hi all! {OPENING_C03}"}}
    organic = {"agent_id": 2, "action_type": "CREATE_POST", "action_args": {"content": "Which app do you use for Zeeland?"}}
    assert orchestration.classify_origin(leaked, private) == ("researcher_instruction", "instruction_leak")
    assert orchestration.classify_origin(organic, private) == ("agent_generated", None)
    assert is_unsafe_public_text(f"C03 - Cold-start opening prompt: {OPENING_C03}", private)[0]
    assert not is_unsafe_public_text("The first campsites on the coast are open", private)[0]


def test_brief_instruction_extraction_ignores_persona_descriptions():
    found = extract_private_instructions(BRIEF_DOC)
    assert [(i["persona_id"], i["kind"], i["text"]) for i in found] == [
        ("C01", "opening", OPENING_C01),
        ("C03", "opening", OPENING_C03),
    ]
    plan = validate_plan(plan_from_brief(BRIEF_DOC, ["C01", "C02", "C03"]))
    assert [s["target"]["persona_ids"] for s in plan["stages"]] == [["C01"], ["C03"]]
    with pytest.raises(ValueError):
        validate_plan({"stages": [{"private_instruction": "x", "trigger_round": 0}]})  # round 0 is seed-only


# ---------------------------------------------------------------- 5

class _Agent:
    def __init__(self):
        self.memory = []

    def update_memory(self, message, role):
        self.memory.append((message.content, role))


class _Logger:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, round_num=None, origin="orchestration", **fields):
        self.events.append({"event_type": event_type, "round": round_num, "origin": origin, **fields})


def test_opening_instructions_are_delivered_once_privately_never_at_round_zero(tmp_path):
    plan = {"stages": [
        {"id": "brief_c03_1", "name": "Opening (C03)", "trigger_round": 1, "resolved_agent_ids": [2],
         "private_instruction": OPENING_C03, "condition": {"type": "always"}, "completion": {"type": "on_delivery"}},
    ], "private_texts": [OPENING_C03]}
    with open(tmp_path / "experiment_plan.json", "w") as f:
        json.dump(plan, f)
    config = {"agent_configs": [{"agent_id": i, "entity_name": f"C0{i + 1}"} for i in range(3)],
              "event_config": {"initial_posts": [{"content": OPENING_C03, "poster_agent_id": 2},
                                                 {"content": "Sunny weekend ahead", "poster_agent_id": 0},
                                                 {"content": "Sunny weekend ahead", "poster_agent_id": 0}]}}
    logger = _Logger()
    runner = orchestration.RunOrchestrator(config, str(tmp_path), logger)
    agents = {i: _Agent() for i in range(3)}

    # Round 0 is seed content only: the instruction is neither posted nor duplicated
    seeds = runner.seed_posts()
    assert [s["content"] for s in seeds] == ["Sunny weekend ahead"]
    deliveries = {r: runner.deliver_due(r, agents.__getitem__) for r in range(1, 6)}
    assert deliveries[1] == [2] and all(not deliveries[r] for r in range(2, 6))
    assert len(agents[2].memory) == 1 and OPENING_C03 in agents[2].memory[0][0]
    assert "never quote" in agents[2].memory[0][0].lower() and "reject" in agents[2].memory[0][0]
    assert not agents[0].memory and not agents[1].memory
    delivered = [e for e in logger.events if e["event_type"] == "private_instruction"]
    assert len(delivered) == 1 and delivered[0]["origin"] == "researcher_instruction" and delivered[0]["round"] == 1


def test_seed_posts_are_not_read_back_from_the_trace_as_round_one_actions(tmp_path):
    db = tmp_path / "twitter_simulation.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE trace (user_id INTEGER, created_at TEXT, action TEXT, info TEXT)")
    conn.execute("INSERT INTO trace VALUES (0, '0', 'sign_up', '{}')")
    conn.execute("INSERT INTO trace VALUES (0, '0', 'create_post', ?)", (json.dumps({"content": "Seed"}),))
    conn.commit()
    after_seed = orchestration.max_trace_rowid(str(db))
    conn.execute("INSERT INTO trace VALUES (1, '1', 'create_post', ?)", (json.dumps({"content": "Own post"}),))
    conn.commit()
    conn.close()
    assert after_seed == 2
    rows = sqlite3.connect(db).execute("SELECT action, info FROM trace WHERE rowid > ?", (after_seed,)).fetchall()
    assert [json.loads(i)["content"] for _, i in rows] == ["Own post"]


def test_stage_conditions_keep_neutral_exposure_optional():
    plan = {"stages": [
        {"id": "exposure", "trigger_round": 3, "resolved_agent_ids": [0, 1],
         "private_instruction": "During your research you encounter Product X. Decide whether you ignore it, inspect it, try it or reject it.",
         "condition": {"type": "not_mentioned", "terms": ["Product X"]},
         "completion": {"type": "mentioned", "terms": ["Product X"]}},
        {"id": "task", "trigger_round": 3, "order": 2, "resolved_agent_ids": [0, 1],
         "private_instruction": "Plan your next trip with the tool of your choice.",
         "condition": {"type": "stage_completed", "stage_id": "exposure"},
         "completion": {"type": "any_action"}},
    ]}
    scheduler = orchestration.StageScheduler(plan, [0, 1])
    scheduler.observe(1, [{"agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "Product X looks fun"}}])
    assert scheduler.due(2) == []
    exposed = scheduler.due(3)
    assert [d["agent_id"] for d in exposed] == [0]  # agent 1 discovered Product X organically
    # Agent 0 rejects it: the follow-up task never triggers for them
    scheduler.observe(3, [{"agent_id": 0, "action_type": "CREATE_POST", "action_args": {"content": "Not interested, sticking with paper maps"}}])
    assert scheduler.due(4) == []
    events = scheduler.observe(4, [{"agent_id": 0, "action_type": "CREATE_COMMENT", "action_args": {"content": "Okay, Product X has offline maps"}}])
    assert [e["stage_id"] for e in events] == ["exposure"]
    assert [d["agent_id"] for d in scheduler.due(5)] == [0]


# ---------------------------------------------------------------- 6

ROLE_AGENTS = [
    {"agent_id": 0, "entity_name": "C01 Dutch family", "entity_type": "Consumer",
     "simulation_role": "prospect", "evidence_role": "target_consumer"},
    {"agent_id": 1, "entity_name": "C02 German couple", "entity_type": "Consumer",
     "simulation_role": "prospect", "evidence_role": "target_consumer"},
    {"agent_id": 2, "entity_name": "Existing user A", "entity_type": "ExistingUser",
     "simulation_role": "existing_free", "evidence_role": "influencer_context"},
    {"agent_id": 3, "entity_name": "Google Maps", "entity_type": "Tool",
     "simulation_role": "brand_platform", "evidence_role": "excluded"},
]


class CitingLLM:
    """Cites every action ref it is shown; records prompts"""

    def __init__(self):
        self.prompts = []

    def chat_json(self, messages, temperature=0.1, max_tokens=None):
        import re

        user = messages[-1]["content"]
        self.prompts.append(user)
        refs = re.findall(r"\[((?:TW|RD)-\d{5})\]", user)
        if "You analyse ONE persona" in messages[0]["content"]:
            return {"fields": [{"field": "outcome", "status": "observed", "summary": "seen", "refs": refs}],
                    "strongest_evidence_for": {"summary": "x", "refs": refs}}
        return {"verdict": "supported", "rationale": "r", "evidence_for": [{"summary": "s", "refs": refs}]}


def test_non_agent_generated_actions_are_excluded_from_consumer_metrics(uploads):
    make_project()
    make_prepared_simulation(agents=ROLE_AGENTS)
    record = RunStore.create_run("sim_test")
    run_dir = RunStore.run_dir("sim_test", record.run_id)
    write_actions(run_dir, "twitter", [
        {"round": 0, "agent_id": 0, "agent_name": "C01 Dutch family", "action_type": "CREATE_POST",
         "action_args": {"content": "Seed: camping season opens"}, "origin": "system_seed"},
        {"round": 1, "agent_id": 0, "agent_name": "C01 Dutch family", "action_type": "CREATE_POST",
         "action_args": {"content": "Looking for a planner app"}, "origin": "agent_generated"},
        {"round": 1, "agent_id": 1, "agent_name": "C02 German couple", "action_type": "CREATE_POST",
         "action_args": {"content": OPENING_C03}, "origin": "researcher_instruction", "origin_flag": "instruction_leak"},
        {"round": 2, "agent_id": 2, "agent_name": "Existing user A", "action_type": "CREATE_POST",
         "action_args": {"content": "I recommend the planner"}, "origin": "agent_generated"},
        {"round": 2, "agent_id": 3, "agent_name": "Google Maps", "action_type": "CREATE_POST",
         "action_args": {"content": "Try our new camping layer"}, "origin": "agent_generated"},
    ], run_id=record.run_id)

    evidence = RunEvidence.load("sim_test", record.run_id)
    metrics = evidence.consumer_metrics()
    assert metrics["actions"] == 1 and metrics["by_persona"] == {"C01": 1}
    assert metrics["excluded_by_origin"] == {"system_seed": 1, "researcher_instruction": 1}
    assert metrics["excluded_by_evidence_role"] == {"influencer_context": 1, "excluded": 1}
    assert metrics["target_consumers_without_actions"] == ["C02"]
    stats = evidence.stats()
    assert stats["by_origin"] == {"system_seed": 1, "agent_generated": 3, "researcher_instruction": 1}
    seed_ref = next(a["ref"] for a in evidence.actions if a["origin"] == "system_seed")
    leak_ref = next(a["ref"] for a in evidence.actions if a["origin"] == "researcher_instruction")
    assert "NOT behavioural evidence" in RunEvidence.format_action(evidence.actions[0])

    llm = CitingLLM()
    spec = AssessmentSpec(persona_ids=[], hypotheses=[{"id": "H1", "text": "Families adopt the planner"}],
                          funnel_fields=["outcome"])
    builder = StructuredReportBuilder(evidence, spec, llm)
    personas = builder.analyse_personas()
    # Without brief ids only target consumers are analysed
    assert [p["canonical_id"] for p in personas] == ["C01", "C02"]
    cited = {r for p in personas for f in p["fields"] for r in f["refs"]}
    assert seed_ref not in cited and leak_ref not in cited
    assert personas[1]["fields"][0]["status"] == "missing"  # C02 only has leaked instruction text
    hypotheses = builder.assess_hypotheses()
    hyp_refs = {r for item in hypotheses[0]["evidence_for"] for r in item["refs"]}
    assert seed_ref not in hyp_refs and leak_ref not in hyp_refs
    appendix = builder.render_appendix({})
    assert "evidence_role = target_consumer AND origin = agent_generated" in appendix
    assert "| C02 | C02 German couple | 1 | prospect | target_consumer |" in appendix


def test_legacy_logs_classify_round_zero_seeds_and_their_duplicates(uploads):
    make_project()
    make_prepared_simulation()
    record = RunStore.create_run("sim_test")
    run_dir = RunStore.run_dir("sim_test", record.run_id)
    seed = {"agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST", "action_args": {"content": "Seed text"}}
    write_actions(run_dir, "twitter", [
        {"round": 0, **seed},
        {"round": 1, **seed},  # old runner re-read the seed from the trace
        {"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST", "action_args": {"content": "Own words"}},
    ], run_id=record.run_id)
    evidence = RunEvidence.load("sim_test", record.run_id)
    assert [a["origin"] for a in evidence.actions] == ["system_seed", "system_seed", "agent_generated"]
    assert evidence.roles_source == "legacy_default" and evidence.consumer_metrics()["actions"] == 1
