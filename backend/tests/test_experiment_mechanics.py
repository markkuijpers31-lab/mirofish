"""
Experiment mechanics: required source coverage, authored-only stage conditions,
factual stage context and orchestration records
"""

import json
import os
import sys

import pytest

from conftest import FakePopen, make_project, select_entities, wait_for_monitor, write_actions

from app.services.assessment_spec import AssessmentSpec
from app.services.local_graph_store import LocalGraphStore
from app.services.report_evidence_sections import StructuredReportBuilder
from app.services.run_evidence import RunEvidence
from app.services.run_store import RunStore
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.services.simulation_manager import SimulationManager
from app.services.source_requirements import check_coverage, section_texts, split_sections, validate_requirements

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_orchestration as orchestration  # noqa: E402

PRODUCT_TRUTH = """# Product Truth {#product-truth}

## Pricing {#pricing}
Free costs nothing. Planner costs 3.99 per month. The Trip Pass costs 29.99 once per trip.

### Trip Pass {#trip-pass}
A Trip Pass applies to one trip and never renews automatically.

## Does not exist {#not-built}
No bookings and no live availability.
"""

REQUIREMENTS = {"documents": [{
    "id": "product_truth", "label": "Product Truth", "filename_pattern": "product[-_ ]?truth",
    "sections": [
        {"id": "pricing", "required_terms": ["Trip Pass", "29.99"], "min_chars": 40},
        {"id": "trip-pass", "required_terms": ["never renews"]},
        {"id": "not-built", "required_terms": ["No bookings"]},
    ],
}]}


def test_coverage_matrix_reports_missing_and_incomplete_sections():
    spec = validate_requirements(REQUIREMENTS)
    docs = [{"original_filename": "kamptribe_product_truth.md", "text": PRODUCT_TRUTH}]
    assert check_coverage(spec, docs)["ok"] is True
    # Nested sections stay inside their parent; anchors are explicit
    sections = split_sections(PRODUCT_TRUTH)
    assert "never renews" in sections["pricing"]["text"] and "No bookings" not in sections["pricing"]["text"]

    broken = PRODUCT_TRUTH.replace("29.99", "thirty").replace("## Does not exist {#not-built}", "## Does not exist")
    coverage = check_coverage(spec, [{"original_filename": "product-truth.md", "text": broken}])
    rows = {r["id"]: r for r in coverage["documents"][0]["sections"]}
    assert coverage["ok"] is False
    assert rows["pricing"]["status"] == "incomplete" and rows["pricing"]["missing_terms"] == ["29.99"]
    assert rows["not-built"]["status"] == "missing"
    assert coverage["missing"] == ["product_truth:pricing", "product_truth:not-built"]
    assert check_coverage(spec, [{"original_filename": "other.md", "text": PRODUCT_TRUTH}])["documents"][0]["files"] == []
    assert section_texts(spec, docs, ["trip-pass"])["trip-pass"].startswith("### Trip Pass")
    with pytest.raises(ValueError):
        section_texts(spec, docs, ["nope"])


def _simulation(client, documents, monkeypatch):
    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry", lambda self, prompt, system_prompt: {})
    make_project(documents=documents)
    store = LocalGraphStore("graph_mech")
    store.set_meta("m", "", {}, "2026-01-01")
    for name in ("C01 Family", "C02 Couple", "A01 Advocate"):
        store.add_node(name=name, labels=["Entity", "Consumer"], summary=name)
    sim_id = SimulationManager().create_simulation(project_id="proj_test", graph_id="graph_mech").simulation_id
    select_entities(sim_id)
    return sim_id


def _prepare(client, sim_id):
    return client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False})


def test_prepare_and_start_refuse_until_required_sections_exist(uploads, client, fake_popen, monkeypatch):
    sim_id = _simulation(client, [("product.md", "# Product\nA planner.")], monkeypatch)
    saved = client.put(f"/api/simulation/{sim_id}/source-requirements", json=REQUIREMENTS)
    assert saved.status_code == 200 and saved.json["data"]["coverage"]["ok"] is False
    assert client.put(f"/api/simulation/{sim_id}/source-requirements", json={"documents": []}).status_code == 400

    refused = _prepare(client, sim_id)
    assert refused.status_code == 400 and "not among the uploaded sources" in refused.json["error"]
    assert refused.json["source_coverage"]["missing"]

    # Upload the Product Truth: coverage passes, prepare and start work, the run snapshots both
    from app.models.project import ProjectManager

    project = ProjectManager.get_project("proj_test")
    text_file = ProjectManager.save_source_text("proj_test", 2, "product_truth.md", PRODUCT_TRUTH)
    project.files.append({"filename": "product_truth.md", "saved_filename": None, "extension": ".md",
                          "size": len(PRODUCT_TRUTH), "text_file": text_file, "text_length": len(PRODUCT_TRUTH),
                          "extraction_status": "ok"})
    ProjectManager.save_project(project)
    assert client.get(f"/api/simulation/{sim_id}/source-requirements").json["data"]["coverage"]["ok"] is True
    import time

    task_id = _prepare(client, sim_id).json["data"]["task_id"]
    for _ in range(200):
        status = client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status == "completed"
    plan = {"stages": [{"id": "exposure", "name": "Neutral exposure", "trigger_round": 2,
                        "target": {"persona_ids": ["C01"]}, "records": ["EXPOSED"],
                        "private_instruction": "During your research you encounter the product. Decide freely.",
                        "context_sections": ["pricing", "not-built"]}]}
    assert client.put(f"/api/simulation/{sim_id}/experiment-plan", json=plan).status_code == 200
    start = client.post("/api/simulation/start", json={"simulation_id": sim_id})
    assert start.status_code == 200, start.json
    wait_for_monitor(sim_id)
    run_dir = RunStore.run_dir(sim_id, start.json["data"]["run_id"])
    resolved = json.load(open(os.path.join(run_dir, "experiment_plan.json")))
    stage = resolved["stages"][0]
    # Context comes verbatim from the authoritative document and is not leak-guarded
    assert "29.99 once per trip" in stage["context"] and "No bookings" in stage["context"]
    assert stage["context"] not in resolved["private_texts"]
    assert stage["records"] == ["EXPOSED"]
    assert json.load(open(os.path.join(run_dir, "source_coverage.json")))["ok"] is True


def test_run_is_refused_when_a_stage_references_a_missing_section(uploads, client, fake_popen, monkeypatch):
    sim_id = _simulation(client, [("product_truth.md", PRODUCT_TRUTH)], monkeypatch)
    client.put(f"/api/simulation/{sim_id}/source-requirements", json=REQUIREMENTS)
    import time

    task_id = _prepare(client, sim_id).json["data"]["task_id"]
    for _ in range(200):
        if client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    client.put(f"/api/simulation/{sim_id}/experiment-plan", json={"stages": [
        {"id": "s", "trigger_round": 1, "private_instruction": "x", "context_sections": ["does-not-exist"]}]})
    refused = client.post("/api/simulation/start", json={"simulation_id": sim_id})
    assert refused.status_code == 400 and "does-not-exist" in refused.json["error"]
    assert RunStore.list_runs(sim_id) == [] and not os.listdir(RunStore.runs_dir(sim_id))
    assert FakePopen.instances == []


def test_authored_conditions_ignore_content_the_agent_only_saw():
    plan = {"stages": [{"id": "task", "trigger_round": 1, "resolved_agent_ids": [0, 1],
                        "private_instruction": "If you decided to look into the product, here is a task.",
                        "context": "The product has a route check.",
                        "condition": {"type": "authored_mentioned", "terms": ["Kamptribe"]},
                        "completion": {"type": "authored_mentioned", "terms": ["route check"]}}]}
    scheduler = orchestration.StageScheduler(plan, [0, 1])
    # Agent 0 only liked an advocate's post about Kamptribe; agent 1 wrote about it
    scheduler.observe(1, [
        {"agent_id": 0, "action_type": "LIKE_POST", "action_args": {"post_content": "Kamptribe saved my trip"}},
        {"agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "Looking at Kamptribe now"}},
    ])
    due = scheduler.due(2)
    assert [d["agent_id"] for d in due] == [1]
    assert "The product has a route check." in due[0]["framed"] and "may still exist" in due[0]["framed"]
    # Seeing the phrase is not completion; writing it is
    assert scheduler.observe(3, [{"agent_id": 1, "action_type": "LIKE_POST", "action_args": {"post_content": "route check"}}]) == []
    done = scheduler.observe(4, [{"agent_id": 1, "action_type": "CREATE_POST", "action_args": {"content": "The route check flagged a ferry"}}])
    assert [e["agent_id"] for e in done] == [1]
    # Factual context repeated in a post is not an instruction leak
    assert orchestration.classify_origin(
        {"action_type": "CREATE_POST", "action_args": {"content": "The product has a route check."}},
        orchestration.private_texts_for_run({}, plan),
    )[0] == "agent_generated"


AGENTS = [
    {"agent_id": 0, "entity_name": "C01 Family", "entity_type": "Consumer", "simulation_role": "prospect_cold",
     "evidence_role": "target_consumer"},
    {"agent_id": 1, "entity_name": "C02 Couple", "entity_type": "Consumer", "simulation_role": "prospect_switcher",
     "evidence_role": "target_consumer"},
    {"agent_id": 2, "entity_name": "U01 Planner user", "entity_type": "ExistingUser", "simulation_role": "existing_planner",
     "evidence_role": "influencer_context"},
]


def test_orchestration_records_decide_exposure_fields_and_cohorts_are_grouped(uploads):
    from conftest import make_prepared_simulation

    make_project(documents=[("product_truth.md", PRODUCT_TRUTH)])
    sim_dir = make_prepared_simulation(agents=AGENTS)
    with open(os.path.join(sim_dir, "source_requirements.json"), "w") as f:
        json.dump(validate_requirements(REQUIREMENTS), f)
    record = RunStore.create_run("sim_test")
    run_dir = RunStore.run_dir("sim_test", record.run_id)
    with open(os.path.join(run_dir, "experiment_plan.json"), "w") as f:
        json.dump({"stages": [{"id": "exposure", "records": ["EXPOSED"]}], "private_texts": []}, f)
    write_actions(run_dir, "twitter", [
        {"event_type": "private_instruction", "origin": "researcher_instruction", "round": 5, "agent_id": 0,
         "stage_id": "exposure", "stage_name": "Neutral exposure", "records": ["EXPOSED"], "status": "delivered"},
        {"round": 6, "agent_id": 0, "agent_name": "C01 Family", "action_type": "CREATE_POST",
         "action_args": {"content": "Had a look at Kamptribe, the budget is nice"}, "origin": "agent_generated"},
        # C02 talks about Kamptribe without any exposure: organic discovery
        {"round": 2, "agent_id": 1, "agent_name": "C02 Couple", "action_type": "CREATE_POST",
         "action_args": {"content": "A friend mentioned Kamptribe"}, "origin": "agent_generated"},
    ], run_id=record.run_id)

    class LLM:
        def __init__(self):
            self.prompts = []

        def chat_json(self, messages, temperature=0.1, max_tokens=None):
            import re
            self.prompts.append(messages[-1]["content"])
            refs = re.findall(r"\[((?:TW|RD)-\d{5})\]", messages[-1]["content"])
            # The LLM wrongly calls C02's organic mention an exposure
            return {"fields": [{"field": "EXPOSED", "status": "observed", "summary": "x", "refs": refs},
                               {"field": "INSPECTED", "status": "observed", "summary": "looked", "refs": refs}]}

    evidence = RunEvidence.load("sim_test", record.run_id)
    assert evidence.persona_records("C01") == {"EXPOSED": 5}
    llm = LLM()
    builder = StructuredReportBuilder(evidence, AssessmentSpec(funnel_fields=["EXPOSED", "INSPECTED"]), llm)
    results = {r["canonical_id"]: r for r in builder.analyse_personas()}
    exposed = {cid: next(f for f in r["fields"] if f["field"] == "EXPOSED") for cid, r in results.items()}
    assert exposed["C01"]["status"] == "observed" and exposed["C01"]["method"] == "orchestration"
    assert exposed["C02"]["status"] == "missing"  # organic mention is not an exposure
    assert next(f for f in results["C02"]["fields"] if f["field"] == "INSPECTED")["status"] == "observed"
    assert "U01" not in results  # existing users are not target consumers
    # The authoritative Product Truth is passed in full, with the timeline as non-citable context
    c01_prompt = next(p for p in llm.prompts if "Persona canonical id: C01" in p)
    assert "AUTHORITATIVE PRODUCT DOCUMENT" in c01_prompt and "No bookings and no live availability" in c01_prompt
    assert "round 5: stage 'Neutral exposure' delivered (records EXPOSED)" in c01_prompt
    rendered = builder.render_personas()
    assert "**Cohort summary**" in rendered
    assert "| EXPOSED | 1 / 0 / 0 | 0 / 0 / 1 |" in rendered
    assert rendered.index("**C01") < rendered.index("**C02")


def test_seed_posts_naming_the_product_under_test_are_rejected(tmp_path):
    from app.services.research_instructions import validate_plan

    plan = validate_plan({"stages": [], "seed_blocked_terms": ["Kamptribe"]})
    assert plan["seed_blocked_terms"] == ["Kamptribe"]
    with open(tmp_path / "experiment_plan.json", "w") as f:
        json.dump({**plan, "private_texts": []}, f)
    config = {"agent_configs": [{"agent_id": 0, "entity_name": "A01"}], "event_config": {"initial_posts": [
        {"content": "Just planned our Utah loop with KampTribe!", "poster_agent_id": 0},
        {"content": "Snow in Yellowstone already?", "poster_agent_id": 0},
    ]}}
    runner = orchestration.RunOrchestrator(config, str(tmp_path))
    assert [p["content"] for p in runner.seed_posts()] == ["Snow in Yellowstone already?"]


def test_graph_build_excludes_assessment_brief_documents(uploads, client, monkeypatch):
    import time

    from app.models.project import ProjectManager
    from app.services import local_graph_extractor as extractor_module

    seen = []

    class CapturingLLM:
        def chat_json(self, messages, temperature=0.2, max_tokens=8192):
            seen.append(messages[-1]["content"])
            return {"entities": [{"name": "C01 Dutch family", "type": "Person", "summary": "family"}], "edges": []}

    project = make_project(documents=[("assessment_brief.md", "Assess C01 against Kamptribe"),
                                      ("personas.md", "C01 is a Dutch family")])
    from app.models.project import ProjectStatus

    project.ontology = {"entity_types": [{"name": "Person"}, {"name": "Organization"}], "edge_types": []}
    project.status = ProjectStatus.ONTOLOGY_GENERATED
    ProjectManager.save_project(project)
    ProjectManager.save_extracted_text("proj_test", "Assess C01 against Kamptribe C01 is a Dutch family")
    monkeypatch.setattr(extractor_module, "LLMClient", lambda: CapturingLLM())
    task_id = client.post("/api/graph/build", json={"project_id": "proj_test"}).json["data"]["task_id"]
    for _ in range(200):
        if client.get(f"/api/graph/task/{task_id}").json["data"]["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    text = " ".join(seen)
    assert "Dutch family" in text and "Kamptribe" not in text


def test_run_is_refused_when_a_saved_stage_reaches_no_agent(uploads, client, fake_popen, monkeypatch):
    import time

    sim_id = _simulation(client, [("product.md", "# Product")], monkeypatch)
    task_id = _prepare(client, sim_id).json["data"]["task_id"]
    for _ in range(200):
        if client.post("/api/simulation/prepare/status", json={"task_id": task_id}).json["data"]["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    client.put(f"/api/simulation/{sim_id}/experiment-plan", json={"stages": [
        {"id": "ghost", "trigger_round": 1, "private_instruction": "x", "target": {"persona_ids": ["C99"]}}]})
    refused = client.post("/api/simulation/start", json={"simulation_id": sim_id})
    assert refused.status_code == 400 and "ghost" in refused.json["error"]
    assert FakePopen.instances == []


def test_discovery_fields_are_derived_from_the_complete_log(uploads):
    from conftest import make_prepared_simulation
    from app.services.research_instructions import validate_plan

    make_project()
    make_prepared_simulation(agents=AGENTS)
    record = RunStore.create_run("sim_test")
    run_dir = RunStore.run_dir("sim_test", record.run_id)
    plan = validate_plan({"stages": [], "derived_fields": {
        "DISCOVERED_ORGANICALLY": {"type": "first_encounter_before_record", "terms": ["Kamptribe"], "record": "EXPOSED"},
        "NOT_DISCOVERED": {"type": "no_encounter_before_record", "terms": ["Kamptribe"], "record": "EXPOSED"}}})
    with open(os.path.join(run_dir, "experiment_plan.json"), "w") as f:
        json.dump({**plan, "stages": [{"id": "exposure", "records": ["EXPOSED"]}], "private_texts": []}, f)
    write_actions(run_dir, "twitter", [
        {"event_type": "private_instruction", "origin": "researcher_instruction", "round": 5, "agent_id": 0,
         "stage_id": "exposure", "records": ["EXPOSED"], "status": "delivered"},
        # C01 only talks about it after the exposure; C02 liked an advocate's post about it first
        {"round": 6, "agent_id": 0, "agent_name": "C01 Family", "action_type": "CREATE_POST",
         "action_args": {"content": "Looking at Kamptribe"}, "origin": "agent_generated"},
        {"round": 2, "agent_id": 1, "agent_name": "C02 Couple", "action_type": "LIKE_POST",
         "action_args": {"post_content": "Kamptribe helped our trip", "post_author_name": "A01"}, "origin": "agent_generated"},
    ], run_id=record.run_id)

    class LLM:
        def chat_json(self, messages, temperature=0.1, max_tokens=None):
            return {"fields": []}

    evidence = RunEvidence.load("sim_test", record.run_id)
    fields = ["NOT_DISCOVERED", "DISCOVERED_ORGANICALLY", "EXPOSED"]
    results = {r["canonical_id"]: {f["field"]: f for f in r["fields"]}
               for r in StructuredReportBuilder(evidence, AssessmentSpec(funnel_fields=fields), LLM()).analyse_personas()}
    assert results["C01"]["NOT_DISCOVERED"]["status"] == "observed"
    assert results["C01"]["DISCOVERED_ORGANICALLY"]["status"] == "missing"
    assert results["C02"]["DISCOVERED_ORGANICALLY"]["status"] == "observed"
    assert results["C02"]["DISCOVERED_ORGANICALLY"]["refs"] == ["TW-00002"]
    assert results["C02"]["NOT_DISCOVERED"]["status"] == "missing"
    assert results["C02"]["EXPOSED"]["status"] == "missing"


def test_funnel_fields_keep_identifier_underscores():
    from app.services.assessment_spec import extract_spec_regex

    spec = extract_spec_regex("For each persona report:\n- NOT_DISCOVERED\n- __PLAN_SELECTED__\n- *trust / friction*\n")
    assert spec.funnel_fields == ["NOT_DISCOVERED", "PLAN_SELECTED", "trust / friction"]
