"""
Run Contract (mirofish.run_contract.v1) integration tests on the committed Kamptribe
fixture: 24 contracted personas, 7 private stages, 50 rounds, Product Truth coverage.
A contracted run must fail closed on every deviation from the contract.
"""

import csv
import hashlib
import json
import os
import sys

import pytest

import contract_fixture as K
from conftest import FakePopen, wait_for_monitor, write_actions

from app.models.task import TaskManager
from app.services.agent_selection import AgentSelectionStore
from app.services.research_instructions import load_project_brief, plan_from_brief
from app.services.run_contract import (
    contract_hash, evaluate_contract_documents, parse_contract_text, validate_contract,
)
from app.services.run_preflight import RunContractPreflight
from app.services.run_store import RunStore
from app.services.simulation_manager import SimulationManager

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_orchestration as orchestration  # noqa: E402

# camel/oasis probe the platform with a subprocess on first import: import them
# before the fake_popen fixture replaces subprocess.Popen
import camel.messages  # noqa: E402,F401
from oasis.social_platform.channel import Channel  # noqa: E402
from oasis.social_platform.platform import Platform  # noqa: E402
from oasis.social_platform.typing import ActionType  # noqa: E402

CONTRACT = K.load_contract()


def _doc(text, name="run_contract.json"):
    return {"original_filename": name, "text": text}


def _sim_file(sim_id, name):
    return os.path.join(RunStore.sim_dir(sim_id), name)


def _read(sim_id, name):
    with open(_sim_file(sim_id, name), encoding="utf-8") as f:
        return json.load(f)


def _write(sim_id, name, data):
    with open(_sim_file(sim_id, name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _runs(sim_id):
    runs_dir = RunStore.runs_dir(sim_id)
    return sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []


def _tree(directory):
    sums = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                sums[os.path.relpath(path, directory)] = hashlib.sha256(f.read()).hexdigest()
    return sums


def _tamper_selection(sim_id, fn):
    selection = _read(sim_id, "agent_selection.json")
    for e in selection["entities"]:
        fn(e)
    _write(sim_id, "agent_selection.json", selection)


def _assert_blocked(client, sim_id):
    """Prepare and start both refuse; no task, no run directory, no process"""
    tasks_before = len(TaskManager()._tasks)
    refused = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False})
    assert refused.status_code == 400 and refused.json["preflight"]["ready_to_prepare"] is False
    assert len(TaskManager()._tasks) == tasks_before
    assert SimulationManager().get_simulation(sim_id).status.value != "preparing"
    started = client.post("/api/simulation/start", json={"simulation_id": sim_id})
    assert started.status_code == 400
    assert _runs(sim_id) == [] and FakePopen.instances == []
    return refused.json["preflight"]


@pytest.fixture
def ready(uploads, client, monkeypatch, fake_popen):
    """Contracted simulation, applied and prepared (ready to start)"""
    sim_id = K.setup(client, monkeypatch)
    K.prepare(client, sim_id)
    return sim_id


# ================================================================ contract parsing

def test_01_exact_contract_loads_with_a_stable_hash():
    state = evaluate_contract_documents([_doc(K.contract_text())])
    assert state["present"] and state["valid"], state["errors"]
    contract = state["contract"]
    assert state["id"] == K.CONTRACT_ID and state["schema"] == "mirofish.run_contract.v1"
    assert contract["counts"] == {"active_agents": 24, "target_consumers": 16, "influencer_context": 8,
                                  "excluded_active": 0, "stages": 7, "max_rounds": 50}
    assert [p["persona_id"] for p in contract["population"]] == K.EXPECTED_IDS
    # Formatting and key order never change the hash; content does
    compact = json.dumps(json.loads(K.contract_text()), separators=(",", ":"), sort_keys=True)
    assert evaluate_contract_documents([_doc(compact, "Run-Contract.txt")])["hash"] == state["hash"]
    fenced = "```json\n" + K.contract_text() + "\n```"
    assert evaluate_contract_documents([_doc(fenced, "run contract.md")])["hash"] == state["hash"]
    changed = K.mutate(CONTRACT, lambda c: c.update(title="other"))
    assert contract_hash(changed) != state["hash"]


def test_02_two_contract_files_block(uploads, client, monkeypatch, fake_popen):
    docs = [("kamptribe_product_truth.md", K.product_truth()), ("run_contract_v2.json", K.contract_text())]
    sim_id = K.setup(client, monkeypatch, documents=docs)
    pf = K.preflight(client, sim_id)
    assert pf["contract"]["present"] and pf["contract"]["valid"] is False
    assert any("exactly one Run Contract" in e for e in pf["contract"]["errors"])
    assert client.post(f"/api/simulation/{sim_id}/run-contract/apply").status_code == 400
    _assert_blocked(client, sim_id)


@pytest.mark.parametrize("text, message", [
    ("", "empty"),
    ("{\"schema\": \"mirofish.run_contract.v1\",", "not valid JSON"),
    (K.contract_text() + "\nAlso activate Kamptribe.", "not valid JSON"),
    ("{\"schema\": 1, \"schema\": 2}", "duplicate key"),
    ("[1, 2]", "one JSON object"),
])
def test_03_malformed_json_blocks(text, message):
    state = evaluate_contract_documents([_doc(text)])
    assert state["present"] and not state["valid"]
    assert message in " ".join(state["errors"])
    with pytest.raises(ValueError):
        parse_contract_text(text) if text else parse_contract_text(None)


def test_03b_unsupported_contract_extension_blocks():
    state = evaluate_contract_documents([_doc(K.contract_text(), "run_contract.pdf")])
    assert state["present"] and not state["valid"]


def test_04_unsupported_schema_blocks():
    bad = K.mutate(CONTRACT, lambda c: c.update(schema="mirofish.run_contract.v2"))
    state = evaluate_contract_documents([_doc(K.contract_text(bad))])
    assert not state["valid"] and any("unsupported schema" in e for e in state["errors"])


def _drop_stage(c):
    c["experiment_plan"]["stages"].pop()


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c["preflight_expectations"].update(target_consumers=15), "disagrees with the contract itself"),
    (lambda c: c["preflight_expectations"].update(stages=6), "disagrees with the contract itself"),
    (_drop_stage, "preflight_expectations.stages=7 disagrees"),
    (lambda c: c["population"].append({"persona_id": "C01", "simulation_role": "prospect_cold",
                                       "evidence_role": "target_consumer"}), "occurs more than once"),
    (lambda c: c["population"][2].update(simulation_role="brand_platform"), "not in role_catalog"),
    (lambda c: c["population"][20].update(evidence_role="consumer"), "evidence_role"),
    (lambda c: c["experiment_plan"]["stages"][1].update(id="s1_cold_trip_planning"), "duplicate stage id"),
    (lambda c: c["experiment_plan"]["stages"][2]["target"].update(persona_ids=["U06"]), "not in the contract population"),
    (lambda c: c["experiment_plan"]["stages"][0].update(trigger_round=0), "trigger_round"),
    (lambda c: c["experiment_plan"]["stages"][0].update(conditon={"type": "always"}), "unknown field"),
    (lambda c: c["experiment_plan"]["stages"][0]["target"].update(agent_ids=[0]), "never agent_ids"),
    (lambda c: c["experiment_plan"]["stages"][2].update(context_sections=["refunds"]), "not a required section"),
    (lambda c: c.update(max_rounds=0), "positive integer"),
    (lambda c: c.update(max_rounds="50"), "positive integer"),
    (lambda c: c.update(max_rounds=True), "positive integer"),
    (lambda c: c.update(contract_id=" "), "contract_id"),
    (lambda c: c.update(population=[]), "population must be a non-empty list"),
    (lambda c: c.update(reject_unlisted_active_entities=False), "reject_unlisted_active_entities"),
    (lambda c: c["source_requirements"].update(documents=[]), "source_requirements"),
    (lambda c: c.update(extra_agents=["Kamptribe"]), "unknown field"),
])
def test_05_invalid_or_self_inconsistent_contracts_block(mutation, message):
    state = evaluate_contract_documents([_doc(K.contract_text(K.mutate(CONTRACT, mutation)))])
    assert not state["valid"], "contract should be refused"
    assert message in " | ".join(state["errors"]), state["errors"]


def test_06_contract_is_excluded_from_ontology_graph_brief_and_report_sources(uploads, client, monkeypatch):
    import io

    from test_ontology_pipeline import ExtractionLLM, OntologyLLM, _wait_task
    from app.api import graph as graph_api
    from app.models.project import ProjectManager
    from app.services import local_graph_extractor as extractor_module
    from app.services.assessment_spec import compose_brief
    from app.services.ontology_generator import OntologyGenerator

    llm = OntologyLLM({"entity_types": [{"name": "Consumer"}], "edges": []})
    monkeypatch.setattr(graph_api, "OntologyGenerator", lambda: OntologyGenerator(llm_client=llm))
    files = [("kamptribe_product_truth.md", K.product_truth().encode()),
             ("run_contract.json", K.contract_text().encode())]
    response = client.post("/api/graph/ontology/generate", content_type="multipart/form-data", data={
        "simulation_requirement": "Assess Kamptribe", "project_name": "k",
        "files": [(io.BytesIO(c), n) for n, c in files]})
    assert response.status_code == 200, response.json
    project_id = response.json["data"]["project_id"]
    assert K.CONTRACT_ID not in llm.user_message and "Product Truth" in llm.user_message
    docs = {d["original_filename"]: d for d in ProjectManager.get_source_documents(project_id)}
    assert docs["run_contract.json"]["extraction_status"] == "ok"  # stored, still parseable
    assert K.CONTRACT_ID not in (ProjectManager.get_extracted_text(project_id) or "")

    seen = []

    class Capture(ExtractionLLM):
        def chat_json(self, messages, **kwargs):
            seen.append(messages[-1]["content"])
            return {"entities": [{"name": "C01 - family", "type": "Consumer"}], "edges": []}

    monkeypatch.setattr(extractor_module, "LLMClient", lambda: Capture([]))
    task_id = client.post("/api/graph/build", json={"project_id": project_id}).json["data"]["task_id"]
    assert _wait_task(client, task_id)["status"] == "completed"
    assert seen and all(K.CONTRACT_ID not in s and "preflight_expectations" not in s for s in seen)

    brief, brief_docs = compose_brief("req", None, [_doc(K.contract_text()), _doc("H1: x", "assessment_brief.md")])
    assert K.CONTRACT_ID not in brief and brief_docs == ["assessment_brief.md"]
    assert K.CONTRACT_ID not in load_project_brief(project_id)["context_text"]


# ================================================================ population

def test_07_exactly_24_requested_ids_resolve_once(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    pf = K.preflight(client, sim_id)
    population = pf["population"]
    assert population["ok"] and population["expected"] == population["resolved"] == population["active"] == 24
    assert [r["persona_id"] for r in population["entities"]] == K.EXPECTED_IDS
    assert {r["persona_id"]: r["name"] for r in population["entities"]} == K.PERSONA_NAMES
    assert population["missing_persona_ids"] == population["duplicate_persona_ids"] == []
    assert population["unexpected_active"] == [] and population["role_mismatches"] == []
    assert pf["roles"] == {**pf["roles"], "target_consumers": 16, "influencer_context": 8, "excluded_active": 0, "ok": True}
    assert pf["experiment_plan"]["ok"] and pf["experiment_plan"]["expected_stages"] == pf["experiment_plan"]["saved_stages"] == 7
    assert pf["rounds"] == {**pf["rounds"], "expected": 50, "ok": True}
    assert pf["sources"]["ok"] and pf["sources"]["coverage_ok"] and pf["sources"]["missing"] == []
    assert pf["ready_to_prepare"] is True and pf["ready_to_start"] is False and pf["errors"] == ["Not prepared yet for this contract (Prepare is required before Start)"]

    selection = AgentSelectionStore.load(sim_id)
    assert selection["confirmed"] is True and selection["contract_hash"] == pf["contract"]["hash"]
    active = {e["name"]: (e["simulation_role"], e["evidence_role"]) for e in selection["entities"] if e["active"]}
    expected = {K.PERSONA_NAMES[p["persona_id"]]: (p["simulation_role"], p["evidence_role"]) for p in CONTRACT["population"]}
    assert active == expected
    for name in K.CONTEXT_NAMES:
        row = next(e for e in selection["entities"] if e["name"] == name)
        assert row["active"] is False, name


def test_08_missing_c16_blocks(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch, drop=("C16",))
    pf = _assert_blocked(client, sim_id)
    assert pf["population"]["missing_persona_ids"] == ["C16"] and pf["population"]["resolved"] == 23
    assert AgentSelectionStore.load(sim_id)["confirmed"] is False


def test_09_two_graph_entities_matching_c16_block(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch, extra=(("C16 Budget students (duplicate)", "Consumer"),))
    pf = _assert_blocked(client, sim_id)
    assert pf["population"]["duplicate_persona_ids"] == ["C16"]
    row = next(r for r in pf["population"]["entities"] if r["persona_id"] == "C16")
    assert sorted(c["name"] for c in row["candidates"]) == ["C16 - Budget student campers", "C16 Budget students (duplicate)"]


def test_10_missing_u05_blocks(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch, drop=("U05",))
    assert _assert_blocked(client, sim_id)["population"]["missing_persona_ids"] == ["U05"]


@pytest.mark.parametrize("name", ["Kamptribe", "Roadtrippers", "RV LIFE / RV Trip Wizard"])
def test_11_12_unexpected_active_brand_platform_blocks(uploads, client, monkeypatch, fake_popen, name):
    sim_id = K.setup(client, monkeypatch)

    def activate(e):
        if e["name"] == name:
            e.update(active=True, simulation_role="context_only", evidence_role="excluded")
    _tamper_selection(sim_id, activate)
    pf = _assert_blocked(client, sim_id)
    assert [u["name"] for u in pf["population"]["unexpected_active"]] == [name]
    assert pf["population"]["active"] == 25 and pf["roles"]["excluded_active"] == 1


def test_13_wrong_role_on_c03_blocks(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    _tamper_selection(sim_id, lambda e: e.update(simulation_role="prospect_switcher") if e["name"].startswith("C03") else None)
    pf = _assert_blocked(client, sim_id)
    assert pf["population"]["role_mismatches"] == [{"persona_id": "C03", "name": K.PERSONA_NAMES["C03"],
                                                    "expected": "prospect_cold", "actual": "prospect_switcher"}]


def test_14_wrong_evidence_role_on_a01_blocks(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    _tamper_selection(sim_id, lambda e: e.update(evidence_role="target_consumer") if e["name"].startswith("A01") else None)
    pf = _assert_blocked(client, sim_id)
    assert [m["persona_id"] for m in pf["population"]["evidence_role_mismatches"]] == ["A01"]
    assert pf["roles"]["target_consumers"] == 17 and pf["roles"]["ok"] is False


def test_15_u03_u04_u05_a03_cannot_silently_stay_inactive(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    dropped = ("U03", "U04", "U05", "A03")
    _tamper_selection(sim_id, lambda e: e.update(active=False) if e["name"][:3] in dropped else None)
    pf = _assert_blocked(client, sim_id)
    assert pf["population"]["inactive_persona_ids"] == list(dropped) and pf["population"]["active"] == 20


def test_16_population_cannot_be_edited_in_contract_mode(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    before = _read(sim_id, "agent_selection.json")
    rows = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    assert rows["contract"]["locked"] is True
    kamptribe = next(e for e in rows["entities"] if e["name"] == "Kamptribe")
    edit = {"entities": [{"entity_uuid": kamptribe["entity_uuid"], "active": True}], "confirmed": True}
    put = client.put(f"/api/simulation/{sim_id}/agent-selection", json=edit)
    assert put.status_code == 409 and put.json["locked_by_contract"] is True
    via_prepare = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "agent_selection": edit})
    assert via_prepare.status_code == 409
    assert _read(sim_id, "agent_selection.json") == before


def test_16b_unapplied_contract_is_not_ready(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch, apply=False)
    pf = _assert_blocked(client, sim_id)
    assert pf["contract"]["valid"] and pf["contract"]["applied"] is False


def test_16c_expected_name_must_match_the_resolved_entity(uploads, client, monkeypatch, fake_popen):
    named = K.mutate(CONTRACT, lambda c: c["population"][0].update(expected_name="C01 - Somebody else"))
    sim_id = K.setup(client, monkeypatch, contract=named)
    assert _assert_blocked(client, sim_id)["population"]["name_mismatch_persona_ids"] == ["C01"]


# ================================================================ plan

def _plan_mismatch(client, monkeypatch, fn):
    sim_id = K.setup(client, monkeypatch)
    plan = _read(sim_id, "experiment_plan.json")
    fn(plan)
    _write(sim_id, "experiment_plan.json", plan)
    return _assert_blocked(client, sim_id)["experiment_plan"]


def test_17_exact_seven_stages_pass(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    plan = K.preflight(client, sim_id)["experiment_plan"]
    assert plan["ok"] and plan["mismatches"] == [] and plan["unresolved_stage_targets"] == []
    assert [s["id"] for s in plan["stages"]] == [s["id"] for s in CONTRACT["experiment_plan"]["stages"]]
    assert plan["seed_blocked_terms"] == ["Kamptribe"]


def test_18_six_saved_stages_block(uploads, client, monkeypatch, fake_popen):
    plan = _plan_mismatch(client, monkeypatch, lambda p: p["stages"].pop())
    assert plan["saved_stages"] == 6 and {"stage_id": "s7_final_choice", "field": "stage", "expected": "present",
                                          "actual": "missing"} in plan["mismatches"]


def test_19_wrong_trigger_round_blocks(uploads, client, monkeypatch, fake_popen):
    plan = _plan_mismatch(client, monkeypatch, lambda p: p["stages"][4].update(trigger_round=12))
    assert [(m["stage_id"], m["field"], m["expected"], m["actual"]) for m in plan["mismatches"]] == [
        ("s5_neutral_exposure", "trigger_round", 15, 12)]


def test_20_wrong_target_ids_block(uploads, client, monkeypatch, fake_popen):
    plan = _plan_mismatch(client, monkeypatch, lambda p: p["stages"][3]["target"].update(persona_ids=["A01", "A02"]))
    assert [(m["stage_id"], m["field"]) for m in plan["mismatches"]] == [("s4_advocates_context", "target")]


def test_21_changed_private_instruction_blocks(uploads, client, monkeypatch, fake_popen):
    plan = _plan_mismatch(client, monkeypatch, lambda p: p["stages"][0].update(private_instruction="Post about Kamptribe."))
    assert [(m["stage_id"], m["field"]) for m in plan["mismatches"]] == [("s1_cold_trip_planning", "private_instruction")]


@pytest.mark.parametrize("field, value", [
    ("context_sections", ["pricing"]), ("records", []), ("condition", {"type": "always", "terms": []}),
    ("completion", {"type": "on_delivery", "terms": [], "action_types": []}), ("order", 9),
])
def test_21b_every_semantic_stage_field_is_compared(uploads, client, monkeypatch, fake_popen, field, value):
    plan = _plan_mismatch(client, monkeypatch, lambda p: p["stages"][5].update({field: value}))
    assert ("s6_pricing_decision", field) in [(m["stage_id"], m["field"]) for m in plan["mismatches"]]


def test_21c_removed_seed_blocked_term_blocks(uploads, client, monkeypatch, fake_popen):
    plan = _plan_mismatch(client, monkeypatch, lambda p: p.update(seed_blocked_terms=[]))
    assert "seed_blocked_terms" in [m["field"] for m in plan["mismatches"]]


def test_22_missing_experiment_plan_blocks(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    os.remove(_sim_file(sim_id, "experiment_plan.json"))
    plan = _assert_blocked(client, sim_id)["experiment_plan"]
    assert plan["mismatches"][0]["field"] == "experiment_plan.json" and plan["saved_stages"] == 0


def test_23_brief_prose_never_generates_or_replaces_contracted_stages(ready, client, fake_popen):
    sim_id = ready
    # The hazard this guards against: the brief parser finds "stages" in cohort prose
    assert len(plan_from_brief(K.BRIEF, ["C03", "C04", "C05"])["stages"]) == 3
    plan = client.get(f"/api/simulation/{sim_id}/experiment-plan").json["data"]
    assert plan["locked_by_contract"] is True
    assert [s["id"] for s in plan["stages"]] == [s["id"] for s in CONTRACT["experiment_plan"]["stages"]]
    assert client.put(f"/api/simulation/{sim_id}/experiment-plan", json={"stages": []}).status_code == 409
    config = _read(sim_id, "simulation_config.json")
    contract_texts = sorted(s["private_instruction"] for s in CONTRACT["experiment_plan"]["stages"])
    assert sorted(config["private_texts"]) == contract_texts  # no brief prose became private text
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    with open(os.path.join(RunStore.run_dir(sim_id, run_id), "experiment_plan.json"), encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["source"] == "saved" and len(resolved["stages"]) == 7
    assert sorted(resolved["private_texts"]) == contract_texts
    assert not any(s["id"].startswith("brief_") for s in resolved["stages"])


def test_23b_unapplied_contract_plan_view_never_falls_back_to_the_brief(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch, apply=False)
    plan = client.get(f"/api/simulation/{sim_id}/experiment-plan").json["data"]
    assert plan["stages"] == [] and plan["locked_by_contract"] is True


# ================================================================ sources

def test_24_missing_source_requirements_blocks(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch)
    os.remove(_sim_file(sim_id, "source_requirements.json"))
    sources = _assert_blocked(client, sim_id)["sources"]
    assert sources["requirements_present"] is False and sources["ok"] is False
    put = client.put(f"/api/simulation/{sim_id}/source-requirements", json=CONTRACT["source_requirements"])
    assert put.status_code == 409


def test_25_missing_product_truth_blocks(uploads, client, monkeypatch, fake_popen):
    docs = [("assessment_brief.md", K.BRIEF), ("notes.md", "# Notes\nCamping.")]
    sim_id = K.setup(client, monkeypatch, documents=docs)
    pf = _assert_blocked(client, sim_id)
    assert pf["sources"]["coverage_ok"] is False and pf["sources"]["documents"][0]["files"] == []
    assert "Required document 'Product Truth' is not among the uploaded sources" in pf["errors"]


def test_26_missing_anchored_section_blocks(uploads, client, monkeypatch, fake_popen):
    truth = K.product_truth().replace("### Trip Pass {#trip-pass}", "### Trip Pass")
    sources = _assert_blocked(client, K.setup(client, monkeypatch, truth=truth))["sources"]
    assert sources["missing"] == ["product_truth:trip-pass"]
    row = next(s for s in sources["documents"][0]["sections"] if s["id"] == "trip-pass")
    assert row["status"] == "missing"


def test_27_missing_required_term_blocks(uploads, client, monkeypatch, fake_popen):
    truth = K.product_truth().replace("19.99", "twenty")
    sources = _assert_blocked(client, K.setup(client, monkeypatch, truth=truth))["sources"]
    row = next(s for s in sources["documents"][0]["sections"] if s["id"] == "pricing")
    assert row["status"] == "incomplete" and row["missing_terms"] == ["19.99"]


def test_28_complete_product_truth_passes(uploads, client, monkeypatch, fake_popen):
    sources = K.preflight(client, K.setup(client, monkeypatch))["sources"]
    assert sources["ok"] and sources["requirements_match"]
    assert sources["documents"][0]["files"] == ["kamptribe_product_truth.md"]
    assert {s["id"]: s["status"] for s in sources["documents"][0]["sections"]} == {
        "product-overview": "covered", "tiers": "covered", "pricing": "covered",
        "trip-pass": "covered", "not-built": "covered"}


# ================================================================ rounds

def test_30_33_omitted_max_rounds_runs_exactly_the_contract_50(ready, client, fake_popen):
    sim_id = ready
    config = _read(sim_id, "simulation_config.json")
    tc = config["time_config"]
    assert tc["total_simulation_hours"] * 60 // tc["minutes_per_round"] == 40  # the generated length is not used
    data = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]
    wait_for_monitor(sim_id)
    record = RunStore.get_run(sim_id, data["run_id"])
    assert record.max_rounds == 50 and record.total_rounds == 50 and data["max_rounds_applied"] == 50
    cmd = FakePopen.instances[0].cmd
    assert cmd[cmd.index("--max-rounds") + 1] == "50"
    run_dir = RunStore.run_dir(sim_id, data["run_id"])
    # The OASIS script resolves the same 50 from the run's own contract snapshot
    assert orchestration.resolve_total_rounds(config, run_dir, None) == (50, "run_contract")
    assert orchestration.resolve_total_rounds(config, run_dir, 50) == (50, "run_contract")
    with pytest.raises(ValueError):
        orchestration.resolve_total_rounds(config, run_dir, 40)
    assert orchestration.resolve_total_rounds(config, RunStore.sim_dir(sim_id) + "/nowhere", None) == (40, "time_config")


@pytest.mark.parametrize("rounds", [40, 51, "40"])
def test_31_32_supplied_max_rounds_mismatch_is_refused_before_run_creation(ready, client, fake_popen, rounds):
    refused = client.post("/api/simulation/start", json={"simulation_id": ready, "max_rounds": rounds})
    assert refused.status_code == 400 and "Locked by Run Contract" in refused.json["error"]
    assert _runs(ready) == [] and FakePopen.instances == []
    explicit = client.post("/api/simulation/start", json={"simulation_id": ready, "max_rounds": 50})
    assert explicit.status_code == 200


def test_31b_platform_mismatch_is_refused(ready, client, fake_popen):
    refused = client.post("/api/simulation/start", json={"simulation_id": ready, "platform": "twitter"})
    assert refused.status_code == 400 and _runs(ready) == []


def test_33b_runner_refuses_a_conflicting_max_rounds_for_a_contracted_run(ready, client, fake_popen):
    from app.services.simulation_runner import SimulationRunner

    pf = RunContractPreflight.evaluate(ready)
    record = RunStore.create_run(ready, "parallel", 50, preflight=pf)
    with pytest.raises(ValueError, match="differs from the Run Contract"):
        SimulationRunner.start_simulation(ready, max_rounds=40, run_id=record.run_id)
    assert FakePopen.instances == []


# ================================================================ prepare

def test_35_applying_and_confirming_never_prepares(uploads, client, monkeypatch, fake_popen):
    def boom(*args, **kwargs):
        raise AssertionError("prepare must not run")

    monkeypatch.setattr(SimulationManager, "prepare_simulation", boom)
    tasks_before = len(TaskManager()._tasks)
    sim_id = K.setup(client, monkeypatch)
    again = client.post(f"/api/simulation/{sim_id}/run-contract/apply")
    assert again.status_code == 200 and again.json["data"]["confirmed"] is True
    assert len(TaskManager()._tasks) == tasks_before
    state = SimulationManager().get_simulation(sim_id)
    assert state.status.value == "created" and state.prepare_task_id is None
    assert not os.path.exists(_sim_file(sim_id, "agent_roster.json"))


def test_36_failing_preflight_creates_no_prepare_task(uploads, client, monkeypatch, fake_popen):
    sim_id = K.setup(client, monkeypatch, drop=("C07",))
    pf = _assert_blocked(client, sim_id)
    assert "C07: no eligible graph entity resolves to this persona id" in pf["errors"]
    assert SimulationManager().get_simulation(sim_id).prepare_task_id is None
    for name in ("agent_roster.json", "reddit_profiles.json", "twitter_profiles.csv", "simulation_config.json"):
        assert not os.path.exists(_sim_file(sim_id, name)), name


def test_37_41_explicit_prepare_freezes_exactly_the_24_contract_agents(ready, client):
    sim_id = ready
    pf = K.preflight(client, sim_id)
    prep = pf["preparation"]
    assert pf["ready_to_start"] is True and pf["errors"] == []
    assert (prep["roster_active"], prep["profile_count_twitter"], prep["profile_count_reddit"], prep["config_agent_count"]) == (24, 24, 24, 24)
    assert prep["selection_hash_matches"] and prep["roster_matches_contract"] and prep["profile_persona_ids_match"]
    roster = _read(sim_id, "agent_roster.json")
    assert [a["name"][:3] for a in roster["active_agents"]] == K.EXPECTED_IDS
    assert [a["agent_id"] for a in roster["active_agents"]] == list(range(24))
    assert sorted(c["name"] for c in roster["context_entities"]) == sorted(K.CONTEXT_NAMES)
    reddit = _read(sim_id, "reddit_profiles.json")
    with open(_sim_file(sim_id, "twitter_profiles.csv"), encoding="utf-8", newline="") as f:
        twitter = list(csv.DictReader(f))
    expected_names = [K.PERSONA_NAMES[p] for p in K.EXPECTED_IDS]
    assert [p["name"] for p in reddit] == [r["name"] for r in twitter] == expected_names
    config = _read(sim_id, "simulation_config.json")
    assert [(a["agent_id"], a["entity_name"], a["simulation_role"], a["evidence_role"]) for a in config["agent_configs"]] == [
        (i, K.PERSONA_NAMES[p["persona_id"]], p["simulation_role"], p["evidence_role"]) for i, p in enumerate(CONTRACT["population"])]


def test_42_post_prepare_mismatch_marks_preparation_failed(uploads, client, monkeypatch, fake_popen):
    from app.services.oasis_profile_generator import OasisProfileGenerator

    original = OasisProfileGenerator.save_profiles

    def drop_last_twitter(self, profiles, file_path, platform="reddit"):
        return original(self, profiles[:-1] if platform == "twitter" else profiles, file_path, platform)

    monkeypatch.setattr(OasisProfileGenerator, "save_profiles", drop_last_twitter)
    sim_id = K.setup(client, monkeypatch)
    failed = K.prepare(client, sim_id, expect="failed")
    assert "Post-prepare verification against the Run Contract failed" in failed["error"]
    assert "twitter_profiles.csv holds 23 profiles" in failed["error"]
    assert SimulationManager().get_simulation(sim_id).status.value == "failed"
    assert client.post("/api/simulation/start", json={"simulation_id": sim_id}).status_code == 400
    assert _runs(sim_id) == []


def test_42b_tampering_after_prepare_makes_the_simulation_non_startable(ready, client, fake_popen):
    reddit = _read(ready, "reddit_profiles.json")
    reddit.append({**reddit[0], "user_id": 24, "name": "Kamptribe"})
    _write(ready, "reddit_profiles.json", reddit)
    pf = K.preflight(client, ready)
    assert pf["ready_to_start"] is False and pf["preparation"]["profile_count_reddit"] == 25
    assert client.post("/api/simulation/start", json={"simulation_id": ready}).status_code == 400
    assert _runs(ready) == []


# ================================================================ start / lifecycle

def test_43_45_reading_state_never_creates_a_run(ready, client):
    for _ in range(2):  # navigation, mount and reload only read
        for path in ("agent-selection", "preflight", "run-status", "runs", "experiment-plan", "source-requirements"):
            assert client.get(f"/api/simulation/{ready}/{path}").status_code == 200, path
        client.post("/api/simulation/prepare/status", json={"simulation_id": ready})
    assert _runs(ready) == [] and FakePopen.instances == []


def test_46_49_explicit_start_creates_one_run_and_revisits_attach(ready, client):
    FakePopen.finish_immediately = False
    first = client.post("/api/simulation/start", json={"simulation_id": ready}).json["data"]
    assert first["started"] is True and len(_runs(ready)) == 1
    again = client.post("/api/simulation/start", json={"simulation_id": ready}).json["data"]
    assert again["already_running"] is True and again["run_id"] == first["run_id"]
    assert len(FakePopen.instances) == 1 and len(_runs(ready)) == 1
    FakePopen.instances[0].finish(0)
    wait_for_monitor(ready)
    after = client.post("/api/simulation/start", json={"simulation_id": ready}).json["data"]
    assert after["run_exists"] is True and after["run_id"] == first["run_id"] and len(_runs(ready)) == 1


def test_47_start_with_failing_preflight_creates_nothing(ready, client):
    _tamper_selection(ready, lambda e: e.update(active=True) if e["name"] == "Kamptribe" else None)
    refused = client.post("/api/simulation/start", json={"simulation_id": ready})
    assert refused.status_code == 400 and refused.json["preflight"]["ready_to_start"] is False
    assert "Unexpected active agents (not in the Run Contract): 'Kamptribe'" in refused.json["preflight"]["errors"]
    assert _runs(ready) == [] and FakePopen.instances == []


def test_50_51_restart_keeps_the_old_run_and_historical_runs_stay_readable(ready, client):
    from app.services.run_evidence import RunEvidence

    # A historical run from before the contract existed (no contract, no preflight)
    historic = os.path.join(RunStore.runs_dir(ready), "run_20250101000000_000000")
    os.makedirs(historic)
    with open(os.path.join(historic, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": "run_20250101000000_000000", "simulation_id": ready, "status": "completed",
                   "created_at": "2025-01-01T00:00:00"}, f)
    write_actions(historic, "twitter", [{"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST",
                                         "action_args": {"content": "old run"}}], run_id="run_20250101000000_000000")
    historic_sums = _tree(historic)

    first = client.post("/api/simulation/start", json={"simulation_id": ready, "new_run": True}).json["data"]
    wait_for_monitor(ready)
    first_sums = _tree(RunStore.run_dir(ready, first["run_id"]))
    second = client.post("/api/simulation/start", json={"simulation_id": ready, "new_run": True}).json["data"]
    wait_for_monitor(ready)
    assert second["previous_run_id"] == first["run_id"] and second["run_id"] != first["run_id"]
    assert _tree(RunStore.run_dir(ready, first["run_id"])) == first_sums
    assert _tree(historic) == historic_sums
    assert [a["content"] for a in RunEvidence.load(ready, "run_20250101000000_000000").actions] == ["old run"]
    assert len(_runs(ready)) == 3


def test_51b_legacy_run_stays_readable_beside_contracted_runs(ready, client):
    sim_dir = RunStore.sim_dir(ready)
    write_actions(sim_dir, "twitter", [{"round": 1, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST",
                                        "action_args": {"content": "legacy"}}])
    with open(os.path.join(sim_dir, "twitter", "actions.jsonl"), "rb") as f:
        legacy = f.read()
    runs = client.get(f"/api/simulation/{ready}/runs").json["data"]["runs"]
    assert runs[0]["run_id"] == "legacy"
    client.post("/api/simulation/start", json={"simulation_id": ready, "new_run": True})
    wait_for_monitor(ready)
    with open(os.path.join(sim_dir, "twitter", "actions.jsonl"), "rb") as f:
        assert f.read() == legacy


# ================================================================ snapshot

def test_52_53_contracted_run_snapshots_contract_preflight_and_every_artifact(ready, client):
    pf = K.preflight(client, ready)
    run_id = client.post("/api/simulation/start", json={"simulation_id": ready}).json["data"]["run_id"]
    wait_for_monitor(ready)
    run_dir = RunStore.run_dir(ready, run_id)
    for name in ("run_contract.json", "preflight.json", "simulation_config.json", "reddit_profiles.json",
                 "twitter_profiles.csv", "agent_roster.json", "agent_selection.json", "experiment_plan.json",
                 "source_requirements.json", "source_coverage.json", os.path.join("assessment", "brief.md"),
                 os.path.join("sources", "manifest.json")):
        assert os.path.exists(os.path.join(run_dir, name)), name
    with open(os.path.join(run_dir, "preflight.json"), encoding="utf-8") as f:
        snap_pf = json.load(f)
    assert snap_pf["ready_to_start"] is True and snap_pf["fingerprint"] == pf["fingerprint"]
    for name, digest in snap_pf["artifacts"].items():
        if name == "experiment_plan.json":
            continue  # replaced by the run's resolved plan (checked below)
        with open(os.path.join(run_dir, name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, name
    with open(os.path.join(run_dir, "experiment_plan.json"), encoding="utf-8") as f:
        resolved = json.load(f)
    assert resolved["contract_hash"] == pf["contract"]["hash"] and len(resolved["stages"]) == 7
    with open(os.path.join(run_dir, "run_contract.json"), encoding="utf-8") as f:
        assert json.load(f)["contract"] == CONTRACT  # the original contract, verbatim
    manifest = json.load(open(os.path.join(run_dir, "sources", "manifest.json"), encoding="utf-8"))
    assert "run_contract.json" in [d["original_filename"] for d in manifest["documents"]]

    contract = RunStore.get_run(ready, run_id).snapshot["contract"]
    assert contract["contract_id"] == K.CONTRACT_ID and contract["contract_hash"] == pf["contract"]["hash"]
    assert (contract["active_agents"], contract["target_consumers"], contract["influencer_context"],
            contract["excluded_active"], contract["max_rounds"], contract["stages"]) == (24, 16, 8, 0, 50, 7)
    assert contract["source_coverage_ok"] is True
    assert contract["selection_hash"] == _read(ready, "agent_selection.json")["selection_hash"]
    assert contract["experiment_plan_hash"] == pf["artifacts"]["experiment_plan.json"]
    assert contract["source_requirements_hash"] == pf["artifacts"]["source_requirements.json"]


def test_54_contract_snapshot_error_refuses_and_cleans_the_partial_run(ready, client, monkeypatch):
    from app.services import run_evidence

    def broken(run_dir, project_id):
        raise OSError("disk full")

    monkeypatch.setattr(run_evidence, "snapshot_run_inputs", broken)
    refused = client.post("/api/simulation/start", json={"simulation_id": ready})
    assert refused.status_code == 400 and "Contracted run refused: disk full" in refused.json["error"]
    assert _runs(ready) == [] and FakePopen.instances == []
    with pytest.raises(ValueError, match="ready run preflight"):
        RunStore.create_run(ready, "parallel", 50)
    assert _runs(ready) == []


def test_54b_files_changed_after_the_preflight_are_never_snapshotted(ready, client):
    pf = RunContractPreflight.evaluate(ready)
    roster = _read(ready, "agent_roster.json")
    roster["active_agents"] = roster["active_agents"][:-1]
    _write(ready, "agent_roster.json", roster)
    with pytest.raises(ValueError, match="agent_roster.json changed after the preflight passed"):
        RunStore.create_run(ready, "parallel", 50, preflight=pf)
    assert _runs(ready) == []


# ================================================================ privacy / evidence

def _run_with_actions(client, sim_id, actions):
    FakePopen.actions = actions
    run_id = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["run_id"]
    wait_for_monitor(sim_id)
    return run_id


def test_55_private_instructions_are_delivered_only_into_intended_agent_memory(ready, client):
    run_id = _run_with_actions(client, ready, {"twitter": [], "reddit": []})
    run_dir = RunStore.run_dir(ready, run_id)
    with open(os.path.join(run_dir, "simulation_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    orch = orchestration.RunOrchestrator(config, run_dir)
    targets = {s["id"]: s["resolved_agent_ids"] for s in orch.plan["stages"]}
    assert targets["s1_cold_trip_planning"] == list(range(0, 8))
    assert targets["s2_switcher_friction"] == list(range(8, 16))
    assert targets["s3_existing_users_experience"] == list(range(16, 21))
    assert targets["s4_advocates_context"] == [21, 22, 23]
    assert targets["s5_neutral_exposure"] == targets["s6_pricing_decision"] == list(range(16))

    memories = {}

    class Agent:
        def __init__(self, aid):
            self.aid = aid

        def update_memory(self, message, role):
            memories.setdefault(self.aid, []).append(message.content)

    delivered = {}
    for rnd in range(1, 51):
        for aid in orch.deliver_due(rnd, lambda aid: Agent(aid)):
            delivered.setdefault(rnd, []).append(aid)
    assert delivered[1] == list(range(16)) and delivered[2] == list(range(16, 21)) and delivered[3] == [21, 22, 23]
    by_text = {s["private_instruction"]: s for s in CONTRACT["experiment_plan"]["stages"]}
    for aid, texts in memories.items():
        for text in texts:
            stage = next(s for i, s in by_text.items() if i in text)
            assert aid in targets[stage["id"]]
    assert all("Kamptribe" not in t for ts in memories.values() for t in ts if "Factual information" not in t)


def test_56_private_instruction_never_reaches_the_public_oasis_db_or_api(ready, client):
    import asyncio
    import sqlite3

    run_id = _run_with_actions(client, ready, {"twitter": [], "reddit": []})
    run_dir = RunStore.run_dir(ready, run_id)
    with open(os.path.join(run_dir, "simulation_config.json"), encoding="utf-8") as f:
        orch = orchestration.RunOrchestrator(json.load(f), run_dir)
    instruction = CONTRACT["experiment_plan"]["stages"][4]["private_instruction"]
    db_path = os.path.join(run_dir, "twitter_simulation.db")

    async def drive():
        channel = Channel()
        platform = Platform(db_path=db_path, channel=channel, recsys_type="twitter")
        assert orch.guard_platform(platform) == ["create_post", "create_comment", "quote_post"]
        task = asyncio.create_task(platform.running())
        results = []
        for content in (instruction, "Found a new planner tonight. " + instruction, "Is 19.99 for a Trip Pass fair?"):
            mid = await channel.write_to_receive_queue((3, content, ActionType.CREATE_POST.value))
            results.append((await channel.read_from_send_queue(mid))[2])
        mid = await channel.write_to_receive_queue((4, (1, instruction), ActionType.CREATE_COMMENT.value))
        results.append((await channel.read_from_send_queue(mid))[2])
        await channel.write_to_receive_queue((None, None, ActionType.EXIT.value))
        await task
        return results

    results = asyncio.run(drive())
    assert [r["success"] for r in results] == [False, True, True, False]
    conn = sqlite3.connect(db_path)
    stored = [r[0] for r in conn.execute("SELECT content FROM post")] + [r[0] for r in conn.execute("SELECT content FROM comment")]
    traces = [r[0] for r in conn.execute("SELECT info FROM trace")]
    conn.close()
    assert stored == ["Found a new planner tonight.", "Is 19.99 for a Trip Pass fair?"]
    for text in stored + traces:
        assert not orchestration.reproduces_private_text(text, [instruction])
    posts = client.get(f"/api/simulation/{ready}/posts", query_string={"platform": "twitter", "run_id": run_id}).json
    assert posts["success"] and len(posts["data"]["posts"]) == 2 and instruction not in json.dumps(posts)
    for path in ("actions", "timeline"):
        assert instruction not in json.dumps(client.get(f"/api/simulation/{ready}/{path}").json)


def test_57_system_seed_never_contains_kamptribe(ready, client):
    config = _read(ready, "simulation_config.json")
    seeds = [p["content"] for p in config["event_config"]["initial_posts"]]
    assert seeds == ["Camping season is open - where are you heading this summer?"]
    rejected = config["event_config"]["rejected_seed_posts"]
    assert rejected[0]["reason"] == "mentions blocked term 'Kamptribe'"
    # Runtime last line of defence, using the run's own plan
    tampered = {**config, "event_config": {"initial_posts": [{"content": "Kamptribe is great", "poster_agent_id": 21}]}}
    publish, refused = orchestration.safe_seed_posts(tampered, [], ["Kamptribe"])
    assert publish == [] and refused[0]["reason"] == "mentions blocked term 'kamptribe'"


def test_58_product_truth_facts_may_become_agent_generated_content(ready, client):
    run_id = _run_with_actions(client, ready, {"twitter": [], "reddit": []})
    with open(os.path.join(RunStore.run_dir(ready, run_id), "experiment_plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    pricing = next(s for s in plan["stages"] if s["id"] == "s6_pricing_decision")
    assert "19.99 USD once" in pricing["context"]  # verbatim Product Truth section
    fact = "Planner costs 3.99 USD per month or 29.99 USD per year. A Trip Pass never renews automatically."
    texts = plan["private_texts"]
    assert orchestration.sanitize_public_text(fact, texts) == (fact, "clean")
    assert orchestration.classify_origin({"action_type": "CREATE_POST", "action_args": {"content": fact}}, texts) == ("agent_generated", None)


def test_59_60_only_agent_generated_target_consumer_actions_count(ready, client):
    from app.services.run_evidence import RunEvidence

    post = lambda aid, name, text, origin="agent_generated": {  # noqa: E731
        "round": 20, "agent_id": aid, "agent_name": name, "action_type": "CREATE_POST",
        "action_args": {"content": text}, "origin": origin}
    run_id = _run_with_actions(client, ready, {"twitter": [
        post(0, K.PERSONA_NAMES["C01"], "Trying the Trip Pass for our summer trip"),
        post(15, K.PERSONA_NAMES["C16"], "Too expensive for students"),
        post(17, K.PERSONA_NAMES["U02"], "Planner saved me hours"),
        post(21, K.PERSONA_NAMES["A01"], "My favourite planner"),
        post(21, K.PERSONA_NAMES["A01"], "Camping season is open", "system_seed"),
    ], "reddit": []})
    evidence = RunEvidence.load(ready, run_id)
    metrics = evidence.consumer_metrics()
    assert metrics["actions"] == 2 and metrics["by_persona"] == {"C01": 1, "C16": 1}
    assert metrics["target_consumer_personas"] == [f"C{i:02d}" for i in range(1, 17)]
    assert metrics["excluded_by_evidence_role"] == {"influencer_context": 2}
    assert metrics["excluded_by_origin"] == {"system_seed": 1}
    roles = {p["canonical_id"]: p["evidence_role"] for p in evidence.population()}
    assert all(roles[p] == "influencer_context" for p in K.EXPECTED_IDS if p[0] in "UA")


# ================================================================ report

class _ReportLLM:
    def chat_json(self, messages, temperature=0.1, max_tokens=None):
        import re
        refs = re.findall(r"\[((?:TW|RD)-\d{5})\]", messages[-1]["content"])
        if "You analyse ONE persona" in messages[0]["content"]:
            return {"fields": [{"field": "outcome", "status": "observed", "summary": "s", "refs": refs}]}
        return {"verdict": "supported", "rationale": "r", "evidence_for": [{"summary": "s", "refs": refs}]}


def test_61_62_report_reads_the_exact_run_and_its_snapshot(ready, client):
    from app.services.report_agent import ReportAgent

    run_a = _run_with_actions(client, ready, {"twitter": [{"round": 1, "agent_id": 0, "agent_name": K.PERSONA_NAMES["C01"],
                                                            "action_type": "CREATE_POST", "action_args": {"content": "run A"}}], "reddit": []})
    run_b = client.post("/api/simulation/start", json={"simulation_id": ready, "new_run": True}).json["data"]["run_id"]
    wait_for_monitor(ready)
    # Later mutable project/simulation state never reaches the report of run A
    roster = _read(ready, "agent_roster.json")
    roster["active_agents"][0]["evidence_role"] = "excluded"
    _write(ready, "agent_roster.json", roster)
    agent = ReportAgent(graph_id=K.GRAPH_ID, simulation_id=ready, simulation_requirement="r",
                        llm_client=_ReportLLM(), graph_tools=object(), run_id=run_a)
    evidence = agent._ensure_evidence()
    assert evidence.run_id == run_a != run_b and [a["content"] for a in evidence.actions] == ["run A"]
    assert evidence.roles_for_agent(0)["evidence_role"] == "target_consumer"
    assert evidence.run_contract["contract_hash"] == K.preflight(client, ready)["contract"]["hash"]
    assert evidence.control_documents == ["run_contract.json"]
    assert "run_contract.json" not in [d["original_filename"] for d in evidence.sources]


def test_63_66_target_consumers_reported_individually_against_product_truth(ready, client):
    from app.services.assessment_spec import extract_spec_regex
    from app.services.report_evidence_sections import StructuredReportBuilder
    from app.services.run_evidence import RunEvidence

    run_id = _run_with_actions(client, ready, {"twitter": [
        {"round": 20, "agent_id": 0, "agent_name": K.PERSONA_NAMES["C01"], "action_type": "CREATE_POST",
         "action_args": {"content": "Bought the Trip Pass"}, "origin": "agent_generated"},
        {"round": 20, "agent_id": 17, "agent_name": K.PERSONA_NAMES["U02"], "action_type": "CREATE_POST",
         "action_args": {"content": "Planner is worth it"}, "origin": "system_seed"}], "reddit": []})
    evidence = RunEvidence.load(ready, run_id)
    spec = extract_spec_regex(evidence.brief_text, {"C", "U", "A"})
    assert spec.persona_ids == [f"C{i:02d}" for i in range(1, 17)]
    builder = StructuredReportBuilder(evidence, spec, _ReportLLM())
    personas = builder.analyse_personas()
    assert [p["canonical_id"] for p in personas] == spec.persona_ids
    assert all(p["evidence_role"] == "target_consumer" and p["counts_in_conversion_metrics"] for p in personas)
    silent = [p for p in personas if p["canonical_id"] != "C01"]
    assert all(f["status"] == "missing" and not f.get("refs") for p in silent for f in p["fields"])
    assert not any(p["canonical_id"].startswith(("U", "A")) for p in personas)
    hypotheses = builder.assess_hypotheses()
    # One citable agent-generated action is never enough for a verdict
    assert [h["verdict"] for h in hypotheses] == ["insufficient_evidence", "insufficient_evidence"]
    context = builder._context_documents()
    assert "AUTHORITATIVE PRODUCT DOCUMENT (Product Truth, complete)" in context
    assert "no live campsite availability" in context and K.CONTRACT_ID not in context


# ================================================================ network / state

def test_69_70_persisted_state_and_reload_reproduce_the_same_preflight(ready, client):
    first = K.preflight(client, ready)
    second = K.preflight(client, ready)
    strip = lambda d: {k: v for k, v in d.items() if k != "checked_at"}  # noqa: E731
    assert strip(first) == strip(second) and first["fingerprint"] == second["fingerprint"]
    view = client.get(f"/api/simulation/{ready}/agent-selection").json["data"]
    assert view["confirmed"] is True and view["summary"]["active_agents"] == 24
    assert view["contract"]["applied_contract_hash"] == first["contract"]["hash"]
    # Re-applying the same contract is idempotent and keeps the preparation
    again = client.post(f"/api/simulation/{ready}/run-contract/apply").json["data"]
    assert again["preparation_invalidated"] is False and again["preflight"]["ready_to_start"] is True


def test_simulations_without_a_contract_keep_existing_behaviour(uploads, client, monkeypatch, fake_popen):
    from conftest import select_entities

    K.install_fake_config(monkeypatch)
    K.make_contract_project(contract=False)
    K.build_graph()
    sim_id = K.create_simulation(client)
    pf = K.preflight(client, sim_id)
    assert pf["mode"] == "standard" and pf["contract"]["present"] is False and pf["ready_to_start"] is None
    assert client.post(f"/api/simulation/{sim_id}/run-contract/apply").status_code == 400
    view = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    assert view["contract"]["locked"] is False
    plan = client.get(f"/api/simulation/{sim_id}/experiment-plan").json["data"]
    assert plan["locked_by_contract"] is False and plan["stages"] and {s["source"] for s in plan["stages"]} == {"brief"}
    select_entities(sim_id, active={K.PERSONA_NAMES["C01"], "Kamptribe"})
    K.prepare(client, sim_id)
    started = client.post("/api/simulation/start", json={"simulation_id": sim_id, "max_rounds": 3})
    assert started.status_code == 200
    wait_for_monitor(sim_id)
    record = RunStore.get_run(sim_id, started.json["data"]["run_id"])
    assert record.max_rounds == 3 and "contract" not in record.snapshot
