"""
Synthetic end-to-end pipeline test (no paid services):
upload sources -> ontology -> local graph -> personas/prepare -> fake OASIS run ->
completed run -> report from raw evidence -> audit ZIP export
"""

import io
import json
import time
import zipfile

import fitz

from conftest import BRIEF, FakePopen, wait_for_monitor
from test_ontology_pipeline import ExtractionLLM, OntologyLLM
from test_report_evidence import FakeLLM

from app.api import graph as graph_api
from app.api import report as report_api
from app.services import local_graph_extractor as extractor_module
from app.services.ontology_generator import OntologyGenerator
from app.services.report_agent import ReportAgent
from app.services.run_store import RunStatus, RunStore
from app.services.simulation_config_generator import SimulationConfigGenerator


def _wait(fn, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


def _pdf(text):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_full_pipeline_from_upload_to_audit_bundle(uploads, client, fake_popen, monkeypatch):
    # 1. Upload TXT + MD + PDF with additional context; ontology from a (fake) LLM
    ontology_llm = OntologyLLM({
        "entity_types": [{"name": "consumer"}, {"name": "campsite owner"}],
        "edge_types": [{"name": "asksAbout", "source_targets": [{"source": "consumer", "target": "campsite owner"}]}],
        "analysis_summary": "Campsite booking",
    })
    monkeypatch.setattr(graph_api, "OntologyGenerator", lambda: OntologyGenerator(llm_client=ontology_llm))
    upload = client.post("/api/graph/ontology/generate", data={
        "simulation_requirement": BRIEF,
        "additional_context": "Dutch market only",
        "project_name": "Kamptribe",
        "files": [
            (io.BytesIO(b"C01 is a Dutch first-time family. C02 is a German camper couple."), "personas.txt"),
            (io.BytesIO(b"# Kamptribe\nThe premium plan costs 4.99 per month."), "product.md"),
            (io.BytesIO(_pdf("Camping De Linde partners with Kamptribe.")), "partners.pdf"),
        ],
    }, content_type="multipart/form-data")
    assert upload.status_code == 200, upload.json
    project_id = upload.json["data"]["project_id"]

    # 2. Build the local graph; the extractor returns name variants of C01
    monkeypatch.setattr(extractor_module, "LLMClient", lambda: ExtractionLLM([
        {"entities": [
            {"name": "C01", "type": "Consumer", "summary": "Dutch first-time family"},
            {"name": "C02 German camper couple", "type": "Consumer", "summary": "German couple"},
        ], "edges": []},
        {"entities": [
            {"name": "C01 Dutch first-time family", "type": "consumer", "summary": "Family with two kids"},
            {"name": "Camping De Linde", "type": "Campsite Owner", "summary": "Campsite"},
        ], "edges": [{"name": "ASKS_ABOUT", "source": "C01", "target": "Camping De Linde", "fact": "C01 asks Camping De Linde about prices."}]},
    ]))
    task_id = client.post("/api/graph/build", json={"project_id": project_id, "chunk_size": 60, "chunk_overlap": 0}).json["data"]["task_id"]
    task = _wait(lambda: (lambda t: t if t["status"] in ("completed", "failed") else None)(
        client.get(f"/api/graph/task/{task_id}").json["data"]))
    assert task["status"] == "completed", task
    assert task["result"]["node_count"] == 3  # "C01" merged into "C01 Dutch first-time family"

    # 3. Create + prepare the simulation (rule-based personas, default config without LLM)
    monkeypatch.setattr(SimulationConfigGenerator, "_call_llm_with_retry", lambda self, prompt, system_prompt: {})
    sim_id = client.post("/api/simulation/create", json={"project_id": project_id}).json["data"]["simulation_id"]

    # Graph entities are not agents until selected: prepare is refused without a selection
    refused = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False})
    assert refused.status_code == 400 and refused.json["agent_selection_required"] is True
    selection = client.get(f"/api/simulation/{sim_id}/agent-selection").json["data"]
    assert selection["summary"]["active_agents"] == 0 and selection["summary"]["graph_entities"] == 3
    roles = {"C01 Dutch first-time family": "prospect", "C02 German camper couple": "prospect",
             "Camping De Linde": "brand_platform"}
    saved = client.put(f"/api/simulation/{sim_id}/agent-selection", json={"confirmed": True, "entities": [
        {"entity_uuid": e["entity_uuid"], "active": True, "simulation_role": roles[e["name"]]}
        for e in selection["entities"]
    ]}).json["data"]
    assert saved["summary"]["active_agents"] == 3 and saved["confirmed"] is True

    prepare_task = client.post("/api/simulation/prepare", json={"simulation_id": sim_id, "use_llm_for_profiles": False}).json["data"]["task_id"]
    prepared = _wait(lambda: (lambda d: d if d["status"] in ("completed", "failed") else None)(
        client.post("/api/simulation/prepare/status", json={"task_id": prepare_task}).json["data"]))
    assert prepared["status"] == "completed", prepared
    profiles = client.get(f"/api/simulation/{sim_id}/profiles").json["data"]["profiles"]
    assert [p["name"] for p in profiles] == ["C01 Dutch first-time family", "C02 German camper couple", "Camping De Linde"]

    # 4. Explicit start; the fake OASIS process writes actions of the configured agents
    FakePopen.actions = {
        "twitter": [
            {"round": 1, "agent_id": 0, "agent_name": "C01 Dutch first-time family", "action_type": "CREATE_POST",
             "action_args": {"content": "Looking for a family campsite planner"}},
            {"round": 2, "agent_id": 0, "agent_name": "C01", "action_type": "CREATE_POST",
             "action_args": {"content": "4.99 per month is too expensive for us"}},
        ],
        "reddit": [
            {"round": 1, "agent_id": 1, "agent_name": "C02 German camper couple", "action_type": "CREATE_COMMENT",
             "action_args": {"content": "We will stick with the free app", "post_content": "Premium plan launched"}},
        ],
    }
    start = client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]
    assert start["started"] is True
    wait_for_monitor(sim_id)
    run_id = start["run_id"]
    status = client.get(f"/api/simulation/{sim_id}/run-status").json["data"]
    assert status["run_status"] == "completed" and status["total_actions_count"] == 3

    # A reload/re-open never starts another run
    assert client.post("/api/simulation/start", json={"simulation_id": sim_id}).json["data"]["started"] is False
    assert len(FakePopen.instances) == 1

    # 5. Report from the raw evidence (real local graph tools as supplementary source)
    monkeypatch.setattr(report_api, "ReportAgent", lambda **kwargs: ReportAgent(llm_client=FakeLLM(), **kwargs))
    report_id = client.post("/api/report/generate", json={"simulation_id": sim_id}).json["data"]["report_id"]
    _wait(lambda: RunStore.get_run(sim_id, run_id).status == RunStatus.REPORT_COMPLETED)
    report = client.get(f"/api/report/{report_id}").json["data"]
    assert report["status"] == "completed"
    assert report["run_id"] == run_id
    markdown = report["markdown_content"]
    for pid in ("C01", "C02", "C03"):
        assert f"**{pid} — " in markdown
    assert "C03 — C03" in markdown and "Not present in the simulation" in markdown

    # 6. Audit bundle
    bundle = client.get(f"/api/report/{report_id}/export")
    assert bundle.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(bundle.data))
    names = set(archive.namelist())
    manifest = json.loads(archive.read("sources/manifest.json"))
    assert [d["original_filename"] for d in manifest["documents"]] == ["personas.txt", "product.md", "partners.pdf"]
    assert "Dutch market only" in archive.read("assessment/brief.md").decode()
    evidence = json.loads(archive.read("evidence/normalized_evidence.json"))
    assert {a["canonical_id"] for a in evidence["actions"] if a["agent_id"] == 0} == {"C01"}
    assert evidence["stats"]["by_platform"] == {"twitter": 2, "reddit": 1}
    report_json = json.loads(archive.read("report/report.json"))
    assert [p["canonical_id"] for p in report_json["personas"]] == ["C01", "C02", "C03"]
    assert {"actions/twitter_actions.jsonl", "actions/reddit_actions.jsonl", "report/full_report.md",
            "report/agent_log.jsonl", "personas/personas_normalized.json", "run/run.json"} <= names
    run_json = json.loads(archive.read("run/run.json"))
    assert run_json["run_id"] == run_id and run_json["status"] in ("reporting", "report_completed")
