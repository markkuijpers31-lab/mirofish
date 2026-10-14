import io
import os

import fitz

from conftest import make_prepared_simulation

from app.models.project import ProjectManager
from app.services import ontology_generator as ontology_module
from app.services.run_evidence import RunEvidence
from app.services.run_store import RunStore


def _pdf_bytes(text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client, monkeypatch, files):
    monkeypatch.setattr(
        ontology_module.OntologyGenerator, "__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        ontology_module.OntologyGenerator,
        "generate",
        lambda self, **kwargs: {"entity_types": [], "edge_types": [], "analysis_summary": ""},
    )
    data = {
        "simulation_requirement": "Assess personas C01-C02",
        "additional_context": "Focus on pricing",
        "files": [(io.BytesIO(content), name) for name, content in files],
    }
    response = client.post("/api/graph/ontology/generate", data=data, content_type="multipart/form-data")
    assert response.status_code == 200, response.json
    return response.json["data"]["project_id"]


def test_pdf_md_txt_sources_are_extracted_and_kept_per_file(uploads, client, monkeypatch):
    project_id = _upload(client, monkeypatch, [
        ("product.pdf", _pdf_bytes("Premium plan costs 4.99 per month")),
        ("notes.md", "# Notes\nFamilies like offline maps.".encode()),
        ("assessment_brief.txt", "Evaluate H1: families pay for premium.".encode()),
        ("broken.pdf", b"%PDF-1.4 this is not a real pdf"),
    ])

    project = ProjectManager.get_project(project_id)
    assert project.additional_context == "Focus on pricing"
    statuses = {f["filename"]: f["extraction_status"] for f in project.files}
    assert statuses["product.pdf"] == "ok"
    assert statuses["notes.md"] == "ok"
    assert statuses["assessment_brief.txt"] == "ok"
    assert statuses["broken.pdf"] in ("failed", "empty")

    docs = {d["original_filename"]: d for d in ProjectManager.get_source_documents(project_id)}
    assert "4.99" in docs["product.pdf"]["text"]
    assert "offline maps" in docs["notes.md"]["text"]
    assert "H1" in docs["assessment_brief.txt"]["text"]
    assert docs["product.pdf"]["original_path"] and os.path.exists(docs["product.pdf"]["original_path"])


def test_sources_are_snapshotted_into_the_run_and_reported(uploads, client, monkeypatch):
    project_id = _upload(client, monkeypatch, [
        ("product.pdf", _pdf_bytes("Premium plan costs 4.99 per month")),
        ("notes.md", b"Families like offline maps."),
        ("assessment_brief.txt", b"H1: families pay for premium."),
    ])
    make_prepared_simulation(project_id=project_id)
    record = RunStore.create_run("sim_test")

    # The project disappearing later must not remove the run's sources
    ProjectManager.delete_project(project_id)

    evidence = RunEvidence.load("sim_test", record.run_id)
    names = [s["original_filename"] for s in evidence.sources]
    assert names == ["product.pdf", "notes.md", "assessment_brief.txt"]
    assert "Focus on pricing" in evidence.brief_text
    assert "H1: families pay for premium." in evidence.brief_text  # brief document included
    assert evidence.brief_documents == ["assessment_brief.txt"]

    excerpt = evidence.source_excerpts(name="product")
    assert "4.99" in excerpt
    manifest = {s["original_filename"]: s for s in evidence.source_manifest()}
    assert manifest["product.pdf"]["used_in_report"] is True
    assert manifest["notes.md"]["used_in_report"] is False
    assert os.path.exists(os.path.join(RunStore.run_dir("sim_test", record.run_id), "sources", "originals", "01_product.pdf"))


def test_legacy_project_text_is_split_per_document(uploads):
    project = ProjectManager.create_project(name="Legacy")
    project.files = [{"filename": "a.md", "size": 1}, {"filename": "b.txt", "size": 1}]
    ProjectManager.save_project(project)
    ProjectManager.save_extracted_text(
        project.project_id, "\n\n=== a.md ===\nAlpha text\n\n=== b.txt ===\nBeta text"
    )
    docs = ProjectManager.get_source_documents(project.project_id)
    assert [(d["original_filename"], d["text"]) for d in docs] == [("a.md", "Alpha text"), ("b.txt", "Beta text")]
