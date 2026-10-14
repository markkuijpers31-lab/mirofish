import io
import time

import fitz
import pytest

from app.api import graph as graph_api
from app.models.project import ProjectManager, ProjectStatus
from app.services import local_graph_extractor as extractor_module
from app.services.entity_reader import EntityReader
from app.services.local_graph_extractor import LocalGraphExtractor
from app.services.local_graph_store import LocalGraphStore
from app.services.ontology_generator import OntologyGenerator

# Malformed output an LLM really produces: strings instead of objects, nulls,
# string attributes, lower-case names, endpoints using the original names
MESSY_ONTOLOGY = {
    "entity_types": [
        "consumer",
        None,
        {"name": "campsite owner", "attributes": ["region", None]},
        {"name": "Consumer", "description": "duplicate"},
    ],
    "edge_types": [
        "unusable edge",
        {"name": "talksTo", "source_targets": [{"source": "consumer", "target": "campsite owner"}, None]},
    ],
    "analysis_summary": 7,
}


class OntologyLLM:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else MESSY_ONTOLOGY
        self.error = error
        self.messages = []

    def chat_json(self, messages, temperature=0.3, max_tokens=8192):
        self.messages.append(messages)
        if self.error:
            raise self.error
        import copy

        return copy.deepcopy(self.response)

    @property
    def user_message(self):
        return self.messages[-1][-1]["content"]


def _pdf_bytes(text):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _upload(client, monkeypatch, files, llm=None, requirement="Assess personas C01-C02", context=""):
    llm = llm or OntologyLLM()
    monkeypatch.setattr(graph_api, "OntologyGenerator", lambda: OntologyGenerator(llm_client=llm))
    data = {
        "simulation_requirement": requirement,
        "additional_context": context,
        "files": [(io.BytesIO(content), name) for name, content in files],
    }
    response = client.post("/api/graph/ontology/generate", data=data, content_type="multipart/form-data")
    return response, llm


def test_malformed_llm_ontology_is_normalized_instead_of_crashing():
    # Regression: 'str' object does not support item assignment
    result = OntologyGenerator(llm_client=OntologyLLM())._validate_and_process(dict(MESSY_ONTOLOGY))

    assert [e["name"] for e in result["entity_types"]] == ["Consumer", "CampsiteOwner", "Person", "Organization"]
    assert result["entity_types"][1]["attributes"] == [{"name": "region", "type": "text", "description": "region"}]
    assert result["edge_types"] == [{
        "name": "TALKS_TO",
        "source_targets": [{"source": "Consumer", "target": "CampsiteOwner"}],
        "attributes": [{"name": "details", "type": "text", "description": "Additional details about this ontology type."}],
        "description": "A TALKS_TO relationship.",
    }]
    assert result["analysis_summary"] == ""


def test_non_object_llm_result_is_a_clear_error():
    with pytest.raises(ValueError, match="JSON object"):
        OntologyGenerator(llm_client=OntologyLLM())._validate_and_process(["not", "an", "object"])


@pytest.mark.parametrize("name,content,expected", [
    ("brief.txt", b"Families compare campsite apps before booking.", "Families compare campsite apps"),
    ("product.md", b"# Kamptribe\nThe premium plan costs 4.99.", "The premium plan costs 4.99."),
    ("product.pdf", None, "Premium plan launched in May"),
])
def test_ontology_generation_from_each_source_type(uploads, client, monkeypatch, name, content, expected):
    content = content if content is not None else _pdf_bytes("Premium plan launched in May")
    response, llm = _upload(client, monkeypatch, [(name, content)])

    assert response.status_code == 200, response.json
    data = response.json["data"]
    assert [e["name"] for e in data["ontology"]["entity_types"]][-2:] == ["Person", "Organization"]
    assert expected in llm.user_message
    assert "Assess personas C01-C02" in llm.user_message
    project = ProjectManager.get_project(data["project_id"])
    assert project.status == ProjectStatus.ONTOLOGY_GENERATED
    assert project.files[0]["filename"] == name


def test_ontology_generation_from_multiple_sources_with_additional_context(uploads, client, monkeypatch):
    response, llm = _upload(
        client, monkeypatch,
        [("a.txt", b"Alpha source text."), ("b.md", b"# Beta\nBeta source text."), ("c.pdf", _pdf_bytes("Gamma source text."))],
        context="Focus on Dutch families",
    )

    assert response.status_code == 200, response.json
    message = llm.user_message
    for expected in ("Alpha source text.", "Beta source text.", "Gamma source text.", "Focus on Dutch families"):
        assert expected in message
    project = ProjectManager.get_project(response.json["data"]["project_id"])
    assert project.additional_context == "Focus on Dutch families"
    assert [f["filename"] for f in project.files] == ["a.txt", "b.md", "c.pdf"]


def test_upload_without_usable_text_is_rejected_and_leaves_no_project(uploads, client, monkeypatch):
    for files in ([("empty.txt", b"   ")], [("broken.pdf", b"%PDF-1.4 not a pdf")], [("notes.docx", b"PK")]):
        response, llm = _upload(client, monkeypatch, files)
        assert response.status_code == 400, files
        assert llm.messages == []
    assert ProjectManager.list_projects() == []


def test_failed_ontology_generation_marks_project_failed(uploads, client, monkeypatch):
    response, _ = _upload(client, monkeypatch, [("a.txt", b"Some text")], llm=OntologyLLM(error=RuntimeError("LLM down")))

    assert response.status_code == 500
    project = ProjectManager.get_project(response.json["project_id"])
    assert project.status == ProjectStatus.FAILED
    assert "LLM down" in project.error
    assert project.files[0]["extraction_status"] == "ok"  # sources stay auditable
    build = client.post("/api/graph/build", json={"project_id": project.project_id})
    assert build.status_code == 400  # no ontology: a graph can never be built from it


class ExtractionLLM:
    """Returns one canned extraction per chunk (None = failure)"""

    def __init__(self, results):
        self.results = list(results)

    def chat_json(self, messages, temperature=0.2, max_tokens=8192):
        result = self.results.pop(0) if self.results else {"entities": [], "edges": []}
        if isinstance(result, Exception):
            raise result
        return result


ONTOLOGY = {
    "entity_types": [{"name": "Consumer"}, {"name": "CampsiteOwner"}, {"name": "Person"}, {"name": "Organization"}],
    "edge_types": [{"name": "TALKS_TO"}],
}


def test_local_graph_extraction_merges_identifier_variants_and_maps_types(uploads):
    llm = ExtractionLLM([
        {"entities": [
            {"name": "C01", "type": "consumer", "summary": ""},
            {"name": "C02 German camper couple", "type": "Consumer", "summary": "Couple"},
            "not an entity",
            {"name": None, "type": "Consumer"},
        ], "edges": [None]},
        {"entities": [
            {"name": "C01 Dutch first-time family", "type": "Consumer", "summary": "Family with two kids"},
            {"name": "Camping De Linde", "type": "campsite owner", "summary": "Owner"},
        ], "edges": [
            {"name": "TALKS_TO", "source": "C01", "target": "Camping De Linde", "fact": "C01 asked Camping De Linde about prices."},
        ]},
        "not even a dict",
    ])
    extractor = LocalGraphExtractor("graph_x", ONTOLOGY, llm=llm)
    extractor.extract_and_store(["chunk 1", "chunk 2", "chunk 3"])

    store = LocalGraphStore("graph_x")
    nodes = store.get_all_nodes()
    assert [n.name for n in nodes] == ["C01 Dutch first-time family", "C02 German camper couple", "Camping De Linde"]
    assert nodes[0].labels == ["Entity", "Consumer"]
    assert nodes[0].attributes["aliases"] == ["C01", "C01 Dutch first-time family"]
    assert nodes[0].summary == "Family with two kids"
    assert nodes[2].labels == ["Entity", "CampsiteOwner"]
    edges = store.get_all_edges()
    assert len(edges) == 1 and edges[0].source_node_uuid == nodes[0].uuid
    assert extractor.stats["entities_merged"] == 1
    assert extractor.stats["chunks_failed"] == 1

    # Entities come back in a stable order: profile user_ids are deterministic
    filtered = EntityReader().filter_defined_entities("graph_x")
    assert [e.name for e in filtered.entities] == [n.name for n in nodes]
    assert [e.get_entity_type() for e in filtered.entities] == ["Consumer", "Consumer", "CampsiteOwner"]


def _wait_task(client, task_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = client.get(f"/api/graph/task/{task_id}").json["data"]
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    raise AssertionError("graph build did not finish")


def _project_with_ontology(text="C01 talks to Camping De Linde."):
    project = ProjectManager.create_project(name="Graph")
    project.simulation_requirement = "req"
    project.ontology = ONTOLOGY
    project.status = ProjectStatus.ONTOLOGY_GENERATED
    ProjectManager.save_project(project)
    ProjectManager.save_extracted_text(project.project_id, text)
    return project


def test_graph_build_uses_the_local_store(uploads, client, monkeypatch):
    project = _project_with_ontology()
    monkeypatch.setattr(extractor_module, "LLMClient", lambda: ExtractionLLM([
        {"entities": [{"name": "C01", "type": "Consumer"}, {"name": "Camping De Linde", "type": "CampsiteOwner"}],
         "edges": [{"name": "TALKS_TO", "source": "C01", "target": "Camping De Linde", "fact": "C01 talks to Camping De Linde."}]},
    ]))

    task_id = client.post("/api/graph/build", json={"project_id": project.project_id}).json["data"]["task_id"]
    task = _wait_task(client, task_id)

    assert task["status"] == "completed", task
    graph_id = task["result"]["graph_id"]
    assert task["result"]["node_count"] == 2
    assert (uploads / "graphs" / graph_id / "graph.db").exists()
    assert LocalGraphStore(graph_id).get_ontology()["entity_types"][0]["name"] == "Consumer"
    assert ProjectManager.get_project(project.project_id).status == ProjectStatus.GRAPH_COMPLETED


def test_graph_build_with_no_extractable_entities_fails_clearly(uploads, client, monkeypatch):
    project = _project_with_ontology()
    monkeypatch.setattr(extractor_module, "LLMClient", lambda: ExtractionLLM([RuntimeError("LLM unavailable")]))

    task_id = client.post("/api/graph/build", json={"project_id": project.project_id}).json["data"]["task_id"]
    task = _wait_task(client, task_id)

    assert task["status"] == "failed"
    saved = ProjectManager.get_project(project.project_id)
    assert saved.status == ProjectStatus.FAILED
    assert "No entities could be extracted" in saved.error
