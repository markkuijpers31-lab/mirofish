from app.services.ontology_generator import OntologyGenerator
from app.services.graph_builder import GraphBuilderService
from app.services.local_graph_store import LocalGraphStore
from app.utils.ontology import (
    MAX_ONTOLOGY_ATTRIBUTES,
    MAX_ONTOLOGY_SOURCE_TARGETS,
    normalize_ontology_attribute,
    normalize_ontology_attributes,
)


def test_normalize_string_attribute():
    assert normalize_ontology_attribute("role") == {
        "name": "role",
        "type": "text",
        "description": "role",
    }


def test_preserve_valid_dictionary_attribute():
    original = {"name": "role", "type": "text", "description": "Public role"}
    assert normalize_ontology_attribute(original) == original
    assert normalize_ontology_attribute(original) is not original


def test_reject_unusable_attribute_shapes():
    for value in (None, 7, [], {}, {"name": None}, {"name": ""}, "   "):
        assert normalize_ontology_attribute(value) is None


def test_attribute_list_is_non_empty_and_capped():
    assert normalize_ontology_attributes(None) == [{
        "name": "details",
        "type": "text",
        "description": "Additional details about this ontology type.",
    }]

    attributes = [None] + [f"field_{index}" for index in range(12)]
    normalized = normalize_ontology_attributes(attributes)

    assert len(normalized) == MAX_ONTOLOGY_ATTRIBUTES
    assert [attribute["name"] for attribute in normalized] == [
        f"field_{index}" for index in range(MAX_ONTOLOGY_ATTRIBUTES)
    ]


def test_generator_normalizes_entity_and_edge_attributes():
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "speaker", "attributes": ["role", None]}],
        "edge_types": [{"name": "quotes", "attributes": ["source_url", {}]}],
    })

    assert result["entity_types"][0]["attributes"] == [{
        "name": "role",
        "type": "text",
        "description": "role",
    }]
    assert result["edge_types"][0]["attributes"] == [{
        "name": "source_url",
        "type": "text",
        "description": "source_url",
    }]


def test_generator_adds_a_property_to_empty_custom_types():
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "speaker", "attributes": []}],
        "edge_types": [{"name": "quotes", "attributes": []}],
    })

    assert result["entity_types"][0]["attributes"][0]["name"] == "details"
    assert result["edge_types"][0]["attributes"][0]["name"] == "details"


def test_graph_builder_stores_the_ontology_as_json_in_the_local_graph(uploads):
    builder = GraphBuilderService()
    graph_id = builder.create_graph("Test graph")
    ontology = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "Speaker", "attributes": ["role", None]}],
        "edge_types": [],
    })

    builder.set_ontology(graph_id, ontology)
    # A second update must still read the stored metadata (was a Python repr before)
    builder.set_ontology(graph_id, {"analysis_summary": "updated"})

    stored = LocalGraphStore(graph_id).get_ontology()
    assert stored["entity_types"][0]["name"] == "Speaker"
    assert stored["entity_types"][0]["attributes"][0]["name"] == "role"
    assert stored["analysis_summary"] == "updated"


def test_edge_only_ontology_round_trips_through_the_local_graph(uploads):
    builder = GraphBuilderService()
    graph_id = builder.create_graph("Edges only")
    builder.set_ontology(graph_id, {
        "entity_types": [],
        "edge_types": [{
            "name": "RELATED_TO",
            "attributes": [{"name": "reason", "type": "text", "description": "reason"}],
            "source_targets": [{"source": "Entity", "target": "Entity"}],
        }],
    })

    stored = LocalGraphStore(graph_id).get_ontology()
    assert stored["entity_types"] == []
    assert stored["edge_types"][0]["source_targets"] == [{"source": "Entity", "target": "Entity"}]


def test_legacy_python_repr_ontology_metadata_does_not_crash(uploads):
    from app.services.local_graph_store import _connect

    store = LocalGraphStore("legacy_graph")
    store.set_meta("legacy", "", {}, "2026-01-01")
    with _connect("legacy_graph") as conn:
        conn.execute("UPDATE graph_meta SET ontology=? WHERE graph_id=?", ("{'entity_types': []}", "legacy_graph"))
        conn.commit()

    assert store.get_ontology() == {}
    GraphBuilderService().set_ontology("legacy_graph", {"entity_types": [{"name": "Person"}]})
    assert store.get_ontology()["entity_types"] == [{"name": "Person"}]


def test_generator_dedupes_and_caps_edge_source_targets():
    entity_names = [f"Type{index}" for index in range(8)]
    pairs = [(a, b) for a in entity_names for b in entity_names if a != b]
    source_targets = [
        {"source": a, "target": b} for a, b in pairs[: MAX_ONTOLOGY_SOURCE_TARGETS + 2]
    ]
    source_targets.insert(1, dict(source_targets[0]))

    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": name} for name in entity_names],
        "edge_types": [{"name": "RELATED_TO", "source_targets": source_targets}],
    })

    targets = result["edge_types"][0]["source_targets"]
    assert len(targets) == MAX_ONTOLOGY_SOURCE_TARGETS
    assert len({(t["source"], t["target"]) for t in targets}) == len(targets)


def test_generator_ignores_invalid_entries_and_normalizes_edge_names():
    source_targets = [
        {"source": "speaker", "target": "news outlet"},
        {"source": "speaker", "target": "news outlet"},
        None,
    ] + [
        {"source": "speaker", "target": "news outlet" if index == 0 else "Person"}
        for index in range(12)
    ]

    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": ["speaker", None, 7, {"name": "news outlet"}],
        "edge_types": [
            "unusable edge",
            None,
            {"name": "worksFor", "source_targets": source_targets},
            {"name": "works-for", "source_targets": []},
        ],
    })

    assert [entity["name"] for entity in result["entity_types"][:2]] == [
        "Speaker",
        "NewsOutlet",
    ]
    assert [edge["name"] for edge in result["edge_types"]] == ["WORKS_FOR"]
    assert result["edge_types"][0]["source_targets"] == [
        {"source": "Speaker", "target": "NewsOutlet"},
        {"source": "Speaker", "target": "Person"},
    ]


def test_generator_caps_after_discarding_invalid_edge_endpoints():
    invalid_first = [
        {"source": f"Removed{index}", "target": "AlsoRemoved"}
        for index in range(MAX_ONTOLOGY_SOURCE_TARGETS)
    ]
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "person"}, {"name": "organization"}],
        "edge_types": [{
            "name": "works_for",
            "source_targets": invalid_first + [
                {"source": "person", "target": "organization"}
            ],
        }],
    })

    assert result["edge_types"][0]["source_targets"] == [
        {"source": "Person", "target": "Organization"}
    ]
