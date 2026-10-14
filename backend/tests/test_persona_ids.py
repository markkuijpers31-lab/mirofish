from app.utils.persona_ids import (
    build_persona_index,
    expand_id_ranges,
    normalize_identifier,
)


def test_short_and_long_name_share_one_canonical_id():
    index = build_persona_index([
        {"agent_id": 0, "name": "C01"},
        {"agent_id": 1, "name": "C01 Dutch first-time family"},
        {"agent_id": 2, "name": "C02 German camper couple"},
    ])
    assert index.resolve(0) == ("C01", "C01 Dutch first-time family")
    assert index.resolve(1) == ("C01", "C01 Dutch first-time family")
    persona = index.personas["C01"]
    assert persona["agent_ids"] == [0, 1]
    assert persona["names"] == ["C01", "C01 Dutch first-time family"]


def test_requested_ids_match_variants_and_padding():
    index = build_persona_index(
        [{"agent_id": 5, "name": "C-1: student"}, {"agent_id": 6, "name": "c01"}],
        known_ids=["C01"],
    )
    assert index.resolve(5)[0] == "C01"
    assert index.resolve(6)[0] == "C01"


def test_explicit_identifier_field_wins():
    index = build_persona_index([{"agent_id": 1, "name": "Anna (urban parent)", "persona_id": "P7"}])
    assert index.resolve(1)[0] == "P7"


def test_names_without_identifier_are_not_mangled():
    index = build_persona_index([
        {"agent_id": 1, "name": "GPT4 enthusiast"},
        {"agent_id": 2, "name": "Campsite Owner Association"},
    ])
    assert index.resolve(1)[0] == "GPT4 enthusiast"
    assert index.resolve(2)[0] == "Campsite Owner Association"


def test_unknown_agent_falls_back_to_name():
    index = build_persona_index([{"agent_id": 0, "name": "C01 family"}, {"agent_id": 1, "name": "C02 couple"}])
    assert index.resolve(99, "C02 couple (duplicate)")[0] == "C02"


def test_range_expansion_and_normalization():
    assert expand_id_ranges("personas C01-C03 and C08 – C10") == ["C01", "C02", "C03", "C08", "C09", "C10"]
    assert normalize_identifier("c-07 Someone") == "C07"
    assert normalize_identifier("Agent_5") is None


def test_assessment_spec_regex_fallback_reads_the_brief():
    from app.services.assessment_spec import extract_spec_regex

    brief = """Analyse consumer personas C01-C10 individually.

For each persona report:
- starting workflow
- discovery/exposure
- plan choice
- checkout outcome
- strongest evidence for
- missing stages

H1: Families convert after seeing the price.
Hypothesis 2: Couples prefer alternatives.
"""
    spec = extract_spec_regex(brief, persona_families=["C"])
    assert spec.persona_ids == [f"C{n:02d}" for n in range(1, 11)]
    assert [h["id"] for h in spec.hypotheses] == ["H1", "H2"]
    assert spec.funnel_fields == ["starting workflow", "discovery/exposure", "plan choice", "checkout outcome"]
    assert spec.funnel_fields_source == "brief"


def test_assessment_spec_without_structure_uses_generic_fields():
    from app.services.assessment_spec import DEFAULT_FUNNEL_FIELDS, extract_spec_regex

    spec = extract_spec_regex("How will people react to the new campsite app?")
    assert spec.persona_ids == []
    assert spec.hypotheses == []
    assert spec.funnel_fields == DEFAULT_FUNNEL_FIELDS
    assert spec.funnel_fields_source == "default"
