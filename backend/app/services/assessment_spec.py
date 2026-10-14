"""
Assessment brief handling: compose the brief and extract requested personas, hypotheses and funnel fields
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.persona_ids import expand_id_ranges, extract_ids_from_text, parse_identifier

logger = get_logger("mirofish.assessment")

# Uploaded documents whose filename marks them as (part of) the assessment brief
BRIEF_FILENAME_RE = re.compile(r"assessment|brief|opdracht|instructie|instruction", re.IGNORECASE)

# Generic funnel fields used only when the brief does not define its own
DEFAULT_FUNNEL_FIELDS = [
    "exposure",
    "first reaction",
    "engagement",
    "trust / friction",
    "outcome",
]

_HYPOTHESIS_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?((?:H|Hyp(?:othes[ie]s?)?)\s*[-_]?\s*\d{1,3})(?:\*\*)?\s*[:.)\-–—]\s*(.+)$",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+(.+?)\s*$")
_FUNNEL_TRIGGER_RE = re.compile(
    r"funnel|per persona|for each persona|each persona|per consumer|for each consumer|"
    r"voor elke|per klant|per segment|for each (?:of )?[A-Z]{1,4}\d|per [A-Z]{1,4}\d",
    re.IGNORECASE,
)
_PERSONA_LINE_RE = re.compile(
    r"persona|consumer|customer|klant|segment|respondent|agent|profile|doelgroep",
    re.IGNORECASE,
)


@dataclass
class AssessmentSpec:
    """What the assessment brief asks the report to cover"""

    persona_ids: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, str]] = field(default_factory=list)
    funnel_fields: List[str] = field(default_factory=list)
    funnel_fields_source: str = "default"  # brief | default
    extraction_method: str = "regex"  # llm | regex
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_ids": self.persona_ids,
            "hypotheses": self.hypotheses,
            "funnel_fields": self.funnel_fields,
            "funnel_fields_source": self.funnel_fields_source,
            "extraction_method": self.extraction_method,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessmentSpec":
        return cls(
            persona_ids=data.get("persona_ids", []),
            hypotheses=data.get("hypotheses", []),
            funnel_fields=data.get("funnel_fields", []),
            funnel_fields_source=data.get("funnel_fields_source", "default"),
            extraction_method=data.get("extraction_method", "regex"),
            notes=data.get("notes", []),
        )


def compose_brief(
    simulation_requirement: Optional[str],
    additional_context: Optional[str],
    source_documents: Iterable[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Compose the assessment brief from the requirement, notes and brief documents

    Args:
        simulation_requirement: Requirement text entered at project creation
        additional_context: Additional notes entered at project creation
        source_documents: Source documents ({original_filename, text, ...})

    Returns:
        (brief markdown, list of filenames used as brief documents)
    """
    parts = ["# Assessment brief", "", "## Simulation requirement", "", (simulation_requirement or "").strip() or "_(empty)_"]
    if additional_context and additional_context.strip():
        parts += ["", "## Additional context", "", additional_context.strip()]
    brief_docs = []
    for doc in source_documents:
        name = doc.get("original_filename") or ""
        if BRIEF_FILENAME_RE.search(name) and (doc.get("text") or "").strip():
            brief_docs.append(name)
            parts += ["", f"## Brief document: {name}", "", doc["text"].strip()]
    return "\n".join(parts) + "\n", brief_docs


# Per-persona items the report always renders separately (not funnel fields)
_NON_FUNNEL_FIELD_RE = re.compile(
    r"strongest evidence|missing (?:stages?|steps?)|ontbrekende|sterkste bewijs", re.IGNORECASE
)


def _funnel_only(fields: List[str]) -> List[str]:
    return [f for f in fields if f and not _NON_FUNNEL_FIELD_RE.search(f)]


def _clean_field(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text).strip()
    text = text.rstrip(":;,.").strip()
    return text


def extract_spec_regex(brief_text: str, persona_families: Optional[Iterable[str]] = None) -> AssessmentSpec:
    """
    Deterministic extraction of the assessment spec (fallback when no LLM is available)

    Args:
        brief_text: Assessment brief text
        persona_families: Identifier letter families present among simulation personas

    Returns:
        AssessmentSpec
    """
    families = {f.upper() for f in (persona_families or [])}
    lines = (brief_text or "").splitlines()

    # Persona ids: ranges and ids on persona-related lines, or of a family that exists
    # among simulation personas. Hypothesis ids (H1...) are excluded.
    persona_ids: List[str] = []
    for line in lines:
        if _HYPOTHESIS_RE.match(line):
            continue
        for candidate in expand_id_ranges(line) + extract_ids_from_text(line):
            parsed = parse_identifier(candidate)
            if not parsed:
                continue
            family = parsed[0].upper()
            if family in families or (_PERSONA_LINE_RE.search(line) and family not in {"H", "Q"}):
                if candidate not in persona_ids:
                    persona_ids.append(candidate)

    hypotheses: List[Dict[str, str]] = []
    for line in lines:
        match = _HYPOTHESIS_RE.match(line)
        if match:
            hid = re.sub(r"[^A-Za-z0-9]", "", match.group(1)).upper()
            hid = re.sub(r"^HYP(?:OTHES[IE]S?)?", "H", hid)
            if not any(h["id"] == hid for h in hypotheses):
                hypotheses.append({"id": hid, "text": _clean_field(match.group(2))})

    funnel_fields: List[str] = []
    i = 0
    while i < len(lines):
        if _FUNNEL_TRIGGER_RE.search(lines[i]) and not _HYPOTHESIS_RE.match(lines[i]):
            j = i + 1
            collected = []
            while j < len(lines):
                bullet = _BULLET_RE.match(lines[j])
                if bullet:
                    collected.append(_clean_field(bullet.group(1)))
                elif lines[j].strip():
                    break
                j += 1
            if len(collected) >= 2:
                for item in collected:
                    if item and item not in funnel_fields:
                        funnel_fields.append(item)
                i = j
                continue
        i += 1

    funnel_fields = _funnel_only(funnel_fields)
    spec = AssessmentSpec(
        persona_ids=persona_ids,
        hypotheses=hypotheses,
        funnel_fields=funnel_fields or list(DEFAULT_FUNNEL_FIELDS),
        funnel_fields_source="brief" if funnel_fields else "default",
        extraction_method="regex",
    )
    return spec


_SPEC_SYSTEM_PROMPT = """You extract the structure of a commercial assessment brief.
Return JSON only:
{"persona_ids": ["..."], "hypotheses": [{"id": "H1", "text": "..."}], "funnel_fields": ["..."]}

Rules:
- persona_ids: identifiers of personas/consumers/segments the brief explicitly asks to analyse individually
  (e.g. "C01".."C10"). Expand ranges. Only identifiers literally present in the brief. Empty list if none.
- hypotheses: hypotheses/claims the brief asks to evaluate, with their id if given (otherwise H1, H2, ...).
- funnel_fields: the per-persona fields/stages the brief asks to report (e.g. "first choice", "pricing exposure").
  Use the brief's own wording. Empty list if the brief does not define them.
- Never invent items that are not in the brief."""


def extract_assessment_spec(
    brief_text: str,
    llm: Any = None,
    persona_families: Optional[Iterable[str]] = None,
) -> AssessmentSpec:
    """
    Extract the assessment spec, using the LLM when available and validating its output

    LLM persona ids are only accepted when they literally occur in the brief (or in an
    expanded range); ranges found deterministically are always included.

    Args:
        brief_text: Assessment brief text
        llm: Object with chat_json(messages=...) (optional)
        persona_families: Identifier letter families present among simulation personas

    Returns:
        AssessmentSpec
    """
    regex_spec = extract_spec_regex(brief_text, persona_families)
    if llm is None or not (brief_text or "").strip():
        return regex_spec

    try:
        data = llm.chat_json(
            messages=[
                {"role": "system", "content": _SPEC_SYSTEM_PROMPT},
                {"role": "user", "content": brief_text[:30000]},
            ],
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(f"Assessment spec LLM extraction failed, using regex fallback: {e}")
        regex_spec.notes.append(f"LLM extraction failed: {e}")
        return regex_spec

    allowed_ids = set(extract_ids_from_text(brief_text))
    persona_ids: List[str] = []
    for raw in (data.get("persona_ids") or []):
        parsed = parse_identifier(str(raw))
        candidate = f"{parsed[0].upper()}{parsed[1]}" if parsed else None
        if candidate and candidate in allowed_ids and candidate not in persona_ids:
            persona_ids.append(candidate)
    for candidate in regex_spec.persona_ids:
        if candidate in expand_id_ranges(brief_text) and candidate not in persona_ids:
            persona_ids.append(candidate)

    hypotheses: List[Dict[str, str]] = []
    for i, h in enumerate(data.get("hypotheses") or [], start=1):
        if isinstance(h, str):
            h = {"text": h}
        text = (h.get("text") or "").strip()
        if text:
            hid = (str(h.get("id") or f"H{i}")).strip()
            hypotheses.append({"id": hid, "text": text})

    funnel_fields = _funnel_only([
        _clean_field(str(f)) for f in (data.get("funnel_fields") or []) if str(f).strip()
    ])

    return AssessmentSpec(
        persona_ids=persona_ids,
        hypotheses=hypotheses or regex_spec.hypotheses,
        funnel_fields=funnel_fields or regex_spec.funnel_fields,
        funnel_fields_source="brief" if (funnel_fields or regex_spec.funnel_fields_source == "brief") else "default",
        extraction_method="llm",
        notes=regex_spec.notes,
    )
