"""
Required source coverage: authoritative documents an experiment needs before it may run
"""

import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.source_requirements")

REQUIREMENTS_FILE = "source_requirements.json"
COVERAGE_FILE = "source_coverage.json"

# A section is a Markdown heading carrying an explicit anchor: "## Pricing {#pricing}"
_ANCHOR_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*\{#([A-Za-z0-9_.-]+)\}\s*$", re.MULTILINE)
_SECTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

STATUS_COVERED = "covered"
STATUS_INCOMPLETE = "incomplete"
STATUS_MISSING = "missing"


def validate_requirements(spec: Any) -> Dict[str, Any]:
    """
    Validate a source requirements specification

    Format:
        {"documents": [{
            "id": "product_truth", "label": "...",
            "filename_pattern": "product[-_ ]?truth",   // regex on the uploaded filename
            "authoritative": true,                      // reports treat it as the authority
            "sections": [{"id": "pricing", "label": "Pricing",
                          "required_terms": ["Trip Pass"], "min_chars": 200}]
        }]}

    Raises:
        ValueError: invalid specification
    """
    if not isinstance(spec, dict) or not isinstance(spec.get("documents"), list) or not spec["documents"]:
        raise ValueError("source requirements need a non-empty 'documents' list")
    documents = []
    for doc in spec["documents"]:
        if not isinstance(doc, dict) or not doc.get("id") or not doc.get("filename_pattern"):
            raise ValueError("each required document needs 'id' and 'filename_pattern'")
        try:
            re.compile(doc["filename_pattern"])
        except re.error as e:
            raise ValueError(f"invalid filename_pattern for {doc['id']}: {e}")
        sections = []
        for section in doc.get("sections") or []:
            sid = str((section or {}).get("id") or "")
            if not _SECTION_ID_RE.match(sid):
                raise ValueError(f"invalid section id in {doc['id']}: {sid!r}")
            if any(s["id"] == sid for s in sections):
                raise ValueError(f"duplicate section id in {doc['id']}: {sid}")
            sections.append({
                "id": sid,
                "label": str(section.get("label") or sid),
                "required_terms": [str(t) for t in section.get("required_terms") or [] if str(t).strip()],
                "min_chars": max(0, int(section.get("min_chars") or 0)),
            })
        if not sections:
            raise ValueError(f"required document {doc['id']} lists no sections")
        documents.append({
            "id": str(doc["id"]),
            "label": str(doc.get("label") or doc["id"]),
            "filename_pattern": doc["filename_pattern"],
            "authoritative": bool(doc.get("authoritative", True)),
            "sections": sections,
        })
    return {"version": 1, "documents": documents}


def split_sections(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Sections of a Markdown document by explicit heading anchors

    A section runs until the next heading of the same or a higher level, so
    nested sub-sections belong to their parent.

    Returns:
        {anchor_id: {"title", "level", "text"}}
    """
    matches = list(_ANCHOR_HEADING_RE.finditer(text or ""))
    all_headings = [(m.start(), len(m.group(1))) for m in re.finditer(r"^(#{1,6})\s", text or "", re.MULTILINE)]
    sections: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        level = len(m.group(1))
        end = len(text)
        for pos, lvl in all_headings:
            if pos > m.start() and lvl <= level:
                end = pos
                break
        sections[m.group(3)] = {"title": m.group(2).strip(), "level": level, "text": text[m.start():end].strip()}
    return sections


def matching_documents(requirement: Dict[str, Any], documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pattern = re.compile(requirement["filename_pattern"], re.IGNORECASE)
    return [d for d in documents or [] if pattern.search(d.get("original_filename") or "")]


def check_coverage(spec: Dict[str, Any], documents: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Coverage matrix of the required sections in the uploaded source documents

    Args:
        spec: Validated requirements
        documents: Source documents ({original_filename, text})

    Returns:
        {"ok": bool, "documents": [{id, label, files, ok, sections: [{id, label, status,
         missing_terms, chars}]}], "missing": ["doc:section", ...]}
    """
    documents = list(documents or [])
    result_docs, missing = [], []
    for req in spec.get("documents", []):
        files = matching_documents(req, documents)
        text = "\n\n".join(d.get("text") or "" for d in files)
        sections = split_sections(text)
        rows = []
        for section in req["sections"]:
            found = sections.get(section["id"])
            if not found:
                status, missing_terms, chars = STATUS_MISSING, list(section["required_terms"]), 0
            else:
                body = found["text"]
                lowered = body.lower()
                missing_terms = [t for t in section["required_terms"] if t.lower() not in lowered]
                chars = len(body)
                status = STATUS_COVERED if not missing_terms and chars >= section["min_chars"] else STATUS_INCOMPLETE
            if status != STATUS_COVERED:
                missing.append(f"{req['id']}:{section['id']}")
            rows.append({
                "id": section["id"],
                "label": section["label"],
                "status": status,
                "missing_terms": missing_terms,
                "chars": chars,
                "min_chars": section["min_chars"],
            })
        result_docs.append({
            "id": req["id"],
            "label": req["label"],
            "authoritative": req["authoritative"],
            "files": [d.get("original_filename") for d in files],
            "ok": bool(files) and all(r["status"] == STATUS_COVERED for r in rows),
            "sections": rows,
        })
    return {
        "ok": all(d["ok"] for d in result_docs),
        "checked_at": datetime.now().isoformat(),
        "documents": result_docs,
        "missing": missing,
    }


def authoritative_texts(spec: Optional[Dict[str, Any]], documents: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Full texts of the documents marked authoritative ({label, filename, text})"""
    result = []
    for req in (spec or {}).get("documents", []):
        if not req.get("authoritative"):
            continue
        for d in matching_documents(req, documents):
            result.append({"label": req["label"], "filename": d.get("original_filename") or "", "text": d.get("text") or ""})
    return result


def section_texts(spec: Optional[Dict[str, Any]], documents: Iterable[Dict[str, Any]], section_ids: Iterable[str]) -> Dict[str, str]:
    """
    Verbatim text of anchored sections from the authoritative documents

    Raises:
        ValueError: a requested section does not exist
    """
    available: Dict[str, str] = {}
    for item in authoritative_texts(spec, documents):
        for sid, section in split_sections(item["text"]).items():
            available.setdefault(sid, section["text"])
    result = {}
    for sid in section_ids or []:
        if sid not in available:
            raise ValueError(f"Authoritative section '{sid}' not found in the uploaded sources")
        result[sid] = available[sid]
    return result


class SourceRequirementsStore:
    """source_requirements.json in the simulation (configuration) directory"""

    _lock = threading.RLock()

    @staticmethod
    def path(simulation_id: str) -> str:
        return os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, REQUIREMENTS_FILE)

    @classmethod
    def load(cls, simulation_id: str) -> Optional[Dict[str, Any]]:
        path = cls.path(simulation_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def save(cls, simulation_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec = validate_requirements(spec)
        path = cls.path(simulation_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with cls._lock:
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        return spec


def simulation_coverage(simulation_id: str, project_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Coverage of the simulation's requirements against its project's sources

    Returns:
        None when the simulation defines no source requirements
    """
    from ..models.project import ProjectManager

    spec = SourceRequirementsStore.load(simulation_id)
    if spec is None:
        return None
    from .run_contract import without_contract_documents

    documents = ProjectManager.get_source_documents(project_id) if project_id else []
    return check_coverage(spec, without_contract_documents(documents))


def coverage_error(coverage: Optional[Dict[str, Any]]) -> Optional[str]:
    """Human-readable refusal reason, or None when coverage is complete / not required"""
    if not coverage or coverage.get("ok"):
        return None
    parts = []
    for doc in coverage["documents"]:
        if not doc["files"]:
            parts.append(f"required document '{doc['label']}' is not among the uploaded sources")
            continue
        bad = [f"{s['id']} ({s['status']}{': missing ' + ', '.join(s['missing_terms']) if s['missing_terms'] else ''})"
               for s in doc["sections"] if s["status"] != STATUS_COVERED]
        if bad:
            parts.append(f"{doc['label']}: " + "; ".join(bad))
    return "Required source coverage is incomplete - " + " | ".join(parts)
