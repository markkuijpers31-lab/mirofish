"""
Deterministic entity name canonicalisation and duplicate detection for agent selection
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.persona_ids import parse_identifier

# Web suffixes stripped from names ("kamptribe.com" -> "kamptribe")
_DOMAIN_SUFFIX_RE = re.compile(
    r"\.(?:com|net|org|io|app|ai|co|nl|de|be|fr|uk|eu|info|biz)(?:\.[a-z]{2})?$"
)
_URL_PREFIX_RE = re.compile(r"^(?:https?://)?(?:www\.)?")
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)

CONFIDENCE_HIGH = "high"
CONFIDENCE_AMBIGUOUS = "ambiguous"


def canonical_key(name: Optional[str]) -> str:
    """
    Normalised comparison key of an entity name

    Case, punctuation, a URL prefix and a trailing web domain are ignored, so
    "Kamptribe", "KAMPTRIBE" and "kamptribe.com" share one key.

    Args:
        name: Entity name

    Returns:
        Space-separated lower-case tokens (empty string for an empty name)
    """
    text = (name or "").strip().lower()
    text = _URL_PREFIX_RE.sub("", text)
    text = text.rstrip("/")
    text = _DOMAIN_SUFFIX_RE.sub("", text)
    return " ".join(t for t in _NON_WORD_RE.split(text) if t and t != "_")


def _persona_key(name: Optional[str]) -> Optional[str]:
    parsed = parse_identifier(name)
    if not parsed:
        return None
    return f"{parsed[0].upper()}{int(parsed[1])}"


def _is_contiguous_suffix(short: List[str], long: List[str]) -> bool:
    return 0 < len(short) < len(long) and long[-len(short):] == short


def _is_contiguous_prefix(short: List[str], long: List[str]) -> bool:
    return 0 < len(short) < len(long) and long[: len(short)] == short


def _preference(entity: Dict[str, Any]) -> tuple:
    """Sort key choosing the canonical member of a group (most informative first)"""
    name = entity.get("name") or ""
    has_domain = bool(_DOMAIN_SUFFIX_RE.search(name.strip().lower()))
    is_shouting = name.isupper() and len(name) > 3
    return (
        has_domain,
        is_shouting,
        -len(entity.get("summary") or ""),
        -len(name),
        entity.get("order", 0),
    )


def find_duplicates(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Detect likely duplicate entities

    High confidence (handled automatically as aliases of one canonical entity):
        - identical canonical_key ("Kamptribe" / "KAMPTRIBE" / "kamptribe.com")
        - the same persona identifier ("C01" / "C01 Dutch family")
    Ambiguous (only flagged, never merged):
        - one name is a contiguous prefix/suffix of another ("Maps" / "Google Maps",
          "My Maps" / "Google My Maps")

    Nothing is merged in the graph; the result only annotates the selection.

    Args:
        entities: [{uuid, name, summary?, order?}]

    Returns:
        {uuid: {"canonical_uuid", "canonical_name", "aliases": [...], "duplicate_of",
                "possible_duplicates": [{uuid, name, reason}]}}
    """
    result: Dict[str, Dict[str, Any]] = {
        e["uuid"]: {
            "canonical_uuid": e["uuid"],
            "canonical_name": e.get("name") or "",
            "duplicate_of": None,
            "duplicate_confidence": None,
            "duplicate_reason": None,
            "aliases": [],
            "possible_duplicates": [],
        }
        for e in entities
    }

    # ---- high confidence groups (union by key) ----
    parent = {e["uuid"]: e["uuid"] for e in entities}

    def find(u: str) -> str:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # An identifier family ("C") only counts when at least two distinct numbers use
    # it, so product names such as "GPT4" / "GPT4 Turbo" are not treated as personas.
    family_numbers: Dict[str, set] = {}
    for e in entities:
        parsed = parse_identifier(e.get("name"))
        if parsed:
            family_numbers.setdefault(parsed[0].upper(), set()).add(int(parsed[1]))
    persona_families = {f for f, nums in family_numbers.items() if len(nums) >= 2}

    reasons: Dict[str, str] = {}
    by_key: Dict[str, str] = {}
    by_persona: Dict[str, str] = {}
    for e in entities:
        key = canonical_key(e.get("name"))
        if key:
            if key in by_key:
                union(by_key[key], e["uuid"])
                reasons[e["uuid"]] = "same normalised name"
            else:
                by_key[key] = e["uuid"]
        pkey = _persona_key(e.get("name"))
        if pkey and pkey.rstrip("0123456789") in persona_families:
            if pkey in by_persona:
                union(by_persona[pkey], e["uuid"])
                reasons.setdefault(e["uuid"], "same persona identifier")
            else:
                by_persona[pkey] = e["uuid"]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in entities:
        groups.setdefault(find(e["uuid"]), []).append(e)
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=_preference)
        canonical = ordered[0]
        for m in members:
            info = result[m["uuid"]]
            info["canonical_uuid"] = canonical["uuid"]
            info["canonical_name"] = canonical.get("name") or ""
            info["aliases"] = [x.get("name") for x in members if x["uuid"] != m["uuid"]]
            if m["uuid"] != canonical["uuid"]:
                info["duplicate_of"] = canonical["uuid"]
                info["duplicate_confidence"] = CONFIDENCE_HIGH
                info["duplicate_reason"] = reasons.get(m["uuid"]) or reasons.get(canonical["uuid"]) or "same normalised name"

    # ---- ambiguous candidates (flag only) ----
    canon_entities = [e for e in entities if result[e["uuid"]]["duplicate_of"] is None]
    tokens = {e["uuid"]: canonical_key(e.get("name")).split() for e in canon_entities}
    for a in canon_entities:
        for b in canon_entities:
            if a["uuid"] == b["uuid"]:
                continue
            ta, tb = tokens[a["uuid"]], tokens[b["uuid"]]
            if _is_contiguous_suffix(ta, tb) or _is_contiguous_prefix(ta, tb):
                reason = f"'{a.get('name')}' may be a short form of '{b.get('name')}'"
                for x, y in ((a, b), (b, a)):
                    flags = result[x["uuid"]]["possible_duplicates"]
                    if not any(f["uuid"] == y["uuid"] for f in flags):
                        flags.append({
                            "uuid": y["uuid"],
                            "name": y.get("name"),
                            "confidence": CONFIDENCE_AMBIGUOUS,
                            "reason": reason,
                        })
    return result
