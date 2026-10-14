"""
Agent selection: which knowledge-graph entities participate as active simulation agents

A graph entity is never automatically an agent. The user chooses the active
population explicitly (per entity: active flag, simulation role, evidence role).
The confirmed selection lives in the simulation directory (agent_selection.json);
prepare freezes it into agent_roster.json, which every run snapshots.
"""

import hashlib
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from ..utils.persona_ids import parse_identifier
from .entity_canonicalizer import find_duplicates

logger = get_logger("mirofish.agent_selection")

SELECTION_FILE = "agent_selection.json"
ROSTER_FILE = "agent_roster.json"
SELECTION_VERSION = 1

# Evidence roles drive reporting logic; they are a fixed vocabulary.
EVIDENCE_TARGET_CONSUMER = "target_consumer"
EVIDENCE_INFLUENCER_CONTEXT = "influencer_context"
EVIDENCE_EXCLUDED = "excluded"
EVIDENCE_ROLES = [
    {"id": EVIDENCE_TARGET_CONSUMER, "label": "Target Consumer",
     "description": "Counts in conversion / adoption metrics"},
    {"id": EVIDENCE_INFLUENCER_CONTEXT, "label": "Influencer / Context",
     "description": "May influence others; never counted in conversion metrics"},
    {"id": EVIDENCE_EXCLUDED, "label": "Exclude From Conversion Metrics",
     "description": "Simulated, but excluded from every conversion metric"},
]
EVIDENCE_ROLE_IDS = [r["id"] for r in EVIDENCE_ROLES]

# Simulation roles are configuration (editable per simulation); nothing in the
# code branches on a specific simulation role.
DEFAULT_ROLE_CATALOG = [
    {"id": "prospect", "label": "Prospect", "default_evidence_role": EVIDENCE_TARGET_CONSUMER},
    {"id": "existing_free", "label": "Existing Free", "default_evidence_role": EVIDENCE_INFLUENCER_CONTEXT},
    {"id": "existing_planner", "label": "Existing Planner", "default_evidence_role": EVIDENCE_INFLUENCER_CONTEXT},
    {"id": "existing_advanced", "label": "Existing Advanced", "default_evidence_role": EVIDENCE_INFLUENCER_CONTEXT},
    {"id": "existing_trip_pass", "label": "Existing Trip Pass", "default_evidence_role": EVIDENCE_INFLUENCER_CONTEXT},
    {"id": "brand_advocate", "label": "Brand Advocate", "default_evidence_role": EVIDENCE_INFLUENCER_CONTEXT},
    {"id": "brand_platform", "label": "Brand / Platform", "default_evidence_role": EVIDENCE_EXCLUDED},
    {"id": "community_member", "label": "Community Member", "default_evidence_role": EVIDENCE_INFLUENCER_CONTEXT},
    {"id": "other", "label": "Other", "default_evidence_role": EVIDENCE_EXCLUDED},
]
FALLBACK_ROLE = "other"

_ROLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,39}$")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Entity-type tokens used only to *suggest* a role; the user decides.
_PERSON_TOKENS = {
    "person", "people", "consumer", "customer", "user", "persona", "student", "traveler",
    "traveller", "family", "couple", "respondent", "individual", "participant", "buyer",
    "shopper", "camper", "parent", "citizen", "resident", "prospect", "lead", "visitor",
}
_ADVOCATE_TOKENS = {"advocate", "ambassador", "influencer", "creator", "fan"}
_COMMUNITY_TOKENS = {"community", "group", "forum", "club", "association", "subreddit", "network"}
_PLATFORM_TOKENS = {
    "company", "brand", "platform", "tool", "product", "app", "application", "service",
    "website", "site", "competitor", "organization", "organisation", "business", "media",
    "outlet", "software", "startup", "retailer", "provider", "marketplace", "map", "maps",
    "agency", "institution", "government", "university", "campsite", "operator",
}

SUMMARY_PREVIEW = 240


def _type_tokens(entity_type: Optional[str]) -> List[str]:
    text = _CAMEL_RE.sub(" ", entity_type or "")
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def suggest_simulation_role(entity_type: Optional[str]) -> str:
    """Suggested simulation role for an entity type (a suggestion only)"""
    tokens = set(_type_tokens(entity_type))
    if tokens & _ADVOCATE_TOKENS:
        return "brand_advocate"
    if tokens & _PERSON_TOKENS:
        return "prospect"
    if tokens & _COMMUNITY_TOKENS:
        return "community_member"
    if tokens & _PLATFORM_TOKENS:
        return "brand_platform"
    return FALLBACK_ROLE


def _catalog_role(catalog: List[Dict[str, Any]], role_id: str) -> Optional[Dict[str, Any]]:
    return next((r for r in catalog if r["id"] == role_id), None)


def default_evidence_role(role_id: str, catalog: List[Dict[str, Any]]) -> str:
    role = _catalog_role(catalog, role_id)
    if role and role.get("default_evidence_role") in EVIDENCE_ROLE_IDS:
        return role["default_evidence_role"]
    return EVIDENCE_EXCLUDED


def _available_role(role_id: str, catalog: List[Dict[str, Any]]) -> str:
    if _catalog_role(catalog, role_id):
        return role_id
    if _catalog_role(catalog, FALLBACK_ROLE):
        return FALLBACK_ROLE
    return catalog[0]["id"]


def validate_role_catalog(catalog: Any) -> List[Dict[str, Any]]:
    """
    Validate a simulation-role catalog

    Args:
        catalog: [{id?, label, default_evidence_role?}]

    Returns:
        Normalised catalog

    Raises:
        ValueError: invalid catalog
    """
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("role_catalog must be a non-empty list")
    result: List[Dict[str, Any]] = []
    for item in catalog:
        if not isinstance(item, dict):
            raise ValueError("role_catalog entries must be objects")
        label = str(item.get("label") or item.get("id") or "").strip()
        role_id = str(item.get("id") or "").strip().lower() or re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        if not label or not _ROLE_ID_RE.match(role_id):
            raise ValueError(f"Invalid simulation role: {item!r}")
        if any(r["id"] == role_id for r in result):
            raise ValueError(f"Duplicate simulation role id: {role_id}")
        evidence = item.get("default_evidence_role") or EVIDENCE_EXCLUDED
        if evidence not in EVIDENCE_ROLE_IDS:
            raise ValueError(f"Invalid default evidence role for {role_id}: {evidence}")
        result.append({"id": role_id, "label": label[:60], "default_evidence_role": evidence})
    return result


def _brief_id_key(name: Optional[str]):
    parsed = parse_identifier(name)
    return (parsed[0].upper(), int(parsed[1])) if parsed else None


def _entity_base(entity: Any, order: int) -> Dict[str, Any]:
    entity_type = entity.get_entity_type() if hasattr(entity, "get_entity_type") else entity.get("entity_type")
    summary = (getattr(entity, "summary", None) if not isinstance(entity, dict) else entity.get("summary")) or ""
    return {
        "uuid": entity.uuid if not isinstance(entity, dict) else entity["uuid"],
        "name": entity.name if not isinstance(entity, dict) else entity.get("name", ""),
        "entity_type": entity_type or "Entity",
        "summary": summary,
        "order": order,
    }


def selection_hash(entries: Iterable[Dict[str, Any]]) -> str:
    """
    Fingerprint of the active population (active entities and their roles)

    Args:
        entries: Selection entries {entity_uuid, active, simulation_role, evidence_role}
    """
    active = sorted(
        (e["entity_uuid"], e.get("simulation_role") or "", e.get("evidence_role") or "")
        for e in entries if e.get("active")
    )
    return hashlib.sha256(json.dumps(active).encode("utf-8")).hexdigest()[:16]


def summarize(entries: List[Dict[str, Any]], catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Counts shown before prepare/start

    Args:
        entries: Selection entries (active flag, roles, entity_type)
        catalog: Simulation-role catalog

    Returns:
        {active_agents, graph_entities, context_entities, by_simulation_role,
         by_evidence_role, by_entity_type}
    """
    active = [e for e in entries if e.get("active")]
    by_role: Dict[str, int] = {}
    by_evidence: Dict[str, int] = {r: 0 for r in EVIDENCE_ROLE_IDS}
    by_type: Dict[str, Dict[str, int]] = {}
    for e in entries:
        t = by_type.setdefault(e.get("entity_type") or "Entity", {"total": 0, "active": 0})
        t["total"] += 1
        if e.get("active"):
            t["active"] += 1
    for e in active:
        by_role[e.get("simulation_role") or FALLBACK_ROLE] = by_role.get(e.get("simulation_role") or FALLBACK_ROLE, 0) + 1
        by_evidence[e.get("evidence_role") or EVIDENCE_EXCLUDED] = by_evidence.get(e.get("evidence_role") or EVIDENCE_EXCLUDED, 0) + 1
    labels = {r["id"]: r["label"] for r in catalog}
    return {
        "active_agents": len(active),
        "graph_entities": len(entries),
        "context_entities": len(entries) - len(active),
        "by_simulation_role": [
            {"id": rid, "label": labels.get(rid, rid), "count": n} for rid, n in by_role.items()
        ],
        "by_evidence_role": [
            {"id": r["id"], "label": r["label"], "count": by_evidence.get(r["id"], 0)} for r in EVIDENCE_ROLES
        ],
        "by_entity_type": by_type,
    }


class AgentSelectionStore:
    """File storage of the selection (editable) and the roster (frozen at prepare)"""

    _lock = threading.RLock()

    @staticmethod
    def sim_dir(simulation_id: str) -> str:
        return os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

    @classmethod
    def _read(cls, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Unreadable selection file {path}: {e}")
            return None

    @classmethod
    def _write(cls, path: str, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, simulation_id: str) -> Optional[Dict[str, Any]]:
        return cls._read(os.path.join(cls.sim_dir(simulation_id), SELECTION_FILE))

    @classmethod
    def save(cls, simulation_id: str, data: Dict[str, Any]):
        with cls._lock:
            cls._write(os.path.join(cls.sim_dir(simulation_id), SELECTION_FILE), data)

    @classmethod
    def load_roster(cls, simulation_id: str) -> Optional[Dict[str, Any]]:
        return cls._read(os.path.join(cls.sim_dir(simulation_id), ROSTER_FILE))

    @classmethod
    def save_roster(cls, simulation_id: str, roster: Dict[str, Any]):
        with cls._lock:
            cls._write(os.path.join(cls.sim_dir(simulation_id), ROSTER_FILE), roster)


def build_selection_view(
    entities: List[Any],
    saved: Optional[Dict[str, Any]],
    brief_persona_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Merge the eligible graph entities with the saved selection

    Entities without a saved decision are inactive (with a suggested role); the
    view is never persisted by reading it.

    Args:
        entities: Eligible graph entities (EntityNode or dicts)
        saved: Saved selection (agent_selection.json) or None
        brief_persona_ids: Persona ids requested by the assessment brief

    Returns:
        {role_catalog, evidence_roles, entities, summary, confirmed, saved,
         selection_hash, missing_saved_entities, brief_persona_ids}
    """
    catalog = (saved or {}).get("role_catalog") or [dict(r) for r in DEFAULT_ROLE_CATALOG]
    decisions = {e["entity_uuid"]: e for e in (saved or {}).get("entities", []) if e.get("entity_uuid")}
    brief_keys = {}
    for pid in brief_persona_ids or []:
        key = _brief_id_key(pid)
        if key:
            brief_keys[key] = pid

    bases = [_entity_base(e, i) for i, e in enumerate(entities)]
    duplicates = find_duplicates(bases)
    rows = []
    for base in bases:
        decision = decisions.get(base["uuid"])
        suggested = _available_role(suggest_simulation_role(base["entity_type"]), catalog)
        if decision:
            sim_role = _available_role(decision.get("simulation_role") or suggested, catalog)
            ev_role = decision.get("evidence_role") if decision.get("evidence_role") in EVIDENCE_ROLE_IDS \
                else default_evidence_role(sim_role, catalog)
            active = bool(decision.get("active"))
        else:
            sim_role = suggested
            ev_role = default_evidence_role(sim_role, catalog)
            active = False
        summary = base["summary"]
        dup = duplicates[base["uuid"]]
        rows.append({
            "entity_uuid": base["uuid"],
            "name": base["name"],
            "entity_type": base["entity_type"],
            "description": summary[:SUMMARY_PREVIEW] + ("..." if len(summary) > SUMMARY_PREVIEW else ""),
            "active": active,
            "simulation_role": sim_role,
            "evidence_role": ev_role,
            "suggested_simulation_role": suggested,
            "has_saved_decision": decision is not None,
            "brief_persona_id": brief_keys.get(_brief_id_key(base["name"])),
            "duplicate_of": dup["duplicate_of"],
            "duplicate_of_name": dup["canonical_name"] if dup["duplicate_of"] else None,
            "duplicate_confidence": dup["duplicate_confidence"],
            "duplicate_reason": dup["duplicate_reason"],
            "aliases": dup["aliases"],
            "possible_duplicates": dup["possible_duplicates"],
        })
    live = {r["entity_uuid"] for r in rows}
    missing = [
        {"entity_uuid": uid, "name": d.get("name"), "active": bool(d.get("active"))}
        for uid, d in decisions.items() if uid not in live
    ]
    return {
        "role_catalog": catalog,
        "evidence_roles": EVIDENCE_ROLES,
        "entities": rows,
        "summary": summarize(rows, catalog),
        "confirmed": bool((saved or {}).get("confirmed")) and not missing,
        "saved": saved is not None,
        "selection_hash": selection_hash(rows),
        "missing_saved_entities": missing,
        "brief_persona_ids": list(brief_persona_ids or []),
        "updated_at": (saved or {}).get("updated_at"),
        "confirmed_at": (saved or {}).get("confirmed_at"),
    }


def apply_selection_update(
    entities: List[Any],
    payload: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    graph_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a selection update and return the document to persist

    Args:
        entities: Eligible graph entities
        payload: {entities: [{entity_uuid, active?, simulation_role?, evidence_role?}],
                  role_catalog?: [...], confirmed?: bool}
        previous: Previously saved selection (decisions not in the payload are kept)
        graph_id: Graph the selection belongs to

    Returns:
        Selection document (agent_selection.json)

    Raises:
        ValueError: unknown entities, unknown roles or an empty confirmed population
    """
    catalog = validate_role_catalog(payload["role_catalog"]) if payload.get("role_catalog") \
        else ((previous or {}).get("role_catalog") or [dict(r) for r in DEFAULT_ROLE_CATALOG])
    view = build_selection_view(entities, previous)
    rows = {r["entity_uuid"]: r for r in view["entities"]}
    role_ids = {r["id"] for r in catalog}

    updates = payload.get("entities") or []
    if not isinstance(updates, list):
        raise ValueError("entities must be a list")
    for item in updates:
        uid = (item or {}).get("entity_uuid")
        if uid not in rows:
            raise ValueError(f"Unknown graph entity: {uid}")
        row = rows[uid]
        if "simulation_role" in item and item["simulation_role"] is not None:
            if item["simulation_role"] not in role_ids:
                raise ValueError(f"Unknown simulation role: {item['simulation_role']}")
            row["simulation_role"] = item["simulation_role"]
        if "evidence_role" in item and item["evidence_role"] is not None:
            if item["evidence_role"] not in EVIDENCE_ROLE_IDS:
                raise ValueError(f"Unknown evidence role: {item['evidence_role']}")
            row["evidence_role"] = item["evidence_role"]
        if "active" in item:
            row["active"] = bool(item["active"])
        row["has_saved_decision"] = True

    for row in rows.values():
        if row["simulation_role"] not in role_ids:
            raise ValueError(
                f"Entity '{row['name']}' uses simulation role '{row['simulation_role']}' "
                f"which is not in the role catalog"
            )

    confirmed = bool(payload.get("confirmed"))
    entries = [
        {
            "entity_uuid": r["entity_uuid"],
            "name": r["name"],
            "entity_type": r["entity_type"],
            "active": r["active"],
            "simulation_role": r["simulation_role"],
            "evidence_role": r["evidence_role"],
        }
        for r in view["entities"]
    ]
    if confirmed and not any(e["active"] for e in entries):
        raise ValueError("Select at least one active agent before confirming the population")
    now = datetime.now().isoformat()
    return {
        "version": SELECTION_VERSION,
        "graph_id": graph_id,
        "confirmed": confirmed,
        "confirmed_at": now if confirmed else None,
        "updated_at": now,
        "role_catalog": catalog,
        "entities": entries,
        "selection_hash": selection_hash(entries),
    }


def build_roster(
    simulation_id: str,
    graph_id: str,
    entities: List[Any],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Freeze a confirmed selection into the population of a preparation

    Active agents get contiguous agent ids (0..N-1) in graph order, which is the
    id OASIS uses for profiles, configs and action logs.

    Args:
        simulation_id: Simulation ID
        graph_id: Graph ID
        entities: Eligible graph entities (graph order)
        selection: Confirmed selection document

    Returns:
        Roster document (agent_roster.json)

    Raises:
        ValueError: unconfirmed selection, empty population, or active entities
            that no longer exist in the graph
    """
    if not selection or not selection.get("confirmed"):
        raise ValueError("Agent selection is not confirmed: choose the active agents first")
    view = build_selection_view(entities, selection)
    missing_active = [m for m in view["missing_saved_entities"] if m["active"]]
    if missing_active:
        names = ", ".join(str(m["name"]) for m in missing_active)
        raise ValueError(f"Selected agents no longer exist in the graph: {names}. Review the agent selection.")
    catalog = view["role_catalog"]
    labels = {r["id"]: r["label"] for r in catalog}
    evidence_labels = {r["id"]: r["label"] for r in EVIDENCE_ROLES}
    by_uuid = {
        (e.uuid if not isinstance(e, dict) else e["uuid"]): e for e in entities
    }

    active_agents, context_entities = [], []
    for row in view["entities"]:
        entity = by_uuid[row["entity_uuid"]]
        summary = (entity.get("summary") if isinstance(entity, dict) else entity.summary) or ""
        item = {
            "entity_uuid": row["entity_uuid"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "simulation_role": row["simulation_role"],
            "simulation_role_label": labels.get(row["simulation_role"], row["simulation_role"]),
            "evidence_role": row["evidence_role"],
            "evidence_role_label": evidence_labels.get(row["evidence_role"], row["evidence_role"]),
            "summary": summary[:SUMMARY_PREVIEW],
        }
        if row["active"]:
            item["agent_id"] = len(active_agents)
            active_agents.append(item)
        else:
            item["duplicate_of"] = row["duplicate_of"]
            context_entities.append(item)
    if not active_agents:
        raise ValueError("The confirmed selection has no active agents")
    return {
        "version": SELECTION_VERSION,
        "simulation_id": simulation_id,
        "graph_id": graph_id,
        "selection_hash": view["selection_hash"],
        "prepared_at": datetime.now().isoformat(),
        "role_catalog": catalog,
        "evidence_roles": EVIDENCE_ROLES,
        "summary": view["summary"],
        "active_agents": active_agents,
        "context_entities": context_entities,
    }


def roster_agent_roles(roster: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """agent_id -> roster entry of an active agent"""
    return {int(a["agent_id"]): a for a in (roster or {}).get("active_agents", []) if a.get("agent_id") is not None}
