"""
Run Contract (mirofish.run_contract.v1): the deterministic control document that owns a contracted experiment

A project may carry at most one uploaded document whose filename matches
``run[-_ ]?contract`` (.txt, .md or .json). Its text is exactly one JSON object,
parsed with json.loads - no LLM, no fuzzy matching. When a valid contract is
present it is the single source of execution truth for the simulation:

    population        which graph entities are agents, with their simulation and evidence roles
    experiment_plan   the private stages (never derived from Assessment Brief prose)
    source_requirements  the authoritative documents/sections that must be covered
    max_rounds        the exact run length
    platform          the platform(s) that run

The contract is control-plane input, never world knowledge: it is excluded from
ontology/graph extraction, the brief, seed generation and ordinary report evidence,
but snapshotted verbatim into every contracted run.
"""

import hashlib
import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import Config
from ..utils.logger import get_logger
from ..utils.persona_ids import parse_identifier
from .agent_selection import (
    EVIDENCE_EXCLUDED, EVIDENCE_INFLUENCER_CONTEXT, EVIDENCE_ROLE_IDS, EVIDENCE_TARGET_CONSUMER,
    selection_hash, validate_role_catalog,
)
from .assessment_spec import is_run_contract_filename

logger = get_logger("mirofish.run_contract")

SCHEMA_V1 = "mirofish.run_contract.v1"
SUPPORTED_SCHEMAS = (SCHEMA_V1,)
CONTRACT_FILE = "run_contract.json"
CONTRACT_EXTENSIONS = (".txt", ".md", ".json")
PLATFORMS = ("parallel", "twitter", "reddit")

TOP_LEVEL_FIELDS = (
    "schema", "contract_id", "title", "platform", "max_rounds", "role_catalog", "population",
    "reject_unlisted_active_entities", "preflight_expectations", "source_requirements", "experiment_plan",
)
POPULATION_FIELDS = {"persona_id", "simulation_role", "evidence_role", "expected_name"}
ROLE_FIELDS = {"id", "label", "default_evidence_role"}
EXPECTATION_FIELDS = (
    "active_agents", "target_consumers", "influencer_context", "excluded_active", "stages", "max_rounds",
)
PLAN_FIELDS = {"stages", "seed_blocked_terms", "derived_fields"}
STAGE_FIELDS = {
    "id", "name", "order", "trigger_round", "target", "private_instruction", "context",
    "context_sections", "records", "condition", "completion",
}
TARGET_FIELDS = {"persona_ids", "simulation_roles", "evidence_roles"}
CONDITION_FIELDS = {"type", "terms", "stage_id"}
COMPLETION_FIELDS = {"type", "terms", "action_types", "rounds"}
DERIVED_FIELD_FIELDS = {"type", "terms", "record", "evidence_roles"}

_CONTRACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_PERSONA_ID_RE = re.compile(r"^[A-Za-z]{1,4}\d{1,4}$")
_STAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,60}$")
_FENCE_RE = re.compile(r"^```[A-Za-z]*[ \t]*\n(.*)\n[ \t]*```$", re.DOTALL)


# ---------------------------------------------------------------- parsing / validation

def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key '{key}'")
        result[key] = value
    return result


def parse_contract_text(text: Optional[str]) -> Dict[str, Any]:
    """
    Parse the extracted text of a Run Contract document

    The text must be exactly one JSON object (optionally wrapped in a single
    Markdown code fence); anything else, trailing data or duplicate keys included,
    is an error.

    Raises:
        ValueError: empty, malformed or non-object content
    """
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Run Contract document is empty or its text could not be extracted")
    fence = _FENCE_RE.match(stripped)
    body = fence.group(1).strip() if fence else stripped
    try:
        data = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as e:
        raise ValueError(f"Run Contract is not valid JSON: {e.msg} (line {e.lineno}, column {e.colno})")
    except ValueError as e:
        raise ValueError(f"Run Contract is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("Run Contract must be one JSON object")
    return data


def canonical_json(data: Any) -> str:
    """Canonical JSON text (sorted keys, no whitespace) used for hashing"""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contract_hash(raw: Dict[str, Any]) -> str:
    """Stable SHA-256 of the contract (canonical JSON of the parsed object)"""
    return sha256_text(canonical_json(raw))


def persona_key(persona_id: Optional[str]) -> Optional[Tuple[str, int]]:
    """Comparison key of a persona identifier with the canonical parser (C01 == C1)"""
    parsed = parse_identifier(persona_id)
    return (parsed[0].upper(), int(parsed[1])) if parsed else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unknown(obj: Dict[str, Any], allowed: Iterable[str], where: str, errors: List[str]):
    extra = sorted(set(obj) - set(allowed))
    if extra:
        errors.append(f"{where}: unknown field(s) {extra}")


def _stage_reach(stage: Dict[str, Any], population: List[Dict[str, Any]]) -> List[str]:
    """Contract persona ids a (raw) stage targets"""
    target = stage.get("target") or {}
    persona_keys = {persona_key(p) for p in target.get("persona_ids") or []}
    roles = set(target.get("simulation_roles") or [])
    evidence = set(target.get("evidence_roles") or [])
    if not (persona_keys or roles or evidence):
        return [p["persona_id"] for p in population]
    return [
        p["persona_id"] for p in population
        if persona_key(p["persona_id"]) in persona_keys
        or p["simulation_role"] in roles or p["evidence_role"] in evidence
    ]


def validate_contract(raw: Any) -> Dict[str, Any]:
    """
    Validate a parsed Run Contract and return its normalised form

    Raises:
        ValueError: every problem found, joined with "; "
    """
    from .research_instructions import validate_plan
    from .source_requirements import validate_requirements

    errors: List[str] = []
    if not isinstance(raw, dict):
        raise ValueError("Run Contract must be one JSON object")
    _unknown(raw, TOP_LEVEL_FIELDS, "contract", errors)
    missing = [f for f in TOP_LEVEL_FIELDS if f not in raw]
    if missing:
        errors.append(f"contract: missing field(s) {missing}")

    schema = raw.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        errors.append(f"unsupported schema {schema!r} (supported: {list(SUPPORTED_SCHEMAS)})")
    contract_id = raw.get("contract_id")
    if not isinstance(contract_id, str) or not contract_id.strip():
        errors.append("contract_id must be a non-empty string")
    elif not _CONTRACT_ID_RE.match(contract_id):
        errors.append(f"contract_id {contract_id!r} may only contain letters, digits, '.', '_' and '-'")
    if not isinstance(raw.get("title"), str) or not raw.get("title", "").strip():
        errors.append("title must be a non-empty string")
    if raw.get("platform") not in PLATFORMS:
        errors.append(f"platform must be one of {list(PLATFORMS)}, got {raw.get('platform')!r}")
    max_rounds = raw.get("max_rounds")
    if not _is_int(max_rounds) or max_rounds <= 0:
        errors.append(f"max_rounds must be a positive integer, got {max_rounds!r}")
    if raw.get("reject_unlisted_active_entities") is not True:
        errors.append("reject_unlisted_active_entities must be true (v1 never admits agents outside the contract)")

    # ---- role catalog ----
    catalog: List[Dict[str, Any]] = []
    raw_catalog = raw.get("role_catalog")
    if not isinstance(raw_catalog, list) or not raw_catalog:
        errors.append("role_catalog must be a non-empty list")
    else:
        for i, role in enumerate(raw_catalog):
            if not isinstance(role, dict) or not isinstance(role.get("id"), str) or not role.get("id"):
                errors.append(f"role_catalog[{i}] needs an explicit string id")
                continue
            _unknown(role, ROLE_FIELDS, f"role_catalog[{i}]", errors)
            if role.get("default_evidence_role") not in EVIDENCE_ROLE_IDS:
                errors.append(f"role_catalog[{i}] ({role['id']}): default_evidence_role must be one of {EVIDENCE_ROLE_IDS}")
        try:
            catalog = validate_role_catalog(raw_catalog)
            for raw_role, role in zip(raw_catalog, catalog):
                if isinstance(raw_role, dict) and raw_role.get("id") != role["id"]:
                    errors.append(f"role id {raw_role.get('id')!r} is not canonical (expected {role['id']!r})")
        except ValueError as e:
            errors.append(f"role_catalog: {e}")
    role_ids = {r["id"] for r in catalog}

    # ---- population ----
    population: List[Dict[str, Any]] = []
    raw_population = raw.get("population")
    if not isinstance(raw_population, list) or not raw_population:
        errors.append("population must be a non-empty list")
        raw_population = []
    seen: Dict[Tuple[str, int], str] = {}
    for i, entry in enumerate(raw_population):
        if not isinstance(entry, dict):
            errors.append(f"population[{i}] must be an object")
            continue
        _unknown(entry, POPULATION_FIELDS, f"population[{i}]", errors)
        pid = entry.get("persona_id")
        if not isinstance(pid, str) or not _PERSONA_ID_RE.match(pid):
            errors.append(f"population[{i}]: persona_id {pid!r} must look like 'C01'")
            continue
        key = persona_key(pid)
        if key in seen:
            errors.append(f"persona_id {pid} occurs more than once (also {seen[key]})")
            continue
        seen[key] = pid
        sim_role = entry.get("simulation_role")
        ev_role = entry.get("evidence_role")
        if sim_role not in role_ids:
            errors.append(f"{pid}: simulation_role {sim_role!r} is not in role_catalog")
        if ev_role not in EVIDENCE_ROLE_IDS:
            errors.append(f"{pid}: evidence_role {ev_role!r} must be one of {EVIDENCE_ROLE_IDS}")
        expected_name = entry.get("expected_name")
        if expected_name is not None and (not isinstance(expected_name, str) or not expected_name.strip()):
            errors.append(f"{pid}: expected_name must be a non-empty string when given")
        population.append({
            "persona_id": pid.upper(),
            "simulation_role": sim_role,
            "evidence_role": ev_role,
            "expected_name": expected_name.strip() if isinstance(expected_name, str) else None,
        })
    population_keys = {persona_key(p["persona_id"]) for p in population}

    # ---- source requirements ----
    requirements: Dict[str, Any] = {}
    try:
        requirements = validate_requirements(raw.get("source_requirements"))
    except (ValueError, TypeError) as e:
        errors.append(f"source_requirements: {e}")
    section_ids = {s["id"] for d in requirements.get("documents", []) for s in d["sections"]}

    # ---- experiment plan ----
    plan: Dict[str, Any] = {"stages": []}
    raw_plan = raw.get("experiment_plan")
    if not isinstance(raw_plan, dict):
        errors.append("experiment_plan must be an object")
    else:
        _unknown(raw_plan, PLAN_FIELDS, "experiment_plan", errors)
        raw_stages = raw_plan.get("stages")
        if not isinstance(raw_stages, list):
            errors.append("experiment_plan.stages must be a list")
            raw_stages = []
        stage_ids: List[str] = []
        for i, stage in enumerate(raw_stages):
            where = f"stage[{i}]"
            if not isinstance(stage, dict):
                errors.append(f"{where} must be an object")
                continue
            sid = stage.get("id")
            if not isinstance(sid, str) or not _STAGE_ID_RE.match(sid):
                errors.append(f"{where}: id {sid!r} must match [A-Za-z0-9_-]{{1,60}}")
            else:
                where = f"stage {sid}"
                if sid in stage_ids:
                    errors.append(f"duplicate stage id {sid}")
                stage_ids.append(sid)
            _unknown(stage, STAGE_FIELDS, where, errors)
            if not _is_int(stage.get("trigger_round")) or stage["trigger_round"] < 1:
                errors.append(f"{where}: trigger_round must be an integer >= 1")
            if "order" in stage and not _is_int(stage["order"]):
                errors.append(f"{where}: order must be an integer")
            target = stage.get("target") or {}
            if not isinstance(target, dict):
                errors.append(f"{where}: target must be an object")
                target = {}
            if "agent_ids" in target:
                errors.append(f"{where}: contract stages target persona_ids or roles, never agent_ids")
            _unknown(target, TARGET_FIELDS, f"{where}.target", errors)
            for pid in target.get("persona_ids") or []:
                if not isinstance(pid, str) or not _PERSONA_ID_RE.match(pid):
                    errors.append(f"{where}: target persona id {pid!r} is not a persona id")
                elif persona_key(pid) not in population_keys:
                    errors.append(f"{where}: target persona {pid} is not in the contract population")
            for part, allowed in (("condition", CONDITION_FIELDS), ("completion", COMPLETION_FIELDS)):
                if isinstance(stage.get(part), dict):
                    _unknown(stage[part], allowed, f"{where}.{part}", errors)
                elif part in stage:
                    errors.append(f"{where}: {part} must be an object")
            for section in stage.get("context_sections") or []:
                if section not in section_ids:
                    errors.append(f"{where}: context section {section!r} is not a required section of source_requirements")
            if population and not _stage_reach(stage, population):
                errors.append(f"{where}: targets no persona of the contract population")
        for name, rule in (raw_plan.get("derived_fields") or {}).items():
            if isinstance(rule, dict):
                _unknown(rule, DERIVED_FIELD_FIELDS, f"derived field {name}", errors)
        try:
            plan = validate_plan(raw_plan, role_ids)
        except ValueError as e:
            errors.append(f"experiment_plan: {e}")
        for stage in plan.get("stages", []):
            stage.pop("source", None)

    # ---- counts and self-declared expectations ----
    counts = {
        "active_agents": len(population),
        "target_consumers": sum(1 for p in population if p["evidence_role"] == EVIDENCE_TARGET_CONSUMER),
        "influencer_context": sum(1 for p in population if p["evidence_role"] == EVIDENCE_INFLUENCER_CONTEXT),
        "excluded_active": sum(1 for p in population if p["evidence_role"] == EVIDENCE_EXCLUDED),
        "stages": len(plan.get("stages", [])),
        "max_rounds": max_rounds if _is_int(max_rounds) else None,
    }
    expectations = raw.get("preflight_expectations")
    if not isinstance(expectations, dict):
        errors.append("preflight_expectations must be an object")
        expectations = {}
    else:
        _unknown(expectations, EXPECTATION_FIELDS, "preflight_expectations", errors)
        for key in EXPECTATION_FIELDS:
            if key not in expectations:
                errors.append(f"preflight_expectations.{key} is required")
            elif not _is_int(expectations[key]):
                errors.append(f"preflight_expectations.{key} must be an integer")
            elif expectations[key] != counts[key]:
                errors.append(
                    f"preflight_expectations.{key}={expectations[key]} disagrees with the contract itself ({counts[key]})"
                )

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema": schema,
        "contract_id": contract_id,
        "title": raw["title"],
        "platform": raw["platform"],
        "max_rounds": max_rounds,
        "role_catalog": catalog,
        "population": population,
        "reject_unlisted_active_entities": True,
        "preflight_expectations": dict(expectations),
        "source_requirements": requirements,
        "experiment_plan": plan,
        "counts": counts,
    }


def contract_plan_document(contract: Dict[str, Any], digest: str) -> Dict[str, Any]:
    """experiment_plan.json content owned by a contract (before save metadata)"""
    return {**contract["experiment_plan"], "contract_id": contract["contract_id"], "contract_hash": digest}


# ---------------------------------------------------------------- project discovery

def contract_documents(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in documents or [] if is_run_contract_filename(d.get("original_filename"))]


def without_contract_documents(documents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Source documents that are world knowledge (Run Contract documents removed)"""
    return [d for d in documents or [] if not is_run_contract_filename(d.get("original_filename"))]


def evaluate_contract_documents(documents: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect and validate the Run Contract among source documents

    Returns:
        {present, valid, errors, documents, id, schema, hash, contract (normalised),
         raw (parsed object)}; present=false when no document carries the name
    """
    matches = contract_documents(documents)
    if not matches:
        return {"present": False, "valid": False, "errors": [], "documents": [],
                "id": None, "schema": None, "hash": None, "contract": None, "raw": None}
    names = [d.get("original_filename") for d in matches]
    result = {"present": True, "valid": False, "errors": [], "documents": names,
              "id": None, "schema": None, "hash": None, "contract": None, "raw": None}
    if len(matches) > 1:
        result["errors"].append(f"exactly one Run Contract document is allowed, found {len(matches)}: {names}")
        return result
    doc = matches[0]
    ext = os.path.splitext(doc.get("original_filename") or "")[1].lower()
    if ext not in CONTRACT_EXTENSIONS:
        result["errors"].append(f"Run Contract must be a {'/'.join(CONTRACT_EXTENSIONS)} file, got {names[0]!r}")
        return result
    try:
        raw = parse_contract_text(doc.get("text"))
    except ValueError as e:
        result["errors"].append(str(e))
        return result
    result["raw"] = raw
    result["schema"] = raw.get("schema")
    result["id"] = raw.get("contract_id") if isinstance(raw.get("contract_id"), str) else None
    try:
        result["contract"] = validate_contract(raw)
    except ValueError as e:
        result["errors"].extend(str(e).split("; "))
        return result
    result["hash"] = contract_hash(raw)
    result["valid"] = True
    return result


def load_project_contract(project_id: Optional[str]) -> Dict[str, Any]:
    """The project's Run Contract state (see evaluate_contract_documents)"""
    from ..models.project import ProjectManager

    documents = ProjectManager.get_source_documents(project_id) if project_id else []
    return evaluate_contract_documents(documents)


def project_has_contract(project_id: Optional[str]) -> bool:
    """Whether the project carries any Run Contract document (valid or not)"""
    from ..models.project import ProjectManager

    project = ProjectManager.get_project(project_id) if project_id else None
    return bool(project and any(is_run_contract_filename(f.get("filename")) for f in project.files))


# ---------------------------------------------------------------- persona resolution

def resolve_population(
    contract: Dict[str, Any],
    graph_nodes: List[Dict[str, Any]],
    eligible_entities: List[Any],
) -> Dict[str, Any]:
    """
    Resolve every contract persona id to exactly one graph entity

    Every live graph node whose name parses (canonical persona-id parser) to the
    requested identifier is a candidate; exactly one candidate that is also an
    eligible (typed) entity resolves the persona. Zero or several candidates block.

    Args:
        contract: Validated contract
        graph_nodes: Every node of the graph ({uuid, name, labels})
        eligible_entities: Entities offered by the agent selection

    Returns:
        {rows: [{persona_id, status, entity_uuid, name, entity_type, simulation_role,
         evidence_role, candidates}], missing, duplicates, name_mismatches, resolved}
    """
    eligible = {}
    for e in eligible_entities:
        uid = e["uuid"] if isinstance(e, dict) else e.uuid
        name = e.get("name") if isinstance(e, dict) else e.name
        etype = e.get("entity_type") if isinstance(e, dict) else e.get_entity_type()
        eligible[uid] = {"name": name, "entity_type": etype}
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for node in graph_nodes:
        key = persona_key(node.get("name"))
        if key:
            by_key.setdefault(key, []).append(node)

    rows, missing, duplicates, mismatches = [], [], [], []
    for entry in contract["population"]:
        pid = entry["persona_id"]
        candidates = by_key.get(persona_key(pid), [])
        row = {
            "persona_id": pid,
            "simulation_role": entry["simulation_role"],
            "evidence_role": entry["evidence_role"],
            "expected_name": entry.get("expected_name"),
            "entity_uuid": None,
            "name": None,
            "entity_type": None,
            "candidates": [{"entity_uuid": c["uuid"], "name": c.get("name"),
                            "eligible": c["uuid"] in eligible} for c in candidates],
        }
        if len(candidates) > 1:
            row["status"] = "duplicate"
            duplicates.append(pid)
        elif not candidates or candidates[0]["uuid"] not in eligible:
            row["status"] = "missing"
            missing.append(pid)
        else:
            node = candidates[0]
            row.update({"status": "resolved", "entity_uuid": node["uuid"], "name": node.get("name"),
                        "entity_type": eligible[node["uuid"]]["entity_type"]})
            expected = entry.get("expected_name")
            if expected and " ".join(expected.split()).lower() != " ".join((node.get("name") or "").split()).lower():
                row["status"] = "name_mismatch"
                mismatches.append(pid)
        rows.append(row)
    return {
        "rows": rows,
        "missing": missing,
        "duplicates": duplicates,
        "name_mismatches": mismatches,
        "resolved": sum(1 for r in rows if r["status"] == "resolved"),
    }


def expected_selection_hash(resolution: Dict[str, Any]) -> str:
    """selection_hash of the population the contract requires"""
    return selection_hash([
        {"entity_uuid": r["entity_uuid"], "active": True,
         "simulation_role": r["simulation_role"], "evidence_role": r["evidence_role"]}
        for r in resolution["rows"] if r["status"] == "resolved"
    ])


def read_graph(graph_id: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """(every graph node, eligible entities) of a graph"""
    from .entity_reader import EntityReader

    reader = EntityReader()
    nodes = reader.get_all_nodes(graph_id)
    eligible = reader.filter_defined_entities(graph_id=graph_id, enrich_with_edges=False).entities
    return nodes, eligible


# ---------------------------------------------------------------- simulation-level store

class RunContractStore:
    """run_contract.json in the simulation (configuration) directory: the applied contract"""

    _lock = threading.RLock()

    @staticmethod
    def path(simulation_id: str) -> str:
        return os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, CONTRACT_FILE)

    @classmethod
    def load(cls, simulation_id: str) -> Optional[Dict[str, Any]]:
        path = cls.path(simulation_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Unreadable run contract {path}: {e}")
            return None

    @classmethod
    def save(cls, simulation_id: str, data: Dict[str, Any]):
        path = cls.path(simulation_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with cls._lock:
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)


def apply_contract(simulation_id: str) -> Dict[str, Any]:
    """
    Persist the configuration a valid Run Contract dictates for a simulation

    Writes agent_selection.json (contract personas active with their roles, every
    other graph entity context-only; confirmed only when every persona resolved
    exactly once), experiment_plan.json, source_requirements.json and
    run_contract.json. A changed population or contract invalidates an existing
    preparation (existing runs keep their own snapshot). Never calls an LLM.

    Callers hold RunStore.lock(simulation_id).

    Returns:
        {"applied": bool, "confirmed": bool, "preparation_invalidated": bool, "errors": [...]}

    Raises:
        ValueError: simulation unknown, no contract or an invalid contract
    """
    from .agent_selection import AgentSelectionStore, apply_selection_update
    from .research_instructions import ExperimentPlanStore
    from .simulation_manager import SimulationManager, SimulationStatus
    from .source_requirements import SourceRequirementsStore

    manager = SimulationManager()
    state = manager.get_simulation(simulation_id)
    if not state:
        raise ValueError(f"Simulation not found: {simulation_id}")
    loaded = load_project_contract(state.project_id)
    if not loaded["present"]:
        raise ValueError("The project has no Run Contract document")
    if not loaded["valid"]:
        raise ValueError("Run Contract invalid: " + "; ".join(loaded["errors"]))
    contract, digest = loaded["contract"], loaded["hash"]

    nodes, eligible = read_graph(state.graph_id)
    resolution = resolve_population(contract, nodes, eligible)
    resolved = {r["entity_uuid"]: r for r in resolution["rows"] if r["status"] == "resolved"}
    all_resolved = resolution["resolved"] == len(contract["population"])
    # Context-only entities carry the contract's excluded role when it has one
    context_role = next((r["id"] for r in contract["role_catalog"] if r["default_evidence_role"] == EVIDENCE_EXCLUDED), None)
    updates = []
    for e in eligible:
        row = resolved.get(e.uuid)
        if row:
            updates.append({"entity_uuid": e.uuid, "active": True,
                            "simulation_role": row["simulation_role"], "evidence_role": row["evidence_role"]})
        else:
            update = {"entity_uuid": e.uuid, "active": False, "evidence_role": EVIDENCE_EXCLUDED}
            if context_role:
                update["simulation_role"] = context_role
            updates.append(update)
    document = apply_selection_update(
        eligible,
        {"entities": updates, "role_catalog": contract["role_catalog"], "confirmed": bool(all_resolved and resolved)},
        None, state.graph_id,
    )
    document["contract_id"] = contract["contract_id"]
    document["contract_hash"] = digest

    roster = AgentSelectionStore.load_roster(simulation_id)
    previous = RunContractStore.load(simulation_id)
    invalidates = (state.profiles_generated or state.config_generated) and (
        roster is None
        or roster.get("selection_hash") != document["selection_hash"]
        or roster.get("contract_hash") != digest
        or (previous or {}).get("contract_hash") != digest
    )

    AgentSelectionStore.save(simulation_id, document)
    ExperimentPlanStore.save(simulation_id, contract_plan_document(contract, digest))
    SourceRequirementsStore.save(simulation_id, contract["source_requirements"])
    RunContractStore.save(simulation_id, {
        "schema": contract["schema"],
        "contract_id": contract["contract_id"],
        "contract_hash": digest,
        "source_document": loaded["documents"][0],
        "applied_at": datetime.now().isoformat(),
        "contract": loaded["raw"],
    })
    if invalidates:
        state.status = SimulationStatus.CREATED
        state.profiles_generated = False
        state.config_generated = False
        manager._save_simulation_state(state)
    logger.info(
        f"[{simulation_id}] Run Contract {contract['contract_id']} applied: "
        f"{resolution['resolved']}/{len(contract['population'])} personas resolved, "
        f"confirmed={document['confirmed']}, preparation_invalidated={bool(invalidates)}"
    )
    errors = [f"{pid}: no graph entity" for pid in resolution["missing"]]
    errors += [f"{pid}: several graph entities" for pid in resolution["duplicates"]]
    errors += [f"{pid}: graph name does not match expected_name" for pid in resolution["name_mismatches"]]
    return {
        "applied": True,
        "confirmed": document["confirmed"],
        "preparation_invalidated": bool(invalidates),
        "errors": errors,
    }


def contract_round_count(run_dir: str) -> Optional[int]:
    """Exact round count of a contracted run (None for runs without a contract)"""
    path = os.path.join(run_dir, CONTRACT_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rounds = (data.get("contract") or {}).get("max_rounds")
    if not _is_int(rounds) or rounds <= 0:
        raise ValueError(f"run_contract.json in {run_dir} has no valid max_rounds")
    return rounds
