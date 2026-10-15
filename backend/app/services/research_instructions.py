"""
Private research instructions and run-scoped experiment stages

Separates four kinds of content that must never be confused:
    1. private agent instructions      (delivered privately into an agent's memory)
    2. experiment/orchestration rules   (stage triggers, conditions, completion)
    3. seed content                     (in-world posts that start the simulation)
    4. agent-generated public actions   (the only behavioural evidence)
"""

import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import Config
from ..utils.logger import get_logger
from ..utils.persona_ids import parse_identifier
from .assessment_spec import BRIEF_FILENAME_RE, is_run_contract_filename

logger = get_logger("mirofish.research_instructions")

PLAN_FILE = "experiment_plan.json"
PLAN_VERSION = 1

# Action / event origins (shared vocabulary with scripts/run_orchestration.py)
ORIGIN_AGENT_GENERATED = "agent_generated"
ORIGIN_SYSTEM_SEED = "system_seed"
ORIGIN_RESEARCHER_INSTRUCTION = "researcher_instruction"
ORIGIN_ORCHESTRATION = "orchestration"
ORIGINS = (ORIGIN_AGENT_GENERATED, ORIGIN_SYSTEM_SEED, ORIGIN_RESEARCHER_INSTRUCTION, ORIGIN_ORCHESTRATION)

# mentioned / not_mentioned look at everything an agent wrote or reacted to (what it
# has encountered); authored_* only at text the agent itself wrote or searched for.
CONDITION_TYPES = (
    "always", "not_mentioned", "mentioned", "authored_mentioned", "authored_not_mentioned",
    "stage_completed", "stage_not_completed",
)
COMPLETION_TYPES = ("on_delivery", "mentioned", "authored_mentioned", "action_types", "any_action", "rounds_elapsed")
# Report fields decided deterministically from the complete action log:
#   first_encounter_before_record: the persona wrote or reacted to content containing a
#       term before the given orchestration record was delivered (or without it)
#   no_encounter_before_record: the opposite
DERIVED_FIELD_TYPES = ("first_encounter_before_record", "no_encounter_before_record")
_RECORD_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,40}$")

# Words that mark research/experiment instructions (never in-world social content)
_INSTRUCTION_MARKERS_RE = re.compile(
    r"\b(?:opening[- ]prompt|cold[- ]start|prompt|instructions?|instructie\w*|opdracht\w*|"
    r"test[- ]task|task\s*:|stage\s*\d|fase\s*\d|step\s*\d\s*:|researchers?\b|"
    r"research (?:task|instruction|question|design)|assessment brief|funnel stage|hypothes[ie]s|"
    r"persona\b|you are (?:a|an|the)\b|act as\b|role[- ]?play|"
    r"scenario\s*:|do not reveal|simulat(?:e|ion|ed) (?:agent|persona|user))",
    re.IGNORECASE,
)
# A persona identifier at the start of a line followed by a separator ("C03 - ...", "C03: ...")
_LEADING_ID_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\*\*)?\[?([A-Za-z]{1,4}[-_]?\d{1,4})\]?(?:\*\*)?\s*[-–—:|]\s*(.+)$")
_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+(.+)|(.{3,80}):\s*)$")
_INSTRUCTION_HEADING_RE = re.compile(
    r"prompt|instruction|instructie|opdracht|task|stage|fase|scenario|script|vraag|question", re.IGNORECASE
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def overlap_ratio(text: str, reference: str, n: int = 5) -> float:
    """
    Share of the n-word shingles of text that occur verbatim in reference

    Args:
        text: Candidate (e.g. a seed post or an agent post)
        reference: Private text

    Returns:
        0.0 - 1.0
    """
    a, b = _words(text), _words(reference)
    if not a or not b:
        return 0.0
    if len(a) < n:
        joined_a, joined_b = " ".join(a), " ".join(b)
        return 1.0 if len(a) >= 3 and joined_a in joined_b else 0.0
    shingles = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    ref = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(shingles & ref) / len(shingles)


def contains_private_text(text: str, private_texts: Iterable[str], threshold: float = 0.5) -> bool:
    """Whether text substantially reproduces any private instruction"""
    for private in private_texts or []:
        if private and (overlap_ratio(text, private) >= threshold or overlap_ratio(private, text) >= 0.8):
            return True
    return False


def looks_like_instruction(text: str) -> bool:
    """Whether text reads like a research/orchestration instruction instead of in-world content"""
    if not text or not text.strip():
        return False
    if _INSTRUCTION_MARKERS_RE.search(text):
        return True
    match = _LEADING_ID_RE.match(text.strip().splitlines()[0])
    return bool(match and parse_identifier(match.group(1)))


def is_unsafe_public_text(text: str, private_texts: Iterable[str] = (), brief_text: str = "") -> Tuple[bool, str]:
    """
    Whether text must not be published as public seed content

    Args:
        text: Candidate public text
        private_texts: Private instruction texts
        brief_text: Assessment brief (verbatim brief copies are rejected)

    Returns:
        (unsafe, reason)
    """
    if contains_private_text(text, private_texts):
        return True, "reproduces a private research instruction"
    if looks_like_instruction(text):
        return True, "reads like a research / experiment instruction"
    if brief_text and len(_words(text)) >= 8 and overlap_ratio(text, brief_text) >= 0.6:
        return True, "copies the assessment brief verbatim"
    return False, ""


def strip_private_text(text: str, private_texts: Iterable[str]) -> str:
    """Remove sentences/lines of text that reproduce a private instruction"""
    private_texts = [p for p in private_texts or [] if p]
    if not text or not private_texts:
        return text
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    kept = [p for p in parts if not (p.strip() and contains_private_text(p, private_texts, threshold=0.6))]
    return " ".join(p.strip() for p in kept if p.strip())


def split_brief_documents(documents: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    (brief documents, context documents) by the brief filename convention

    Run Contract documents are control-plane input: they are in neither list.
    """
    brief, context = [], []
    for doc in documents or []:
        name = doc.get("original_filename") or ""
        if is_run_contract_filename(name):
            continue
        (brief if BRIEF_FILENAME_RE.search(name) else context).append(doc)
    return brief, context


def extract_private_instructions(brief_text: str, persona_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """
    Find persona-targeted research instructions in the assessment brief

    A line "<ID> - <text>" is an instruction when the text (or the heading it
    sits under) mentions a prompt/instruction/task/stage. Persona descriptions
    ("C01 - Dutch family with two kids") are not instructions.

    Args:
        brief_text: Assessment brief
        persona_ids: Persona ids requested by the brief (optional filter)

    Returns:
        [{persona_id, kind, label, text}]
    """
    wanted = set()
    for pid in persona_ids or []:
        parsed = parse_identifier(pid)
        if parsed:
            wanted.add((parsed[0].upper(), int(parsed[1])))
    lines = (brief_text or "").splitlines()
    results: List[Dict[str, Any]] = []
    heading_is_instruction = False
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _HEADING_RE.match(line)
        if heading and not _LEADING_ID_RE.match(line):
            heading_is_instruction = bool(_INSTRUCTION_HEADING_RE.search(heading.group(1) or heading.group(2) or ""))
            i += 1
            continue
        match = _LEADING_ID_RE.match(line)
        parsed = parse_identifier(match.group(1)) if match else None
        if not parsed or (parsed[0].upper() in ("H", "Q")):
            i += 1
            continue
        key = (parsed[0].upper(), int(parsed[1]))
        body = [match.group(2).strip()]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not _LEADING_ID_RE.match(lines[j]) and not _HEADING_RE.match(lines[j]):
            body.append(lines[j].strip())
            j += 1
        text = " ".join(body).strip()
        is_instruction = heading_is_instruction or bool(_INSTRUCTION_MARKERS_RE.search(text))
        if is_instruction and text and (not wanted or key in wanted):
            label_match = re.match(r"^([^:]{3,60}):\s*(.+)$", text)
            label = label_match.group(1).strip() if label_match else "Instruction"
            instruction = label_match.group(2).strip() if label_match else text
            kind = "opening" if re.search(r"opening|cold[- ]start|start|first|eerste", label + " " + text[:80], re.IGNORECASE) else "instruction"
            results.append({
                "persona_id": f"{parsed[0].upper()}{parsed[1]}",
                "kind": kind,
                "label": label,
                "text": instruction,
            })
        i = j
    return results


# ---------------------------------------------------------------- plan

def _default_stage(index: int) -> Dict[str, Any]:
    return {
        "id": f"stage_{index}",
        "name": f"Stage {index}",
        "order": index,
        "trigger_round": 1,
        "target": {"agent_ids": [], "persona_ids": [], "simulation_roles": [], "evidence_roles": []},
        "private_instruction": "",
        "context": "",
        "context_sections": [],
        "records": [],
        "condition": {"type": "always"},
        "completion": {"type": "on_delivery"},
        "source": "user",
    }


def plan_from_brief(brief_text: str, persona_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Suggested experiment plan: one private round-1 stage per brief instruction"""
    stages = []
    for n, ins in enumerate(extract_private_instructions(brief_text, persona_ids), start=1):
        stage = _default_stage(n)
        stage.update({
            "id": f"brief_{ins['persona_id'].lower()}_{n}",
            "name": f"{ins['label']} ({ins['persona_id']})",
            "trigger_round": 1,
            "private_instruction": ins["text"],
            "source": "brief",
        })
        stage["target"]["persona_ids"] = [ins["persona_id"]]
        stages.append(stage)
    return {"version": PLAN_VERSION, "stages": stages, "saved": False}


def _str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\n]", value)]
    return [str(v).strip() for v in value if str(v).strip()]


def validate_plan(plan: Any, role_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Validate an experiment plan

    Stage fields: id, name, order, trigger_round (>=1), target {agent_ids, persona_ids,
    simulation_roles, evidence_roles} (empty = every active agent), private_instruction,
    condition {type, terms?, stage_id?}, completion {type, terms?, action_types?, rounds?}

    Raises:
        ValueError: invalid plan
    """
    from .agent_selection import EVIDENCE_ROLE_IDS

    if not isinstance(plan, dict):
        raise ValueError("experiment plan must be an object")
    stages_in = plan.get("stages") or []
    if not isinstance(stages_in, list):
        raise ValueError("stages must be a list")
    role_ids = set(role_ids or [])
    stages: List[Dict[str, Any]] = []
    for n, raw in enumerate(stages_in, start=1):
        if not isinstance(raw, dict):
            raise ValueError("each stage must be an object")
        stage = _default_stage(n)
        stage_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw.get("id") or stage["id"]))[:60]
        if any(s["id"] == stage_id for s in stages):
            raise ValueError(f"Duplicate stage id: {stage_id}")
        try:
            trigger = int(raw.get("trigger_round", 1))
        except (TypeError, ValueError):
            raise ValueError(f"Stage {stage_id}: trigger_round must be an integer")
        if trigger < 1:
            raise ValueError(f"Stage {stage_id}: trigger_round must be >= 1 (round 0 is seed content only)")
        instruction = str(raw.get("private_instruction") or "").strip()
        if not instruction:
            raise ValueError(f"Stage {stage_id}: private_instruction is required")
        target_in = raw.get("target") or {}
        agent_ids = []
        for a in target_in.get("agent_ids") or []:
            try:
                agent_ids.append(int(a))
            except (TypeError, ValueError):
                raise ValueError(f"Stage {stage_id}: invalid agent id {a!r}")
        sim_roles = _str_list(target_in.get("simulation_roles"))
        if role_ids and any(r not in role_ids for r in sim_roles):
            raise ValueError(f"Stage {stage_id}: unknown simulation role in {sim_roles}")
        ev_roles = _str_list(target_in.get("evidence_roles"))
        if any(r not in EVIDENCE_ROLE_IDS for r in ev_roles):
            raise ValueError(f"Stage {stage_id}: unknown evidence role in {ev_roles}")
        condition = dict(raw.get("condition") or {"type": "always"})
        if condition.get("type", "always") not in CONDITION_TYPES:
            raise ValueError(f"Stage {stage_id}: unknown condition type {condition.get('type')}")
        condition["type"] = condition.get("type", "always")
        condition["terms"] = _str_list(condition.get("terms"))
        if condition["type"] in ("mentioned", "not_mentioned", "authored_mentioned", "authored_not_mentioned") and not condition["terms"]:
            raise ValueError(f"Stage {stage_id}: condition '{condition['type']}' needs terms")
        if condition["type"] in ("stage_completed", "stage_not_completed") and not condition.get("stage_id"):
            raise ValueError(f"Stage {stage_id}: condition '{condition['type']}' needs stage_id")
        completion = dict(raw.get("completion") or {"type": "on_delivery"})
        completion["type"] = completion.get("type", "on_delivery")
        if completion["type"] not in COMPLETION_TYPES:
            raise ValueError(f"Stage {stage_id}: unknown completion type {completion['type']}")
        completion["terms"] = _str_list(completion.get("terms"))
        completion["action_types"] = [a.upper() for a in _str_list(completion.get("action_types"))]
        if completion["type"] in ("mentioned", "authored_mentioned") and not completion["terms"]:
            raise ValueError(f"Stage {stage_id}: completion '{completion['type']}' needs terms")
        records = [str(r).strip().upper() for r in raw.get("records") or [] if str(r).strip()]
        if any(not _RECORD_LABEL_RE.match(r) for r in records):
            raise ValueError(f"Stage {stage_id}: records must be UPPER_SNAKE labels, got {records}")
        context_sections = _str_list(raw.get("context_sections"))
        if completion["type"] == "action_types" and not completion["action_types"]:
            raise ValueError(f"Stage {stage_id}: completion 'action_types' needs action_types")
        if completion["type"] == "rounds_elapsed":
            try:
                completion["rounds"] = max(1, int(completion.get("rounds", 1)))
            except (TypeError, ValueError):
                raise ValueError(f"Stage {stage_id}: completion rounds must be an integer")
        stage.update({
            "id": stage_id,
            "name": str(raw.get("name") or stage_id)[:120],
            "order": int(raw.get("order", n)) if str(raw.get("order", n)).lstrip("-").isdigit() else n,
            "trigger_round": trigger,
            "target": {
                "agent_ids": agent_ids,
                "persona_ids": [p.upper() for p in _str_list(target_in.get("persona_ids"))],
                "simulation_roles": sim_roles,
                "evidence_roles": ev_roles,
            },
            "private_instruction": instruction,
            # Factual product context delivered with the instruction. It is public
            # information (e.g. product documentation), so it is not part of the
            # leak guard: agents may legitimately repeat these facts.
            "context": str(raw.get("context") or "").strip(),
            "context_sections": context_sections,
            # Orchestration facts recorded for the targeted agents on delivery
            # (e.g. EXPOSED). Facts about the experiment, never behaviour.
            "records": records,
            "condition": condition,
            "completion": completion,
            "source": raw.get("source") if raw.get("source") in ("user", "brief") else "user",
        })
        stages.append(stage)
    # Terms seed posts may not contain (e.g. the product under test), so system
    # content never causes "organic" discovery
    seed_blocked_terms = _str_list(plan.get("seed_blocked_terms"))
    derived_fields = {}
    for name, rule in (plan.get("derived_fields") or {}).items():
        label = re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")
        if not isinstance(rule, dict) or rule.get("type") not in DERIVED_FIELD_TYPES:
            raise ValueError(f"derived field {name}: type must be one of {DERIVED_FIELD_TYPES}")
        terms = _str_list(rule.get("terms"))
        if not terms:
            raise ValueError(f"derived field {name}: terms are required")
        derived_fields[label] = {"type": rule["type"], "terms": terms, "record": str(rule.get("record") or "").upper(),
                                 # Only decide the field for these evidence roles (empty = everyone)
                                 "evidence_roles": _str_list(rule.get("evidence_roles"))}
    known = {s["id"] for s in stages}
    for s in stages:
        ref = s["condition"].get("stage_id")
        if ref and ref not in known:
            raise ValueError(f"Stage {s['id']}: condition refers to unknown stage {ref}")
    stages.sort(key=lambda s: (s["order"], s["trigger_round"]))
    return {"version": PLAN_VERSION, "stages": stages, "seed_blocked_terms": seed_blocked_terms,
            "derived_fields": derived_fields}


def resolve_plan(plan: Dict[str, Any], roster: Dict[str, Any], extra_private_texts: Iterable[str] = ()) -> Dict[str, Any]:
    """
    Resolve stage targets to the active agent ids of a prepared roster

    Args:
        plan: Validated plan
        roster: agent_roster.json
        extra_private_texts: Further private texts the leak guard must know
            (e.g. brief instructions the user removed from the plan)

    Returns:
        Plan for simulation_config.json: stages with resolved_agent_ids, private_texts
    """
    agents = roster.get("active_agents", [])
    by_persona: Dict[Tuple[str, int], List[int]] = {}
    for a in agents:
        parsed = parse_identifier(a.get("name"))
        if parsed:
            by_persona.setdefault((parsed[0].upper(), int(parsed[1])), []).append(int(a["agent_id"]))
    resolved_stages = []
    for stage in plan.get("stages", []):
        target = stage.get("target") or {}
        ids = set()
        has_filter = any(target.get(k) for k in ("agent_ids", "persona_ids", "simulation_roles", "evidence_roles"))
        for a in agents:
            aid = int(a["agent_id"])
            if not has_filter:
                ids.add(aid)
                continue
            if aid in (target.get("agent_ids") or []):
                ids.add(aid)
            if a.get("simulation_role") in (target.get("simulation_roles") or []):
                ids.add(aid)
            if a.get("evidence_role") in (target.get("evidence_roles") or []):
                ids.add(aid)
        for pid in target.get("persona_ids") or []:
            parsed = parse_identifier(pid)
            if parsed:
                ids.update(by_persona.get((parsed[0].upper(), int(parsed[1])), []))
        resolved_stages.append({**stage, "resolved_agent_ids": sorted(ids)})
    private_texts = [s["private_instruction"] for s in resolved_stages if s.get("private_instruction")]
    for text in extra_private_texts or []:
        if text and text not in private_texts:
            private_texts.append(text)
    return {
        "version": PLAN_VERSION,
        "stages": resolved_stages,
        "seed_blocked_terms": list(plan.get("seed_blocked_terms") or []),
        "derived_fields": dict(plan.get("derived_fields") or {}),
        "private_texts": private_texts,
        "unresolved_stages": [s["id"] for s in resolved_stages if not s["resolved_agent_ids"]],
    }


class ExperimentPlanStore:
    """experiment_plan.json in the simulation (configuration) directory"""

    _lock = threading.RLock()

    @staticmethod
    def path(simulation_id: str) -> str:
        return os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id, PLAN_FILE)

    @classmethod
    def load(cls, simulation_id: str) -> Optional[Dict[str, Any]]:
        path = cls.path(simulation_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Unreadable experiment plan {path}: {e}")
            return None

    @classmethod
    def save(cls, simulation_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        data = {**plan, "saved": True, "updated_at": datetime.now().isoformat()}
        path = cls.path(simulation_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with cls._lock:
            tmp = f"{path}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        return data


def load_project_brief(project_id: Optional[str]) -> Dict[str, Any]:
    """
    The project's assessment brief, separated from its context documents

    Returns:
        {brief_text, context_text, persona_ids, instructions}
    """
    from ..models.project import ProjectManager
    from .assessment_spec import compose_brief, extract_spec_regex

    project = ProjectManager.get_project(project_id) if project_id else None
    if not project:
        return {"brief_text": "", "context_text": "", "persona_ids": [], "instructions": []}
    documents = ProjectManager.get_source_documents(project_id)
    brief_text, _ = compose_brief(project.simulation_requirement, project.additional_context, documents)
    _brief_docs, context_docs = split_brief_documents(documents)
    if documents:
        context_text = "\n\n".join(
            f"=== {d.get('original_filename')} ===\n{d.get('text') or ''}" for d in context_docs if (d.get("text") or "").strip()
        )
    else:
        context_text = ProjectManager.get_extracted_text(project_id) or ""
    spec = extract_spec_regex(brief_text)
    instructions = extract_private_instructions(brief_text)
    return {
        "brief_text": brief_text,
        "context_text": context_text,
        "persona_ids": spec.persona_ids,
        "instructions": instructions,
    }


def _run_source_documents(run_dir: str) -> List[Dict[str, Any]]:
    """Source documents of a run from its own snapshot (sources/manifest.json)"""
    manifest_path = os.path.join(run_dir, "sources", "manifest.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    documents = []
    for doc in manifest.get("documents", []):
        if is_run_contract_filename(doc.get("original_filename")):
            continue  # control document, never a source
        text_path = os.path.join(run_dir, "sources", doc.get("text_file") or "")
        text = ""
        if doc.get("text_file") and os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
        documents.append({**doc, "text": text})
    return documents


def snapshot_run_plan(run_dir: str, simulation_id: str) -> Dict[str, Any]:
    """
    Resolve the experiment plan for one run and write it into the run directory

    Uses the run's own snapshots (agent_roster.json, assessment/brief.md,
    simulation_config.json), so later edits never change an existing run.
    Without a roster (simulations prepared before agent selection) every
    configured agent is treated as active.

    Args:
        run_dir: Run directory (already holding its snapshot files)
        simulation_id: Simulation ID

    Returns:
        Summary {stages, private_texts, unresolved_stages, source}
    """
    from .assessment_spec import extract_spec_regex

    def _read(name: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    roster = _read("agent_roster.json")
    config = _read("simulation_config.json") or {}
    if roster is None:
        roster = {"active_agents": [
            {"agent_id": a.get("agent_id"), "name": a.get("entity_name"), "entity_uuid": a.get("entity_uuid"),
             "simulation_role": a.get("simulation_role") or "", "evidence_role": a.get("evidence_role") or ""}
            for a in config.get("agent_configs", []) if a.get("agent_id") is not None
        ]}
    brief_text = ""
    brief_path = os.path.join(run_dir, "assessment", "brief.md")
    if os.path.exists(brief_path):
        with open(brief_path, "r", encoding="utf-8") as f:
            brief_text = f.read()
    saved = _read(PLAN_FILE)
    contract = _read("run_contract.json")
    if contract is not None:
        # Contracted run: the Run Contract owns the stages. Brief prose never becomes
        # a stage or a private text, and a plan of another contract is refused.
        if not saved or not saved.get("saved") or saved.get("contract_hash") != contract.get("contract_hash"):
            raise ValueError("experiment_plan.json of this run does not belong to its Run Contract")
        brief_instructions = []
        plan, source = validate_plan(saved), "saved"
    elif saved is not None and saved.get("saved"):
        brief_instructions = [i["text"] for i in extract_private_instructions(brief_text)]
        plan, source = validate_plan(saved), "saved"
    else:
        brief_instructions = [i["text"] for i in extract_private_instructions(brief_text)]
        plan, source = validate_plan(plan_from_brief(brief_text, extract_spec_regex(brief_text).persona_ids)), "brief"
    wanted = [sid for s in plan["stages"] for sid in s.get("context_sections") or []]
    if wanted:
        from .source_requirements import REQUIREMENTS_FILE, section_texts

        texts = section_texts(_read(REQUIREMENTS_FILE), _run_source_documents(run_dir), wanted)
        for stage in plan["stages"]:
            parts = [stage.get("context") or ""] + [texts[sid] for sid in stage.get("context_sections") or []]
            stage["context"] = "\n\n".join(p for p in parts if p)
    resolved = resolve_plan(plan, roster, list(config.get("private_texts") or []) + brief_instructions)
    if source == "saved" and resolved["unresolved_stages"]:
        # A saved stage that reaches nobody would silently never fire (e.g. a persona
        # id that is not part of any active agent's name): refuse the run instead
        raise ValueError(
            f"Stages target no active agent: {resolved['unresolved_stages']}. "
            "Check the persona ids / roles of these stages against the agent selection."
        )
    resolved["source"] = source
    if contract is not None:
        resolved["contract_id"] = contract.get("contract_id")
        resolved["contract_hash"] = contract.get("contract_hash")
    resolved["simulation_id"] = simulation_id
    resolved["resolved_at"] = datetime.now().isoformat()
    with open(os.path.join(run_dir, PLAN_FILE), "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    return {
        "stages": len(resolved["stages"]),
        "private_texts": len(resolved["private_texts"]),
        "unresolved_stages": resolved["unresolved_stages"],
        "source": source,
    }
