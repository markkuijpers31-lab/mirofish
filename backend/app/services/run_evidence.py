"""
Run evidence: direct, graph-independent access to everything one simulation execution produced
"""

import csv
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_run_logger
from ..utils.persona_ids import PersonaIndex, build_persona_index, parse_identifier
from .assessment_spec import BRIEF_FILENAME_RE, compose_brief, extract_spec_regex, is_run_contract_filename
from .agent_selection import EVIDENCE_TARGET_CONSUMER, ROSTER_FILE
from .research_instructions import (
    ORIGIN_AGENT_GENERATED, ORIGIN_ORCHESTRATION, ORIGIN_RESEARCHER_INSTRUCTION, ORIGIN_SYSTEM_SEED, ORIGINS,
)
from .run_store import LEGACY_RUN_ID, RunRecord, RunStore

PLATFORMS = ("twitter", "reddit")

# Default scope of conversion / adoption metrics
CONSUMER_METRIC_SCOPE = f"evidence_role = {EVIDENCE_TARGET_CONSUMER} AND origin = {ORIGIN_AGENT_GENERATED}"

# Events written by the runner for inspection (never counted as actions)
ORCHESTRATION_EVENT_TYPES = ("private_instruction", "stage_completed", "seed_post_rejected")
PLATFORM_REF_PREFIX = {"twitter": "TW", "reddit": "RD"}

# action_args keys holding text the actor wrote themselves
_AUTHORED_KEYS = ("content", "query")
# action_args keys describing what the actor reacted to
_TARGET_KEYS = (
    ("post_content", "post_author_name"),
    ("comment_content", "comment_author_name"),
    ("original_content", "original_author_name"),
    ("quote_content", None),
    ("target_user_name", None),
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name or "document")[:80] or "document"


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_agents(run_dir: str) -> List[Dict[str, Any]]:
    """
    Load the simulation agents (personas) of a run from its config and profile files

    Args:
        run_dir: Run directory (or legacy simulation directory)

    Returns:
        List of {agent_id, name, entity_type, username, bio, persona, ...}
    """
    config = _read_json(os.path.join(run_dir, "simulation_config.json"), {}) or {}
    agents: Dict[int, Dict[str, Any]] = {}

    for ac in config.get("agent_configs", []) or []:
        try:
            agent_id = int(ac.get("agent_id"))
        except (TypeError, ValueError):
            continue
        agents[agent_id] = {
            "agent_id": agent_id,
            "name": ac.get("entity_name") or "",
            "entity_type": ac.get("entity_type"),
            "entity_uuid": ac.get("entity_uuid"),
            "stance": ac.get("stance"),
            "activity_level": ac.get("activity_level"),
        }
        if ac.get("simulation_role"):
            agents[agent_id]["simulation_role"] = ac["simulation_role"]
        if ac.get("evidence_role"):
            agents[agent_id]["evidence_role"] = ac["evidence_role"]

    reddit_profiles = _read_json(os.path.join(run_dir, "reddit_profiles.json"), []) or []
    for p in reddit_profiles:
        try:
            agent_id = int(p.get("user_id"))
        except (TypeError, ValueError):
            continue
        agent = agents.setdefault(agent_id, {"agent_id": agent_id, "name": p.get("name") or ""})
        for key in ("username", "bio", "persona", "age", "gender", "mbti", "country", "profession",
                    "interested_topics", "persona_id", "canonical_id"):
            if p.get(key) not in (None, "") and key not in agent:
                agent[key] = p.get(key)
        if not agent.get("name"):
            agent["name"] = p.get("name") or ""

    twitter_csv = os.path.join(run_dir, "twitter_profiles.csv")
    if os.path.exists(twitter_csv):
        try:
            with open(twitter_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        agent_id = int(row.get("user_id"))
                    except (TypeError, ValueError):
                        continue
                    agent = agents.setdefault(agent_id, {"agent_id": agent_id, "name": row.get("name") or ""})
                    if not agent.get("name"):
                        agent["name"] = row.get("name") or ""
                    agent.setdefault("username", row.get("username"))
                    if row.get("user_char") and "persona" not in agent:
                        agent["persona"] = row.get("user_char")
        except Exception:
            pass

    # The frozen roster (agent selection) is the authority on roles
    roster = _read_json(os.path.join(run_dir, ROSTER_FILE), None)
    for entry in (roster or {}).get("active_agents", []) or []:
        try:
            agent_id = int(entry.get("agent_id"))
        except (TypeError, ValueError):
            continue
        agent = agents.setdefault(agent_id, {"agent_id": agent_id, "name": entry.get("name") or ""})
        for key in ("simulation_role", "evidence_role", "entity_uuid"):
            if entry.get(key):
                agent[key] = entry[key]
        agent["simulation_role_label"] = entry.get("simulation_role_label")
        agent["evidence_role_label"] = entry.get("evidence_role_label")

    return [agents[k] for k in sorted(agents)]


def snapshot_run_inputs(run_dir: str, project_id: Optional[str]) -> Dict[str, Any]:
    """
    Copy source documents, the assessment brief and the persona index into a run

    Args:
        run_dir: Target run directory
        project_id: Project the simulation belongs to

    Returns:
        Summary {sources, sources_failed, brief_documents, personas}
    """
    from ..models.project import ProjectManager

    summary: Dict[str, Any] = {"sources": 0, "sources_failed": 0, "brief_documents": [], "personas": 0}
    project = ProjectManager.get_project(project_id) if project_id else None
    documents = ProjectManager.get_source_documents(project_id) if project else []

    sources_dir = os.path.join(run_dir, "sources")
    originals_dir = os.path.join(sources_dir, "originals")
    os.makedirs(sources_dir, exist_ok=True)
    manifest = []
    for i, doc in enumerate(documents, start=1):
        name = doc.get("original_filename") or f"document_{i}"
        text_file = f"{i:02d}_{_safe_name(os.path.splitext(name)[0])}.txt"
        with open(os.path.join(sources_dir, text_file), "w", encoding="utf-8") as f:
            f.write(doc.get("text") or "")
        original_file = None
        if doc.get("original_path"):
            os.makedirs(originals_dir, exist_ok=True)
            original_file = f"{i:02d}_{_safe_name(name)}"
            shutil.copy2(doc["original_path"], os.path.join(originals_dir, original_file))
        if doc.get("extraction_status") not in ("ok",):
            summary["sources_failed"] += 1
        manifest.append({
            "index": i,
            "original_filename": name,
            "extension": doc.get("extension"),
            "size": doc.get("size"),
            "text_length": doc.get("text_length", len(doc.get("text") or "")),
            "extraction_status": doc.get("extraction_status"),
            "error": doc.get("error"),
            "origin": doc.get("origin"),
            "text_file": text_file,
            "original_file": original_file,
        })
    with open(os.path.join(sources_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"project_id": project_id, "documents": manifest}, f, ensure_ascii=False, indent=2)
    summary["sources"] = len(manifest)

    brief, brief_docs = compose_brief(
        project.simulation_requirement if project else "",
        project.additional_context if project else "",
        documents,
    )
    assessment_dir = os.path.join(run_dir, "assessment")
    os.makedirs(assessment_dir, exist_ok=True)
    with open(os.path.join(assessment_dir, "brief.md"), "w", encoding="utf-8") as f:
        f.write(brief)
    summary["brief_documents"] = brief_docs

    agents = load_agents(run_dir)
    families = {parse_identifier(a.get("name"))[0].upper() for a in agents if parse_identifier(a.get("name"))}
    spec = extract_spec_regex(brief, families)
    index = build_persona_index(agents, spec.persona_ids)
    with open(os.path.join(run_dir, "personas.json"), "w", encoding="utf-8") as f:
        json.dump({"requested_ids": spec.persona_ids, "personas": index.to_list()}, f, ensure_ascii=False, indent=2)
    summary["personas"] = len(index.personas)
    return summary


def _describe_action(args: Dict[str, Any]) -> Tuple[str, str]:
    authored = ""
    for key in _AUTHORED_KEYS:
        if args.get(key):
            authored = str(args[key])
            break
    target_parts = []
    for content_key, author_key in _TARGET_KEYS:
        if args.get(content_key):
            author = args.get(author_key) if author_key else None
            prefix = f"{author}: " if author else ""
            target_parts.append(f"{prefix}{args[content_key]}")
    return authored, " | ".join(target_parts)


class RunEvidence:
    """
    All evidence of one execution, read directly from its run directory

    Raw action logs are the primary source of truth for reports; nothing here
    depends on the graph store.
    """

    def __init__(self, simulation_id: str, run_id: str, run_dir: str, record: Optional[RunRecord]):
        self.simulation_id = simulation_id
        self.run_id = run_id
        self.run_dir = run_dir
        self.record = record
        self.log = get_run_logger("mirofish.evidence", simulation_id, run_id, "evidence")

        self.config: Dict[str, Any] = {}
        self.agents: List[Dict[str, Any]] = []
        self.persona_index: PersonaIndex = PersonaIndex()
        self.requested_ids: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.events: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PLATFORMS}
        self.integrity: Dict[str, Any] = {"warnings": [], "platforms": {}}
        self.sources: List[Dict[str, Any]] = []
        # Run Contract documents: snapshotted control input, never report evidence
        self.control_documents: List[str] = []
        self.run_contract: Optional[Dict[str, Any]] = None
        self.sources_origin = "run_snapshot"
        self.brief_text = ""
        self.brief_documents: List[str] = []
        self.used_sources: set = set()
        self.roster: Optional[Dict[str, Any]] = None
        self.roles_source = "legacy_default"
        self.roles_note = ""
        self.agent_roles: Dict[int, Dict[str, Any]] = {}
        self.orchestration_events: List[Dict[str, Any]] = []
        self.source_requirements: Optional[Dict[str, Any]] = None
        self.plan_record_labels: List[str] = []
        self.derived_fields: Dict[str, Dict[str, Any]] = {}

    # ---------- loading ----------

    @classmethod
    def load(cls, simulation_id: str, run_id: Optional[str] = None) -> "RunEvidence":
        """
        Load the evidence of a run (default: the simulation's current run)

        Raises:
            ValueError: if the simulation has no run
        """
        resolved = RunStore.resolve_run_id(simulation_id, run_id)
        if not resolved:
            raise ValueError(f"Simulation {simulation_id} has no run to report on")
        record = RunStore.get_run(simulation_id, resolved)
        if record is None:
            raise ValueError(f"Run {resolved} not found for simulation {simulation_id}")
        evidence = cls(simulation_id, resolved, RunStore.run_dir(simulation_id, resolved), record)
        evidence._load_all()
        return evidence

    def _load_all(self):
        self.config = _read_json(os.path.join(self.run_dir, "simulation_config.json"), {}) or {}
        self._load_sources_and_brief()
        self.source_requirements = _read_json(os.path.join(self.run_dir, "source_requirements.json"), None)
        plan = _read_json(os.path.join(self.run_dir, "experiment_plan.json"), {}) or {}
        self.plan_record_labels = sorted({r for st in plan.get("stages", []) for r in st.get("records") or []})
        self.derived_fields: Dict[str, Dict[str, Any]] = plan.get("derived_fields") or {}
        self.agents = load_agents(self.run_dir)
        self._load_roles()
        personas_file = _read_json(os.path.join(self.run_dir, "personas.json"), {}) or {}
        self.set_requested_ids(personas_file.get("requested_ids") or [])
        self._load_actions()
        counts = self.platform_counts()
        self.log.info(
            f"Action evidence loaded: twitter={counts.get('twitter', 0)} reddit={counts.get('reddit', 0)} "
            f"personas={len(self.persona_index.personas)} warnings={len(self.integrity['warnings'])}"
        )
        self.log.info(
            f"Source files loaded: {len(self.sources)} ({self.sources_origin}), brief_chars={len(self.brief_text)}"
        )

    def _load_sources_and_brief(self):
        manifest = _read_json(os.path.join(self.run_dir, "sources", "manifest.json"))
        if manifest and self.run_id != LEGACY_RUN_ID:
            for doc in manifest.get("documents", []):
                text_path = os.path.join(self.run_dir, "sources", doc.get("text_file") or "")
                text = ""
                if doc.get("text_file") and os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.read()
                self.sources.append({**doc, "text": text})
        else:
            # Legacy run or failed snapshot: read the project's documents (read-only)
            self.sources_origin = "project_live"
            project_id = (self.record.project_id if self.record else None) or self._project_id_from_state()
            if project_id:
                from ..models.project import ProjectManager

                for i, doc in enumerate(ProjectManager.get_source_documents(project_id), start=1):
                    self.sources.append({
                        "index": i,
                        "original_filename": doc.get("original_filename"),
                        "extension": doc.get("extension"),
                        "size": doc.get("size"),
                        "text_length": doc.get("text_length"),
                        "extraction_status": doc.get("extraction_status"),
                        "error": doc.get("error"),
                        "origin": doc.get("origin"),
                        "text": doc.get("text") or "",
                    })

        self.control_documents = [
            d.get("original_filename") for d in self.sources if is_run_contract_filename(d.get("original_filename"))
        ]
        self.sources = [d for d in self.sources if not is_run_contract_filename(d.get("original_filename"))]
        self.run_contract = _read_json(os.path.join(self.run_dir, "run_contract.json"), None)
        self.brief_documents = [
            d.get("original_filename")
            for d in self.sources
            if BRIEF_FILENAME_RE.search(d.get("original_filename") or "") and (d.get("text") or "").strip()
        ]
        brief_path = os.path.join(self.run_dir, "assessment", "brief.md")
        if os.path.exists(brief_path) and self.run_id != LEGACY_RUN_ID:
            with open(brief_path, "r", encoding="utf-8") as f:
                self.brief_text = f.read()
        else:
            project_id = (self.record.project_id if self.record else None) or self._project_id_from_state()
            project = None
            if project_id:
                from ..models.project import ProjectManager

                project = ProjectManager.get_project(project_id)
            self.brief_text, _ = compose_brief(
                project.simulation_requirement if project else self.config.get("simulation_requirement"),
                project.additional_context if project else None,
                self.sources,
            )

    def _load_roles(self):
        """Simulation / evidence role per agent (roster > config > legacy default)"""
        self.roster = _read_json(os.path.join(self.run_dir, ROSTER_FILE), None)
        if self.roster:
            self.roles_source = "roster"
        elif any(a.get("evidence_role") for a in self.agents):
            self.roles_source = "config"
        else:
            self.roles_source = "legacy_default"
            self.roles_note = (
                "This run was prepared before explicit agent selection: no simulation/evidence roles are "
                "recorded, so every agent is treated as a target consumer."
            )
        for a in self.agents:
            self.agent_roles[int(a["agent_id"])] = {
                "simulation_role": a.get("simulation_role") or ("unassigned" if self.roles_source != "legacy_default" else "legacy"),
                "evidence_role": a.get("evidence_role") or EVIDENCE_TARGET_CONSUMER,
            }

    def roles_for_agent(self, agent_id: Any) -> Dict[str, Any]:
        try:
            agent_id = int(agent_id)
        except (TypeError, ValueError):
            pass
        return self.agent_roles.get(agent_id) or {"simulation_role": "unknown", "evidence_role": "unknown"}

    def persona_evidence_role(self, canonical_id: str) -> str:
        """Evidence role of a persona (target consumer if any of its agents is one)"""
        persona = self.persona_index.personas.get(canonical_id)
        roles = [self.roles_for_agent(aid)["evidence_role"] for aid in (persona or {}).get("agent_ids", [])]
        if EVIDENCE_TARGET_CONSUMER in roles:
            return EVIDENCE_TARGET_CONSUMER
        return roles[0] if roles else "unknown"

    def persona_simulation_role(self, canonical_id: str) -> str:
        persona = self.persona_index.personas.get(canonical_id)
        roles = [self.roles_for_agent(aid)["simulation_role"] for aid in (persona or {}).get("agent_ids", [])]
        return roles[0] if roles else "unknown"

    def _project_id_from_state(self) -> Optional[str]:
        state = _read_json(os.path.join(RunStore.sim_dir(self.simulation_id), "state.json"), {}) or {}
        return state.get("project_id") or self.config.get("project_id")

    def set_requested_ids(self, requested_ids: List[str]):
        """(Re)build the persona index with the identifiers the brief asks for"""
        self.requested_ids = list(requested_ids or [])
        self.persona_index = build_persona_index(self.agents, self.requested_ids)
        for action in self.actions:
            action["canonical_id"], action["display_name"] = self.persona_index.resolve(
                action.get("agent_id"), action.get("agent_name")
            )

    def _platform_log_paths(self) -> List[Tuple[str, Optional[str]]]:
        paths = [(os.path.join(self.run_dir, p, "actions.jsonl"), p) for p in PLATFORMS]
        if self.run_id == LEGACY_RUN_ID:
            single = os.path.join(self.run_dir, "actions.jsonl")
            if os.path.exists(single) and not any(os.path.exists(p) for p, _ in paths):
                paths.append((single, None))
        return paths

    def _load_actions(self):
        counters = {p: 0 for p in PLATFORMS}
        for path, platform in self._platform_log_paths():
            if not os.path.exists(path):
                continue
            label = platform or "combined"
            segments: List[Dict[str, Any]] = [{"actions": [], "events": [], "start": None}]
            foreign_run_ids = set()
            malformed = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    line_run_id = data.get("run_id")
                    if line_run_id and self.run_id != LEGACY_RUN_ID and line_run_id != self.run_id:
                        foreign_run_ids.add(line_run_id)
                        continue
                    if "event_type" in data:
                        if data["event_type"] == "simulation_start":
                            if segments[-1]["actions"] or segments[-1]["start"] is not None:
                                segments.append({"actions": [], "events": [], "start": data})
                            else:
                                segments[-1]["start"] = data
                        segments[-1]["events"].append(data)
                        continue
                    if "agent_id" not in data:
                        continue
                    segments[-1]["actions"].append(data)

            info = {
                "file": os.path.relpath(path, self.run_dir),
                "executions_in_file": len(segments),
                "malformed_lines": malformed,
                "foreign_run_ids": sorted(foreign_run_ids),
                "completed": any(e.get("event_type") == "simulation_end" for e in segments[-1]["events"]),
            }
            if len(segments) > 1:
                dropped = sum(len(s["actions"]) for s in segments[:-1])
                info["dropped_actions_from_earlier_executions"] = dropped
                self.integrity["warnings"].append(
                    f"{info['file']} contains {len(segments)} executions (restarts appended to one file). "
                    f"Only the last execution is used; {dropped} actions from earlier executions were excluded."
                )
            if foreign_run_ids:
                self.integrity["warnings"].append(
                    f"{info['file']} contains lines from other runs {sorted(foreign_run_ids)}; they were excluded."
                )
            if malformed:
                self.integrity["warnings"].append(f"{info['file']} has {malformed} malformed lines (skipped).")
            self.integrity["platforms"][label] = info

            for event in segments[-1]["events"]:
                if event.get("event_type") in ORCHESTRATION_EVENT_TYPES:
                    self.orchestration_events.append({**event, "platform": platform or event.get("platform")})

            # Older logs carry no origin: round-0 posts were system seed posts, and the
            # runner re-logged them at round 1 (trace re-read) - both are seed content.
            seeds = {
                (d.get("agent_id"), (d.get("action_args") or {}).get("content"))
                for d in segments[-1]["actions"]
                if d.get("round", 0) == 0 and d.get("action_type") == "CREATE_POST"
            }
            seed_repeats_left = set(seeds)
            for data in segments[-1]["actions"]:
                plat = platform or data.get("platform") or "unknown"
                counters[plat] = counters.get(plat, 0) + 1
                ref = f"{PLATFORM_REF_PREFIX.get(plat, plat[:2].upper())}-{counters[plat]:05d}"
                args = data.get("action_args") or {}
                authored, target = _describe_action(args)
                canonical, display = self.persona_index.resolve(data.get("agent_id"), data.get("agent_name"))
                origin, origin_flag = data.get("origin"), data.get("origin_flag")
                if origin not in ORIGINS:
                    origin, origin_flag = ORIGIN_AGENT_GENERATED, None
                    key = (data.get("agent_id"), args.get("content"))
                    if data.get("round", 0) == 0 and data.get("action_type") == "CREATE_POST":
                        origin, origin_flag = ORIGIN_SYSTEM_SEED, "legacy_round0_seed"
                    elif data.get("action_type") == "CREATE_POST" and key in seed_repeats_left:
                        seed_repeats_left.discard(key)
                        origin, origin_flag = ORIGIN_SYSTEM_SEED, "legacy_duplicate_seed"
                roles = self.roles_for_agent(data.get("agent_id"))
                self.actions.append({
                    "ref": ref,
                    "platform": plat,
                    "round": data.get("round", 0),
                    "timestamp": data.get("timestamp", ""),
                    "agent_id": data.get("agent_id"),
                    "agent_name": data.get("agent_name", ""),
                    "canonical_id": canonical,
                    "display_name": display,
                    "action_type": data.get("action_type", ""),
                    "content": authored,
                    "target": target,
                    "success": data.get("success", True),
                    "args": args,
                    "origin": origin,
                    "origin_flag": origin_flag,
                    "simulation_role": roles["simulation_role"],
                    "evidence_role": roles["evidence_role"],
                })
            if platform:
                self.events[platform] = segments[-1]["events"]

    # ---------- queries ----------

    @staticmethod
    def is_behavioural(a: Dict[str, Any]) -> bool:
        """Only actions the agent itself generated are behavioural evidence"""
        return a.get("origin", ORIGIN_AGENT_GENERATED) == ORIGIN_AGENT_GENERATED

    @classmethod
    def is_consumer_evidence(cls, a: Dict[str, Any]) -> bool:
        """Counts in conversion / adoption metrics (CONSUMER_METRIC_SCOPE)"""
        return cls.is_behavioural(a) and a.get("evidence_role") == EVIDENCE_TARGET_CONSUMER

    def behavioural_actions(self) -> List[Dict[str, Any]]:
        return [a for a in self.actions if self.is_behavioural(a)]

    def consumer_actions(self) -> List[Dict[str, Any]]:
        return [a for a in self.actions if self.is_consumer_evidence(a)]

    def consumer_metrics(self) -> Dict[str, Any]:
        """
        Conversion / adoption counts, restricted to target consumers' own actions

        Seed content, researcher instructions and the actions of existing users,
        advocates or platform accounts are reported as excluded, never counted.
        """
        consumer = self.consumer_actions()
        by_type: Dict[str, int] = {}
        by_persona: Dict[str, int] = {}
        for a in consumer:
            by_type[a["action_type"]] = by_type.get(a["action_type"], 0) + 1
            by_persona[a["canonical_id"]] = by_persona.get(a["canonical_id"], 0) + 1
        target_personas = sorted(
            cid for cid in self.persona_index.personas if self.persona_evidence_role(cid) == EVIDENCE_TARGET_CONSUMER
        )
        excluded_origin: Dict[str, int] = {}
        excluded_role: Dict[str, int] = {}
        for a in self.actions:
            if not self.is_behavioural(a):
                excluded_origin[a["origin"]] = excluded_origin.get(a["origin"], 0) + 1
            elif a.get("evidence_role") != EVIDENCE_TARGET_CONSUMER:
                excluded_role[a.get("evidence_role") or "unknown"] = excluded_role.get(a.get("evidence_role") or "unknown", 0) + 1
        return {
            "scope": CONSUMER_METRIC_SCOPE,
            "roles_source": self.roles_source,
            "actions": len(consumer),
            "by_action_type": by_type,
            "by_persona": by_persona,
            "target_consumer_personas": target_personas,
            "target_consumers_without_actions": [cid for cid in target_personas if cid not in by_persona],
            "excluded_by_origin": excluded_origin,
            "excluded_by_evidence_role": excluded_role,
        }

    def population(self) -> List[Dict[str, Any]]:
        """Personas with their roles and action counts by origin"""
        rows = []
        for p in self.persona_index.to_list():
            cid = p["canonical_id"]
            own = [a for a in self.actions if a["canonical_id"] == cid]
            rows.append({
                **p,
                "simulation_role": self.persona_simulation_role(cid),
                "evidence_role": self.persona_evidence_role(cid),
                "behavioural_actions": sum(1 for a in own if self.is_behavioural(a)),
                "excluded_actions": sum(1 for a in own if not self.is_behavioural(a)),
            })
        return rows

    def authoritative_documents(self) -> List[Dict[str, str]]:
        """Full texts of the sources the experiment declared authoritative (e.g. Product Truth)"""
        from .source_requirements import authoritative_texts

        return authoritative_texts(self.source_requirements, self.sources)

    def persona_stage_timeline(self, canonical_id: str) -> List[Dict[str, Any]]:
        """
        Private stage deliveries and completions of a persona (orchestration facts, not behaviour)

        Each stage is listed once (earliest round over both platforms).
        """
        persona = self.persona_index.personas.get(canonical_id)
        agent_ids = set((persona or {}).get("agent_ids", []))
        timeline: Dict[tuple, Dict[str, Any]] = {}
        for e in self.orchestration_events:
            if e.get("event_type") not in ("private_instruction", "stage_completed"):
                continue
            try:
                if int(e.get("agent_id")) not in agent_ids:
                    continue
            except (TypeError, ValueError):
                continue
            if e.get("event_type") == "private_instruction" and e.get("status", "delivered") != "delivered":
                continue
            kind = "delivered" if e["event_type"] == "private_instruction" else "completed"
            key = (e.get("stage_id"), kind)
            round_num = e.get("round", e.get("completed_round", 0)) or 0
            if key not in timeline or round_num < timeline[key]["round"]:
                timeline[key] = {
                    "stage_id": e.get("stage_id"),
                    "stage_name": e.get("stage_name") or e.get("stage_id"),
                    "event": kind,
                    "round": round_num,
                    "records": list(e.get("records") or []) if kind == "delivered" else [],
                }
        return sorted(timeline.values(), key=lambda x: (x["round"], x["event"] != "delivered", x["stage_id"] or ""))

    def first_encounter(self, canonical_id: str, terms: List[str]) -> Optional[Dict[str, Any]]:
        """
        Earliest action of a persona whose own text or reacted-to content contains a term

        Only the persona's own agent-generated actions count (seen content enters the
        log through the persona's reaction to it).
        """
        lowered = [t.lower() for t in terms or [] if t]
        hits = [
            a for a in self.persona_actions(canonical_id, behavioural_only=True)
            if any(t in f"{a.get('content') or ''} {a.get('target') or ''}".lower() for t in lowered)
        ]
        return min(hits, key=lambda a: (a.get("round", 0), a["timestamp"], a["ref"])) if hits else None

    def persona_records(self, canonical_id: str) -> Dict[str, int]:
        """Orchestration labels recorded for a persona (e.g. EXPOSED) with the first round"""
        records: Dict[str, int] = {}
        for item in self.persona_stage_timeline(canonical_id):
            for label in item["records"]:
                records.setdefault(label, item["round"])
        return records

    def platform_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in self.actions:
            counts[a["platform"]] = counts.get(a["platform"], 0) + 1
        return counts

    def resolve_persona(self, persona: Optional[str]) -> Optional[str]:
        """Resolve a persona reference (canonical id, id variant or name) to a canonical id"""
        if not persona:
            return None
        persona = str(persona).strip()
        if persona in self.persona_index.personas:
            return persona
        canonical = self.persona_index.canonical_for_name(persona)
        if canonical in self.persona_index.personas:
            return canonical
        lowered = persona.lower()
        for cid, p in self.persona_index.personas.items():
            if lowered == cid.lower() or any(lowered == n.lower() for n in p["names"]):
                return cid
        for cid, p in self.persona_index.personas.items():
            if any(lowered in n.lower() for n in p["names"]):
                return cid
        return canonical

    def search_actions(
        self,
        query: Optional[str] = None,
        persona: Optional[str] = None,
        platform: Optional[str] = None,
        action_type: Optional[str] = None,
        limit: int = 50,
        scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter/search raw actions of this run

        Args:
            query: Keywords matched against authored text and reaction targets (any term)
            persona: Canonical id / name of the actor
            platform: twitter / reddit
            action_type: e.g. CREATE_POST
            limit: Max results (0 = all)
            scope: None (all, incl. seed/system content), "behavioural" (agent_generated)
                or "consumer" (target consumers' agent_generated actions)

        Returns:
            Matching normalized actions, ordered by score then chronology
        """
        canonical = self.resolve_persona(persona) if persona else None
        terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) >= 3]
        results = []
        for a in self.actions:
            if scope == "behavioural" and not self.is_behavioural(a):
                continue
            if scope == "consumer" and not self.is_consumer_evidence(a):
                continue
            if canonical and a["canonical_id"] != canonical:
                continue
            if platform and a["platform"] != platform.lower():
                continue
            if action_type and a["action_type"].upper() != action_type.upper():
                continue
            score = 0
            if terms:
                haystack = f"{a['content']} {a['target']} {a['display_name']} {a['action_type']}".lower()
                score = sum(1 for t in terms if t in haystack)
                if score == 0:
                    continue
            results.append((score, a))
        results.sort(key=lambda x: (-x[0], x[1]["platform"], x[1]["round"], x[1]["ref"]))
        found = [a for _, a in results]
        return found[:limit] if limit else found

    def persona_actions(self, canonical_id: str, behavioural_only: bool = False) -> List[Dict[str, Any]]:
        return [
            a for a in self.actions
            if a["canonical_id"] == canonical_id and (not behavioural_only or self.is_behavioural(a))
        ]

    @staticmethod
    def format_action(a: Dict[str, Any], max_chars: int = 400) -> str:
        """One-line, citeable rendering of an action"""
        text = f"[{a['ref']}] round {a['round']} {a['platform']} {a['canonical_id']} ({a['display_name']})"
        if a.get("evidence_role"):
            text += f" [{a.get('simulation_role')}/{a.get('evidence_role')}]"
        text += f" {a['action_type']}"
        if a.get("origin", ORIGIN_AGENT_GENERATED) != ORIGIN_AGENT_GENERATED:
            text += f" [origin={a['origin']}: NOT behavioural evidence]"
        if a.get("content"):
            text += f' wrote: "{a["content"]}"'
        if a.get("target"):
            text += f' on: "{a["target"]}"'
        if not a.get("success", True):
            text += " [action failed]"
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text

    def persona_transcript(self, canonical_id: str, max_chars: int = 40000, behavioural_only: bool = False) -> str:
        """Chronological transcript of one persona's actions on both platforms"""
        actions = sorted(
            self.persona_actions(canonical_id, behavioural_only=behavioural_only),
            key=lambda a: (a["round"], a["timestamp"], a["ref"]),
        )
        lines = []
        total = 0
        for a in actions:
            line = self.format_action(a)
            if total + len(line) > max_chars:
                lines.append(f"... ({len(actions) - len(lines)} more actions truncated for length; all are in normalized_evidence.json)")
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """Aggregate counts of the run's raw actions"""
        by_type: Dict[str, int] = {}
        by_persona: Dict[str, Dict[str, int]] = {}
        rounds = [a["round"] for a in self.actions if isinstance(a.get("round"), int)]
        for a in self.actions:
            by_type[a["action_type"]] = by_type.get(a["action_type"], 0) + 1
            p = by_persona.setdefault(a["canonical_id"], {"total": 0, "twitter": 0, "reddit": 0})
            p["total"] += 1
            p[a["platform"]] = p.get(a["platform"], 0) + 1
        silent = [cid for cid in self.persona_index.personas if cid not in by_persona]
        by_origin: Dict[str, int] = {}
        for a in self.actions:
            by_origin[a.get("origin", ORIGIN_AGENT_GENERATED)] = by_origin.get(a.get("origin", ORIGIN_AGENT_GENERATED), 0) + 1
        return {
            "by_origin": by_origin,
            "behavioural_actions": by_origin.get(ORIGIN_AGENT_GENERATED, 0),
            "consumer_metrics": self.consumer_metrics(),
            "private_instructions_delivered": sum(
                1 for e in self.orchestration_events
                if e.get("event_type") == "private_instruction" and e.get("status", "delivered") == "delivered"
            ),
            "total_actions": len(self.actions),
            "by_platform": self.platform_counts(),
            "by_action_type": by_type,
            "by_persona": by_persona,
            "personas_total": len(self.persona_index.personas),
            "personas_without_actions": silent,
            "rounds_observed": [min(rounds), max(rounds)] if rounds else [],
        }

    def source_excerpts(self, name: Optional[str] = None, query: Optional[str] = None, max_chars: int = 6000) -> str:
        """
        Text from the run's source documents (marks the returned documents as used)

        Args:
            name: Optional (partial) filename
            query: Optional keywords; returns the best matching paragraphs
            max_chars: Output budget
        """
        docs = [d for d in self.sources if not name or name.lower() in (d.get("original_filename") or "").lower()]
        if not docs:
            return "No source document matches. Available: " + ", ".join(
                d.get("original_filename") or "?" for d in self.sources
            )
        terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) >= 3]
        budget = max(500, max_chars // max(len(docs), 1))
        parts = []
        for d in docs:
            text = d.get("text") or ""
            if not text.strip():
                parts.append(f"### {d.get('original_filename')}\n(no extracted text: {d.get('extraction_status')} {d.get('error') or ''})")
                continue
            if terms:
                paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
                scored = sorted(paragraphs, key=lambda p: -sum(1 for t in terms if t in p.lower()))
                excerpt = "\n\n".join(scored)[:budget]
            else:
                excerpt = text[:budget]
            self.used_sources.add(d.get("original_filename"))
            parts.append(f"### {d.get('original_filename')}\n{excerpt}")
        return "\n\n".join(parts)

    def mark_source_used(self, filename: str):
        self.used_sources.add(filename)

    def source_manifest(self) -> List[Dict[str, Any]]:
        """Available sources with an explicit used_in_report flag"""
        return [
            {
                "original_filename": d.get("original_filename"),
                "extension": d.get("extension"),
                "size": d.get("size"),
                "text_length": d.get("text_length", len(d.get("text") or "")),
                "extraction_status": d.get("extraction_status"),
                "error": d.get("error"),
                "used_in_report": d.get("original_filename") in self.used_sources,
            }
            for d in self.sources
        ]

    def to_dict(self, include_actions: bool = True) -> Dict[str, Any]:
        """Machine-readable normalized evidence"""
        data = {
            "simulation_id": self.simulation_id,
            "run_id": self.run_id,
            "run": self.record.to_dict() if self.record else None,
            "legacy_run": self.run_id == LEGACY_RUN_ID,
            "integrity": self.integrity,
            "requested_persona_ids": self.requested_ids,
            "personas": self.persona_index.to_list(),
            "population": self.population(),
            "roles_source": self.roles_source,
            "roles_note": self.roles_note,
            "metric_scope": CONSUMER_METRIC_SCOPE,
            "orchestration_events": self.orchestration_events,
            "stage_timelines": {cid: self.persona_stage_timeline(cid) for cid in self.persona_index.personas},
            "sources_origin": self.sources_origin,
            "sources": self.source_manifest(),
            "control_documents": self.control_documents,
            "run_contract": {
                "contract_id": self.run_contract.get("contract_id"),
                "contract_hash": self.run_contract.get("contract_hash"),
            } if self.run_contract else None,
            "stats": self.stats(),
            "evidence_type": "simulated (synthetic agents in an OASIS social-media simulation; not real-world customer data)",
        }
        if include_actions:
            data["actions"] = self.actions
        return data
