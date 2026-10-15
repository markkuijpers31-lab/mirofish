"""
Run preflight: the one canonical check that a contracted simulation is exactly what its Run Contract says

RunContractPreflight.evaluate() compares the persisted configuration of a
simulation (agent_selection.json, experiment_plan.json, source_requirements.json,
run_contract.json, and after preparation agent_roster.json, both profile files and
simulation_config.json) with what the project's Run Contract requires. Prepare,
start and the UI all use this result; the frontend's own state never decides.

Every persisted file is read exactly once per evaluation and its SHA-256 is
recorded, so a run snapshot can prove it copied the very bytes that passed.
"""

import csv
import hashlib
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger
from .agent_selection import (
    EVIDENCE_EXCLUDED, EVIDENCE_INFLUENCER_CONTEXT, EVIDENCE_TARGET_CONSUMER, selection_hash,
)
from .run_contract import (
    CONTRACT_FILE, canonical_json, expected_selection_hash, load_project_contract, persona_key,
    read_graph, resolve_population, sha256_text, without_contract_documents,
)

logger = get_logger("mirofish.run_preflight")

PREFLIGHT_FILE = "preflight.json"

# Persisted files a contracted run is built from (all must exist and match)
CONTRACT_RUN_FILES = (
    "simulation_config.json",
    "reddit_profiles.json",
    "twitter_profiles.csv",
    "agent_roster.json",
    "agent_selection.json",
    "experiment_plan.json",
    "source_requirements.json",
    CONTRACT_FILE,
)
_PREPARED_STATUSES = {"ready", "running", "paused", "completed", "stopped"}
_PLAN_META_KEYS = {"saved", "updated_at", "contract_id", "contract_hash", "applies_to"}


def file_sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class _Artifacts:
    """Reads each persisted file of a simulation once and remembers its hash"""

    def __init__(self, sim_dir: str):
        self.sim_dir = sim_dir
        self.hashes: Dict[str, Optional[str]] = {}
        self._cache: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}

    def _bytes(self, name: str) -> Optional[bytes]:
        path = os.path.join(self.sim_dir, name)
        if not os.path.exists(path):
            self.hashes[name] = None
            return None
        with open(path, "rb") as f:
            data = f.read()
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, name: str) -> Any:
        if name not in self._cache:
            data = self._bytes(name)
            value = None
            if data is not None:
                try:
                    value = json.loads(data.decode("utf-8"))
                except Exception as e:
                    self.errors[name] = f"unreadable: {e}"
            self._cache[name] = value
        return self._cache[name]

    def csv_rows(self, name: str) -> Optional[List[Dict[str, str]]]:
        if name not in self._cache:
            data = self._bytes(name)
            value = None
            if data is not None:
                try:
                    value = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
                except Exception as e:
                    self.errors[name] = f"unreadable: {e}"
            self._cache[name] = value
        return self._cache[name]


def _prepare_task_running(state: Any) -> bool:
    from ..models.task import TaskManager, TaskStatus

    if (state.status.value if hasattr(state.status, "value") else state.status) != "preparing" or not state.prepare_task_id:
        return False
    task = TaskManager().get_task(state.prepare_task_id)
    return bool(task and task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING))


def _standard_result(simulation_id: str) -> Dict[str, Any]:
    return {
        "simulation_id": simulation_id,
        "mode": "standard",
        "contract": {"present": False, "id": None, "schema": None, "hash": None, "valid": False,
                     "applied": False, "documents": [], "errors": []},
        "population": None, "roles": None, "experiment_plan": None, "rounds": None,
        "sources": None, "platform": None, "preparation": None,
        "ready_to_prepare": None, "ready_to_start": None,
        "errors": [], "artifacts": {}, "checked_at": datetime.now().isoformat(),
    }


class RunContractPreflight:
    """Canonical validator of contracted simulations (see module docstring)"""

    @classmethod
    def evaluate(
        cls,
        simulation_id: str,
        requested: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a simulation against its project's Run Contract

        Args:
            simulation_id: Simulation ID
            requested: Values of a start request to check against the contract
                ({"max_rounds": int|None, "platform": str|None})

        Returns:
            Preflight document (contract, population, roles, experiment_plan, rounds,
            sources, platform, preparation, ready_to_prepare, ready_to_start, errors,
            artifacts, fingerprint). For simulations without a Run Contract:
            mode="standard", contract.present=false and nothing else is judged.

        Raises:
            ValueError: unknown simulation
        """
        from ..models.project import ProjectManager
        from .simulation_manager import SimulationManager
        from .source_requirements import check_coverage

        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")
        loaded = load_project_contract(state.project_id)
        if not loaded["present"]:
            return _standard_result(simulation_id)

        requested = requested or {}
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        art = _Artifacts(sim_dir)
        errors: List[str] = []
        digest = loaded["hash"]
        applied = art.json(CONTRACT_FILE)
        contract_block = {
            "present": True,
            "id": loaded["id"],
            "schema": loaded["schema"],
            "hash": digest,
            "valid": loaded["valid"],
            "documents": loaded["documents"],
            "errors": list(loaded["errors"]),
            "applied": bool(loaded["valid"] and applied and applied.get("contract_hash") == digest
                            and canonical_json(applied.get("contract")) == canonical_json(loaded["raw"])),
            "applied_hash": (applied or {}).get("contract_hash"),
            "title": (loaded["contract"] or {}).get("title"),
        }
        result: Dict[str, Any] = {
            "simulation_id": simulation_id,
            "mode": "contract",
            "contract": contract_block,
            "requested": {"max_rounds": requested.get("max_rounds"), "platform": requested.get("platform")},
        }
        if not loaded["valid"]:
            errors.extend(f"Run Contract invalid: {e}" for e in loaded["errors"])
            result.update({
                "population": {"ok": False}, "roles": {"ok": False}, "experiment_plan": {"ok": False},
                "rounds": {"expected": None, "ok": False}, "sources": {"ok": False},
                "platform": {"ok": False}, "preparation": {"ok": False, "prepared": False},
                "ready_to_prepare": False, "ready_to_start": False, "errors": errors,
                "artifacts": {}, "checked_at": datetime.now().isoformat(),
            })
            result["fingerprint"] = cls.fingerprint(result)
            return result
        if not contract_block["applied"]:
            errors.append("Run Contract has not been applied to this simulation (or changed since): apply it first")

        contract = loaded["contract"]
        nodes, eligible = read_graph(state.graph_id)
        resolution = resolve_population(contract, nodes, eligible)
        eligible_order = [e.uuid for e in eligible]
        resolved_by_uuid = {r["entity_uuid"]: r for r in resolution["rows"] if r["status"] == "resolved"}
        expected_agents = [resolved_by_uuid[u] for u in eligible_order if u in resolved_by_uuid]
        expected_hash = expected_selection_hash(resolution)

        # ---------------- population / roles ----------------
        selection = art.json("agent_selection.json")
        sel_rows = {e.get("entity_uuid"): e for e in (selection or {}).get("entities", [])}
        names = {e.uuid: e.name for e in eligible}
        population_rows, inactive, role_mismatches, evidence_mismatches = [], [], [], []
        for row in resolution["rows"]:
            sel = sel_rows.get(row["entity_uuid"]) if row["entity_uuid"] else None
            out = {k: row[k] for k in ("persona_id", "status", "entity_uuid", "name", "entity_type",
                                         "simulation_role", "evidence_role", "expected_name", "candidates")}
            out.update({
                "selection_active": bool(sel and sel.get("active")),
                "selection_simulation_role": (sel or {}).get("simulation_role"),
                "selection_evidence_role": (sel or {}).get("evidence_role"),
            })
            if row["status"] == "resolved":
                if not out["selection_active"]:
                    inactive.append(row["persona_id"])
                else:
                    if sel.get("simulation_role") != row["simulation_role"]:
                        role_mismatches.append({"persona_id": row["persona_id"], "name": row["name"],
                                                "expected": row["simulation_role"], "actual": sel.get("simulation_role")})
                    if sel.get("evidence_role") != row["evidence_role"]:
                        evidence_mismatches.append({"persona_id": row["persona_id"], "name": row["name"],
                                                    "expected": row["evidence_role"], "actual": sel.get("evidence_role")})
            population_rows.append(out)
        unexpected = [
            {"entity_uuid": uid, "name": e.get("name") or names.get(uid)}
            for uid, e in sel_rows.items() if e.get("active") and uid not in resolved_by_uuid
        ]
        active_sel = [e for e in (selection or {}).get("entities", []) if e.get("active")]
        selection_checks = {
            "saved": selection is not None,
            "confirmed": bool(selection and selection.get("confirmed")),
            "role_catalog_matches": bool(selection) and canonical_json(selection.get("role_catalog")) == canonical_json(contract["role_catalog"]),
            "selection_hash_matches": bool(selection) and selection.get("selection_hash") == expected_hash
            and selection_hash(selection.get("entities", [])) == expected_hash,
            "contract_hash_matches": bool(selection) and selection.get("contract_hash") == digest,
        }
        population = {
            "expected": len(contract["population"]),
            "resolved": resolution["resolved"],
            "active": len(active_sel),
            "missing_persona_ids": resolution["missing"],
            "duplicate_persona_ids": resolution["duplicates"],
            "name_mismatch_persona_ids": resolution["name_mismatches"],
            "inactive_persona_ids": inactive,
            "unexpected_active": unexpected,
            "role_mismatches": role_mismatches,
            "evidence_role_mismatches": evidence_mismatches,
            "selection": selection_checks,
            "entities": population_rows,
        }
        population["ok"] = bool(
            resolution["resolved"] == population["expected"]
            and not (resolution["missing"] or resolution["duplicates"] or resolution["name_mismatches"])
            and not (inactive or unexpected or role_mismatches or evidence_mismatches)
            and len(active_sel) == population["expected"]
            and all(selection_checks.values())
        )
        for pid in resolution["missing"]:
            errors.append(f"{pid}: no eligible graph entity resolves to this persona id")
        for row in population_rows:
            if row["status"] == "duplicate":
                errors.append(f"{row['persona_id']}: several graph entities match: "
                              + ", ".join(repr(c["name"]) for c in row["candidates"]))
            elif row["status"] == "name_mismatch":
                errors.append(f"{row['persona_id']}: graph entity {row['candidates'][0]['name']!r} "
                              f"does not match expected_name {row['expected_name']!r}")
        if inactive:
            errors.append(f"Contract personas inactive in the saved selection: {inactive}")
        if unexpected:
            errors.append("Unexpected active agents (not in the Run Contract): " + ", ".join(repr(u["name"]) for u in unexpected))
        for m in role_mismatches:
            errors.append(f"{m['persona_id']} ({m['name']}): simulation role {m['actual']!r}, contract requires {m['expected']!r}")
        for m in evidence_mismatches:
            errors.append(f"{m['persona_id']} ({m['name']}): evidence role {m['actual']!r}, contract requires {m['expected']!r}")
        for check, ok in selection_checks.items():
            if not ok:
                errors.append(f"Saved agent selection: {check.replace('_', ' ')} failed")

        counts = {EVIDENCE_TARGET_CONSUMER: 0, EVIDENCE_INFLUENCER_CONTEXT: 0, EVIDENCE_EXCLUDED: 0}
        for e in active_sel:
            counts[e.get("evidence_role")] = counts.get(e.get("evidence_role"), 0) + 1
        roles = {
            "target_consumers": counts[EVIDENCE_TARGET_CONSUMER],
            "influencer_context": counts[EVIDENCE_INFLUENCER_CONTEXT],
            "excluded_active": counts[EVIDENCE_EXCLUDED],
            "expected_target_consumers": contract["counts"]["target_consumers"],
            "expected_influencer_context": contract["counts"]["influencer_context"],
            "expected_excluded_active": contract["counts"]["excluded_active"],
        }
        roles["ok"] = bool(
            roles["target_consumers"] == roles["expected_target_consumers"]
            and roles["influencer_context"] == roles["expected_influencer_context"]
            and roles["excluded_active"] == roles["expected_excluded_active"]
            and population["ok"]
        )
        if not roles["ok"]:
            errors.append(
                f"Roles: target consumers {roles['target_consumers']}/{roles['expected_target_consumers']}, "
                f"influencer/context {roles['influencer_context']}/{roles['expected_influencer_context']}, "
                f"excluded active {roles['excluded_active']}/{roles['expected_excluded_active']}"
            )

        # ---------------- experiment plan ----------------
        expected_plan = contract["experiment_plan"]
        saved_plan = art.json("experiment_plan.json")
        plan_mismatches: List[Dict[str, Any]] = []
        if saved_plan is None:
            plan_mismatches.append({"stage_id": None, "field": "experiment_plan.json", "expected": "saved", "actual": "missing"})
        else:
            if saved_plan.get("contract_hash") != digest:
                plan_mismatches.append({"stage_id": None, "field": "contract_hash", "expected": digest,
                                        "actual": saved_plan.get("contract_hash")})
            for key in sorted((set(saved_plan) | set(expected_plan)) - _PLAN_META_KEYS - {"stages"}):
                if canonical_json(saved_plan.get(key)) != canonical_json(expected_plan.get(key)):
                    plan_mismatches.append({"stage_id": None, "field": key, "expected": expected_plan.get(key),
                                            "actual": saved_plan.get(key)})
            saved_stages = saved_plan.get("stages") if isinstance(saved_plan.get("stages"), list) else []
            exp_ids = [s["id"] for s in expected_plan["stages"]]
            got_ids = [s.get("id") for s in saved_stages if isinstance(s, dict)]
            for sid in exp_ids:
                if sid not in got_ids:
                    plan_mismatches.append({"stage_id": sid, "field": "stage", "expected": "present", "actual": "missing"})
            for sid in got_ids:
                if sid not in exp_ids:
                    plan_mismatches.append({"stage_id": sid, "field": "stage", "expected": "absent", "actual": "unexpected"})
            if [i for i in got_ids if i in exp_ids] != [i for i in exp_ids if i in got_ids]:
                plan_mismatches.append({"stage_id": None, "field": "stage_order", "expected": exp_ids, "actual": got_ids})
            by_id = {s.get("id"): s for s in saved_stages if isinstance(s, dict)}
            for stage in expected_plan["stages"]:
                got = by_id.get(stage["id"])
                if got is None:
                    continue
                for key in sorted(set(stage) | set(got)):
                    if canonical_json(stage.get(key)) != canonical_json(got.get(key)):
                        plan_mismatches.append({"stage_id": stage["id"], "field": key,
                                                "expected": stage.get(key), "actual": got.get(key)})
        unresolved_targets = []
        resolved_keys = {persona_key(r["persona_id"]) for r in resolution["rows"] if r["status"] == "resolved"}
        for stage in expected_plan["stages"]:
            for pid in stage["target"].get("persona_ids") or []:
                if persona_key(pid) not in resolved_keys:
                    unresolved_targets.append({"stage_id": stage["id"], "persona_id": pid})
        experiment_plan = {
            "expected_stages": len(expected_plan["stages"]),
            "saved_stages": len((saved_plan or {}).get("stages") or []) if saved_plan is not None else 0,
            "mismatches": plan_mismatches,
            "unresolved_stage_targets": unresolved_targets,
            "seed_blocked_terms": list(expected_plan.get("seed_blocked_terms") or []),
            "stages": [
                {"id": s["id"], "name": s["name"], "order": s["order"], "trigger_round": s["trigger_round"],
                 "persona_ids": s["target"].get("persona_ids") or [],
                 "simulation_roles": s["target"].get("simulation_roles") or [],
                 "evidence_roles": s["target"].get("evidence_roles") or [],
                 "condition": s["condition"].get("type"), "completion": s["completion"].get("type"),
                 "records": s.get("records") or [], "context_sections": s.get("context_sections") or []}
                for s in expected_plan["stages"]
            ],
        }
        experiment_plan["ok"] = not plan_mismatches and not unresolved_targets
        for m in plan_mismatches:
            errors.append(f"Experiment plan: stage {m['stage_id'] or '-'} field {m['field']} differs from the Run Contract")
        for u in unresolved_targets:
            errors.append(f"Experiment plan: stage {u['stage_id']} targets unresolved persona {u['persona_id']}")

        # ---------------- rounds / platform ----------------
        expected_rounds = contract["max_rounds"]
        req_rounds = requested.get("max_rounds")
        rounds = {
            "expected": expected_rounds,
            "requested": req_rounds,
            "requested_ok": req_rounds is None or req_rounds == expected_rounds,
            "locked": True,
        }
        rounds["ok"] = bool(rounds["requested_ok"])
        if not rounds["requested_ok"]:
            errors.append(f"max_rounds {req_rounds} requested; the Run Contract fixes {expected_rounds}")
        req_platform = requested.get("platform")
        platform = {
            "expected": contract["platform"],
            "requested": req_platform,
            "twitter_enabled": bool(state.enable_twitter),
            "reddit_enabled": bool(state.enable_reddit),
        }
        platform["ok"] = bool(state.enable_twitter and state.enable_reddit
                              and (req_platform is None or req_platform == contract["platform"]))
        if not (state.enable_twitter and state.enable_reddit):
            errors.append("Contracted simulations need both platforms enabled (profiles for both are snapshotted)")
        if req_platform is not None and req_platform != contract["platform"]:
            errors.append(f"platform {req_platform!r} requested; the Run Contract fixes {contract['platform']!r}")

        # ---------------- sources ----------------
        saved_requirements = art.json("source_requirements.json")
        documents = without_contract_documents(ProjectManager.get_source_documents(state.project_id))
        coverage = check_coverage(contract["source_requirements"], documents)
        coverage.pop("checked_at", None)
        sources = {
            "requirements_present": saved_requirements is not None,
            "requirements_match": saved_requirements is not None
            and canonical_json(saved_requirements) == canonical_json(contract["source_requirements"]),
            "coverage_ok": bool(coverage["ok"]),
            "missing": coverage["missing"],
            "documents": coverage["documents"],
        }
        sources["ok"] = bool(sources["requirements_present"] and sources["requirements_match"] and sources["coverage_ok"])
        if not sources["requirements_present"]:
            errors.append("source_requirements.json is missing (required by the Run Contract)")
        elif not sources["requirements_match"]:
            errors.append("source_requirements.json differs from the Run Contract")
        for doc in coverage["documents"]:
            if not doc["files"]:
                errors.append(f"Required document '{doc['label']}' is not among the uploaded sources")
            for s in doc["sections"]:
                if s["status"] != "covered":
                    detail = f": missing terms {s['missing_terms']}" if s["missing_terms"] else ""
                    errors.append(f"{doc['label']} section #{s['id']} is {s['status']}{detail}")

        # ---------------- preparation ----------------
        preparing = _prepare_task_running(state)
        status = state.status.value
        roster = art.json("agent_roster.json")
        config = art.json("simulation_config.json")
        reddit = art.json("reddit_profiles.json")
        twitter = art.csv_rows("twitter_profiles.csv")
        prep_problems: List[str] = []
        prepared = bool(state.profiles_generated and state.config_generated and status in _PREPARED_STATUSES)

        def _agent_ok(i: int, agent: Dict[str, Any]) -> bool:
            exp = expected_agents[i]
            return (agent.get("agent_id") == i and agent.get("entity_uuid") == exp["entity_uuid"]
                    and agent.get("name") == exp["name"] and agent.get("simulation_role") == exp["simulation_role"]
                    and agent.get("evidence_role") == exp["evidence_role"])

        roster_active = (roster or {}).get("active_agents") or []
        roster_ok = bool(
            roster and len(roster_active) == len(expected_agents) == population["expected"]
            and all(_agent_ok(i, a) for i, a in enumerate(roster_active))
            and sorted(c.get("entity_uuid") for c in roster.get("context_entities", []))
            == sorted(u for u in eligible_order if u not in resolved_by_uuid)
            and canonical_json(roster.get("role_catalog")) == canonical_json(contract["role_catalog"])
        )
        if roster is not None and not roster_ok:
            prep_problems.append("agent_roster.json does not match the contract-resolved population")
        sel_hash_ok = bool(roster and selection and roster.get("selection_hash") == selection.get("selection_hash") == expected_hash)
        roster_contract_ok = bool(roster and roster.get("contract_hash") == digest)
        if roster is not None and not roster_contract_ok:
            prep_problems.append("agent_roster.json was prepared for a different Run Contract")

        def _profiles_ok(rows: Optional[List[Dict[str, Any]]]) -> bool:
            if not isinstance(rows, list) or len(rows) != len(expected_agents):
                return False
            for i, p in enumerate(rows):
                exp = expected_agents[i]
                if str(p.get("user_id")) != str(i) or p.get("name") != exp["name"]:
                    return False
                if persona_key(p.get("name")) != persona_key(exp["persona_id"]):
                    return False
            return True

        reddit_ok, twitter_ok = _profiles_ok(reddit), _profiles_ok(twitter)
        if reddit is not None and not reddit_ok:
            prep_problems.append(f"reddit_profiles.json holds {len(reddit) if isinstance(reddit, list) else '?'} profiles "
                                 f"that do not match the {len(expected_agents)} contract agents")
        if twitter is not None and not twitter_ok:
            prep_problems.append(f"twitter_profiles.csv holds {len(twitter)} profiles that do not match the "
                                 f"{len(expected_agents)} contract agents")
        agent_configs = (config or {}).get("agent_configs") or []
        config_ok = bool(config) and len(agent_configs) == len(expected_agents) and all(
            c.get("agent_id") == i and c.get("entity_uuid") == expected_agents[i]["entity_uuid"]
            and c.get("entity_name") == expected_agents[i]["name"]
            and c.get("simulation_role") == expected_agents[i]["simulation_role"]
            and c.get("evidence_role") == expected_agents[i]["evidence_role"]
            for i, c in enumerate(agent_configs)
        )
        if config is not None and not config_ok:
            prep_problems.append(f"simulation_config.json agent_configs ({len(agent_configs)}) do not match the contract agents")
        blocked = [t.lower() for t in expected_plan.get("seed_blocked_terms") or []]
        bad_seeds = [
            p.get("content") for p in ((config or {}).get("event_config") or {}).get("initial_posts") or []
            if any(t in str(p.get("content") or "").lower() for t in blocked)
            or not (isinstance(p.get("poster_agent_id"), int) and 0 <= p["poster_agent_id"] < len(expected_agents))
        ]
        if bad_seeds:
            prep_problems.append(f"{len(bad_seeds)} seed post(s) mention a blocked term or have no valid poster")
        for name, problem in art.errors.items():
            prep_problems.append(f"{name} {problem}")
        preparation = {
            "prepared": prepared,
            "preparing": preparing,
            "status": status,
            "selection_hash_matches": sel_hash_ok,
            "roster_matches_contract": roster_ok,
            "roster_contract_hash_matches": roster_contract_ok,
            "roster_active": len(roster_active) if roster is not None else None,
            "profile_count_twitter": len(twitter) if isinstance(twitter, list) else None,
            "profile_count_reddit": len(reddit) if isinstance(reddit, list) else None,
            "profile_persona_ids_match": reddit_ok and twitter_ok,
            "config_agent_count": len(agent_configs) if config is not None else None,
            "config_agents_match": config_ok,
            "seed_posts_ok": not bad_seeds,
            "expected_agents": len(expected_agents),
            "problems": prep_problems,
        }
        preparation["ok"] = bool(
            prepared and not preparing and sel_hash_ok and roster_ok and roster_contract_ok
            and reddit_ok and twitter_ok and config_ok and not bad_seeds and not art.errors
        )

        base_ok = bool(contract_block["applied"] and population["ok"] and roles["ok"] and experiment_plan["ok"]
                       and sources["ok"] and state.enable_twitter and state.enable_reddit)
        ready_to_prepare = base_ok and not preparing
        ready_to_start = base_ok and not preparing and preparation["ok"] and rounds["ok"] and platform["ok"]
        if preparing:
            errors.append("A preparation is running")
        elif not preparation["ok"]:
            if not prepared:
                errors.append("Not prepared yet for this contract (Prepare is required before Start)")
            errors.extend(prep_problems)

        # Every file a contracted run snapshots must be checked bytes
        for name in CONTRACT_RUN_FILES:
            if name not in art.hashes:
                art._bytes(name)

        result.update({
            "population": population,
            "roles": roles,
            "experiment_plan": experiment_plan,
            "rounds": rounds,
            "platform": platform,
            "sources": sources,
            "preparation": preparation,
            "ready_to_prepare": ready_to_prepare,
            "ready_to_start": ready_to_start,
            "errors": errors,
            "artifacts": dict(sorted(art.hashes.items())),
            "checked_at": datetime.now().isoformat(),
        })
        result["fingerprint"] = cls.fingerprint(result)
        return result

    @staticmethod
    def fingerprint(result: Dict[str, Any]) -> str:
        """Hash of a preflight without its timestamp: equal state gives an equal fingerprint"""
        stable = {k: v for k, v in result.items() if k not in ("checked_at", "fingerprint", "requested")}
        return sha256_text(canonical_json(stable))


def summarize_for_run(preflight: Dict[str, Any]) -> Dict[str, Any]:
    """Counts and hashes of a ready preflight recorded in run.json.snapshot"""
    population = preflight.get("population") or {}
    roles = preflight.get("roles") or {}
    return {
        "contract_id": preflight["contract"]["id"],
        "contract_hash": preflight["contract"]["hash"],
        "preflight_fingerprint": preflight.get("fingerprint"),
        "active_agents": population.get("active"),
        "target_consumers": roles.get("target_consumers"),
        "influencer_context": roles.get("influencer_context"),
        "excluded_active": roles.get("excluded_active"),
        "max_rounds": (preflight.get("rounds") or {}).get("expected"),
        "platform": (preflight.get("platform") or {}).get("expected"),
        "source_coverage_ok": (preflight.get("sources") or {}).get("coverage_ok"),
    }
