"""
Simulation-related API routes
Step2: Entity reading and filtering, OASIS simulation preparation and execution (fully automated)
"""

import os
import traceback
from typing import Any, Dict, Optional
from flask import request, jsonify, send_file

from . import simulation_bp
from ..config import Config
from ..services.agent_selection import (
    DEFAULT_ROLE_CATALOG, AgentSelectionStore, apply_selection_update, build_selection_view,
)
from ..services.entity_reader import EntityReader
from ..services.oasis_profile_generator import OasisProfileGenerator
from ..services.simulation_manager import SimulationManager, SimulationStatus
from ..services.simulation_runner import SimulationRunner, RunnerStatus
from ..services.run_store import RunStatus, RunStore
from ..services.run_contract import RunContractStore, apply_contract, project_has_contract
from ..services.run_preflight import RunContractPreflight
from ..utils.logger import get_logger, get_run_logger
from ..utils.locale import t, get_locale, set_locale
from ..models.project import ProjectManager

logger = get_logger("mirofish.api.simulation")


# Interview prompt optimization prefix
# Adding this prefix prevents the Agent from calling tools and makes it reply with text directly
INTERVIEW_PROMPT_PREFIX = "Based on your persona, all past memories and actions, reply to me with text directly without calling any tools: "


def optimize_interview_prompt(prompt: str) -> str:
    """
    Optimize the interview prompt by adding a prefix to prevent the Agent from calling tools

    Args:
        prompt: Original prompt

    Returns:
        Optimized prompt
    """
    if not prompt:
        return prompt
    # Avoid adding the prefix repeatedly
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


# ============== Entity reading endpoints ==============


@simulation_bp.route("/entities/<graph_id>", methods=["GET"])
def get_graph_entities(graph_id: str):
    """
    Get all entities in the graph (filtered)

    Only returns nodes matching predefined entity types (nodes whose Labels are not just Entity)

    Query parameters:
        entity_types: Comma-separated list of entity types (optional, for further filtering)
        enrich: Whether to fetch related edge information (default true)
    """
    try:
        entity_types_str = request.args.get("entity_types", "")
        entity_types = (
            [t.strip() for t in entity_types_str.split(",") if t.strip()]
            if entity_types_str
            else None
        )
        enrich = request.args.get("enrich", "true").lower() == "true"

        logger.info(
            f"Getting graph entities: graph_id={graph_id}, entity_types={entity_types}, enrich={enrich}"
        )

        reader = EntityReader()
        result = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=enrich,
        )

        return jsonify({"success": True, "data": result.to_dict()})

    except Exception as e:
        logger.error(f"Failed to get graph entities: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/entities/<graph_id>/<entity_uuid>", methods=["GET"])
def get_entity_detail(graph_id: str, entity_uuid: str):
    """Get details of a single entity"""
    try:
        reader = EntityReader()
        entity = reader.get_entity_with_context(graph_id, entity_uuid)

        if not entity:
            return jsonify(
                {"success": False, "error": t("api.entityNotFound", id=entity_uuid)}
            ), 404

        return jsonify({"success": True, "data": entity.to_dict()})

    except Exception as e:
        logger.error(f"Failed to get entity details: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/entities/<graph_id>/by-type/<entity_type>", methods=["GET"])
def get_entities_by_type(graph_id: str, entity_type: str):
    """Get all entities of a specified type"""
    try:
        enrich = request.args.get("enrich", "true").lower() == "true"

        reader = EntityReader()
        entities = reader.get_entities_by_type(
            graph_id=graph_id, entity_type=entity_type, enrich_with_edges=enrich
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "entity_type": entity_type,
                    "count": len(entities),
                    "entities": [e.to_dict() for e in entities],
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to get entities: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Agent selection & experiment plan ==============


def _active_run_exists(simulation_id: str) -> bool:
    """Whether a run of the simulation is currently executing"""
    current = SimulationRunner.reconcile_run(simulation_id, RunStore.get_current_run(simulation_id))
    return bool(current and current.status == RunStatus.RUNNING) or SimulationRunner.is_process_alive(simulation_id)


def _prepare_task_running(state) -> bool:
    """Whether the simulation's prepare task is still pending/processing"""
    from ..models.task import TaskManager, TaskStatus

    if state.status != SimulationStatus.PREPARING or not state.prepare_task_id:
        return False
    task = TaskManager().get_task(state.prepare_task_id)
    return bool(task and task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING))


def _population_status(simulation_id: str, state, saved: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Relation between the saved selection, the prepared roster and the simulation state

    prepared  = profiles + config exist for the current population
    in_sync   = the prepared roster was built from the saved (confirmed) selection
    legacy    = prepared before explicit agent selection existed (no roster)
    """
    roster = AgentSelectionStore.load_roster(simulation_id)
    saved = saved if saved is not None else AgentSelectionStore.load(simulation_id)
    flags_ok = state.profiles_generated and state.config_generated and state.status not in (
        SimulationStatus.CREATED, SimulationStatus.PREPARING, SimulationStatus.FAILED
    )
    in_sync = bool(
        roster and saved and saved.get("confirmed")
        and saved.get("selection_hash") == roster.get("selection_hash")
    )
    legacy = flags_ok and roster is None and saved is None
    legacy_agents = 0
    if legacy:
        config = SimulationManager().get_simulation_config(simulation_id) or {}
        legacy_agents = len(config.get("agent_configs", []))
    return {
        "prepared": bool(flags_ok and (in_sync or legacy)),
        "in_sync": in_sync,
        "legacy_prepared": legacy,
        "legacy_agents_count": legacy_agents,
        "selection_confirmed": bool(saved and saved.get("confirmed")),
        "prepared_selection_hash": roster.get("selection_hash") if roster else None,
        "roster_summary": roster.get("summary") if roster else None,
        "prepared_at": roster.get("prepared_at") if roster else None,
        "preparing": _prepare_task_running(state),
        "prepare_task_id": state.prepare_task_id,
        "active_run": _active_run_exists(simulation_id),
        "status": state.status.value,
    }


def _selection_response(simulation_id: str, state, saved: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from ..services.research_instructions import load_project_brief

    saved = saved if saved is not None else AgentSelectionStore.load(simulation_id)
    entities = EntityReader().filter_defined_entities(
        graph_id=state.graph_id, enrich_with_edges=False
    ).entities
    brief = load_project_brief(state.project_id)
    view = build_selection_view(entities, saved, brief.get("persona_ids"))
    view.update({
        "simulation_id": simulation_id,
        "graph_id": state.graph_id,
        "population": _population_status(simulation_id, state, saved),
        "brief_instructions_count": len(brief.get("instructions", [])),
        "contract": _contract_lock_info(simulation_id, state),
    })
    return view


CONTRACT_LOCKED_MESSAGE = "Locked by Run Contract: population, stages, source requirements and rounds come from the contract"


def _contract_locked(state) -> bool:
    """Whether the simulation's project carries a Run Contract (valid or not)"""
    return project_has_contract(state.project_id)


def _contract_lock_info(simulation_id: str, state) -> Dict[str, Any]:
    applied = RunContractStore.load(simulation_id) or {}
    return {
        "locked": _contract_locked(state),
        "applied_contract_id": applied.get("contract_id"),
        "applied_contract_hash": applied.get("contract_hash"),
    }


def _contract_locked_response():
    return jsonify({"success": False, "error": CONTRACT_LOCKED_MESSAGE, "locked_by_contract": True}), 409


def _save_selection(simulation_id: str, state, payload: Dict[str, Any], confirmed_default: bool = False) -> Dict[str, Any]:
    """
    Validate + persist a selection; a changed population invalidates the preparation

    Raises:
        ValueError: invalid selection
        RuntimeError: population cannot change right now (prepare/run in progress)
    """
    if _prepare_task_running(state):
        raise RuntimeError("Preparation is in progress; wait for it to finish before changing the population")
    previous = AgentSelectionStore.load(simulation_id)
    entities = EntityReader().filter_defined_entities(
        graph_id=state.graph_id, enrich_with_edges=False
    ).entities
    payload = dict(payload or {})
    payload.setdefault("confirmed", confirmed_default)
    document = apply_selection_update(entities, payload, previous, state.graph_id)
    roster = AgentSelectionStore.load_roster(simulation_id)
    changes_population = roster is not None and roster.get("selection_hash") != document["selection_hash"]
    legacy_prepared = roster is None and state.profiles_generated and state.config_generated
    invalidates = (changes_population or legacy_prepared) and (state.profiles_generated or state.config_generated)
    if invalidates and _active_run_exists(simulation_id):
        raise RuntimeError("A run is executing; stop it before changing the population (existing runs keep their own snapshot)")
    AgentSelectionStore.save(simulation_id, document)
    if invalidates:
        # Prepared profiles belong to the previous population: re-prepare before the next run.
        # Existing runs are untouched (each holds its own roster snapshot).
        state.status = SimulationStatus.CREATED
        state.profiles_generated = False
        state.config_generated = False
        SimulationManager()._save_simulation_state(state)
    get_logger("mirofish.api.simulation").info(
        f"[{simulation_id}] Agent selection saved: active={sum(1 for e in document['entities'] if e['active'])}/"
        f"{len(document['entities'])} confirmed={document['confirmed']} invalidated_prepare={bool(invalidates)}"
    )
    return {"document": document, "preparation_invalidated": bool(invalidates)}


@simulation_bp.route("/<simulation_id>/agent-selection", methods=["GET"])
def get_agent_selection(simulation_id: str):
    """
    The agent selection: every eligible graph entity with its active flag and roles

    Reading never persists anything (an unsaved selection has every entity inactive).

    Returns:
        {
            "success": true,
            "data": {
                "entities": [{"entity_uuid", "name", "entity_type", "description", "active",
                              "simulation_role", "evidence_role", "duplicate_of", "possible_duplicates", ...}],
                "role_catalog": [{"id", "label", "default_evidence_role"}],
                "evidence_roles": [{"id", "label"}],
                "summary": {"active_agents": 25, "graph_entities": 67, "context_entities": 42,
                            "by_simulation_role": [...], "by_evidence_role": [...], "by_entity_type": {...}},
                "confirmed": true,
                "population": {"prepared", "in_sync", "legacy_prepared", "preparing", "prepare_task_id", ...}
            }
        }
    """
    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        return jsonify({"success": True, "data": _selection_response(simulation_id, state)})
    except Exception as e:
        logger.error(f"Failed to get agent selection: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/agent-selection", methods=["PUT", "POST"])
def save_agent_selection(simulation_id: str):
    """
    Save the agent selection

    Request (JSON):
        {
            "entities": [{"entity_uuid": "...", "active": true,
                          "simulation_role": "prospect", "evidence_role": "target_consumer"}],
            "role_catalog": [{"id": "prospect", "label": "Prospect", "default_evidence_role": "target_consumer"}],
            "confirmed": true      // confirmed selections can be prepared
        }

    Entities missing from the request keep their previous decision. Changing the
    active population of a prepared simulation invalidates the preparation
    (preparation_invalidated=true); existing runs keep their own snapshot.

    Returns:
        {"success": true, "data": {...same as GET..., "preparation_invalidated": false}}
    """
    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        if _contract_locked(state):
            return _contract_locked_response()
        try:
            result = _save_selection(simulation_id, state, request.get_json() or {})
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"success": False, "error": str(e)}), 409
        data = _selection_response(simulation_id, state, result["document"])
        data["preparation_invalidated"] = result["preparation_invalidated"]
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Failed to save agent selection: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/experiment-plan", methods=["GET"])
def get_experiment_plan(simulation_id: str):
    """
    Private experiment plan (run-scoped stages delivered privately to agents)

    Without a saved plan, the suggestion derived from the assessment brief is
    returned (saved=false); it is what prepare uses until a plan is saved.

    Returns:
        {"success": true, "data": {"stages": [...], "saved": false,
         "condition_types": [...], "completion_types": [...]}}
    """
    from ..services.research_instructions import (
        COMPLETION_TYPES, CONDITION_TYPES, ExperimentPlanStore, load_project_brief, plan_from_brief,
    )

    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        plan = ExperimentPlanStore.load(simulation_id)
        locked = _contract_locked(state)
        if locked:
            # The Run Contract owns the stages: never derive them from brief prose
            applied = RunContractStore.load(simulation_id) or {}
            if plan is None or plan.get("contract_hash") != applied.get("contract_hash"):
                plan = {"stages": [], "saved": False}
        elif plan is None:
            brief = load_project_brief(state.project_id)
            plan = plan_from_brief(brief["brief_text"], brief["persona_ids"])
        return jsonify({"success": True, "data": {
            **plan,
            "locked_by_contract": locked,
            "condition_types": list(CONDITION_TYPES),
            "completion_types": list(COMPLETION_TYPES),
        }})
    except Exception as e:
        logger.error(f"Failed to get experiment plan: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/experiment-plan", methods=["PUT", "POST"])
def save_experiment_plan(simulation_id: str):
    """
    Save the private experiment plan

    Request (JSON):
        {
            "stages": [{
                "id": "exposure", "name": "Neutral product exposure", "trigger_round": 5,
                "target": {"simulation_roles": ["prospect"], "agent_ids": [], "persona_ids": [], "evidence_roles": []},
                "private_instruction": "During your research you encounter Product X ... Decide whether you ignore, inspect, try or reject it.",
                "condition": {"type": "not_mentioned", "terms": ["Product X"]},
                "completion": {"type": "mentioned", "terms": ["Product X"]}
            }]
        }

    Stage instructions are delivered privately into the targeted agents' memory at
    the trigger round; they are never posted. The plan is resolved into every new
    run's directory when the run is created (existing runs keep theirs).

    Returns:
        {"success": true, "data": {"stages": [...], "saved": true, "applies_to": "next_run"}}
    """
    from ..services.research_instructions import ExperimentPlanStore, validate_plan

    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        if _contract_locked(state):
            return _contract_locked_response()
        selection = AgentSelectionStore.load(simulation_id) or {}
        catalog = selection.get("role_catalog") or DEFAULT_ROLE_CATALOG
        try:
            plan = validate_plan(request.get_json() or {}, [r["id"] for r in catalog])
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        # Stages are run-scoped: the plan is resolved into each new run's directory
        # at run creation; runs that already exist keep their own plan.
        saved = ExperimentPlanStore.save(simulation_id, plan)
        return jsonify({"success": True, "data": {**saved, "applies_to": "next_run"}})
    except Exception as e:
        logger.error(f"Failed to save experiment plan: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/source-requirements", methods=["GET"])
def get_source_requirements(simulation_id: str):
    """
    Required source coverage of this simulation and the current coverage matrix

    Returns:
        {"success": true, "data": {"requirements": {...} | null, "coverage": {"ok": bool,
         "documents": [{"id", "label", "files", "ok", "sections": [{"id", "status", "missing_terms"}]}],
         "missing": ["product_truth:pricing"]} | null}}
    """
    from ..services.source_requirements import SourceRequirementsStore, simulation_coverage

    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        return jsonify({"success": True, "data": {
            "requirements": SourceRequirementsStore.load(simulation_id),
            "coverage": simulation_coverage(simulation_id, state.project_id),
        }})
    except Exception as e:
        logger.error(f"Failed to get source requirements: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/source-requirements", methods=["PUT", "POST"])
def save_source_requirements(simulation_id: str):
    """
    Define the authoritative sources an experiment needs (prepare/start refuse until covered)

    Request (JSON):
        {"documents": [{"id": "product_truth", "label": "Product Truth",
                        "filename_pattern": "product[-_ ]?truth", "authoritative": true,
                        "sections": [{"id": "pricing", "label": "Pricing",
                                      "required_terms": ["Trip Pass"], "min_chars": 200}]}]}

    Sections are Markdown headings with an explicit anchor, e.g. "## Pricing {#pricing}".

    Returns:
        {"success": true, "data": {"requirements": {...}, "coverage": {...}}}
    """
    from ..services.source_requirements import SourceRequirementsStore, simulation_coverage

    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        if _contract_locked(state):
            return _contract_locked_response()
        try:
            spec = SourceRequirementsStore.save(simulation_id, request.get_json() or {})
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        return jsonify({"success": True, "data": {
            "requirements": spec,
            "coverage": simulation_coverage(simulation_id, state.project_id),
        }})
    except Exception as e:
        logger.error(f"Failed to save source requirements: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/preflight", methods=["GET"])
def get_preflight(simulation_id: str):
    """
    Run preflight: the persisted configuration checked against the project's Run Contract

    Query parameters:
        max_rounds, platform: optional values of a planned start request

    Returns:
        {"success": true, "data": {
            "mode": "contract" | "standard",
            "contract": {"present", "id", "schema", "hash", "valid", "applied", "errors"},
            "population": {"expected", "resolved", "active", "missing_persona_ids",
                           "duplicate_persona_ids", "unexpected_active", "role_mismatches",
                           "evidence_role_mismatches", "entities": [...], "ok"},
            "roles": {"target_consumers", "influencer_context", "excluded_active", "ok"},
            "experiment_plan": {"expected_stages", "saved_stages", "mismatches",
                                "unresolved_stage_targets", "ok"},
            "rounds": {"expected", "ok"},
            "sources": {"requirements_present", "coverage_ok", "missing", "documents", "ok"},
            "preparation": {"prepared", "selection_hash_matches", "roster_matches_contract",
                            "profile_count_twitter", "profile_count_reddit",
                            "profile_persona_ids_match", "config_agent_count", "ok"},
            "ready_to_prepare": bool, "ready_to_start": bool, "errors": [...],
            "fingerprint": "..."}}
        For simulations without a Run Contract: mode="standard", contract.present=false.
    """
    try:
        requested = {}
        if request.args.get("max_rounds") not in (None, ""):
            try:
                requested["max_rounds"] = int(request.args["max_rounds"])
            except ValueError:
                return jsonify({"success": False, "error": t("api.maxRoundsInvalid")}), 400
        if request.args.get("platform"):
            requested["platform"] = request.args["platform"]
        try:
            data = RunContractPreflight.evaluate(simulation_id, requested)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 404
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Failed to evaluate preflight: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@simulation_bp.route("/<simulation_id>/run-contract/apply", methods=["POST"])
def apply_run_contract(simulation_id: str):
    """
    Apply the project's Run Contract to the simulation (deterministic, never calls an LLM)

    Persists agent_selection.json (contract personas active with the contract roles,
    every other graph entity context-only; confirmed only when every persona id
    resolved to exactly one graph entity), experiment_plan.json,
    source_requirements.json and run_contract.json, then returns the preflight.
    Preparation is never started.

    Returns:
        {"success": true, "data": {"applied": true, "confirmed": bool,
         "preparation_invalidated": bool, "errors": [...], "preflight": {...}}}
    """
    try:
        state = SimulationManager().get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": t("api.simulationNotFound", id=simulation_id)}), 404
        with RunStore.lock(simulation_id):
            if _prepare_task_running(state):
                return jsonify({"success": False, "error": "Preparation is in progress; wait for it to finish"}), 409
            if _active_run_exists(simulation_id):
                return jsonify({"success": False, "error": "A run is executing; stop it before applying the Run Contract"}), 409
            try:
                result = apply_contract(simulation_id)
            except ValueError as e:
                return jsonify({"success": False, "error": str(e),
                                "preflight": RunContractPreflight.evaluate(simulation_id)}), 400
            result["preflight"] = RunContractPreflight.evaluate(simulation_id)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Failed to apply run contract: {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


def _preflight_refusal(preflight: Dict[str, Any], action: str):
    """400 response carrying the complete preflight"""
    errors = preflight.get("errors") or ["preflight failed"]
    return jsonify({
        "success": False,
        "error": f"Run preflight failed, {action} refused: " + "; ".join(errors[:12])
        + (f" (+{len(errors) - 12} more)" if len(errors) > 12 else ""),
        "preflight": preflight,
    }), 400


def _coverage_refusal(simulation_id: str, project_id: str):
    """400 response when required sources are incomplete, else None"""
    from ..services.source_requirements import coverage_error, simulation_coverage

    coverage = simulation_coverage(simulation_id, project_id)
    error = coverage_error(coverage)
    if error:
        return jsonify({"success": False, "error": error, "source_coverage": coverage}), 400
    return None


# ============== Simulation management endpoints ==============


@simulation_bp.route("/create", methods=["POST"])
def create_simulation():
    """
    Create a new simulation

    Note: Parameters such as max_rounds are intelligently generated by the LLM; no manual setup is required

    Request (JSON):
        {
            "project_id": "proj_xxxx",      // required
            "graph_id": "mirofish_xxxx",    // optional; if not provided, obtained from the project
            "enable_twitter": true,          // optional, default true
            "enable_reddit": true            // optional, default true
        }

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}

        project_id = data.get("project_id")
        if not project_id:
            return jsonify({"success": False, "error": t("api.requireProjectId")}), 400

        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify(
                {"success": False, "error": t("api.projectNotFound", id=project_id)}
            ), 404

        graph_id = data.get("graph_id") or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": t("api.graphNotBuilt")}), 400

        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
        )

        return jsonify({"success": True, "data": state.to_dict()})

    except Exception as e:
        logger.error(f"Failed to create simulation: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Check whether the simulation has finished preparing

    Checks:
    1. state.json exists and status is "ready"
    2. Required files exist: reddit_profiles.json, twitter_profiles.csv, simulation_config.json

    Note: The run scripts (run_*.py) remain in the backend/scripts/ directory and are no longer copied to the simulation directory

    Args:
        simulation_id: Simulation ID

    Returns:
        (is_prepared: bool, info: dict)
    """
    import os
    from ..config import Config

    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

    # Check if directory exists
    if not os.path.exists(simulation_dir):
        return False, {"reason": t("api.simDirNotExist")}

    # Required files list (excluding scripts; scripts are in backend/scripts/)
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv",
    ]

    # Check if files exist
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)

    if missing_files:
        return False, {
            "reason": t("api.simMissingFiles"),
            "missing_files": missing_files,
            "existing_files": existing_files,
        }

    # Check status in state.json
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        import json

        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)

        # Detailed log
        logger.debug(
            f"Checking simulation preparation status: {simulation_id}, status={status}, config_generated={config_generated}"
        )

        # If config_generated=True and files exist, consider preparation complete
        # The following statuses all indicate preparation work is done:
        # - ready: preparation complete, can run
        # - preparing: if config_generated=True, preparation is done
        # - running: currently running, preparation was done earlier
        # - completed: run completed, preparation was done earlier
        # - stopped: stopped, preparation was done earlier
        # "failed" is a failed preparation: never startable, even if an older
        # simulation_config.json is still on disk.
        prepared_statuses = [
            "ready",
            "preparing",
            "running",
            "paused",
            "completed",
            "stopped",
        ]
        profiles_generated = state_data.get("profiles_generated", config_generated)
        # A saved agent selection must match the population the profiles were
        # prepared for; simulations prepared before agent selection (no selection,
        # no roster) stay startable as before.
        selection = AgentSelectionStore.load(simulation_id)
        roster = AgentSelectionStore.load_roster(simulation_id)
        if selection is not None and not (
            selection.get("confirmed") and roster
            and roster.get("selection_hash") == selection.get("selection_hash")
        ):
            return False, {
                "reason": "The agent selection changed since preparation (or is unconfirmed); prepare again",
                "status": status,
                "config_generated": config_generated,
                "selection_out_of_sync": True,
            }

        if status in prepared_statuses and config_generated and profiles_generated:
            # Get file statistics
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            config_file = os.path.join(simulation_dir, "simulation_config.json")

            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, "r", encoding="utf-8") as f:
                    profiles_data = json.load(f)
                    profiles_count = (
                        len(profiles_data) if isinstance(profiles_data, list) else 0
                    )

            # If status is preparing but files are done, auto-update status to ready
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    from datetime import datetime

                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, "w", encoding="utf-8") as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"Auto-updating simulation status: {simulation_id} preparing -> ready"
                    )
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Failed to auto-update status: {e}")

            logger.info(
                f"Simulation {simulation_id} check result: preparation complete (status={status}, config_generated={config_generated})"
            )
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "active_agents_count": state_data.get("active_agents_count") or profiles_count,
                "context_entities_count": state_data.get("context_entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files,
            }
        else:
            logger.warning(
                f"Simulation {simulation_id} check result: not prepared (status={status}, config_generated={config_generated})"
            )
            return False, {
                "reason": t(
                    "api.simNotPrepared",
                    status=status,
                    config_generated=config_generated,
                ),
                "status": status,
                "config_generated": config_generated,
            }

    except Exception as e:
        return False, {"reason": t("api.simStateReadFailed", error=str(e))}


@simulation_bp.route("/prepare", methods=["POST"])
def prepare_simulation():
    """
    Prepare the simulation environment (async task; LLM intelligently generates all parameters)

    This is a time-consuming operation. The endpoint returns a task_id immediately;
    use GET /api/simulation/prepare/status to query progress

    Features:
    - Automatically detects completed preparation work to avoid duplicate generation
    - If already prepared, returns the existing result directly
    - Supports forced regeneration (force_regenerate=true)

    Steps:
    1. Check whether preparation work has already been completed
    2. Read and filter entities from the graph store
    3. Generate an OASIS Agent Profile for each entity (with retry mechanism)
    4. LLM intelligently generates the simulation configuration (with retry mechanism)
    5. Save configuration files and preset scripts

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",                   // required, simulation ID
            "entity_types": ["Student", "PublicFigure"],  // optional, specified entity types
            "use_llm_for_profiles": true,                 // optional, whether to use LLM to generate personas
            "parallel_profile_count": 5,                  // optional, number of personas generated in parallel, default 5
            "force_regenerate": false,                    // optional, force regeneration, default false
            "agent_selection": {...}                      // optional, saved + confirmed before preparing
                                                          // (same body as PUT /<id>/agent-selection)
        }

    Only the entities marked active in the confirmed agent selection become OASIS
    agents; without a confirmed selection the request fails with 400. A prepare
    request while a preparation is running returns that task (no second task).

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // returned for new tasks
                "status": "preparing|ready",
                "message": "Preparation task started|Preparation already completed",
                "already_prepared": true|false    // whether already prepared
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        # Check-and-set of the preparation runs under the simulation lock, so a
        # start can never snapshot files a new preparation is rewriting
        with RunStore.lock(simulation_id):
            return _prepare_locked(data, simulation_id)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404

    except Exception as e:
        logger.error(f"Failed to start preparation task: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _prepare_locked(data: Dict[str, Any], simulation_id: str):
    """Body of POST /prepare (caller holds RunStore.lock(simulation_id))"""
    import threading
    from ..models.task import TaskManager, TaskStatus

    manager = SimulationManager()
    state = manager.get_simulation(simulation_id)

    if not state:
        return jsonify(
            {
                "success": False,
                "error": t("api.simulationNotFound", id=simulation_id),
            }
        ), 404

    # A Run Contract owns the population: no selection may come with the request
    contract_mode = _contract_locked(state)
    if contract_mode and data.get("agent_selection") is not None:
        return _contract_locked_response()

    # An explicit selection in the request is saved (and confirmed) first
    if data.get("agent_selection") is not None:
        try:
            _save_selection(simulation_id, state, data["agent_selection"], confirmed_default=True)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"success": False, "error": str(e)}), 409
        state = manager.get_simulation(simulation_id)

    # A running preparation is attached to, never duplicated (reload-safe)
    if _prepare_task_running(state):
        return jsonify({"success": True, "data": {
            "simulation_id": simulation_id,
            "task_id": state.prepare_task_id,
            "status": "preparing",
            "message": t("api.prepareStarted"),
            "already_prepared": False,
            "already_running": True,
            "expected_entities_count": state.active_agents_count,
            "graph_entities_count": state.entities_count,
        }})

    # Check if force regenerate
    force_regenerate = data.get("force_regenerate", False)
    logger.info(
        f"Processing /prepare request: simulation_id={simulation_id}, force_regenerate={force_regenerate}"
    )

    # Contracted simulation: the backend preflight decides, before any task exists
    preflight = RunContractPreflight.evaluate(simulation_id) if contract_mode else None
    if contract_mode and not force_regenerate and preflight["ready_to_start"]:
        return jsonify({"success": True, "data": {
            "simulation_id": simulation_id,
            "status": "ready",
            "message": t("api.alreadyPrepared"),
            "already_prepared": True,
            "preflight": preflight,
        }})
    if contract_mode and not preflight["ready_to_prepare"]:
        get_logger("mirofish.api.simulation").warning(
            f"[{simulation_id}] Prepare refused by run preflight: {preflight['errors']}"
        )
        return _preflight_refusal(preflight, "preparation")

    # Check if already prepared (avoid duplicate generation)
    if not force_regenerate and not contract_mode:
        logger.debug(f"Checking simulation {simulation_id}  is already prepared...")
        is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
        logger.debug(
            f"Check result: is_prepared={is_prepared}, prepare_info={prepare_info}"
        )
        if is_prepared:
            logger.info(f"Simulation {simulation_id} already prepared, skipping duplicate generation")
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": t("api.alreadyPrepared"),
                        "already_prepared": True,
                        "prepare_info": prepare_info,
                    },
                }
            )
        else:
            logger.info(f"Simulation {simulation_id} not prepared, will start preparation task")

    # The population must be chosen explicitly: graph entity != agent
    selection = AgentSelectionStore.load(simulation_id)
    active_selected = sum(1 for e in (selection or {}).get("entities", []) if e.get("active"))
    if not selection or not selection.get("confirmed") or active_selected == 0:
        return jsonify({
            "success": False,
            "error": t("api.agentSelectionRequired"),
            "agent_selection_required": True,
        }), 400

    # Required authoritative sources (e.g. a complete Product Truth) must be present
    refusal = _coverage_refusal(simulation_id, state.project_id)
    if refusal:
        return refusal

    # Get required info from project
    project = ProjectManager.get_project(state.project_id)
    if not project:
        return jsonify(
            {
                "success": False,
                "error": t("api.projectNotFound", id=state.project_id),
            }
        ), 404

    # Get simulation requirement
    simulation_requirement = project.simulation_requirement or ""
    if not simulation_requirement:
        return jsonify(
            {"success": False, "error": t("api.projectMissingRequirement")}
        ), 400

    # Context documents only: the assessment brief is research design, never
    # simulation content (its instructions are delivered privately per run)
    from ..services.research_instructions import load_project_brief

    brief = load_project_brief(state.project_id)
    document_text = brief["context_text"]

    # A contracted population is never narrowed by entity types
    entity_types_list = None if contract_mode else data.get("entity_types")
    use_llm_for_profiles = data.get("use_llm_for_profiles", True)
    parallel_profile_count = data.get("parallel_profile_count", 5)

    # ========== Synchronously get entity count (before background task starts) ==========
    # So frontend can immediately get expected Agent total after calling prepare
    try:
        logger.info(f"Synchronously getting entity count: graph_id={state.graph_id}")
        reader = EntityReader()
        # Quick entity read (no edge info needed, just count)
        filtered_preview = reader.filter_defined_entities(
            graph_id=state.graph_id,
            defined_entity_types=entity_types_list,
            enrich_with_edges=False,  # Don't fetch edge info, speed up
        )
        # Save entity count to state (for frontend immediate access)
        state.entities_count = filtered_preview.filtered_count
        state.entity_types = list(filtered_preview.entity_types)
        state.active_agents_count = active_selected
        state.context_entities_count = max(filtered_preview.filtered_count - active_selected, 0)
        logger.info(
            f"Population: {active_selected} active agents / {filtered_preview.filtered_count} graph entities, "
            f"types: {filtered_preview.entity_types}"
        )
    except Exception as e:
        logger.warning(f"Synchronously getting entity count failed (will retry in background task): {e}")
        # Failure doesn't affect subsequent flow, background task will retry

    # Create async task
    task_manager = TaskManager()
    task_id = task_manager.create_task(
        task_type="simulation_prepare",
        metadata={"simulation_id": simulation_id, "project_id": state.project_id},
    )

    # Update simulation state (including pre-fetched entity count). The
    # previous preparation is invalid until the new one completes.
    state.status = SimulationStatus.PREPARING
    state.profiles_generated = False
    state.config_generated = False
    state.error = None
    state.prepare_task_id = task_id
    manager._save_simulation_state(state)

    # Capture locale before spawning background thread
    current_locale = get_locale()

    # Define background task
    def run_prepare():
        set_locale(current_locale)
        try:
            task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=0,
                message=t("progress.startPreparingEnv"),
            )

            # Prepare simulation (with progress callback)
            # Store phase progress details
            stage_details = {}

            def progress_callback(stage, progress, message, **kwargs):
                # Calculate total progress
                stage_weights = {
                    "reading": (0, 20),  # 0-20%
                    "generating_profiles": (20, 70),  # 20-70%
                    "generating_config": (70, 90),  # 70-90%
                    "copying_scripts": (90, 100),  # 90-100%
                }

                start, end = stage_weights.get(stage, (0, 100))
                current_progress = int(start + (end - start) * progress / 100)

                # Build detailed progress info
                stage_names = {
                    "reading": t("progress.readingGraphEntities"),
                    "generating_profiles": t("progress.generatingProfiles"),
                    "generating_config": t("progress.generatingSimConfig"),
                    "copying_scripts": t("progress.preparingScripts"),
                }

                stage_index = (
                    list(stage_weights.keys()).index(stage) + 1
                    if stage in stage_weights
                    else 1
                )
                total_stages = len(stage_weights)

                # Update phase details
                stage_details[stage] = {
                    "stage_name": stage_names.get(stage, stage),
                    "stage_progress": progress,
                    "current": kwargs.get("current", 0),
                    "total": kwargs.get("total", 0),
                    "item_name": kwargs.get("item_name", ""),
                }

                # Build detailed progress info
                detail = stage_details[stage]
                progress_detail_data = {
                    "current_stage": stage,
                    "current_stage_name": stage_names.get(stage, stage),
                    "stage_index": stage_index,
                    "total_stages": total_stages,
                    "stage_progress": progress,
                    "current_item": detail["current"],
                    "total_items": detail["total"],
                    "item_description": message,
                }

                # Build concise message
                if detail["total"] > 0:
                    detailed_message = (
                        f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                        f"{detail['current']}/{detail['total']} - {message}"
                    )
                else:
                    detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"

                task_manager.update_task(
                    task_id,
                    progress=current_progress,
                    message=detailed_message,
                    progress_detail=progress_detail_data,
                )

            result_state = manager.prepare_simulation(
                simulation_id=simulation_id,
                simulation_requirement=simulation_requirement,
                document_text=document_text,
                defined_entity_types=entity_types_list,
                use_llm_for_profiles=use_llm_for_profiles,
                progress_callback=progress_callback,
                parallel_profile_count=parallel_profile_count,
                brief=brief,
            )

            if contract_mode:
                # Verify the frozen roster, both profile files and the config against
                # the Run Contract; any mismatch makes the preparation non-startable
                with RunStore.lock(simulation_id):
                    verified = RunContractPreflight.evaluate(simulation_id)
                    if not verified["ready_to_start"]:
                        raise ValueError(
                            "Post-prepare verification against the Run Contract failed: "
                            + "; ".join(verified["errors"])
                        )

            # Task complete
            task_manager.complete_task(
                task_id, result=result_state.to_simple_dict()
            )

        except Exception as e:
            logger.error(f"Simulation preparation failed: {str(e)}")
            task_manager.fail_task(task_id, str(e))

            # Update simulation state to failed
            state = manager.get_simulation(simulation_id)
            if state:
                state.status = SimulationStatus.FAILED
                state.error = str(e)
                manager._save_simulation_state(state)

    # Start background thread
    thread = threading.Thread(target=run_prepare, daemon=True)
    thread.start()

    return jsonify(
        {
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": t("api.prepareStarted"),
                "already_prepared": False,
                "expected_entities_count": state.active_agents_count,  # Expected Agent total (active only)
                "graph_entities_count": state.entities_count,
                "context_entities_count": state.context_entities_count,
                "entity_types": state.entity_types,  # Entity type list
            },
        }
    )


@simulation_bp.route("/prepare/status", methods=["POST"])
def get_prepare_status():
    """
    Query the progress of a preparation task

    Supports two query modes:
    1. Query an in-progress task's progress via task_id
    2. Check whether preparation work has already been completed via simulation_id

    Request (JSON):
        {
            "task_id": "task_xxxx",          // optional, the task_id returned by prepare
            "simulation_id": "sim_xxxx"      // optional, simulation ID (used to check completed preparation)
        }

    Returns:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // whether preparation is already complete
                "prepare_info": {...}            // detailed info when already prepared
            }
        }
    """
    from ..models.task import TaskManager

    try:
        data = request.get_json() or {}

        task_id = data.get("task_id")
        simulation_id = data.get("simulation_id")

        # If simulation_id provided, check if already prepared first
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "ready",
                            "progress": 100,
                            "message": t("api.alreadyPrepared"),
                            "already_prepared": True,
                            "prepare_info": prepare_info,
                        },
                    }
                )

        # If no task_id, return error
        if not task_id:
            if simulation_id:
                # Has simulation_id but not prepared
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "not_started",
                            "progress": 0,
                            "message": t("api.notStartedPrepare"),
                            "already_prepared": False,
                        },
                    }
                )
            return jsonify(
                {"success": False, "error": t("api.requireTaskOrSimId")}
            ), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            # Task not found, but if simulation_id provided, check if already prepared
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify(
                        {
                            "success": True,
                            "data": {
                                "simulation_id": simulation_id,
                                "task_id": task_id,
                                "status": "ready",
                                "progress": 100,
                                "message": t("api.taskCompletedPrepared"),
                                "already_prepared": True,
                                "prepare_info": prepare_info,
                            },
                        }
                    )

            return jsonify(
                {"success": False, "error": t("api.taskNotFound", id=task_id)}
            ), 404

        task_dict = task.to_dict()
        task_dict["already_prepared"] = False

        return jsonify({"success": True, "data": task_dict})

    except Exception as e:
        logger.error(f"Failed to query task status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@simulation_bp.route("/<simulation_id>", methods=["GET"])
def get_simulation(simulation_id: str):
    """Get simulation state"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {
                    "success": False,
                    "error": t("api.simulationNotFound", id=simulation_id),
                }
            ), 404

        result = state.to_dict()

        # If simulation is ready, attach run instructions
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Failed to get simulation status: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/list", methods=["GET"])
def list_simulations():
    """
    List all simulations

    Query parameters:
        project_id: Filter by project ID (optional)
    """
    try:
        project_id = request.args.get("project_id")

        manager = SimulationManager()
        simulations = manager.list_simulations(project_id=project_id)

        return jsonify(
            {
                "success": True,
                "data": [s.to_dict() for s in simulations],
                "count": len(simulations),
            }
        )

    except Exception as e:
        logger.error(f"Failed to list simulations: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _get_report_id_for_simulation(simulation_id: str) -> Optional[str]:
    """
    Get the latest report_id for a simulation

    Iterates through the reports directory to find reports whose simulation_id matches;
    if there are multiple, returns the latest one (sorted by created_at)

    Args:
        simulation_id: Simulation ID

    Returns:
        report_id or None
    """
    import json
    from datetime import datetime

    # Reports directory path: backend/uploads/reports
    # __file__ is app/api/simulation.py, need to go up two levels to backend/
    reports_dir = os.path.join(os.path.dirname(__file__), "../../uploads/reports")
    if not os.path.exists(reports_dir):
        return None

    matching_reports = []

    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue

            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append(
                        {
                            "report_id": meta.get("report_id"),
                            "created_at": meta.get("created_at", ""),
                            "status": meta.get("status", ""),
                        }
                    )
            except Exception:
                continue

        if not matching_reports:
            return None

        # Sort by creation time descending, return newest
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")

    except Exception as e:
        logger.warning(f"Finding simulation {simulation_id}  report failed: {e}")
        return None


@simulation_bp.route("/history", methods=["GET"])
def get_simulation_history():
    """
    Get the history simulation list (with project details)

    Used for the homepage history project display; returns a simulation list enriched with project name, description, and other information

    Query parameters:
        limit: Return count limit (default 20)

    Returns:
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "Wuhan University public opinion analysis",
                    "simulation_requirement": "If Wuhan University releases...",
                    "status": "completed",
                    "entities_count": 68,
                    "profiles_count": 68,
                    "entity_types": ["Student", "Professor", ...],
                    "created_at": "2024-12-10",
                    "updated_at": "2024-12-10",
                    "total_rounds": 120,
                    "current_round": 120,
                    "report_id": "report_xxxx",
                    "version": "v1.0.2"
                },
                ...
            ],
            "count": 7
        }
    """
    try:
        limit = request.args.get("limit", 20, type=int)

        manager = SimulationManager()
        simulations = manager.list_simulations()[:limit]

        # Enhance simulation data, read only from Simulation files
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()

            # Get simulation config info (read simulation_requirement from simulation_config.json)
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get(
                    "simulation_requirement", ""
                )
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get(
                    "total_simulation_hours", 0
                )
                # Recommended rounds (fallback value)
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0)
                    * 60
                    / max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0

            # Get run state of the current run (read user-set actual rounds from run_state.json)
            current_run = RunStore.get_current_run(sim.simulation_id)
            sim_dict["run_id"] = current_run.run_id if current_run else None
            sim_dict["run_status"] = current_run.status.value if current_run else "not_started"
            run_state = (
                SimulationRunner.get_run_state(sim.simulation_id, current_run.run_id)
                if current_run
                else None
            )
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Use user-set total_rounds, fallback to recommended rounds
                sim_dict["total_rounds"] = (
                    run_state.total_rounds
                    if run_state.total_rounds > 0
                    else recommended_rounds
                )
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds

            # Get associated project file list (max 3)
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, "files") and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "Unknown file")}
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []

            # Get associated report_id (find newest report for this simulation)
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)

            # Add version number
            sim_dict["version"] = "v1.0.2"

            # Format date
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""

            enriched_simulations.append(sim_dict)

        return jsonify(
            {
                "success": True,
                "data": enriched_simulations,
                "count": len(enriched_simulations),
            }
        )

    except Exception as e:
        logger.error(f"Failed to get simulation history: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/profiles", methods=["GET"])
def get_simulation_profiles(simulation_id: str):
    """
    Get the simulation's Agent Profile

    Query parameters:
        platform: Platform type (reddit/twitter, default reddit)
    """
    try:
        platform = request.args.get("platform")

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if not state:
            return jsonify(
                {"success": False, "error": t("api.simulationNotFound", id=simulation_id)}
            ), 404
        if platform is None:
            platform = state.get_default_platform()
        if platform not in ("reddit", "twitter"):
            return jsonify({"success": False, "error": f"Unsupported platform: {platform}"}), 400
        profiles = manager.get_profiles(simulation_id, platform=platform)

        return jsonify(
            {
                "success": True,
                "data": {
                    "platform": platform,
                    "count": len(profiles),
                    "profiles": profiles,
                },
            }
        )

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404

    except Exception as e:
        logger.error(f"Failed to get Profile: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/profiles/realtime", methods=["GET"])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    Get the simulation's Agent Profile in real time (used to view progress during generation)

    Differences from the /profiles endpoint:
    - Reads the file directly, without going through SimulationManager
    - Suitable for real-time viewing during generation
    - Returns additional metadata (such as file modification time, whether generation is in progress)

    Query parameters:
        platform: Platform type (reddit/twitter, default reddit)

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // expected total (if any)
                "is_generating": true,  // whether generation is in progress
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "profiles": [...]
            }
        }
    """
    import json
    import csv
    from datetime import datetime

    try:
        platform = request.args.get("platform", "reddit")
        if platform not in ("reddit", "twitter"):
            return jsonify({"success": False, "error": f"Unsupported platform: {platform}"}), 400

        # Get simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify(
                {
                    "success": False,
                    "error": t("api.simulationNotFound", id=simulation_id),
                }
            ), 404

        # Determine file path
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")

        # Check if file exists
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None

        if file_exists:
            # Get file modification time
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                if platform == "reddit":
                    with open(profiles_file, "r", encoding="utf-8") as f:
                        profiles = json.load(f)
                else:
                    with open(profiles_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        profiles = list(reader)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read profiles file (may be writing): {e}")
                profiles = []

        # Check if generating (via state.json); a failed preparation is
        # reported explicitly so the UI can stop polling and show the error
        is_generating = False
        total_expected = None
        status = None
        error = None
        profiles_generated = False

        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    total_expected = state_data.get("active_agents_count") or state_data.get("entities_count")
                    error = state_data.get("error")
                    profiles_generated = state_data.get(
                        "profiles_generated", state_data.get("config_generated", False)
                    )
            except Exception:
                pass

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "platform": platform,
                    "count": len(profiles),
                    "total_expected": total_expected,
                    "is_generating": is_generating,
                    "status": status,
                    "error": error,
                    "profiles_generated": profiles_generated,
                    "file_exists": file_exists,
                    "file_modified_at": file_modified_at,
                    "profiles": profiles,
                },
            }
        )

    except Exception as e:
        logger.error(f"Real-time Profile fetch failed: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config/realtime", methods=["GET"])
def get_simulation_config_realtime(simulation_id: str):
    """
    Get the simulation configuration in real time (used to view progress during generation)

    Differences from the /config endpoint:
    - Reads the file directly, without going through SimulationManager
    - Suitable for real-time viewing during generation
    - Returns additional metadata (such as file modification time, whether generation is in progress)
    - Can return partial info even before the configuration is fully generated

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // whether generation is in progress
                "generation_stage": "generating_config",  // current generation stage
                "config": {...}  // configuration content (if it exists)
            }
        }
    """
    import json
    from datetime import datetime

    try:
        # Get simulation directory
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify(
                {
                    "success": False,
                    "error": t("api.simulationNotFound", id=simulation_id),
                }
            ), 404

        # Config file path
        config_file = os.path.join(sim_dir, "simulation_config.json")

        # Check if file exists
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None

        if file_exists:
            # Get file modification time
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to read config file (may be writing): {e}")
                config = None

        # Check if generating (via state.json)
        is_generating = False
        generation_stage = None
        config_generated = False
        profiles_generated = False
        status = None
        error = None

        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    error = state_data.get("error")
                    is_generating = status == "preparing"
                    config_generated = state_data.get("config_generated", False)
                    profiles_generated = state_data.get("profiles_generated", config_generated)

                    # Determine current phase
                    if is_generating:
                        if profiles_generated:
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "failed":
                        generation_stage = "failed"
                    elif status == "ready" or config_generated:
                        generation_stage = "completed"
            except Exception:
                pass

        # Build return data
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "status": status,
            "error": error,
            "profiles_generated": profiles_generated,
            "config_generated": config_generated,
            # A config file left over from an earlier preparation is not valid
            # while the current preparation has not (successfully) finished
            "config": config if config_generated else None,
        }

        # If config exists, extract key statistics
        if config and config_generated:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get(
                    "total_simulation_hours"
                ),
                "initial_posts_count": len(
                    config.get("event_config", {}).get("initial_posts", [])
                ),
                "hot_topics_count": len(
                    config.get("event_config", {}).get("hot_topics", [])
                ),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model"),
            }

        return jsonify({"success": True, "data": response_data})

    except Exception as e:
        logger.error(f"Failed to get Config in real-time: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config", methods=["GET"])
def get_simulation_config(simulation_id: str):
    """
    Get the simulation configuration (the complete configuration intelligently generated by the LLM)

    Returns:
        - time_config: Time configuration (simulation duration, rounds, peak/off-peak periods)
        - agent_configs: Activity configuration for each Agent (activity level, posting frequency, stance, etc.)
        - event_config: Event configuration (initial posts, hot topics)
        - platform_configs: Platform configuration
        - generation_reasoning: LLM's configuration reasoning explanation
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)

        if not config:
            return jsonify({"success": False, "error": t("api.configNotFound")}), 404

        return jsonify({"success": True, "data": config})

    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config/download", methods=["GET"])
def download_simulation_config(simulation_id: str):
    """Download the simulation configuration file"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            return jsonify(
                {"success": False, "error": t("api.configFileNotFound")}
            ), 404

        return send_file(
            config_path, as_attachment=True, download_name="simulation_config.json"
        )

    except Exception as e:
        logger.error(f"Failed to download config: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/script/<script_name>/download", methods=["GET"])
def download_simulation_script(script_name: str):
    """
    Download a simulation run script file (generic scripts, located in backend/scripts/)

    Possible values for script_name:
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # Scripts are in backend/scripts/ directory
        scripts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scripts")
        )

        # Validate script name
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py",
            "run_parallel_simulation.py",
            "action_logger.py",
        ]

        if script_name not in allowed_scripts:
            return jsonify(
                {
                    "success": False,
                    "error": t(
                        "api.unknownScript", name=script_name, allowed=allowed_scripts
                    ),
                }
            ), 400

        script_path = os.path.join(scripts_dir, script_name)

        if not os.path.exists(script_path):
            return jsonify(
                {
                    "success": False,
                    "error": t("api.scriptFileNotFound", name=script_name),
                }
            ), 404

        return send_file(script_path, as_attachment=True, download_name=script_name)

    except Exception as e:
        logger.error(f"Failed to download script: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Profile Generation API (standalone) ==============


@simulation_bp.route("/generate-profiles", methods=["POST"])
def generate_profiles():
    """
    Directly generate an OASIS Agent Profile from the graph (without creating a simulation)

    Request (JSON):
        {
            "graph_id": "mirofish_xxxx",     // required
            "entity_types": ["Student"],      // optional
            "use_llm": true,                  // optional
            "platform": "reddit"              // optional
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")
        if not graph_id:
            return jsonify({"success": False, "error": t("api.requireGraphId")}), 400

        entity_types = data.get("entity_types")
        use_llm = data.get("use_llm", True)
        platform = data.get("platform", "reddit")

        reader = EntityReader()
        filtered = reader.filter_defined_entities(
            graph_id=graph_id, defined_entity_types=entity_types, enrich_with_edges=True
        )

        if filtered.filtered_count == 0:
            return jsonify(
                {"success": False, "error": t("api.noMatchingEntities")}
            ), 400

        generator = OasisProfileGenerator()
        profiles = generator.generate_profiles_from_entities(
            entities=filtered.entities, use_llm=use_llm
        )

        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]

        return jsonify(
            {
                "success": True,
                "data": {
                    "platform": platform,
                    "entity_types": list(filtered.entity_types),
                    "count": len(profiles_data),
                    "profiles": profiles_data,
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to generate Profile: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Simulation Run Control API ==============


@simulation_bp.route("/start", methods=["POST"])
def start_simulation():
    """
    Start a simulation run (idempotent)

    Every execution is a separate run with its own immutable directory
    (uploads/simulations/<simulation_id>/runs/<run_id>/). A plain start never
    replaces or deletes an existing run.

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",          // required, simulation ID
            "platform": "parallel",                // optional: twitter / reddit / parallel (default)
            "max_rounds": 100,                     // optional: max simulation rounds
            "enable_graph_memory_update": false,   // optional: kept for compatibility (not applied)
            "new_run": false,                      // optional: explicit restart -> create a NEW run, keep the old one
            "force": false                         // deprecated and ignored: never deletes or restarts anything
        }

    Behaviour:
        - no run yet                    -> create run, start it            (started=true)
        - current run running           -> nothing happens                 (started=false, already_running=true)
        - current run finished          -> nothing happens                 (started=false, run_exists=true)
        - new_run=true                  -> stop the old run if it is still active (evidence kept),
                                           create a new run and start it   (started=true)

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "run_id": "run_20250101120000_ab12cd",
                "run_status": "running",
                "started": true,
                "already_running": false,
                "run_exists": false,
                "previous_run_id": null,
                "runner_status": "running",
                "process_pid": 12345,
                ...
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        platform = data.get("platform", "parallel")
        max_rounds = data.get("max_rounds")  # Optional: max simulation rounds
        enable_graph_memory_update = bool(data.get("enable_graph_memory_update", False))
        new_run = data.get("new_run", False) is True
        force = data.get("force", False)

        # Validate max_rounds parameter
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify(
                        {"success": False, "error": t("api.maxRoundsPositive")}
                    ), 400
            except (ValueError, TypeError):
                return jsonify(
                    {"success": False, "error": t("api.maxRoundsInvalid")}
                ), 400

        if platform not in ["twitter", "reddit", "parallel"]:
            return jsonify(
                {"success": False, "error": t("api.invalidPlatform", platform=platform)}
            ), 400

        if force:
            logger.warning(
                f"[{simulation_id}] Deprecated 'force' flag ignored: runs are never deleted; "
                f"use new_run=true for an explicit restart"
            )

        with RunStore.lock(simulation_id):
            manager = SimulationManager()
            state = manager.get_simulation(simulation_id)

            if not state:
                return jsonify(
                    {
                        "success": False,
                        "error": t("api.simulationNotFound", id=simulation_id),
                    }
                ), 404

            # Run Contract: run length and platform are fixed by the contract. A
            # differing request is refused before anything else happens.
            contract_mode = _contract_locked(state)
            if contract_mode:
                applied = RunContractStore.load(simulation_id) or {}
                contract_rounds = (applied.get("contract") or {}).get("max_rounds")
                contract_platform = (applied.get("contract") or {}).get("platform")
                if max_rounds is not None and max_rounds != contract_rounds:
                    return jsonify({"success": False, "error": (
                        f"max_rounds {max_rounds} refused: Locked by Run Contract ({contract_rounds} rounds)")}), 400
                if "platform" in data and data.get("platform") != contract_platform:
                    return jsonify({"success": False, "error": (
                        f"platform {data.get('platform')!r} refused: Locked by Run Contract ({contract_platform!r})")}), 400

            current = SimulationRunner.reconcile_run(
                simulation_id, RunStore.get_current_run(simulation_id)
            )

            # Idempotent start: an existing execution is reported, never replaced
            if current is not None and not new_run:
                if current.status == RunStatus.RUNNING or (
                    current.status == RunStatus.NOT_STARTED
                    and SimulationRunner.is_process_alive(simulation_id)
                ):
                    get_run_logger("mirofish.api.simulation", simulation_id, current.run_id).info(
                        "Start requested while run is active: attaching (no new run)"
                    )
                    return jsonify({"success": True, "data": _start_response(
                        simulation_id, current.run_id, started=False, already_running=True
                    )})
                if current.status != RunStatus.NOT_STARTED:
                    get_run_logger("mirofish.api.simulation", simulation_id, current.run_id).info(
                        f"Start requested for existing run with status {current.status.value}: "
                        f"nothing started (use explicit restart for a new run)"
                    )
                    return jsonify({"success": True, "data": _start_response(
                        simulation_id, current.run_id, started=False, run_exists=True
                    )})

            preflight = None
            if contract_mode:
                # The same canonical preflight the UI shows, re-evaluated here under the
                # simulation lock: nothing can change between this check and the snapshot
                preflight = RunContractPreflight.evaluate(
                    simulation_id, {"max_rounds": max_rounds, "platform": data.get("platform")}
                )
                if not preflight["ready_to_start"]:
                    get_logger("mirofish.api.simulation").warning(
                        f"[{simulation_id}] Start refused by run preflight: {preflight['errors']}"
                    )
                    return _preflight_refusal(preflight, "start")
                max_rounds = preflight["rounds"]["expected"]
                platform = preflight["platform"]["expected"]
            else:
                # A new execution requires a prepared simulation
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if not is_prepared:
                    return jsonify(
                        {
                            "success": False,
                            "error": t("api.simNotReady", status=state.status.value),
                        }
                    ), 400

                # ...and complete required sources (checked again when the run snapshots them)
                refusal = _coverage_refusal(simulation_id, state.project_id)
                if refusal:
                    return refusal

            previous_run_id = None
            if current is not None and new_run:
                previous_run_id = current.run_id
                prev_log = get_run_logger("mirofish.api.simulation", simulation_id, current.run_id)
                if current.status == RunStatus.RUNNING:
                    prev_log.info("Explicit restart: stopping active run (its evidence is kept)")
                    try:
                        SimulationRunner.stop_simulation(simulation_id, current.run_id)
                    except Exception as e:
                        prev_log.warning(f"Stop before restart: {e}")
                # A finished run may still have its process alive for interviews
                SimulationRunner.terminate_run_process(simulation_id)

            if SimulationRunner.is_process_alive(simulation_id):
                return jsonify(
                    {"success": False, "error": t("api.simRunningForceHint")}
                ), 409

            # Reuse a never-started run (e.g. its process failed to launch) only when
            # it holds no evidence; otherwise create a fresh run directory
            reusable = current is not None and not new_run and current.status == RunStatus.NOT_STARTED and not current.legacy
            if reusable and contract_mode:
                # A never-started contracted run is reused only when its snapshot was
                # taken from exactly the configuration that passes now
                reusable = (current.snapshot.get("contract") or {}).get("preflight_fingerprint") == preflight["fingerprint"]
            if reusable:
                run_id = current.run_id
            else:
                run_id = RunStore.create_run(
                    simulation_id, platform, max_rounds, previous_run_id=previous_run_id,
                    preflight=preflight,
                ).run_id

            graph_id = state.graph_id
            if not graph_id:
                project = ProjectManager.get_project(state.project_id)
                if project:
                    graph_id = project.graph_id

            run_state = SimulationRunner.start_simulation(
                simulation_id=simulation_id,
                platform=platform,
                max_rounds=max_rounds,
                enable_graph_memory_update=enable_graph_memory_update,
                graph_id=graph_id,
                run_id=run_id,
            )

            # Update simulation state
            state.status = SimulationStatus.RUNNING
            manager._save_simulation_state(state)

        response_data = _start_response(simulation_id, run_id, started=True, run_state=run_state)
        response_data["previous_run_id"] = previous_run_id
        if max_rounds:
            response_data["max_rounds_applied"] = max_rounds
        return jsonify({"success": True, "data": response_data})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"Failed to start simulation: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _run_info(simulation_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Lifecycle info of a run (default: current run) for API responses"""
    record = RunStore.get_run(simulation_id, run_id) if run_id else RunStore.get_current_run(simulation_id)
    record = SimulationRunner.reconcile_run(simulation_id, record)
    runs = RunStore.list_runs(simulation_id)
    if record is None:
        return {
            "run_id": None,
            "run_status": RunStatus.NOT_STARTED.value,
            "runs_count": len(runs),
            "legacy_run": False,
        }
    return {
        "run_id": record.run_id,
        "run_status": record.status.value,
        "runs_count": len(runs),
        "legacy_run": record.legacy,
        "run_created_at": record.created_at,
        "run_completed_at": record.completed_at,
        "latest_report_id": record.latest_report_id,
        "report_ids": record.report_ids,
        "run_error": record.error,
    }


def _start_response(
    simulation_id: str,
    run_id: str,
    started: bool,
    already_running: bool = False,
    run_exists: bool = False,
    run_state: Any = None,
) -> Dict[str, Any]:
    run_state = run_state or SimulationRunner.get_run_state(simulation_id, run_id)
    data = run_state.to_dict() if run_state else {"simulation_id": simulation_id, "runner_status": "idle"}
    data.update(_run_info(simulation_id, run_id))
    data.update({
        "started": started,
        "already_running": already_running,
        "run_exists": run_exists,
        # Kept for older frontends: a start never clears earlier logs anymore
        "force_restarted": False,
        "graph_memory_update_enabled": False,
    })
    return data


@simulation_bp.route("/<simulation_id>/runs/<run_id>/export", methods=["GET"])
def export_run_bundle(simulation_id: str, run_id: str):
    """
    Download the evidence bundle (ZIP) of one run, without a report

    Query parameters:
        report_id: optional report of this run to include

    Returns:
        application/zip attachment
    """
    try:
        import io
        from ..services.run_export import build_run_bundle

        data, filename = build_run_bundle(
            simulation_id, run_id, report_id=request.args.get("report_id")
        )
        return send_file(
            io.BytesIO(data),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/runs", methods=["GET"])
def list_simulation_runs(simulation_id: str):
    """
    List all runs (executions) of a simulation, oldest first

    Returns:
        {
            "success": true,
            "data": {"current_run_id": "run_...", "runs": [RunRecord...]},
            "count": 2
        }
    """
    try:
        runs = RunStore.list_runs(simulation_id)
        runs = [SimulationRunner.reconcile_run(simulation_id, r) for r in runs]
        return jsonify({
            "success": True,
            "data": {
                "current_run_id": runs[-1].run_id if runs else None,
                "runs": [r.to_dict() for r in runs],
            },
            "count": len(runs),
        })
    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/stop", methods=["POST"])
def stop_simulation():
    """
    Stop the simulation

    Request (JSON):
        {
            "simulation_id": "sim_xxxx"  // required, simulation ID
        }

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "stopped",
                "completed_at": "2025-12-01T12:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        with RunStore.lock(simulation_id):
            run_state = SimulationRunner.stop_simulation(simulation_id, data.get("run_id"))

            # Update simulation state (stopped runs keep their evidence; the
            # simulation stays prepared so a new run can be started explicitly)
            manager = SimulationManager()
            state = manager.get_simulation(simulation_id)
            if state:
                state.status = SimulationStatus.STOPPED
                manager._save_simulation_state(state)

        result = run_state.to_dict()
        result.update(_run_info(simulation_id, run_state.run_id))
        return jsonify({"success": True, "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"Failed to stop simulation: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Real-time Status Monitoring API ==============


@simulation_bp.route("/<simulation_id>/run-status", methods=["GET"])
def get_run_status(simulation_id: str):
    """
    Get the real-time running status of a run (for frontend polling)

    Query parameters:
        run_id: Run ID (optional, default: current run)

    The response always includes the persistent lifecycle fields
    run_id / run_status (not_started|running|completed|failed|stopped|reporting|report_completed)
    / runs_count, so a reloaded page can show the real backend state.

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                "total_rounds": 144,
                "progress_percent": 3.5,
                "simulated_hours": 2,
                "total_simulation_hours": 72,
                "twitter_running": true,
                "reddit_running": true,
                "twitter_actions_count": 150,
                "reddit_actions_count": 200,
                "total_actions_count": 350,
                "started_at": "2025-12-01T10:00:00",
                "updated_at": "2025-12-01T10:30:00"
            }
        }
    """
    try:
        run_id = request.args.get("run_id")
        info = _run_info(simulation_id, run_id)
        run_state = (
            SimulationRunner.get_run_state(simulation_id, info["run_id"])
            if info["run_id"]
            else None
        )

        if not run_state:
            data = {
                "simulation_id": simulation_id,
                "runner_status": "idle",
                "current_round": 0,
                "total_rounds": 0,
                "progress_percent": 0,
                "twitter_actions_count": 0,
                "reddit_actions_count": 0,
                "total_actions_count": 0,
            }
            data.update(info)
            return jsonify({"success": True, "data": data})

        data = run_state.to_dict()
        data.update(info)
        return jsonify({"success": True, "data": data})

    except Exception as e:
        logger.error(f"Failed to get run state: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/run-status/detail", methods=["GET"])
def get_run_status_detail(simulation_id: str):
    """
    Get the detailed running status of the simulation (including all actions)

    Used for the frontend to display real-time dynamics

    Query parameters:
        platform: Filter by platform (twitter/reddit, optional)

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                ...
                "all_actions": [
                    {
                        "round_num": 5,
                        "timestamp": "2025-12-01T10:30:00",
                        "platform": "twitter",
                        "agent_id": 3,
                        "agent_name": "Agent Name",
                        "action_type": "CREATE_POST",
                        "action_args": {"content": "..."},
                        "result": null,
                        "success": true
                    },
                    ...
                ],
                "twitter_actions": [...],  # all actions on the Twitter platform
                "reddit_actions": [...]    # all actions on the Reddit platform
            }
        }
    """
    try:
        info = _run_info(simulation_id, request.args.get("run_id"))
        run_id = info["run_id"]
        run_state = SimulationRunner.get_run_state(simulation_id, run_id) if run_id else None
        platform_filter = request.args.get("platform")

        if not run_state:
            data = {
                "simulation_id": simulation_id,
                "runner_status": "idle",
                "all_actions": [],
                "twitter_actions": [],
                "reddit_actions": [],
            }
            data.update(info)
            return jsonify({"success": True, "data": data})

        # Get full action list
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id, platform=platform_filter, run_id=run_id
        )

        # Get actions by platform
        twitter_actions = (
            SimulationRunner.get_all_actions(
                simulation_id=simulation_id, platform="twitter", run_id=run_id
            )
            if not platform_filter or platform_filter == "twitter"
            else []
        )

        reddit_actions = (
            SimulationRunner.get_all_actions(
                simulation_id=simulation_id, platform="reddit", run_id=run_id
            )
            if not platform_filter or platform_filter == "reddit"
            else []
        )

        # Get current round actions (recent_actions only shows latest round)
        current_round = run_state.current_round
        recent_actions = (
            SimulationRunner.get_all_actions(
                simulation_id=simulation_id,
                platform=platform_filter,
                round_num=current_round,
                run_id=run_id,
            )
            if current_round > 0
            else []
        )

        # Get basic status info
        result = run_state.to_dict()
        result.update(info)
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions only shows latest round content from both platforms
        result["recent_actions"] = [a.to_dict() for a in recent_actions]

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Failed to get detailed status: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/actions", methods=["GET"])
def get_simulation_actions(simulation_id: str):
    """
    Get the history of Agent actions in the simulation

    Query parameters:
        limit: Return count (default 100)
        offset: Offset (default 0)
        platform: Filter by platform (twitter/reddit)
        agent_id: Filter by Agent ID
        round_num: Filter by round number

    Returns:
        {
            "success": true,
            "data": {
                "count": 100,
                "actions": [...]
            }
        }
    """
    try:
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)
        platform = request.args.get("platform")
        agent_id = request.args.get("agent_id", type=int)
        round_num = request.args.get("round_num", type=int)
        run_id = request.args.get("run_id")

        actions = SimulationRunner.get_actions(
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num,
            run_id=run_id,
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "count": len(actions),
                    "actions": [a.to_dict() for a in actions],
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to get action history: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/timeline", methods=["GET"])
def get_simulation_timeline(simulation_id: str):
    """
    Get the simulation timeline (aggregated by round)

    Used for the frontend to display the progress bar and timeline view

    Query parameters:
        start_round: Starting round (default 0)
        end_round: Ending round (default all)

    Returns aggregated info for each round
    """
    try:
        start_round = request.args.get("start_round", 0, type=int)
        end_round = request.args.get("end_round", type=int)

        timeline = SimulationRunner.get_timeline(
            simulation_id=simulation_id,
            start_round=start_round,
            end_round=end_round,
            run_id=request.args.get("run_id"),
        )

        return jsonify(
            {
                "success": True,
                "data": {"rounds_count": len(timeline), "timeline": timeline},
            }
        )

    except Exception as e:
        logger.error(f"Failed to get timeline: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/agent-stats", methods=["GET"])
def get_agent_stats(simulation_id: str):
    """
    Get statistics for each Agent

    Used for the frontend to display Agent activity rankings, action distributions, etc.
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id, request.args.get("run_id"))

        return jsonify(
            {"success": True, "data": {"agents_count": len(stats), "stats": stats}}
        )

    except Exception as e:
        logger.error(f"Failed to get Agent statistics: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Database Query API ==============


@simulation_bp.route("/<simulation_id>/posts", methods=["GET"])
def get_simulation_posts(simulation_id: str):
    """
    Get posts from the simulation

    Query parameters:
        platform: Platform type (twitter/reddit)
        limit: Return count (default 50)
        offset: Offset

    Returns a list of posts (read from the SQLite database)
    """
    try:
        platform = request.args.get("platform", "reddit")
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)

        sim_dir = RunStore.get_run_dir(simulation_id, request.args.get("run_id"))

        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)

        if not os.path.exists(db_path):
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "platform": platform,
                        "count": 0,
                        "posts": [],
                        "message": t("api.dbNotExist"),
                    },
                }
            )

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM post 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

            posts = [dict(row) for row in cursor.fetchall()]

            cursor.execute("SELECT COUNT(*) FROM post")
            total = cursor.fetchone()[0]

        except sqlite3.OperationalError:
            posts = []
            total = 0

        conn.close()

        return jsonify(
            {
                "success": True,
                "data": {
                    "platform": platform,
                    "total": total,
                    "count": len(posts),
                    "posts": posts,
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to get posts: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/comments", methods=["GET"])
def get_simulation_comments(simulation_id: str):
    """
    Get comments from the simulation (Reddit only)

    Query parameters:
        post_id: Filter by post ID (optional)
        limit: Return count
        offset: Offset
    """
    try:
        post_id = request.args.get("post_id")
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)

        sim_dir = RunStore.get_run_dir(simulation_id, request.args.get("run_id"))

        db_path = os.path.join(sim_dir, "reddit_simulation.db")

        if not os.path.exists(db_path):
            return jsonify({"success": True, "data": {"count": 0, "comments": []}})

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            if post_id:
                cursor.execute(
                    """
                    SELECT * FROM comment 
                    WHERE post_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """,
                    (post_id, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM comment 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )

            comments = [dict(row) for row in cursor.fetchall()]

        except sqlite3.OperationalError:
            comments = []

        conn.close()

        return jsonify(
            {"success": True, "data": {"count": len(comments), "comments": comments}}
        )

    except Exception as e:
        logger.error(f"Failed to get comments: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interview API ==============


@simulation_bp.route("/interview", methods=["POST"])
def interview_agent():
    """
    Interview a single Agent

    Note: This feature requires the simulation environment to be running (after completing the simulation loop, it enters wait-for-command mode)

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",       // required, simulation ID
            "agent_id": 0,                     // required, Agent ID
            "prompt": "What do you think about this?",  // required, interview question
            "platform": "twitter",             // optional, specified platform (twitter/reddit)
                                               // if not specified: in dual-platform mode, both platforms are interviewed simultaneously
            "timeout": 60                      // optional, timeout (seconds), default 60
        }

    Returns (no platform specified, dual-platform mode):
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "What do you think about this?",
                "result": {
                    "agent_id": 0,
                    "prompt": "...",
                    "platforms": {
                        "twitter": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit": {"agent_id": 0, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }

    Returns (platform specified):
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "What do you think about this?",
                "result": {
                    "agent_id": 0,
                    "response": "I think...",
                    "platform": "twitter",
                    "timestamp": "2025-12-08T10:00:00"
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        agent_id = data.get("agent_id")
        prompt = data.get("prompt")
        platform = data.get("platform")  # Optional: twitter/reddit/None
        timeout = data.get("timeout", 60)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        if agent_id is None:
            return jsonify({"success": False, "error": t("api.requireAgentId")}), 400

        if not prompt:
            return jsonify({"success": False, "error": t("api.requirePrompt")}), 400

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": t("api.invalidInterviewPlatform")}
            ), 400

        # Check environment status
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({"success": False, "error": t("api.envNotRunning")}), 400

        # Optimize prompt, add prefix to prevent Agent from calling tools
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_agent(
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout,
        )

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except TimeoutError as e:
        return jsonify(
            {"success": False, "error": t("api.interviewTimeout", error=str(e))}
        ), 504

    except Exception as e:
        logger.error(f"Interview failed: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/batch", methods=["POST"])
def interview_agents_batch():
    """
    Batch interview multiple Agents

    Note: This feature requires the simulation environment to be running

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",       // required, simulation ID
            "interviews": [                    // required, interview list
                {
                    "agent_id": 0,
                    "prompt": "What do you think about A?",
                    "platform": "twitter"      // optional, specifies the interview platform for this Agent
                },
                {
                    "agent_id": 1,
                    "prompt": "What do you think about B?"  // if platform is not specified, the default is used
                }
            ],
            "platform": "reddit",              // optional, default platform (overridden by each item's platform)
                                               // if not specified: in dual-platform mode, each Agent is interviewed on both platforms simultaneously
            "timeout": 120                     // optional, timeout (seconds), default 120
        }

    Returns:
        {
            "success": true,
            "data": {
                "interviews_count": 2,
                "result": {
                    "interviews_count": 4,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        "twitter_1": {"agent_id": 1, "response": "...", "platform": "twitter"},
                        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        interviews = data.get("interviews")
        platform = data.get("platform")  # Optional: twitter/reddit/None
        timeout = data.get("timeout", 120)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify({"success": False, "error": t("api.requireInterviews")}), 400

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": t("api.invalidInterviewPlatform")}
            ), 400

        # Validate each interview item
        for i, interview in enumerate(interviews):
            if "agent_id" not in interview:
                return jsonify(
                    {
                        "success": False,
                        "error": t("api.interviewListMissingAgentId", index=i + 1),
                    }
                ), 400
            if "prompt" not in interview:
                return jsonify(
                    {
                        "success": False,
                        "error": t("api.interviewListMissingPrompt", index=i + 1),
                    }
                ), 400
            # Validate platform for each item (if present)
            item_platform = interview.get("platform")
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify(
                    {
                        "success": False,
                        "error": t("api.interviewListInvalidPlatform", index=i + 1),
                    }
                ), 400

        # Check environment status
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({"success": False, "error": t("api.envNotRunning")}), 400

        # Optimize each interview item's prompt, add prefix to prevent Agent from calling tools
        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview["prompt"] = optimize_interview_prompt(
                interview.get("prompt", "")
            )
            optimized_interviews.append(optimized_interview)

        result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout,
        )

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except TimeoutError as e:
        return jsonify(
            {"success": False, "error": t("api.batchInterviewTimeout", error=str(e))}
        ), 504

    except Exception as e:
        logger.error(f"Batch Interview failed: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/all", methods=["POST"])
def interview_all_agents():
    """
    Global interview - interview all Agents with the same question

    Note: This feature requires the simulation environment to be running

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",            // required, simulation ID
            "prompt": "What do you think about this whole thing?",  // required, interview question (same question for all Agents)
            "platform": "reddit",                   // optional, specified platform (twitter/reddit)
                                                    // if not specified: in dual-platform mode, each Agent is interviewed on both platforms simultaneously
            "timeout": 180                          // optional, timeout (seconds), default 180
        }

    Returns:
        {
            "success": true,
            "data": {
                "interviews_count": 50,
                "result": {
                    "interviews_count": 100,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        ...
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        prompt = data.get("prompt")
        platform = data.get("platform")  # Optional: twitter/reddit/None
        timeout = data.get("timeout", 180)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        if not prompt:
            return jsonify({"success": False, "error": t("api.requirePrompt")}), 400

        # Validate platform parameter
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": t("api.invalidInterviewPlatform")}
            ), 400

        # Check environment status
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({"success": False, "error": t("api.envNotRunning")}), 400

        # Optimize prompt, add prefix to prevent Agent from calling tools
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_all_agents(
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout,
        )

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except TimeoutError as e:
        return jsonify(
            {"success": False, "error": t("api.globalInterviewTimeout", error=str(e))}
        ), 504

    except Exception as e:
        logger.error(f"Global Interview failed: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/history", methods=["POST"])
def get_interview_history():
    """
    Get interview history records

    Reads all interview records from the simulation database

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",  // required, simulation ID
            "platform": "reddit",          // optional, platform type (reddit/twitter)
                                           // if not specified, returns all history for both platforms
            "agent_id": 0,                 // optional, only get this Agent's interview history
            "limit": 100                   // optional, return count, default 100
        }

    Returns:
        {
            "success": true,
            "data": {
                "count": 10,
                "history": [
                    {
                        "agent_id": 0,
                        "response": "I think...",
                        "prompt": "What do you think about this?",
                        "timestamp": "2025-12-08T10:00:00",
                        "platform": "reddit"
                    },
                    ...
                ]
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        platform = data.get("platform")  # If not specified, return history from both platforms
        agent_id = data.get("agent_id")
        limit = data.get("limit", 100)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        history = SimulationRunner.get_interview_history(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit,
        )

        return jsonify(
            {"success": True, "data": {"count": len(history), "history": history}}
        )

    except Exception as e:
        logger.error(f"Failed to get Interview history: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/env-status", methods=["POST"])
def get_env_status():
    """
    Get the simulation environment status

    Checks whether the simulation environment is alive (can receive Interview commands)

    Request (JSON):
        {
            "simulation_id": "sim_xxxx"  // required, simulation ID
        }

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "Environment is running and can receive Interview commands"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")

        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        env_alive = SimulationRunner.check_env_alive(simulation_id)

        # Get more detailed status info
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = t("api.envRunning")
        else:
            message = t("api.envNotRunningShort")

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "env_alive": env_alive,
                    "twitter_available": env_status.get("twitter_available", False),
                    "reddit_available": env_status.get("reddit_available", False),
                    "message": message,
                },
            }
        )

    except Exception as e:
        logger.error(f"Failed to get environment status: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/close-env", methods=["POST"])
def close_simulation_env():
    """
    Close the simulation environment

    Sends a close-environment command to the simulation so that it gracefully exits wait-for-command mode.

    Note: This differs from the /stop endpoint, which forcibly terminates the process;
    this endpoint lets the simulation gracefully close the environment and exit.

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",  // required, simulation ID
            "timeout": 30                  // optional, timeout (seconds), default 30
        }

    Returns:
        {
            "success": true,
            "data": {
                "message": "Environment close command sent",
                "result": {...},
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        timeout = data.get("timeout", 30)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        result = SimulationRunner.close_simulation_env(
            simulation_id=simulation_id, timeout=timeout
        )

        # Update simulation state
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"Failed to close environment: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
