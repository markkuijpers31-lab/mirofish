"""
Run export: one ZIP bundle that makes a finished run auditable without re-running it
"""

import hashlib
import io
import json
import os
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_run_logger
from .run_evidence import RunEvidence
from .run_store import LEGACY_RUN_ID, RunStore

README = """# MiroFish run export

This archive contains the complete evidence of one simulation run (execution).

- run/            run metadata (run.json), runner state and the simulation process log
- config/         simulation configuration, agent selection and the private experiment plan of this run
- personas/       agent profiles, the agent roster (active agents with simulation/evidence roles and
                  context-only graph entities) and the normalized persona index (canonical IDs)
- sources/        source documents (extracted text + originals when available) and their manifest
- assessment/     assessment brief and the spec derived from it (personas, hypotheses, funnel fields)
- actions/        raw Twitter and Reddit action logs of this run (actions.jsonl); every action has an
                  origin (agent_generated / system_seed / researcher_instruction), private instruction
                  deliveries are logged as events
- evidence/       normalized evidence (every action with its reference, e.g. TW-00012)
- report/         final report (Markdown + JSON), sections and report-agent logs

All behaviour in this archive is simulated (synthetic agents), not real-world customer data.
MANIFEST.json lists every file with its SHA-256 checksum.
"""


def _read_bytes(path: str) -> Optional[bytes]:
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    return None


def build_run_bundle(
    simulation_id: str,
    run_id: Optional[str] = None,
    report_id: Optional[str] = None,
) -> Tuple[bytes, str]:
    """
    Build the export ZIP of a run (and optionally one of its reports)

    Args:
        simulation_id: Simulation ID
        run_id: Run ID (default: the report's run, else the current run)
        report_id: Report to include (optional)

    Returns:
        (zip bytes, download filename)

    Raises:
        ValueError: if the run cannot be found
    """
    from .report_agent import ReportManager

    report_folder = ReportManager._get_report_folder(report_id) if report_id else None
    report_meta: Dict[str, Any] = {}
    if report_folder and os.path.exists(os.path.join(report_folder, "meta.json")):
        with open(os.path.join(report_folder, "meta.json"), "r", encoding="utf-8") as f:
            report_meta = json.load(f)
    run_resolution = "requested"
    if not run_id:
        run_id = report_meta.get("run_id")
        run_resolution = "report_meta" if run_id else "current_run"
    if not run_id:
        run_id = RunStore.resolve_run_id(simulation_id)
    record = RunStore.get_run(simulation_id, run_id) if run_id else None
    if record is None:
        raise ValueError(f"Run not found for simulation {simulation_id}")

    log = get_run_logger("mirofish.export", simulation_id, run_id, "export")
    run_dir = RunStore.run_dir(simulation_id, run_id)
    sim_dir = RunStore.sim_dir(simulation_id)
    entries: List[Tuple[str, bytes]] = []

    def add_file(arcname: str, path: str):
        data = _read_bytes(path)
        if data is not None:
            entries.append((arcname, data))

    def add_json(arcname: str, obj: Any):
        entries.append((arcname, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")))

    entries.append(("README.md", README.encode("utf-8")))

    # run/
    add_json("run/run.json", record.to_dict())
    add_file("run/run_state.json", os.path.join(run_dir, "run_state.json"))
    add_file("run/simulation.log", os.path.join(run_dir, "simulation.log"))

    # config/
    add_file("config/simulation_config.json", os.path.join(run_dir, "simulation_config.json"))
    add_file("config/simulation_state.json", os.path.join(sim_dir, "state.json"))
    add_file("config/agent_selection.json", os.path.join(run_dir, "agent_selection.json"))
    add_file("config/experiment_plan.json", os.path.join(run_dir, "experiment_plan.json"))
    add_file("personas/agent_roster.json", os.path.join(run_dir, "agent_roster.json"))

    # personas/
    add_file("personas/reddit_profiles.json", os.path.join(run_dir, "reddit_profiles.json"))
    add_file("personas/twitter_profiles.csv", os.path.join(run_dir, "twitter_profiles.csv"))

    # actions/ (raw, byte-identical copies)
    add_file("actions/twitter_actions.jsonl", os.path.join(run_dir, "twitter", "actions.jsonl"))
    add_file("actions/reddit_actions.jsonl", os.path.join(run_dir, "reddit", "actions.jsonl"))
    if run_id == LEGACY_RUN_ID:
        add_file("actions/legacy_actions.jsonl", os.path.join(run_dir, "actions.jsonl"))

    # evidence/, sources/, assessment/
    evidence = RunEvidence.load(simulation_id, run_id)
    report_json_path = os.path.join(report_folder, "report.json") if report_folder else None
    report_json = None
    if report_json_path and os.path.exists(report_json_path):
        with open(report_json_path, "r", encoding="utf-8") as f:
            report_json = json.load(f)
        spec = report_json.get("assessment_spec") or {}
        if spec.get("persona_ids"):
            evidence.set_requested_ids(spec["persona_ids"])
        add_json("assessment/spec.json", spec)

    evidence_path = os.path.join(report_folder, "evidence.json") if report_folder else None
    if evidence_path and os.path.exists(evidence_path):
        add_file("evidence/normalized_evidence.json", evidence_path)
    else:
        add_json("evidence/normalized_evidence.json", evidence.to_dict())
    add_json("personas/personas_normalized.json", {
        "requested_ids": evidence.requested_ids,
        "personas": evidence.persona_index.to_list(),
    })

    sources_dir = os.path.join(run_dir, "sources")
    if run_id != LEGACY_RUN_ID and os.path.isdir(sources_dir):
        for root, _dirs, files in os.walk(sources_dir):
            for name in sorted(files):
                full = os.path.join(root, name)
                add_file("sources/" + os.path.relpath(full, sources_dir).replace(os.sep, "/"), full)
    else:
        manifest = []
        for i, doc in enumerate(evidence.sources, start=1):
            text_name = f"{i:02d}_{os.path.splitext(doc.get('original_filename') or 'document')[0]}.txt"
            entries.append((f"sources/{text_name}", (doc.get("text") or "").encode("utf-8")))
            manifest.append({k: v for k, v in doc.items() if k != "text"} | {"text_file": text_name})
        add_json("sources/manifest.json", {"origin": evidence.sources_origin, "documents": manifest})

    brief_path = os.path.join(run_dir, "assessment", "brief.md")
    if run_id != LEGACY_RUN_ID and os.path.exists(brief_path):
        add_file("assessment/brief.md", brief_path)
    else:
        entries.append(("assessment/brief.md", evidence.brief_text.encode("utf-8")))

    # report/
    if report_folder and os.path.isdir(report_folder):
        for name in ("full_report.md", "report.json", "meta.json", "outline.json", "progress.json",
                     "agent_log.jsonl", "console_log.txt"):
            add_file(f"report/{name}", os.path.join(report_folder, name))
        for name in sorted(os.listdir(report_folder)):
            if name.startswith("section_") and name.endswith(".md"):
                add_file(f"report/sections/{name}", os.path.join(report_folder, name))

    manifest = {
        "simulation_id": simulation_id,
        "run_id": run_id,
        "run_resolution": run_resolution,
        "report_id": report_id,
        "generated_at": datetime.now().isoformat(),
        "evidence_type": "simulated (synthetic agents); not real-world customer data",
        "files": [
            {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in entries
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries:
            zf.writestr(name, data)
        zf.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    filename = f"mirofish_{simulation_id}_{run_id}" + (f"_{report_id}" if report_id else "") + ".zip"
    log.info(f"Run export built: {len(entries)} files, report={report_id}")
    return buffer.getvalue(), filename
