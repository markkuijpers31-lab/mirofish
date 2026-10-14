"""
Evidence-bound report sections: per-persona funnel analysis, hypothesis assessment and evidence appendix
"""

import json
import re
from typing import Any, Dict, List, Optional

from .agent_selection import EVIDENCE_TARGET_CONSUMER
from .assessment_spec import AssessmentSpec
from .run_evidence import CONSUMER_METRIC_SCOPE, RunEvidence

PERSONA_SECTION_TITLE = "Persona Analysis"
HYPOTHESIS_SECTION_TITLE = "Hypothesis Assessment"
APPENDIX_SECTION_TITLE = "Evidence Base & Methodology"
STRUCTURED_SECTION_TITLES = (PERSONA_SECTION_TITLE, HYPOTHESIS_SECTION_TITLE, APPENDIX_SECTION_TITLE)

FIELD_STATUSES = ("observed", "ambiguous", "missing")
VERDICTS = ("supported", "contradicted", "mixed", "insufficient_evidence")

# A hypothesis verdict other than insufficient_evidence needs at least this many
# distinct, valid action citations
MIN_HYPOTHESIS_REFS = 2

# Upper bound of persona LLM calls per report (personas beyond it get a
# deterministic summary only)
MAX_PERSONA_LLM_CALLS = 40

EVIDENCE_DISCLAIMER = (
    "All findings below describe **simulated** behaviour of synthetic agents in an "
    "agent-based social-media simulation (OASIS: simulated Twitter and Reddit). They are "
    "not observations of real customers. Counts are simulation counts, not conversion "
    "rates or market shares; no real-world percentages can be derived from them."
)

_PERSONA_SYSTEM_PROMPT = """You analyse ONE persona in an agent-based social-media simulation.
You receive the persona definition, the funnel fields the assessment asks for, context from the
assessment brief and product documents, and the COMPLETE list of this persona's simulated actions.
Every action has a reference like [TW-00012] (Twitter) or [RD-00003] (Reddit).

Rules:
- Use ONLY the action list as evidence of what the persona did. Brief and product documents are
  context, never evidence of behaviour. The list holds only the persona's own (agent-generated)
  actions; seed posts and private research instructions are excluded by design.
- For each funnel field decide: "observed" (clearly shown by cited actions), "ambiguous" (actions
  touch on it but do not settle it) or "missing" (not shown). If in doubt, use "missing".
- Every "observed" or "ambiguous" field MUST cite the refs that show it. Never cite refs that are
  not in the list.
- Keep negative evidence (objections, distrust, complaints, choosing an alternative, abandoning).
- Never invent steps, motives, prices, plans or outcomes that the actions do not show.
- Product capabilities: when an AUTHORITATIVE product document is provided, it is the only basis for
  saying what the product can or cannot do. A capability that is absent from a snippet or from the
  persona's posts is NOT evidence that the product lacks it. If the persona believes something is
  missing, report it as the persona's belief and say whether the authoritative document contradicts it.
- The private orchestration timeline (instructions and exposures the experiment delivered) is context:
  it tells you what the persona was told and when, but it is never evidence of the persona's behaviour.
- Likes, favourable comments, questions or "sounds useful" are not trial, adoption, payment or return.
- No percentages or conversion rates.
- Write summaries in the requested report language.

Return JSON only:
{"fields": [{"field": "<exact field name>", "status": "observed|ambiguous|missing", "summary": "...", "refs": ["TW-00012"]}],
 "strongest_evidence_for": {"summary": "...", "refs": []},
 "strongest_evidence_against": {"summary": "...", "refs": []},
 "notes": "..."}"""

_HYPOTHESIS_SYSTEM_PROMPT = """You assess ONE hypothesis against evidence from an agent-based social-media simulation.
You receive the hypothesis, per-persona funnel findings (already validated) and relevant simulated
actions, each with a reference like [TW-00012] or [RD-00003].

Rules:
- Only simulated actions count as evidence. Cite their refs; never cite refs not provided.
- When an AUTHORITATIVE product document is provided it decides what the product can and cannot do;
  absence of a capability from snippets or posts is never evidence that it does not exist.
- Each action shows [simulation_role/evidence_role]. Conversion, adoption or acquisition claims may
  only rest on actions of evidence role target_consumer. Existing users, advocates and platform
  accounts can show influence on prospects, but never count as conversions.
- verdict: "supported", "contradicted", "mixed" or "insufficient_evidence".
  Use "insufficient_evidence" whenever the simulation does not clearly speak to the hypothesis.
- Keep evidence against the hypothesis even when the verdict is "supported".
- This is simulated evidence: never claim real-world validity, no percentages or conversion rates.
- Write in the requested report language.

Return JSON only:
{"verdict": "...", "rationale": "...",
 "evidence_for": [{"summary": "...", "refs": []}],
 "evidence_against": [{"summary": "...", "refs": []}],
 "missing_evidence": "what would be needed to assess it"}"""


def _cell(text: Any, limit: int = 300) -> str:
    """Make text safe for a Markdown table cell"""
    value = " ".join(str(text or "").split())
    if len(value) > limit:
        value = value[: limit - 3] + "..."
    return value.replace("|", "\\|") or "—"


def _clean_refs(refs: Any, valid: set) -> List[str]:
    if isinstance(refs, str):
        refs = [refs]
    cleaned = []
    for r in refs or []:
        r = str(r).strip().strip("[]")
        if r in valid and r not in cleaned:
            cleaned.append(r)
    return cleaned


class StructuredReportBuilder:
    """
    Builds the evidence-bound report sections from a run's raw evidence

    Validation happens in code, not in the prompt: an LLM claim without a valid
    citation of this run's actions is downgraded to "missing", and a hypothesis
    without enough cited evidence becomes "insufficient_evidence".
    """

    def __init__(
        self,
        evidence: Optional[RunEvidence],
        spec: AssessmentSpec,
        llm: Any,
        language_instruction: str = "",
        log: Any = None,
    ):
        self.evidence = evidence
        self.spec = spec
        self.llm = llm
        self.language_instruction = language_instruction
        self.log = log
        self.persona_results: List[Dict[str, Any]] = []
        self.hypothesis_results: List[Dict[str, Any]] = []

    # ---------- helpers ----------

    def _info(self, msg: str):
        if self.log:
            self.log.info(msg)

    def _warn(self, msg: str):
        if self.log:
            self.log.warning(msg)

    def target_persona_ids(self) -> List[str]:
        """
        Persona ids to analyse: the brief's ids, else the target consumers of the
        selected population (every persona for runs without recorded roles)
        """
        if self.spec.persona_ids:
            return list(self.spec.persona_ids)
        if not self.evidence:
            return []
        personas = sorted(self.evidence.persona_index.personas.keys())
        if self.evidence.roles_source == "legacy_default":
            return personas
        return [cid for cid in personas if self.evidence.persona_evidence_role(cid) == EVIDENCE_TARGET_CONSUMER]

    # Authoritative documents are passed in full (up to this budget) so capability
    # judgements never rest on a partial excerpt
    AUTHORITATIVE_BUDGET = 60000

    def _context_documents(self, budget: int = 3000) -> str:
        if not self.evidence or not self.evidence.sources:
            return "(no source documents available)"
        parts = []
        authoritative = self.evidence.authoritative_documents()
        for doc in authoritative:
            text = doc["text"]
            if len(text) > self.AUTHORITATIVE_BUDGET:
                self._warn(f"Authoritative document {doc['filename']} truncated to {self.AUTHORITATIVE_BUDGET} chars")
                text = text[: self.AUTHORITATIVE_BUDGET] + "\n...(truncated)"
            self.evidence.mark_source_used(doc["filename"])
            parts.append(
                f"AUTHORITATIVE PRODUCT DOCUMENT ({doc['label']}, complete): the only valid basis for what the "
                f"product does and does not do.\n{text}"
            )
        parts.append("Other source excerpts (context only):\n" + self.evidence.source_excerpts(max_chars=budget))
        return "\n\n".join(parts)

    def _timeline_text(self, cid: str) -> str:
        timeline = self.evidence.persona_stage_timeline(cid) if self.evidence else []
        if not timeline:
            return "(no private instructions or exposures were delivered to this persona)"
        return "\n".join(
            f"- round {t['round']}: stage '{t['stage_name']}' {t['event']}"
            + (f" (records {', '.join(t['records'])})" if t["records"] else "")
            for t in timeline
        )

    def _apply_orchestration_records(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fields named after an orchestration record (e.g. EXPOSED) are decided by the
        delivered stages only: they describe what the experiment did, not behaviour.
        """
        if not self.evidence or not (self.evidence.plan_record_labels or self.evidence.derived_fields):
            return result
        cid = result["canonical_id"]
        records = self.evidence.persona_records(cid) if result.get("present_in_simulation") else {}
        result["stage_timeline"] = self.evidence.persona_stage_timeline(cid) if result.get("present_in_simulation") else []
        labels = set(self.evidence.plan_record_labels)
        for field in result.get("fields", []):
            label = re.sub(r"[^A-Z0-9]+", "_", field["field"].upper()).strip("_")
            if label not in labels:
                continue
            if label in records:
                field.update({"status": "observed", "refs": [], "method": "orchestration",
                              "summary": f"Orchestration fact: recorded in round {records[label]} (not behaviour)."})
            else:
                field.update({"status": "missing", "refs": [], "method": "orchestration",
                              "summary": "Not delivered to this persona by the experiment."})
        self._apply_derived_fields(result, records)
        result["missing_stages"] = [f["field"] for f in result.get("fields", []) if f["status"] == "missing"]
        return result

    def _apply_derived_fields(self, result: Dict[str, Any], records: Dict[str, int]):
        """Fields the plan defines as derived from the complete log (e.g. organic discovery)"""
        rules = self.evidence.derived_fields if self.evidence else {}
        if not rules:
            return
        present = bool(result.get("present_in_simulation"))
        for field in result.get("fields", []):
            label = re.sub(r"[^A-Z0-9]+", "_", field["field"].upper()).strip("_")
            rule = rules.get(label)
            if not rule:
                continue
            if rule.get("evidence_roles") and result.get("evidence_role") not in rule["evidence_roles"]:
                field.update({"status": "missing", "refs": [], "method": "derived",
                              "summary": f"Not applicable to evidence role {result.get('evidence_role')}."})
                continue
            encounter = self.evidence.first_encounter(result["canonical_id"], rule["terms"]) if present else None
            record_round = records.get(rule.get("record") or "")
            before = encounter is not None and (record_round is None or encounter["round"] < record_round)
            terms = ", ".join(rule["terms"])
            when = f"before {rule['record']} (round {record_round})" if record_round is not None else "during the run"
            if rule["type"] == "first_encounter_before_record":
                if before:
                    field.update({"status": "observed", "refs": [encounter["ref"]], "method": "derived",
                                  "summary": f"First own or reacted-to mention of {terms} in round {encounter['round']}, {when}."})
                else:
                    field.update({"status": "missing", "refs": [], "method": "derived",
                                  "summary": f"No mention of {terms} {when}."})
            elif present:
                if before:
                    field.update({"status": "missing", "refs": [], "method": "derived",
                                  "summary": f"Encountered {terms} in round {encounter['round']}, {when}."})
                else:
                    field.update({"status": "observed", "refs": [], "method": "derived",
                                  "summary": f"No own or reacted-to mention of {terms} {when} (complete log checked)."})

    def _missing_result(self, cid: str, display: str, reason: str, present: bool, counts: Dict[str, int]) -> Dict[str, Any]:
        fields = [
            {"field": f, "status": "missing", "summary": reason, "refs": []}
            for f in self.spec.funnel_fields
        ]
        return {
            "canonical_id": cid,
            "display_name": display,
            "present_in_simulation": present,
            "action_counts": counts,
            "fields": fields,
            "strongest_evidence_for": {"summary": "None observed in the simulation.", "refs": []},
            "strongest_evidence_against": {"summary": "None observed in the simulation.", "refs": []},
            "missing_stages": list(self.spec.funnel_fields),
            "notes": reason,
            "method": "deterministic",
        }

    # ---------- personas ----------

    def analyse_personas(self) -> List[Dict[str, Any]]:
        """Analyse every target persona individually"""
        results = []
        llm_calls = 0
        for cid in self.target_persona_ids():
            persona = self.evidence.persona_index.personas.get(cid) if self.evidence else None
            if persona is None:
                results.append(self._missing_result(
                    cid, cid,
                    "Persona requested by the assessment brief but not present among the simulated agents.",
                    present=False, counts={"total": 0},
                ))
                continue
            actions = self.evidence.persona_actions(cid, behavioural_only=True)
            counts = {"total": len(actions)}
            for a in actions:
                counts[a["platform"]] = counts.get(a["platform"], 0) + 1
            excluded = len(self.evidence.persona_actions(cid)) - len(actions)
            if excluded:
                counts["excluded_non_agent_generated"] = excluded
            if not actions:
                results.append(self._missing_result(
                    cid, persona["display_name"],
                    "No simulation evidence observed: this persona took no actions in this run.",
                    present=True, counts=counts,
                ))
                continue
            if llm_calls >= MAX_PERSONA_LLM_CALLS:
                result = self._missing_result(
                    cid, persona["display_name"],
                    "Not analysed individually (persona limit reached); see raw actions in the evidence bundle.",
                    present=True, counts=counts,
                )
                results.append(result)
                continue
            llm_calls += 1
            results.append(self._analyse_persona_llm(cid, persona, actions, counts))
        for r in results:
            if self.evidence and r.get("present_in_simulation"):
                r["simulation_role"] = self.evidence.persona_simulation_role(r["canonical_id"])
                r["evidence_role"] = self.evidence.persona_evidence_role(r["canonical_id"])
                r["counts_in_conversion_metrics"] = r["evidence_role"] == EVIDENCE_TARGET_CONSUMER
            self._apply_orchestration_records(r)
        self.persona_results = results
        self._info(f"Persona analysis done: {len(results)} personas, {llm_calls} LLM calls")
        return results

    def _analyse_persona_llm(self, cid: str, persona: Dict[str, Any], actions: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, Any]:
        valid_refs = {a["ref"] for a in actions}
        profile = persona["profiles"][0] if persona.get("profiles") else {}
        profile_text = json.dumps(
            {k: v for k, v in profile.items() if k in ("name", "entity_type", "bio", "persona", "profession", "stance")},
            ensure_ascii=False,
        )[:2000]
        user_prompt = "\n".join([
            f"Persona canonical id: {cid}",
            f"Display name: {persona['display_name']}",
            f"Name variants in the simulation: {', '.join(persona['names'])}",
            f"Persona definition: {profile_text}",
            "",
            "Funnel fields to assess (use these exact names):",
            *[f"- {f}" for f in self.spec.funnel_fields],
            "",
            "Assessment brief (context):",
            self.evidence.brief_text[:12000],
            "",
            "Private orchestration timeline (experiment facts, NOT behaviour, not citable):",
            self._timeline_text(cid),
            "",
            "Product / source documents (context, not evidence of behaviour):",
            self._context_documents(),
            "",
            f"COMPLETE agent-generated action list of {cid} ({len(actions)} actions):",
            self.evidence.persona_transcript(cid, behavioural_only=True),
        ])
        try:
            data = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": f"{_PERSONA_SYSTEM_PROMPT}\n\n{self.language_instruction}"},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
        except Exception as e:
            self._warn(f"Persona {cid}: LLM analysis failed ({e}); fields marked missing")
            result = self._missing_result(
                cid, persona["display_name"],
                f"Automatic analysis failed ({e}); {len(actions)} raw actions are available in the evidence bundle.",
                present=True, counts=counts,
            )
            return result
        return self._validate_persona(cid, persona["display_name"], data, valid_refs, counts)

    def _validate_persona(self, cid: str, display: str, data: Dict[str, Any], valid_refs: set, counts: Dict[str, int]) -> Dict[str, Any]:
        by_name = {}
        for entry in (data.get("fields") or []):
            if isinstance(entry, dict) and entry.get("field"):
                by_name[str(entry["field"]).strip().lower()] = entry
        fields = []
        downgraded = 0
        for name in self.spec.funnel_fields:
            entry = by_name.get(name.lower(), {})
            status = str(entry.get("status", "missing")).lower()
            if status not in FIELD_STATUSES:
                status = "missing"
            refs = _clean_refs(entry.get("refs"), valid_refs)
            summary = str(entry.get("summary") or "").strip()
            if status in ("observed", "ambiguous") and not refs:
                downgraded += 1
                status = "missing"
                summary = "Not shown by any citable simulated action (claim without evidence discarded)."
            if status == "missing" and not summary:
                summary = "Not observed in the simulation."
            fields.append({"field": name, "status": status, "summary": summary, "refs": refs if status != "missing" else []})

        def _strongest(key: str) -> Dict[str, Any]:
            item = data.get(key) or {}
            if not isinstance(item, dict):
                item = {"summary": str(item)}
            refs = _clean_refs(item.get("refs"), valid_refs)
            if not refs:
                return {"summary": "None observed in the simulation.", "refs": []}
            return {"summary": str(item.get("summary") or "").strip(), "refs": refs}

        if downgraded:
            self._warn(f"Persona {cid}: {downgraded} uncited claims downgraded to missing")
        return {
            "canonical_id": cid,
            "display_name": display,
            "present_in_simulation": True,
            "action_counts": counts,
            "fields": fields,
            "strongest_evidence_for": _strongest("strongest_evidence_for"),
            "strongest_evidence_against": _strongest("strongest_evidence_against"),
            "missing_stages": [f["field"] for f in fields if f["status"] == "missing"],
            "notes": str(data.get("notes") or "").strip(),
            "uncited_claims_discarded": downgraded,
            "method": "llm_validated",
        }

    def render_personas(self) -> str:
        """Markdown for the persona analysis section"""
        lines = [EVIDENCE_DISCLAIMER, ""]
        if self.spec.persona_ids:
            lines.append(f"Personas requested by the assessment brief: {', '.join(self.spec.persona_ids)}.")
        else:
            lines.append("The assessment brief requests no specific persona IDs; every simulated persona is analysed.")
        lines.append(
            f"Funnel fields ({'from the brief' if self.spec.funnel_fields_source == 'brief' else 'generic default, the brief defines none'}): "
            f"{', '.join(self.spec.funnel_fields)}."
        )
        lines.append("")
        if not self.persona_results:
            lines.append("No personas available to analyse: this run has no simulation evidence.")
        lines += self._render_cohort_summary()
        for r in self._results_by_cohort():
            counts = r.get("action_counts", {})
            count_text = ", ".join(f"{k} {v}" for k, v in counts.items() if k != "total")
            lines.append(f"**{r['canonical_id']} — {_cell(r['display_name'], 120)}**")
            lines.append("")
            if r.get("evidence_role"):
                lines.append(
                    f"Role: {r.get('simulation_role')} · evidence role: {r['evidence_role']}"
                    + ("" if r.get("counts_in_conversion_metrics") else " (not counted in conversion / adoption metrics)")
                )
                lines.append("")
            if not r.get("present_in_simulation"):
                lines.append("_Not present in the simulation: no agent maps to this ID. Every field is missing._")
            else:
                count_text = ", ".join(f"{k} {v}" for k, v in counts.items() if k in ("twitter", "reddit"))
                lines.append(f"Agent-generated actions: {counts.get('total', 0)}" + (f" ({count_text})" if count_text else ""))
                if counts.get("excluded_non_agent_generated"):
                    lines.append(
                        f"Excluded from behavioural evidence: {counts['excluded_non_agent_generated']} seed / instruction actions"
                    )
            lines.append("")
            lines.append("| Field | Status | Simulated evidence | Refs |")
            lines.append("|---|---|---|---|")
            for f in r["fields"]:
                status = f["status"].upper() if f["status"] == "missing" else f["status"]
                if f.get("method") in ("orchestration", "derived"):
                    status += f" ({f['method']})"
                lines.append(f"| {_cell(f['field'], 60)} | {status} | {_cell(f['summary'])} | {_cell(', '.join(f['refs']), 120)} |")
            lines.append("")
            for label, key in (("Strongest evidence for", "strongest_evidence_for"), ("Strongest evidence against", "strongest_evidence_against")):
                item = r[key]
                refs = f" ({', '.join(item['refs'])})" if item["refs"] else ""
                lines.append(f"- **{label}:** {_cell(item['summary'])}{refs}")
            missing = r.get("missing_stages") or []
            lines.append(f"- **Missing / unobserved stages:** {', '.join(missing) if missing else 'none'}")
            lines.append("")
        return "\n".join(lines).strip()

    def _results_by_cohort(self) -> List[Dict[str, Any]]:
        """Persona results ordered by cohort (simulation role), keeping the brief's order within a cohort"""
        order: List[str] = []
        for r in self.persona_results:
            role = r.get("simulation_role") or "not present"
            if role not in order:
                order.append(role)
        return sorted(self.persona_results, key=lambda r: order.index(r.get("simulation_role") or "not present"))

    def _render_cohort_summary(self) -> List[str]:
        """Per-cohort counts of observed / ambiguous / missing for every funnel field"""
        cohorts: Dict[str, List[Dict[str, Any]]] = {}
        for r in self._results_by_cohort():
            cohorts.setdefault(r.get("simulation_role") or "not present", []).append(r)
        if not cohorts or not self.spec.funnel_fields:
            return []
        names = list(cohorts)
        lines = [
            "**Cohort summary** (observed / ambiguous / missing personas per field; acquisition and conversion "
            "denominators use target-consumer cohorts only)",
            "",
            "| Field | " + " | ".join(
                f"{_cell(n, 40)} (n={len(cohorts[n])}{', target' if any(x.get('counts_in_conversion_metrics') for x in cohorts[n]) else ''})"
                for n in names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for field in self.spec.funnel_fields:
            cells = []
            for n in names:
                statuses = [next((f["status"] for f in r["fields"] if f["field"] == field), "missing") for r in cohorts[n]]
                cells.append(f"{statuses.count('observed')} / {statuses.count('ambiguous')} / {statuses.count('missing')}")
            lines.append(f"| {_cell(field, 60)} | " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    # ---------- hypotheses ----------

    def assess_hypotheses(self) -> List[Dict[str, Any]]:
        """Assess every hypothesis of the brief against the run's evidence"""
        results = []
        behavioural = self.evidence.behavioural_actions() if self.evidence else []
        all_refs = {a["ref"] for a in behavioural}
        persona_summary = [
            {
                "persona": r["canonical_id"],
                "fields": [
                    {"field": f["field"], "status": f["status"], "summary": f["summary"], "refs": f["refs"]}
                    for f in r["fields"]
                ],
            }
            for r in self.persona_results
        ]
        for h in self.spec.hypotheses:
            if not all_refs:
                results.append({
                    **h,
                    "verdict": "insufficient_evidence",
                    "rationale": "This run contains no agent-generated simulated actions.",
                    "evidence_for": [],
                    "evidence_against": [],
                    "missing_evidence": "Simulated actions relevant to the hypothesis.",
                    "method": "deterministic",
                })
                continue
            relevant = self.evidence.search_actions(query=h["text"], limit=60, scope="behavioural")
            if len(relevant) < 10:
                seen = {a["ref"] for a in relevant}
                relevant += [a for a in behavioural if a["ref"] not in seen][: 60 - len(relevant)]
            user_prompt = "\n".join([
                f"Hypothesis {h['id']}: {h['text']}",
                "",
                "Per-persona funnel findings (validated):",
                json.dumps(persona_summary, ensure_ascii=False)[:20000],
                "",
                f"Conversion / adoption counts (scope: {CONSUMER_METRIC_SCOPE}; simulation counts, not real-world rates):",
                json.dumps(self.evidence.consumer_metrics(), ensure_ascii=False)[:3000],
                "",
                "Relevant simulated actions:",
                "\n".join(RunEvidence.format_action(a) for a in relevant),
                "",
                "Product documents (context, not evidence of behaviour):",
                self._context_documents(budget=1500),
            ])
            try:
                data = self.llm.chat_json(
                    messages=[
                        {"role": "system", "content": f"{_HYPOTHESIS_SYSTEM_PROMPT}\n\n{self.language_instruction}"},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
            except Exception as e:
                self._warn(f"Hypothesis {h['id']}: LLM assessment failed ({e})")
                data = {"verdict": "insufficient_evidence", "rationale": f"Automatic assessment failed: {e}"}
            results.append(self._validate_hypothesis(h, data, all_refs))
        self.hypothesis_results = results
        self._info(f"Hypothesis assessment done: {len(results)} hypotheses")
        return results

    def _validate_hypothesis(self, h: Dict[str, str], data: Dict[str, Any], valid_refs: set) -> Dict[str, Any]:
        def _items(key: str) -> List[Dict[str, Any]]:
            items = []
            for item in data.get(key) or []:
                if isinstance(item, str):
                    item = {"summary": item}
                refs = _clean_refs(item.get("refs"), valid_refs)
                if refs:
                    items.append({"summary": str(item.get("summary") or "").strip(), "refs": refs})
            return items

        evidence_for = _items("evidence_for")
        evidence_against = _items("evidence_against")
        verdict = str(data.get("verdict") or "insufficient_evidence").lower()
        if verdict not in VERDICTS:
            verdict = "insufficient_evidence"
        distinct_refs = {r for item in evidence_for + evidence_against for r in item["refs"]}
        rationale = str(data.get("rationale") or "").strip()
        forced = False
        if verdict != "insufficient_evidence" and len(distinct_refs) < MIN_HYPOTHESIS_REFS:
            forced = True
            verdict = "insufficient_evidence"
            rationale = (
                f"Fewer than {MIN_HYPOTHESIS_REFS} citable simulated actions support an assessment. "
                + rationale
            ).strip()
        return {
            **h,
            "verdict": verdict,
            "rationale": rationale,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "missing_evidence": str(data.get("missing_evidence") or "").strip(),
            "verdict_forced_by_evidence_rule": forced,
            "method": "llm_validated",
        }

    def render_hypotheses(self) -> str:
        """Markdown for the hypothesis section"""
        lines = [
            "Verdicts are based on simulated behaviour only and are not real-world validation. "
            f"A verdict other than INSUFFICIENT EVIDENCE requires at least {MIN_HYPOTHESIS_REFS} cited simulated actions.",
            "",
        ]
        if not self.spec.hypotheses:
            lines.append("The assessment brief defines no hypotheses; none were evaluated.")
            return "\n".join(lines)
        lines.append("| Hypothesis | Verdict | Rationale |")
        lines.append("|---|---|---|")
        for r in self.hypothesis_results:
            lines.append(f"| {_cell(r['id'], 20)}: {_cell(r['text'], 200)} | {r['verdict'].replace('_', ' ').upper()} | {_cell(r['rationale'], 400)} |")
        lines.append("")
        for r in self.hypothesis_results:
            lines.append(f"**{r['id']} — {r['verdict'].replace('_', ' ').upper()}**")
            lines.append("")
            for label, key in (("Evidence for", "evidence_for"), ("Evidence against", "evidence_against")):
                items = r.get(key) or []
                if items:
                    for item in items:
                        lines.append(f"- {label}: {_cell(item['summary'])} ({', '.join(item['refs'])})")
                else:
                    lines.append(f"- {label}: none observed in the simulation")
            if r.get("missing_evidence"):
                lines.append(f"- Missing evidence: {_cell(r['missing_evidence'])}")
            lines.append("")
        return "\n".join(lines).strip()

    # ---------- appendix ----------

    def render_appendix(self, graph_stats: Dict[str, Any]) -> str:
        """Deterministic evidence appendix (no LLM)"""
        lines = [EVIDENCE_DISCLAIMER, ""]
        ev = self.evidence
        if ev is None:
            lines.append("**No simulation run evidence was available for this report.**")
            return "\n".join(lines)
        record = ev.record
        stats = ev.stats()
        lines += [
            "**Run**",
            "",
            f"- Simulation: `{ev.simulation_id}` · Run: `{ev.run_id}`" + (" (legacy layout, read-only)" if ev.run_id == "legacy" else ""),
            f"- Run status: {record.status.value if record else 'unknown'} · started {(record.started_at if record else None) or '—'} · completed {(record.completed_at if record else None) or '—'}",
            f"- Rounds observed in action logs: {stats['rounds_observed'] or 'none'}",
            "",
            "**Simulation action evidence (primary source)**",
            "",
            f"- Twitter actions: {stats['by_platform'].get('twitter', 0)} · Reddit actions: {stats['by_platform'].get('reddit', 0)} · Total: {stats['total_actions']}",
            f"- Action types: {', '.join(f'{k} {v}' for k, v in sorted(stats['by_action_type'].items())) or 'none'}",
        ]
        origins = stats.get("by_origin", {})
        lines.append(
            "- Action origins: " + (", ".join(f"{k} {v}" for k, v in sorted(origins.items())) or "none")
            + ". Only agent_generated actions are behavioural evidence; seed posts and private research "
              "instructions stay in the raw logs for inspection."
        )
        lines.append(f"- Private instructions delivered (agent memory, never posted): {stats.get('private_instructions_delivered', 0)}")
        for warning in ev.integrity.get("warnings", []):
            lines.append(f"- ⚠️ Integrity: {warning}")
        cm = stats.get("consumer_metrics", {})
        lines += [
            "",
            "**Conversion metric scope**",
            "",
            f"- Default scope: `{CONSUMER_METRIC_SCOPE}` (roles from: {ev.roles_source})"
            + (f". {ev.roles_note}" if ev.roles_note else ""),
            f"- Target-consumer personas: {', '.join(cm.get('target_consumer_personas', [])) or 'none'}",
            f"- Counted actions: {cm.get('actions', 0)} · excluded by origin: "
            f"{', '.join(f'{k} {v}' for k, v in cm.get('excluded_by_origin', {}).items()) or 'none'} · "
            f"excluded by evidence role: {', '.join(f'{k} {v}' for k, v in cm.get('excluded_by_evidence_role', {}).items()) or 'none'}",
        ]
        lines += ["", "**Simulation population (canonical IDs)**", "",
                  "| Canonical ID | Display name | Agent IDs | Simulation role | Evidence role | Twitter | Reddit | Total | Excluded (non-agent) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in ev.population():
            c = stats["by_persona"].get(p["canonical_id"], {})
            lines.append(
                f"| {_cell(p['canonical_id'], 40)} | {_cell(p['display_name'], 80)} | {_cell(', '.join(str(i) for i in p['agent_ids']), 40)} | "
                f"{_cell(p['simulation_role'], 30)} | {_cell(p['evidence_role'], 30)} | "
                f"{c.get('twitter', 0)} | {c.get('reddit', 0)} | {c.get('total', 0)} | {p['excluded_actions']} |"
            )
        if ev.roster:
            context = ev.roster.get("context_entities", [])
            lines.append("")
            lines.append(
                f"Population: {len(ev.roster.get('active_agents', []))} active agents / "
                f"{len(ev.roster.get('active_agents', [])) + len(context)} graph entities. "
                f"{len(context)} context-only entities (in the knowledge graph, never simulation accounts): "
                f"{_cell(', '.join(c.get('name', '') for c in context[:40]), 600)}"
            )
        not_present = [pid for pid in self.spec.persona_ids if pid not in ev.persona_index.personas]
        if not_present:
            lines.append("")
            lines.append(f"Requested by the brief but not present in the simulation: {', '.join(not_present)}.")
        lines += ["", "**Source documents**", "", "| File | Type | Extracted chars | Extraction | Used in this report |", "|---|---|---|---|---|"]
        for s in ev.source_manifest():
            lines.append(
                f"| {_cell(s['original_filename'], 80)} | {_cell(s['extension'], 10)} | {s.get('text_length') or 0} | "
                f"{_cell(s['extraction_status'], 20)}{(' — ' + _cell(s['error'], 80)) if s.get('error') else ''} | {'yes' if s['used_in_report'] else 'no'} |"
            )
        if not ev.sources:
            lines.append("| (none) | | | | |")
        lines.append(f"\nSource snapshot origin: {ev.sources_origin}.")
        lines += [
            "",
            "**Assessment spec (derived from the brief)**",
            "",
            f"- Requested persona IDs: {', '.join(self.spec.persona_ids) or 'none'}",
            f"- Hypotheses: {', '.join(h['id'] for h in self.spec.hypotheses) or 'none'}",
            f"- Funnel fields ({self.spec.funnel_fields_source}): {', '.join(self.spec.funnel_fields)}",
            f"- Extraction method: {self.spec.extraction_method}",
            "",
            "**Supplementary graph retrieval**",
            "",
            f"- Graph tool calls: {graph_stats.get('calls', 0)} · results returned: {graph_stats.get('hits', 0)}",
            "- The local graph holds facts extracted from the source documents; it is a supplementary source and never the basis for statements about simulated behaviour.",
            "",
            "**Method**",
            "",
            "- Persona and hypothesis findings are generated from the complete raw action logs of this run and validated in code: claims without a valid action reference are reported as missing, hypotheses without enough cited actions as insufficient evidence.",
            "- Action references (e.g. TW-00012, RD-00003) point to entries in `normalized_evidence.json` of the run export bundle.",
        ]
        return "\n".join(lines).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_spec": self.spec.to_dict(),
            "personas": self.persona_results,
            "hypotheses": self.hypothesis_results,
        }
