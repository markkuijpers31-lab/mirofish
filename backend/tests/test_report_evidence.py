import io
import json
import os
import re
import time
import zipfile

import pytest

from conftest import default_actions, make_prepared_simulation, make_project, write_actions

from app.api import report as report_api
from app.services import simulation_runner as runner_module
from app.services.assessment_spec import AssessmentSpec
from app.services.graph_tools import SearchResult
from app.services.report_agent import ReportAgent, ReportManager, ReportStatus
from app.services.report_evidence_sections import StructuredReportBuilder
from app.services.run_evidence import RunEvidence
from app.services.run_store import RunStatus, RunStore
from app.services.simulation_runner import SimulationRunner


class EmptyGraphTools:
    """Graph retrieval that never finds anything (e.g. failed graph ingestion)"""

    def __init__(self):
        self.calls = 0

    def get_simulation_context(self, **kwargs):
        self.calls += 1
        return {"graph_statistics": {}, "related_facts": [], "total_entities": 0}

    def quick_search(self, graph_id, query, limit=10):
        self.calls += 1
        return SearchResult(facts=[], edges=[], nodes=[], query=query, total_count=0)


class FakeLLM:
    """Deterministic LLM: answers by prompt type, cites refs from the given evidence"""

    def __init__(self):
        self.persona_prompts = {}

    def chat_json(self, messages, temperature=0.3, max_tokens=8192):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "structure of a commercial assessment brief" in system:
            return {
                "persona_ids": ["C01", "C02", "C03"],
                "hypotheses": [
                    {"id": "H1", "text": "Families are willing to pay for the premium plan."},
                    {"id": "H2", "text": "Couples prefer the free alternative."},
                ],
                "funnel_fields": ["starting workflow", "pricing exposure", "checkout outcome"],
            }
        if "You analyse ONE persona" in system:
            cid = re.search(r"Persona canonical id: (\S+)", user).group(1)
            self.persona_prompts[cid] = user
            if cid == "C01":
                return {
                    "fields": [
                        {"field": "starting workflow", "status": "observed", "summary": "Searched for a planner", "refs": ["TW-00001"]},
                        {"field": "pricing exposure", "status": "observed", "summary": "Saw 4.99 and found it expensive", "refs": ["TW-00002"]},
                        # Invented outcome without any citation: must become missing
                        {"field": "checkout outcome", "status": "observed", "summary": "Bought the premium plan", "refs": []},
                    ],
                    "strongest_evidence_for": {"summary": "Actively looked for a planner", "refs": ["TW-00001"]},
                    "strongest_evidence_against": {"summary": "Price objection", "refs": ["TW-00002"]},
                }
            return {
                "fields": [
                    {"field": "starting workflow", "status": "ambiguous", "summary": "Commented on launch", "refs": ["RD-00001"]},
                    # Cites another persona's action: invalid, must become missing
                    {"field": "pricing exposure", "status": "observed", "summary": "Saw the price", "refs": ["TW-00002"]},
                ],
                "strongest_evidence_against": {"summary": "Sticks with free app", "refs": ["RD-00001"]},
            }
        if "You assess ONE hypothesis" in system:
            if "H1" in user.split("\n")[0]:
                return {"verdict": "supported", "rationale": "One family post", "evidence_for": [{"summary": "post", "refs": ["TW-00001"]}]}
            return {
                "verdict": "supported",
                "rationale": "Couple sticks with free app; family finds price high",
                "evidence_for": [{"summary": "free app", "refs": ["RD-00001"]}],
                "evidence_against": [{"summary": "price", "refs": ["TW-00002"]}],
            }
        # Outline planning
        return {"title": "Kamptribe test", "summary": "Simulated findings", "sections": [{"title": "Cross-Persona Findings"}]}

    def chat(self, messages, temperature=0.5, max_tokens=8192, response_format=None):
        observations = sum(1 for m in messages if m["role"] == "user" and "Observation" in m["content"])
        plan = [
            '<tool>{"name": "simulation_actions", "parameters": {"query": "price"}}</tool>',
            '<tool>{"name": "quick_search", "parameters": {"query": "premium plan price"}}</tool>',
            '<tool>{"name": "persona_evidence", "parameters": {"persona": "C01"}}</tool>',
        ]
        if observations < len(plan):
            return "Thought: need evidence\n" + plan[observations]
        return "Final Answer: In the simulation, C01 objected to the price [TW-00002]."


def _completed_run(simulation_id="sim_test"):
    make_project()
    make_prepared_simulation(simulation_id=simulation_id)
    record = RunStore.create_run(simulation_id)
    run_dir = RunStore.run_dir(simulation_id, record.run_id)
    for platform in ("twitter", "reddit"):
        write_actions(run_dir, platform, default_actions(platform), run_id=record.run_id)
    RunStore.set_status(simulation_id, record.run_id, RunStatus.COMPLETED)
    return record.run_id


def _generate(run_id):
    graph = EmptyGraphTools()
    agent = ReportAgent(
        graph_id="graph_test",
        simulation_id="sim_test",
        simulation_requirement="req",
        llm_client=FakeLLM(),
        graph_tools=graph,
        run_id=run_id,
    )
    report = agent.generate_report(report_id="report_test")
    return agent, report, graph


def test_report_reads_raw_actions_when_graph_retrieval_is_empty(uploads):
    run_id = _completed_run()

    agent, report, graph = _generate(run_id)

    assert report.status == ReportStatus.COMPLETED, report.error
    assert report.run_id == run_id
    assert graph.calls > 0  # the graph was asked, returned nothing, and the report still has evidence
    folder = ReportManager._get_report_folder("report_test")
    data = json.load(open(os.path.join(folder, "report.json")))
    c01 = next(p for p in data["personas"] if p["canonical_id"] == "C01")
    assert {f["field"]: f["status"] for f in c01["fields"]} == {
        "starting workflow": "observed",
        "pricing exposure": "observed",
        "checkout outcome": "missing",
    }
    assert c01["action_counts"]["total"] == 2  # both "C01" and "C01 Dutch first-time family"
    assert data["graph_retrieval"]["hits"] == 0

    evidence = json.load(open(os.path.join(folder, "evidence.json")))
    assert {a["canonical_id"] for a in evidence["actions"] if a["agent_id"] in (0, 1)} == {"C01"}
    assert evidence["stats"]["by_platform"] == {"twitter": 2, "reddit": 1}

    # The ReACT agent called the raw evidence tool and saw real actions
    log_lines = [json.loads(line) for line in open(os.path.join(folder, "agent_log.jsonl"))]
    results = [l for l in log_lines if l["action"] == "tool_result" and l["details"]["tool_name"] == "simulation_actions"]
    assert results and "TW-00002" in results[0]["details"]["result"]


def test_report_covers_every_requested_persona_and_names_missing_evidence(uploads):
    run_id = _completed_run()
    _agent, report, _graph = _generate(run_id)

    markdown = report.markdown_content
    for pid in ("C01", "C02", "C03"):
        assert f"**{pid} — " in markdown, pid
    # C03 is requested by the brief but never simulated: explicitly missing, nothing invented
    c03_block = markdown.split("**C03 — ")[1].split("**Strongest evidence for:**")[0]
    assert "Not present in the simulation" in c03_block
    assert c03_block.count("| MISSING |") == 3
    # C02 claimed a price exposure citing C01's action: discarded as missing
    c02_block = markdown.split("**C02 — ")[1].split("**C03 — ")[0]
    assert "| pricing exposure | MISSING |" in c02_block
    assert "Bought the premium plan" not in markdown
    assert "not real customers" in markdown or "not observations of real customers" in markdown


def test_hypotheses_need_enough_cited_evidence(uploads):
    run_id = _completed_run()
    agent, _report, _graph = _generate(run_id)

    verdicts = {h["id"]: h for h in agent.structured.hypothesis_results}
    assert verdicts["H1"]["verdict"] == "insufficient_evidence"
    assert verdicts["H1"]["verdict_forced_by_evidence_rule"] is True
    assert verdicts["H2"]["verdict"] == "supported"
    assert verdicts["H2"]["evidence_against"]  # negative evidence is kept


def test_persona_without_actions_is_missing_without_llm(uploads):
    run_id = _completed_run()
    evidence = RunEvidence.load("sim_test", run_id)
    spec = AssessmentSpec(persona_ids=["C01"], funnel_fields=["first choice"])
    evidence.actions = []

    class NoLLM:
        def chat_json(self, **kwargs):
            raise AssertionError("LLM must not be called without evidence")

    builder = StructuredReportBuilder(evidence, spec, NoLLM())
    result = builder.analyse_personas()[0]
    assert result["fields"][0]["status"] == "missing"
    assert result["missing_stages"] == ["first choice"]


def test_report_generation_never_starts_a_simulation(uploads, client, monkeypatch):
    run_id = _completed_run()
    runs_before = len(RunStore.list_runs("sim_test"))

    def forbidden(*args, **kwargs):
        raise AssertionError("report generation must not start a simulation")

    monkeypatch.setattr(SimulationRunner, "start_simulation", classmethod(forbidden))
    monkeypatch.setattr(runner_module.subprocess, "Popen", forbidden)
    monkeypatch.setattr(
        report_api,
        "ReportAgent",
        lambda **kwargs: ReportAgent(llm_client=FakeLLM(), graph_tools=EmptyGraphTools(), **kwargs),
    )

    response = client.post("/api/report/generate", json={"simulation_id": "sim_test", "force_regenerate": True})
    assert response.status_code == 200, response.json
    report_id = response.json["data"]["report_id"]
    assert response.json["data"]["run_id"] == run_id

    deadline = time.time() + 20
    while time.time() < deadline:
        if RunStore.get_run("sim_test", run_id).status == RunStatus.REPORT_COMPLETED:
            break
        time.sleep(0.05)
    record = RunStore.get_run("sim_test", run_id)
    assert record.status == RunStatus.REPORT_COMPLETED
    assert record.latest_report_id == report_id
    assert len(RunStore.list_runs("sim_test")) == runs_before

    # Regenerating from the same immutable evidence works and still starts nothing
    again = client.post("/api/report/generate", json={"simulation_id": "sim_test", "force_regenerate": True})
    assert again.status_code == 200

    # The export bundle holds everything needed for an audit
    deadline = time.time() + 20
    while time.time() < deadline and RunStore.get_run("sim_test", run_id).status != RunStatus.REPORT_COMPLETED:
        time.sleep(0.05)
    bundle = client.get(f"/api/report/{report_id}/export")
    assert bundle.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(bundle.data)).namelist())
    for required in (
        "MANIFEST.json", "run/run.json", "config/simulation_config.json", "sources/manifest.json",
        "assessment/brief.md", "assessment/spec.json", "personas/reddit_profiles.json",
        "personas/personas_normalized.json", "actions/twitter_actions.jsonl", "actions/reddit_actions.jsonl",
        "evidence/normalized_evidence.json", "report/full_report.md", "report/report.json",
        "report/agent_log.jsonl", "report/console_log.txt",
    ):
        assert required in names, required
    run_dir = RunStore.run_dir("sim_test", run_id)
    archive = zipfile.ZipFile(io.BytesIO(bundle.data))
    assert archive.read("actions/twitter_actions.jsonl") == open(os.path.join(run_dir, "twitter", "actions.jsonl"), "rb").read()


def test_report_is_refused_for_a_running_run(uploads, client):
    make_project()
    make_prepared_simulation()
    record = RunStore.create_run("sim_test")
    RunStore.set_status("sim_test", record.run_id, RunStatus.RUNNING, process_pid=os.getpid())
    SimulationRunner._active_runs["sim_test"] = record.run_id

    class Alive:
        def poll(self):
            return None

    SimulationRunner._processes["sim_test"] = Alive()
    response = client.post("/api/report/generate", json={"simulation_id": "sim_test"})
    assert response.status_code == 400


@pytest.mark.parametrize("wrapper", ["tool", "tool_call"])
def test_tool_call_parser_accepts_prompted_format(uploads, wrapper):
    agent = ReportAgent.__new__(ReportAgent)
    response = f'Thought\n<{wrapper}>{{"name": "simulation_actions", "parameters": {{"persona": "C01"}}}}</{wrapper}>'
    calls = agent._parse_tool_calls(response)
    assert calls == [{"name": "simulation_actions", "parameters": {"persona": "C01"}}]


def _prompt_tool_example(prompt):
    """The tool-call example a prompt shows the model, e.g. <tool>{...}</tool>"""
    match = re.search(r"(<(\w+)>\s*(\{.*?\})\s*</\2>)", prompt, re.DOTALL)
    assert match, "prompt no longer contains a tool-call example"
    return match.group(2), match.group(3)


def test_every_prompted_tool_syntax_is_understood_by_the_parser():
    from app.services.report_agent import CHAT_SYSTEM_PROMPT_TEMPLATE, SECTION_SYSTEM_PROMPT_TEMPLATE

    section_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
        report_title="t", report_summary="s", simulation_requirement="r", section_title="x", tools_description=""
    )
    chat_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
        simulation_requirement="r", report_content="c", tools_description=""
    )
    agent = ReportAgent.__new__(ReportAgent)
    for prompt in (section_prompt, chat_prompt):
        tag, example = _prompt_tool_example(prompt)
        example = example.replace("tool_name", "simulation_actions")
        response = f"Thought: I need evidence\n<{tag}>\n{example}\n</{tag}>"
        assert agent._parse_tool_calls(response) == [
            {"name": "simulation_actions", "parameters": {"param_name": "param_value"}}
        ], tag


class PromptFollowingLLM(FakeLLM):
    """Calls tools with exactly the syntax the section prompt teaches, and records what it sees"""

    def __init__(self):
        super().__init__()
        self.observations = []

    def chat(self, messages, temperature=0.5, max_tokens=8192, response_format=None):
        tag, _ = _prompt_tool_example(messages[0]["content"])
        seen = [m["content"] for m in messages if m["role"] == "user" and "Observation" in m["content"]]
        if not seen:
            return f'Thought: look at the raw actions\n<{tag}>\n{{"name": "simulation_actions", "parameters": {{"query": "expensive"}}}}\n</{tag}>'
        self.observations.append(seen[-1])
        return "Final Answer: In the simulation, C01 found the plan too expensive [TW-00002]."


def test_prompted_tool_call_reaches_the_tool_and_its_result_reaches_the_report(uploads):
    run_id = _completed_run()
    llm = PromptFollowingLLM()
    agent = ReportAgent(
        graph_id="graph_test", simulation_id="sim_test", simulation_requirement="req",
        llm_client=llm, graph_tools=EmptyGraphTools(), run_id=run_id,
    )
    report = agent.generate_report(report_id="report_parse")

    assert report.status == ReportStatus.COMPLETED, report.error
    # The tool really ran on the raw actions and its output went back to the model
    assert llm.observations, "the tool call was never executed"
    assert "TW-00002" in llm.observations[0] and "too expensive" in llm.observations[0]
    folder = ReportManager._get_report_folder("report_parse")
    log_lines = [json.loads(line) for line in open(os.path.join(folder, "agent_log.jsonl"))]
    assert any(l["action"] == "tool_call" and l["details"]["tool_name"] == "simulation_actions" for l in log_lines)
    assert "[TW-00002]" in report.markdown_content
