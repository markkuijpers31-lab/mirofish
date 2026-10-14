"""
Report Agent service.

ReACT-pattern simulation report generator.

Evidence model: the raw action logs of ONE simulation run are the primary source
of truth (run_evidence.RunEvidence); the local graph store (facts extracted from
the source documents) is a supplementary retrieval source only.

Features:
1. Generate reports from a run's raw simulation evidence, source documents and assessment brief.
2. Plan the outline structure first, then generate section by section.
3. Each section uses a ReACT multi-round thinking-and-reflection pattern.
4. Support chatting with the user, autonomously invoking retrieval tools during the conversation.
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger, get_run_logger
from ..utils.locale import get_language_instruction, t
from ..utils.persona_ids import parse_identifier
from .assessment_spec import AssessmentSpec, extract_assessment_spec
from .run_evidence import RunEvidence
from .report_evidence_sections import (
    APPENDIX_SECTION_TITLE,
    HYPOTHESIS_SECTION_TITLE,
    PERSONA_SECTION_TITLE,
    STRUCTURED_SECTION_TITLES,
    StructuredReportBuilder,
)
from .graph_tools import (
    GraphToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult,
)

logger = get_logger("mirofish.report_agent")


class ReportLogger:
    """
    Detailed logger for the Report Agent.

    Generates an agent_log.jsonl file in the report folder, recording every
    detailed action. Each line is a complete JSON object containing a timestamp,
    action type, detailed content, etc.
    """

    def __init__(self, report_id: str):
        """
        Initialize the logger.

        Args:
            report_id: Report ID, used to determine the log file path.
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "agent_log.jsonl"
        )
        self.start_time = datetime.now()
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure the directory containing the log file exists."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """Return elapsed time in seconds since the start."""
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None,
    ):
        """
        Record a single log entry.

        Args:
            action: Action type, e.g. 'start', 'tool_call', 'llm_response',
                'section_complete', etc.
            stage: Current stage, e.g. 'planning', 'generating', 'completed'.
            details: Detailed content dict, not truncated.
            section_title: Title of the current section (optional).
            section_index: Index of the current section (optional).
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details,
        }

        # Append to JSONL file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Record the start of report generation."""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": t("report.taskStarted"),
            },
        )

    def log_planning_start(self):
        """Record the start of outline planning."""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": t("report.planningStart")},
        )

    def log_planning_context(self, context: Dict[str, Any]):
        """Record the context information obtained during planning."""
        self.log(
            action="planning_context",
            stage="planning",
            details={"message": t("report.fetchSimContext"), "context": context},
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Record outline planning completion."""
        self.log(
            action="planning_complete",
            stage="planning",
            details={"message": t("report.planningComplete"), "outline": outline_dict},
        )

    def log_section_start(self, section_title: str, section_index: int):
        """Record the start of section generation."""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": t("report.sectionStart", title=section_title)},
        )

    def log_react_thought(
        self, section_title: str, section_index: int, iteration: int, thought: str
    ):
        """Record a ReACT thinking step."""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": t("report.reactThought", iteration=iteration),
            },
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int,
    ):
        """Record a tool call."""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": t("report.toolCall", toolName=tool_name),
            },
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int,
    ):
        """Record a tool call result (full content, not truncated)."""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Full result, no truncation
                "result_length": len(result),
                "message": t("report.toolResult", toolName=tool_name),
            },
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool,
    ):
        """Record an LLM response (full content, not truncated)."""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Full response, no truncation
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": t(
                    "report.llmResponse",
                    hasToolCalls=has_tool_calls,
                    hasFinalAnswer=has_final_answer,
                ),
            },
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int,
    ):
        """Record that section content generation is complete (records content only; does not mean the whole section is done)."""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Full content, no truncation
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": t("report.sectionContentDone", title=section_title),
            },
        )

    def log_section_full_complete(
        self, section_title: str, section_index: int, full_content: str
    ):
        """
        Record that section generation is complete.

        The frontend should listen for this log to determine whether a section
        is truly finished and to obtain the full content.
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": t("report.sectionComplete", title=section_title),
            },
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Record report generation completion."""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": t("report.reportComplete"),
            },
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Record an error."""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": t("report.errorOccurred", error=error_message),
            },
        )


class ReportConsoleLogger:
    """
    Console logger for the Report Agent.

    Writes console-style logs (INFO, WARNING, etc.) to a console_log.txt file
    inside the report folder. These logs are plain-text console output, distinct
    from agent_log.jsonl.
    """

    def __init__(self, report_id: str):
        """
        Initialize the console logger.

        Args:
            report_id: Report ID, used to determine the log file path.
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "console_log.txt"
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()

    def _ensure_log_file(self):
        """Ensure the directory containing the log file exists."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """Set up the file handler so logs are also written to the file."""
        import logging

        # Create file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path, mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.INFO)

        # Use same concise format as console
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self._file_handler.setFormatter(formatter)

        # Add to report_agent logger
        loggers_to_attach = [
            "mirofish.report_agent",
            "mirofish.graph_tools",
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate handlers
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        """Close the file handler and remove it from the loggers."""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                "mirofish.report_agent",
                "mirofish.graph_tools",
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """Ensure the file handler is closed on destruction."""
        self.close()


class ReportStatus(str, Enum):
    """Report status."""

    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """A report section."""

    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content}

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown format."""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline."""

    title: str
    summary: str
    sections: List[ReportSection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """A complete report."""

    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════
# Prompt template constants
# ═══════════════════════════════════════════════════════════════

# ── Tool Descriptions ──

TOOL_DESC_SIMULATION_ACTIONS = """\
[PRIMARY EVIDENCE - Raw simulation actions of this run]
Searches the complete Twitter and Reddit action logs (actions.jsonl) of the run this report is about.
Every result carries a citeable reference such as [TW-00012] or [RD-00003].
Filters: query (keywords, any term), persona (canonical id such as C01, or a name), platform
(twitter/reddit), action_type (e.g. CREATE_POST, CREATE_COMMENT, LIKE_POST), limit.
Use this for every statement about what agents did or said."""

TOOL_DESC_PERSONA_EVIDENCE = """\
[PRIMARY EVIDENCE - Everything one persona did]
Returns the persona definition and the complete chronological list of that persona's simulated
actions on both platforms. Parameter: persona (canonical id such as C01, or a name)."""

TOOL_DESC_EVIDENCE_STATS = """\
[PRIMARY EVIDENCE - Counts and integrity]
Returns simulation counts per platform, action type and persona, personas without any action,
the personas requested by the assessment brief and data-integrity warnings. No parameters.
Counts are simulation counts, never real-world rates."""

TOOL_DESC_SOURCE_DOCUMENTS = """\
[CONTEXT - Source documents and assessment brief]
Returns text from the uploaded source documents (PDF/MD/TXT) of this run's project.
Parameters: name (optional, part of a filename), query (optional keywords).
Source documents describe the product/context; they are not evidence of simulated behaviour."""

TOOL_DESC_SIMULATION_CONFIG = """\
[CONTEXT - Simulation configuration]
Returns the run's simulation configuration: time settings, initial posts/events, hot topics and
the configured agents (entity names and types). No parameters."""

TOOL_DESC_INSIGHT_FORGE = """\
[SUPPLEMENTARY - graph of facts extracted from the SOURCE DOCUMENTS, not simulation actions]
[Deep Insight Retrieval - Powerful Retrieval Tool]
This is our powerful retrieval function, designed for deep analysis. It will:
1. Automatically decompose your question into multiple sub-questions
2. Retrieve information from the simulation graph across multiple dimensions
3. Integrate results from semantic search, entity analysis, and relationship chain tracing
4. Return the most comprehensive and in-depth retrieval content

[Use Cases]
- When you need to deeply analyze a topic
- When you need to understand multiple aspects of an event
- When you need to gather rich material to support a report section

[Returned Content]
- Relevant fact originals (directly quotable)
- Core entity insights
- Relationship chain analysis"""

TOOL_DESC_PANORAMA_SEARCH = """\
[SUPPLEMENTARY - graph of facts extracted from the SOURCE DOCUMENTS, not simulation actions]
[Breadth Search - Get the Full Picture View]
This tool is used to get the complete picture of simulation results, especially suited for understanding how events evolve. It will:
1. Retrieve all relevant nodes and relationships
2. Distinguish currently valid facts from historical/expired facts
3. Help you understand how public opinion evolved

[Use Cases]
- When you need to understand the complete development trajectory of an event
- When you need to compare changes in public opinion across different stages
- When you need to obtain comprehensive entity and relationship information

[Returned Content]
- Currently valid facts (latest simulation results)
- Historical/expired facts (evolution records)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\
[SUPPLEMENTARY - graph of facts extracted from the SOURCE DOCUMENTS, not simulation actions]
[Quick Search - Fast Retrieval]
A lightweight fast retrieval tool, suitable for simple, direct information queries.

[Use Cases]
- When you need to quickly find a specific piece of information
- When you need to verify a fact
- Simple information retrieval

[Returned Content]
- A list of facts most relevant to the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[In-Depth Interview - Real Agent Interviewing (Dual-Platform)]
Calls the OASIS simulation environment's interview API to conduct real interviews with running simulation agents!
This is not an LLM simulation, but a call to the real interview interface to get raw answers from simulation agents.
By default, interviews are conducted simultaneously on both Twitter and Reddit platforms to obtain more comprehensive views.

Functional flow:
1. Automatically reads persona files to understand all simulation agents
2. Intelligently selects the agents most relevant to the interview topic (e.g., students, media, officials, etc.)
3. Automatically generates interview questions
4. Calls the /api/simulation/interview/batch interface to conduct real interviews on both platforms
5. Integrates all interview results to provide multi-perspective analysis

[Use Cases]
- When you need to understand views on an event from different role perspectives (What do students think? What does the media think? What do officials say?)
- When you need to collect multiple opinions and positions
- When you need to get real answers from simulation agents (from the OASIS simulation environment)
- When you want to make the report more vivid by including "interview transcripts"

[Returned Content]
- Identity information of interviewed agents
- Each agent's interview answers on both Twitter and Reddit platforms
- Key quotes (directly quotable)
- Interview summaries and perspective comparisons

[Important] Requires the OASIS simulation environment of this run to be running. Interview answers
are produced after the simulation ended; label them as post-simulation interviews, not as observed behaviour."""

# ── Outline Planning Prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an analyst writing an evidence-based assessment report about ONE run of an agent-based
social-media simulation (OASIS: simulated Twitter and Reddit populated by synthetic personas).

[Evidence Model]
- Primary evidence: the raw action logs of this run (what each simulated persona posted, commented,
  liked, searched...). They are synthetic: they show how simulated personas behaved, not how real
  customers behave.
- Context: the assessment brief, the source/product documents and the simulation configuration.
- Supplementary: a graph of facts extracted from the source documents.

[Fixed Sections - already generated separately, do NOT plan them]
- "Persona Analysis" (every requested persona individually, funnel fields, missing stages)
- "Hypothesis Assessment" (each hypothesis of the brief with a verdict)
- "Evidence Base & Methodology" (counts, sources, integrity)

[Your Task]
Plan 2 to 4 additional synthesis sections answering the assessment brief across personas, e.g.
cross-persona patterns, frictions and objections, open questions, recommendations for real-world
validation. Never plan sections that would need data the simulation does not contain.

Output JSON:
{
    "title": "Report Title",
    "summary": "One sentence summarizing the main simulated finding (label it as simulated)",
    "sections": [
        {"title": "Section Title", "description": "Section Content Description"}
    ]
}

Note: the sections array must contain 2 to 4 elements."""

PLAN_USER_PROMPT_TEMPLATE = """\
[Assessment Brief]
{simulation_requirement}

[Simulation Evidence of This Run (simulation counts)]
{evidence_stats_json}

[Data Integrity]
{integrity_warnings}

[Sample of Simulated Actions]
{action_sample}

[Supplementary: graph of source-document facts]
- Entities: {total_nodes}, relationships: {total_edges}, entity types: {entity_types}
- Sample facts: {related_facts_json}

Plan the synthesis sections (2-4)."""

# ── Section Generation Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert writer of "Future Prediction Reports", currently writing one section of the report.

Report Title: {report_title}
Report Summary: {report_summary}
Prediction Scenario (Simulation Requirement): {simulation_requirement}

Current Section to Write: {section_title}

═══════════════════════════════════════════════════════════════
[Core Concept]
═══════════════════════════════════════════════════════════════

This report is about ONE run of an agent-based social-media simulation (simulated Twitter and
Reddit, synthetic personas). The simulation shows how simulated personas behaved under the
conditions of the assessment brief. It is not real-world customer data.

═══════════════════════════════════════════════════════════════
[Most Important Rules - Must Be Followed]
═══════════════════════════════════════════════════════════════

1. [Raw Simulation Actions Are the Primary Evidence]
   - Use simulation_actions, persona_evidence and evidence_stats to observe what happened in this run
   - Every statement about simulated behaviour must cite action references, e.g. [TW-00012] or [RD-00003]
   - Graph tools (insight_forge, panorama_search, quick_search) return facts extracted from the
     SOURCE DOCUMENTS; use them only as context and never present them as simulated behaviour
   - Forbidden to use your own knowledge to fill gaps

2. [Simulated Is Not Real-World]
   - Write "in the simulation", "simulated persona C01 ...", never "customers will ..."
   - Do NOT present percentages, conversion rates or market shares; mention counts only as
     simulation counts (e.g. "3 of the 10 simulated personas")

3. [Missing Evidence Stays Missing]
   - If the simulation does not show something, write explicitly that it was not observed
   - Never fill gaps with assumptions; keep negative evidence (objections, distrust, abandonment)

4. [Quote Simulated Statements]
   - Quote simulated posts/comments with their reference:
     > "original content" [RD-00003]

5. [Language Consistency - Quoted Content Must Be Translated to the Report Language]
   - The report must be written entirely in the user-specified language
   - Translate quoted tool content into the report language, keeping its meaning and the reference

6. [Consistency With the Fixed Sections]
   - The completed sections (Persona Analysis, Hypothesis Assessment) were validated against the raw
     actions; do not contradict them

═══════════════════════════════════════════════════════════════
[⚠️ Format Specifications - Extremely Important!]
═══════════════════════════════════════════════════════════════

[One Section = Minimum Content Unit]
- Each section is the minimum block unit of the report
- ❌ Forbidden to use any Markdown headings (#, ##, ###, ####, etc.) within a section
- ❌ Forbidden to add the section main title at the beginning of content
- ✅ Section titles are automatically added by the system; you only need to write pure body content
- ✅ Use **bold**, paragraph breaks, quotes, lists to organize content, but do not use headings

[Correct Example]
```
In the simulation, price was the most discussed friction point among the family personas...

**Price Objections**

Simulated persona C03 rejected the paid plan after comparing it with a free alternative:

> "Why would I pay for this if the campsite app is free?" [RD-00041]

**Not Observed**

- No simulated persona reached a checkout step in this run
```

[Wrong Example]
```
## Executive Summary          ← Wrong! Do not add any headings
### I. Initial Stage          ← Wrong! Do not use ### for subsections
#### 1.1 Detailed Analysis    ← Wrong! Do not use #### for subdivisions

This section analyzes...
```

═══════════════════════════════════════════════════════════════
[Available Tools] (Call 3-5 times per section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Tool Usage Suggestions]
- Start with evidence_stats and simulation_actions / persona_evidence (primary evidence)
- source_documents / simulation_config: context about the product and the simulation setup
- insight_forge / panorama_search / quick_search: supplementary source-document facts
- interview_agents: only if the run's environment is still running; label answers as post-simulation interviews

═══════════════════════════════════════════════════════════════
[Workflow]
═══════════════════════════════════════════════════════════════

Each reply you can only do one of the following two things (not both at the same time):

Option A - Call a tool:
Output your thought, then call a tool using the following format:
<tool>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool>
The system will execute the tool and return the result to you. You do not need to and cannot write tool return results yourself.

Option B - Output final content:
When you have obtained enough information through tools, output the section content starting with "Final Answer:".

⚠️ Strictly forbidden:
- Forbidden to include both a tool call and a Final Answer in the same reply
- Forbidden to fabricate tool return results (Observation) yourself; all tool results are injected by the system
- At most one tool call per reply

═══════════════════════════════════════════════════════════════
[Section Content Requirements]
═══════════════════════════════════════════════════════════════

1. Content must be based on this run's simulated actions retrieved by tools, with action references
2. Quote simulated statements with their references
3. Use Markdown format (but headings are forbidden):
   - Use **bold text** to highlight key points (instead of subheadings)
   - Use lists (- or 1.2.3.) to organize points
   - Use blank lines to separate different paragraphs
   - ❌ Forbidden to use any heading syntax such as #, ##, ###, ####
4. [Quote Format Specification - Must Be a Separate Paragraph]
   Quotes must be independent paragraphs, with a blank line before and after each, and cannot be mixed within paragraphs:

   ✅ Correct format:
   ```
   The school's response was considered to lack substance.

   > "The school's response pattern appeared rigid and sluggish in the rapidly changing social media environment."

   This assessment reflects the public's general dissatisfaction.
   ```

   ❌ Wrong format:
   ```
   The school's response was considered to lack substance.> "The school's response pattern..." This assessment reflects...
   ```
5. Maintain logical coherence with other sections
6. [Avoid Repetition] Carefully read the completed section content below and do not repeat the same information
7. [Reiterated] Do not add any headings! Use **bold** instead of subheadings"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed section content (please read carefully to avoid repetition):
{previous_content}

═══════════════════════════════════════════════════════════════
[Current Task] Write section: {section_title}
═══════════════════════════════════════════════════════════════

[Important Reminders]
1. Carefully read the completed sections above to avoid repeating the same content!
2. Must call tools to get this run's simulated actions before starting
3. Cite action references for statements about simulated behaviour
4. Report content must come from tool results; state explicitly what was not observed

[⚠️ Format Warning - Must Be Followed]
- ❌ Do not write any headings (#, ##, ###, #### are all not allowed)
- ❌ Do not write "{section_title}" as the beginning
- ✅ Section titles are automatically added by the system
- ✅ Write body text directly, use **bold** instead of subheadings

Please begin:
1. First think (Thought) about what information this section needs
2. Then call a tool (Action) to get simulation data
3. After collecting enough information, output Final Answer (pure body text, no headings at all)"""

# ── ReACT Loop Message Templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (Retrieval Result):

═══ Tool {tool_name} Returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tools called {tool_calls_count}/{max_tool_calls} times (used: {used_tools_str}){unused_hint}
- If information is sufficient: output section content starting with "Final Answer:" (must quote the original text above)
- If more information is needed: call a tool to continue retrieving
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Note] You have only called tools {tool_calls_count} times; at least {min_tool_calls} times are required."
    "Please call tools again to get more simulation data, then output Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Currently only {tool_calls_count} tool calls have been made; at least {min_tool_calls} are required."
    "Please call a tool to get simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}); cannot call tools anymore."
    'Please immediately output section content starting with "Final Answer:" based on the information already obtained.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 You have not yet used: {unused_list}; it is recommended to try different tools to get multi-angle information"

REACT_FORCE_FINAL_MSG = "Tool call limit reached. Please output Final Answer: directly and generate the section content."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient assistant for one simulation run's assessment report.

[Background]
Assessment brief: {simulation_requirement}
Evidence rules: simulated actions are synthetic, not real-world data; cite action references; say
explicitly when something was not observed; no percentages or conversion rates.

[Generated Analysis Report]
{report_content}

[Rules]
1. Prioritize answering questions based on the report content above
2. Answer questions directly; avoid lengthy reasoning and discussion
3. Only call tools to retrieve more data when the report content is insufficient to answer
4. Answers should be concise, clear, and well-organized

[Available Tools] (use only when needed, at most 1-2 calls)
{tools_description}

[Tool Call Format]
<tool>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool>

[Answer Style]
- Concise and direct; no long-winded explanations
- Use > format to quote key content
- Give conclusions first, then explain reasons"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent main class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - the simulation-report-generation agent.

    Uses a ReACT (Reasoning + Acting) pattern:
    1. Planning stage: analyze the simulation requirement and plan the report outline.
    2. Generation stage: generate content section by section; each section may call
       tools multiple times to retrieve information.
    3. Reflection stage: check content completeness and accuracy.
    """

    # Max tool calls (per section)
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Max reflection rounds
    MAX_REFLECTION_ROUNDS = 3

    # Max tool calls in conversation
    MAX_TOOL_CALLS_PER_CHAT = 2

    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        graph_tools: Optional[GraphToolsService] = None,
        run_id: Optional[str] = None,
        evidence: Optional[RunEvidence] = None,
    ):
        """
        Initialize the Report Agent.

        Args:
            graph_id: Graph ID (supplementary retrieval source).
            simulation_id: Simulation ID.
            simulation_requirement: Description of the simulation requirement.
            llm_client: LLM client (optional).
            graph_tools: Graph tools service (optional).
            run_id: Run whose evidence the report is built from (default: current run).
            evidence: Pre-loaded RunEvidence (optional, mainly for tests).
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        self.graph_tools = graph_tools or GraphToolsService()

        self.evidence: Optional[RunEvidence] = evidence
        self._evidence_loaded = evidence is not None
        self.evidence_error: Optional[str] = None
        self.run_id = run_id or (evidence.run_id if evidence else None)
        self.spec: Optional[AssessmentSpec] = None
        self.structured: Optional[StructuredReportBuilder] = None
        self.graph_stats = {"calls": 0, "hits": 0}
        self.log = get_run_logger("mirofish.report_agent", simulation_id, self.run_id, "report")

        # Tool definitions
        self.tools = self._define_tools()

        # Logger (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(
            t("report.agentInitDone", graphId=graph_id, simulationId=simulation_id)
        )

    def _ensure_evidence(self) -> Optional[RunEvidence]:
        """Load the run's raw evidence once (None if the simulation has no run)"""
        if not self._evidence_loaded:
            self._evidence_loaded = True
            try:
                self.evidence = RunEvidence.load(self.simulation_id, self.run_id)
                self.run_id = self.evidence.run_id
                self.log = get_run_logger("mirofish.report_agent", self.simulation_id, self.run_id, "report")
            except Exception as e:
                self.evidence_error = str(e)
                self.log.warning(f"No run evidence available: {e}")
        return self.evidence

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define the available tools (raw evidence first, graph as supplement)."""
        return {
            "simulation_actions": {
                "name": "simulation_actions",
                "description": TOOL_DESC_SIMULATION_ACTIONS,
                "parameters": {
                    "query": "Keywords (optional)",
                    "persona": "Canonical persona id or name (optional)",
                    "platform": "twitter or reddit (optional)",
                    "action_type": "e.g. CREATE_POST (optional)",
                    "limit": "Max results (optional, default 40)",
                },
            },
            "persona_evidence": {
                "name": "persona_evidence",
                "description": TOOL_DESC_PERSONA_EVIDENCE,
                "parameters": {"persona": "Canonical persona id (e.g. C01) or name"},
            },
            "evidence_stats": {
                "name": "evidence_stats",
                "description": TOOL_DESC_EVIDENCE_STATS,
                "parameters": {},
            },
            "source_documents": {
                "name": "source_documents",
                "description": TOOL_DESC_SOURCE_DOCUMENTS,
                "parameters": {
                    "name": "Part of a filename (optional)",
                    "query": "Keywords (optional)",
                },
            },
            "simulation_config": {
                "name": "simulation_config",
                "description": TOOL_DESC_SIMULATION_CONFIG,
                "parameters": {},
            },
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to deeply analyze",
                    "report_context": "Context of the current report section (optional, helps generate more precise sub-questions)",
                },
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query, used for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default True)",
                },
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default 10)",
                },
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g.: 'Understand students' views on the dormitory formaldehyde incident')",
                    "max_agents": "Maximum number of agents to interview (optional, default 5, max 10)",
                },
            },
        }

    def _execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], report_context: str = ""
    ) -> str:
        """
        Execute a tool call.

        Args:
            tool_name: Tool name.
            parameters: Tool parameters.
            report_context: Report context (used by InsightForge).

        Returns:
            Tool execution result (text format).
        """
        self.log.info(t("report.executingTool", toolName=tool_name, params=parameters))

        try:
            if tool_name in self.EVIDENCE_TOOL_NAMES:
                return self._execute_evidence_tool(tool_name, parameters or {})

            if tool_name in ("insight_forge", "panorama_search", "quick_search"):
                result_text = self._execute_graph_tool(tool_name, parameters or {}, report_context)
                return (
                    "[Supplementary graph retrieval: facts extracted from the source documents, "
                    "NOT simulated behaviour]\n" + result_text
                )

            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.graph_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx,
                )
                return result.to_text()

            elif tool_name == "panorama_search":
                # Broad search - get full picture
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ["true", "1", "yes"]
                result = self.graph_tools.panorama_search(
                    graph_id=self.graph_id, query=query, include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":
                # Simple search - fast retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.graph_tools.quick_search(
                    graph_id=self.graph_id, query=query, limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":
                # Deep interview - call real OASIS interview API for simulated Agent responses (dual platform)
                interview_topic = parameters.get(
                    "interview_topic", parameters.get("query", "")
                )
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.graph_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents,
                )
                return result.to_text()

            # ========== Backward-compatible legacy tools (internally redirected to new tools) ==========

            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info(t("report.redirectToQuickSearch"))
                return self._execute_tool("quick_search", parameters, report_context)

            elif tool_name == "get_graph_statistics":
                result = self.graph_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.graph_tools.get_entity_summary(
                    graph_id=self.graph_id, entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge (more powerful)
                logger.info(t("report.redirectToInsightForge"))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool(
                    "insight_forge", {"query": query}, report_context
                )

            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.graph_tools.get_entities_by_type(
                    graph_id=self.graph_id, entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)

            else:
                return f"{t('api.unknownTool', name=tool_name)}"

        except Exception as e:
            logger.error(t("report.toolExecFailed", toolName=tool_name, error=str(e)))
            return f"Tool execution failed: {str(e)}"

    def _execute_graph_tool(
        self, tool_name: str, parameters: Dict[str, Any], report_context: str = ""
    ) -> str:
        """Run a supplementary graph tool and count its retrieval results"""
        query = parameters.get("query", "")
        self.graph_stats["calls"] += 1
        if tool_name == "insight_forge":
            ctx = parameters.get("report_context", "") or report_context
            result = self.graph_tools.insight_forge(
                graph_id=self.graph_id,
                query=query,
                simulation_requirement=self.simulation_requirement,
                report_context=ctx,
            )
            hits = getattr(result, "total_facts", 0) or 0
        elif tool_name == "panorama_search":
            include_expired = parameters.get("include_expired", True)
            if isinstance(include_expired, str):
                include_expired = include_expired.lower() in ["true", "1", "yes"]
            result = self.graph_tools.panorama_search(
                graph_id=self.graph_id, query=query, include_expired=include_expired
            )
            hits = len(getattr(result, "active_facts", []) or [])
        else:
            limit = parameters.get("limit", 10)
            if isinstance(limit, str):
                limit = int(limit)
            result = self.graph_tools.quick_search(
                graph_id=self.graph_id, query=query, limit=limit
            )
            hits = getattr(result, "total_count", 0) or 0
        self.graph_stats["hits"] += hits
        self.log.info(f"Graph retrieval {tool_name}: {hits} results (query={query[:80]!r})")
        return result.to_text()

    def _execute_evidence_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Tools reading the run's raw evidence directly (no graph involved)"""
        evidence = self._ensure_evidence()
        if evidence is None:
            return (
                "No simulation run evidence is available for this report "
                f"({self.evidence_error or 'no run found'}). State that simulated behaviour cannot be assessed."
            )

        if tool_name == "simulation_actions":
            limit = parameters.get("limit", 40)
            try:
                limit = max(1, min(int(limit), 200))
            except (TypeError, ValueError):
                limit = 40
            actions = evidence.search_actions(
                query=parameters.get("query"),
                persona=parameters.get("persona"),
                platform=parameters.get("platform"),
                action_type=parameters.get("action_type"),
                limit=0,
            )
            header = (
                f"Run {evidence.run_id}: {len(actions)} matching simulated actions "
                f"(showing {min(len(actions), limit)})."
            )
            if not actions and parameters.get("query"):
                # Keyword search is literal; never let a vocabulary mismatch look like
                # "no events": fall back to every action matching the other filters
                actions = evidence.search_actions(
                    persona=parameters.get("persona"),
                    platform=parameters.get("platform"),
                    action_type=parameters.get("action_type"),
                    limit=0,
                )
                header = (
                    f"Run {evidence.run_id}: no action contains the keywords {parameters.get('query')!r}. "
                    f"Showing all {len(actions)} actions matching the other filters "
                    f"(first {min(len(actions), limit)}); read them before concluding anything is unobserved."
                )
            if not actions:
                return header + " This run contains no simulated action for these filters: NOT OBSERVED."
            return header + "\n" + "\n".join(RunEvidence.format_action(a) for a in actions[:limit])

        if tool_name == "persona_evidence":
            cid = evidence.resolve_persona(parameters.get("persona") or parameters.get("query"))
            persona = evidence.persona_index.personas.get(cid) if cid else None
            if persona is None:
                return (
                    f"Persona {parameters.get('persona')!r} is not present among the simulated agents. "
                    f"Known personas: {', '.join(sorted(evidence.persona_index.personas))}"
                )
            actions = evidence.persona_actions(cid)
            profile = persona["profiles"][0] if persona.get("profiles") else {}
            lines = [
                f"Persona {cid} — {persona['display_name']} (agent ids {persona['agent_ids']}, names {persona['names']})",
                f"Definition: {json.dumps({k: v for k, v in profile.items() if k in ('entity_type', 'bio', 'persona', 'profession', 'stance')}, ensure_ascii=False)[:1500]}",
                f"Simulated actions: {len(actions)}",
            ]
            lines.append(evidence.persona_transcript(cid, max_chars=12000) if actions else "NO simulated actions: nothing was observed for this persona.")
            return "\n".join(lines)

        if tool_name == "evidence_stats":
            return json.dumps(
                {
                    "run_id": evidence.run_id,
                    "stats": evidence.stats(),
                    "requested_persona_ids": evidence.requested_ids,
                    "personas": evidence.persona_index.to_list(),
                    "integrity_warnings": evidence.integrity.get("warnings", []),
                    "note": "Simulation counts of synthetic agents, not real-world rates.",
                },
                ensure_ascii=False,
                indent=2,
            )[:12000]

        if tool_name == "source_documents":
            return evidence.source_excerpts(
                name=parameters.get("name"), query=parameters.get("query"), max_chars=6000
            )

        if tool_name == "simulation_config":
            cfg = evidence.config or {}
            summary = {
                "time_config": cfg.get("time_config"),
                "event_config": cfg.get("event_config"),
                "agents": [
                    {"agent_id": a.get("agent_id"), "entity_name": a.get("entity_name"), "entity_type": a.get("entity_type"), "stance": a.get("stance")}
                    for a in cfg.get("agent_configs", [])
                ],
            }
            return json.dumps(summary, ensure_ascii=False, indent=2)[:10000]

        return f"{t('api.unknownTool', name=tool_name)}"

    EVIDENCE_TOOL_NAMES = {
        "simulation_actions",
        "persona_evidence",
        "evidence_stats",
        "source_documents",
        "simulation_config",
    }

    # Set of valid tool names for bare JSON fallback parsing validation
    VALID_TOOL_NAMES = EVIDENCE_TOOL_NAMES | {
        "insight_forge",
        "panorama_search",
        "quick_search",
        "interview_agents",
    }

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from the LLM response.

        Supported formats (in priority order):
        1. <tool>{"name": "tool_name", "parameters": {...}}</tool>
        2. Bare JSON (the whole response or a single line is a tool-call JSON)
        """
        tool_calls = []

        # Format 1: XML style (standard format)
        # The prompts instruct <tool>...</tool>; <tool_call> is accepted for compatibility
        xml_pattern = r"<(tool_call|tool)>\s*(\{.*?\})\s*</\1>"
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(2))
                if "tool" in call_data and "name" not in call_data:
                    call_data["name"] = call_data.pop("tool")
                if "params" in call_data and "parameters" not in call_data:
                    call_data["parameters"] = call_data.pop("params")
                if call_data.get("name"):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: fallback - LLM outputs bare JSON directly (no <tool_call> wrapper)
        # Only try if format 1 didn't match, to avoid false matching JSON in body text
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Response may contain thinking text + bare JSON, try extracting last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate whether the parsed JSON is a legitimate tool call."""
        # Support both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...} key names
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize key names to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        """Generate the tool description text."""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join(
                [f"{k}: {v}" for k, v in tool["parameters"].items()]
            )
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)

    @staticmethod
    def _strip_fake_tool_results(response: str) -> str:
        """Strip any <tool_result> blocks the LLM fabricated in its response.

        When the LLM generates a tool call block and then continues to generate
        a <tool_result> block in the same response, we must strip the fake result
        before appending to message history. The real tool result will be injected
        separately by the system.
        """
        tag_pattern = re.compile(r'</?tool_result\b[^>]*>', flags=re.IGNORECASE)
        parts = []
        cursor = 0
        depth = 0

        for match in tag_pattern.finditer(response):
            if depth == 0:
                parts.append(response[cursor:match.start()])

            if match.group(0).lstrip().startswith('</'):
                depth = max(0, depth - 1)
            else:
                depth += 1
            cursor = match.end()

        if depth == 0:
            parts.append(response[cursor:])

        cleaned = ''.join(parts)
        # Treat a malformed opening tag without a closing > as unsafe too.
        cleaned = re.sub(r'<tool_result\b.*$', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def plan_outline(
        self, progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan the report outline.

        Uses the LLM to analyze the simulation requirement and plan the
        report's directory structure.

        Args:
            progress_callback: Progress callback function.

        Returns:
            ReportOutline: The report outline.
        """
        logger.info(t("report.startPlanningOutline"))

        if progress_callback:
            progress_callback("planning", 0, t("progress.analyzingRequirements"))

        # Supplementary graph context (facts from the source documents)
        try:
            context = self.graph_tools.get_simulation_context(
                graph_id=self.graph_id, simulation_requirement=self.simulation_requirement
            )
        except Exception as e:
            self.log.warning(f"Graph context unavailable: {e}")
            context = {}
        self.graph_stats["calls"] += 1
        self.graph_stats["hits"] += len(context.get("related_facts", []) or [])

        if progress_callback:
            progress_callback("planning", 30, t("progress.generatingOutline"))

        # Primary evidence: the run's raw actions
        evidence = self._ensure_evidence()
        if evidence is not None:
            evidence_stats = evidence.stats()
            integrity = "\n".join(evidence.integrity.get("warnings", [])) or "No integrity warnings."
            sample = "\n".join(
                RunEvidence.format_action(a, max_chars=300) for a in evidence.actions[:20]
            ) or "(no simulated actions in this run)"
        else:
            evidence_stats = {"error": self.evidence_error or "no run evidence"}
            integrity = "No run evidence available."
            sample = "(no simulated actions available)"

        brief = evidence.brief_text if evidence is not None and evidence.brief_text else self.simulation_requirement
        system_prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{get_language_instruction()}"
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=brief[:6000],
            evidence_stats_json=json.dumps(evidence_stats, ensure_ascii=False, indent=2)[:4000],
            integrity_warnings=integrity,
            action_sample=sample,
            total_nodes=context.get("graph_statistics", {}).get("total_nodes", 0),
            total_edges=context.get("graph_statistics", {}).get("total_edges", 0),
            entity_types=list(
                context.get("graph_statistics", {}).get("entity_types", {}).keys()
            ),
            related_facts_json=json.dumps(
                context.get("related_facts", [])[:10], ensure_ascii=False, indent=2
            ),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            if progress_callback:
                progress_callback("planning", 80, t("progress.parsingOutline"))

            # Parse outline (fixed evidence sections are added by generate_report)
            sections = []
            for section_data in response.get("sections", [])[:4]:
                title = (section_data.get("title") or "").strip()
                if title and title not in STRUCTURED_SECTION_TITLES:
                    sections.append(ReportSection(title=title, content=""))

            outline = ReportOutline(
                title=response.get("title", t("api.defaultReportTitle")),
                summary=response.get("summary", ""),
                sections=sections,
            )

            if progress_callback:
                progress_callback("planning", 100, t("progress.outlinePlanComplete"))

            logger.info(t("report.outlinePlanDone", count=len(sections)))
            return outline

        except Exception as e:
            logger.error(t("report.outlinePlanFailed", error=str(e)))
            # Return default outline (3 sections, as fallback)
            return ReportOutline(
                title="Simulation Assessment Report",
                summary="Assessment of simulated persona behaviour (synthetic evidence, not real-world data)",
                sections=[
                    ReportSection(title="Cross-Persona Findings"),
                    ReportSection(title="Open Questions and Real-World Validation"),
                ],
            )

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0,
    ) -> str:
        """
        Generate a single section's content using the ReACT pattern.

        ReACT loop:
        1. Thought - analyze what information is needed.
        2. Action - call a tool to retrieve information.
        3. Observation - analyze the tool's returned result.
        4. Repeat until information is sufficient or the maximum count is reached.
        5. Final Answer - generate the section content.

        Args:
            section: The section to generate.
            outline: The complete outline.
            previous_sections: Contents of previous sections (used to maintain coherence).
            progress_callback: Progress callback.
            section_index: Section index (used for logging).

        Returns:
            Section content (Markdown format).
        """
        logger.info(t("report.reactGenerateSection", title=section.title))

        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build user prompt - pass up to 4000 chars per completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Max 4000 chars per section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first chapter)"

        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # Max iterations
        min_tool_calls = 3  # Min tool calls
        conflict_retries = 0  # Consecutive conflict count (tool call + Final Answer both present)
        used_tools = set()  # Track called tool names
        all_tools = {
            "simulation_actions",
            "persona_evidence",
            "evidence_stats",
            "source_documents",
            "insight_forge",
        }

        # Report context, used for InsightForge sub-query generation
        report_context = f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}"

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    t(
                        "progress.deepSearchAndWrite",
                        current=tool_calls_count,
                        max=self.MAX_TOOL_CALLS_PER_SECTION,
                    ),
                )

            # Call LLM
            response = self.llm.chat(
                messages=messages, temperature=0.5, max_tokens=8192
            )

            # Check if LLM returned None (API error or empty content)
            if response is None:
                logger.warning(
                    t(
                        "report.sectionIterNone",
                        title=section.title,
                        iteration=iteration + 1,
                    )
                )
                # If iterations remain, add message and retry
                if iteration < max_iterations - 1:
                    messages.append(
                        {"role": "assistant", "content": "(Response was empty)"}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue generating content.",
                        }
                    )
                    continue
                # Last iteration also returned None, break loop and force finish
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once, reuse result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict handling: LLM output both tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t(
                        "report.sectionConflict",
                        title=section.title,
                        iteration=iteration + 1,
                        conflictCount=conflict_retries,
                    )
                )

                if conflict_retries <= 2:
                    # First two times: discard this response, ask LLM to re-respond
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[Format Error] You included both a tool call and a Final Answer in the same reply, which is not allowed.\n"
                                "Each reply can only do one of the following two things:\n"
                                "- Call a tool (output a <tool> block, do not write Final Answer)\n"
                                "- Output final content (start with 'Final Answer:', do not include <tool>)\n"
                                "Please reply again, doing only one of them."
                            ),
                        }
                    )
                    continue
                else:
                    # Third time: downgrade, truncate to first tool call, force execute
                    logger.warning(
                        t(
                            "report.sectionConflictDowngrade",
                            title=section.title,
                            conflictCount=conflict_retries,
                        )
                    )
                    end_match = re.search(r"</tool(?:_call)?>", response)
                    if end_match:
                        response = response[: end_match.end()]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer,
                )

            # ── Case 1: LLM output Final Answer ──
            if has_final_answer:
                # Insufficient tool calls, reject and ask to continue calling tools
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = (
                        f"(These tools have not been used yet, recommend trying them: {', '.join(unused_tools)})"
                        if unused_tools
                        else ""
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                                tool_calls_count=tool_calls_count,
                                min_tool_calls=min_tool_calls,
                                unused_hint=unused_hint,
                            ),
                        }
                    )
                    continue

                # Normal completion
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(
                    t(
                        "report.sectionGenDone",
                        title=section.title,
                        count=tool_calls_count,
                    )
                )

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count,
                    )
                return final_answer

            # ── Case 2: LLM attempting tool call ──
            if has_tool_calls:
                # Tool quota exhausted → explicitly inform, ask for Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_TOOL_LIMIT_MSG.format(
                                tool_calls_count=tool_calls_count,
                                max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            ),
                        }
                    )
                    continue

                # Only execute first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(
                        t(
                            "report.multiToolOnlyFirst",
                            total=len(tool_calls),
                            toolName=call["name"],
                        )
                    )

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1,
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context,
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1,
                    )

                tool_calls_count += 1
                used_tools.add(call["name"])

                # Build unused tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(
                        unused_list=", ".join(unused_tools)
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": REACT_OBSERVATION_TEMPLATE.format(
                            tool_name=call["name"],
                            result=result,
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            used_tools_str=", ".join(used_tools),
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # ── Case 3: Neither tool call nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Insufficient tool calls, recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f"(These tools have not been used yet, recommend trying them: {', '.join(unused_tools)})"
                    if unused_tools
                    else ""
                )

                messages.append(
                    {
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # Tool calls sufficient, LLM output content but missing "Final Answer:" prefix
            # Use this content directly as final answer, no more idle loops
            logger.info(
                t("report.sectionNoPrefix", title=section.title, count=tool_calls_count)
            )
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count,
                )
            return final_answer

        # Reached max iterations, force content generation
        logger.warning(t("report.sectionMaxIter", title=section.title))
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(messages=messages, temperature=0.5, max_tokens=8192)

        # Check if force-finish LLM returned None
        if response is None:
            logger.error(t("report.sectionForceFailed", title=section.title))
            final_answer = t("report.sectionGenFailedContent")
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # Log section content generation complete
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count,
            )

        return final_answer

    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None,
    ) -> Report:
        """
        Generate the full report (streaming output section by section).

        Each section is saved to a folder immediately after it is generated,
        without waiting for the whole report to complete.
        File structure:
        reports/{report_id}/
            meta.json       - report metadata
            outline.json    - report outline
            progress.json   - generation progress
            section_01.md   - section 1
            section_02.md   - section 2
            ...
            full_report.md  - full report

        Args:
            progress_callback: Progress callback function (stage, progress, message).
            report_id: Report ID (optional; auto-generated if not provided).

        Returns:
            Report: The complete report.
        """
        import uuid

        # Auto-generate report_id if not provided
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        # Load the run's raw evidence before anything else: it is the primary source
        evidence = self._ensure_evidence()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat(),
            run_id=self.run_id,
        )
        self.log.info(f"Report generation started: {report_id}")

        # Completed section titles list (for progress tracking)
        completed_section_titles = []

        try:
            # Initialize: create report folder and save initial state
            ReportManager._ensure_report_folder(report_id)

            # Initialize logger (structured log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement,
            )

            # Initialize console logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)

            # Assessment spec: which personas / hypotheses / funnel fields the brief asks for
            brief = evidence.brief_text if evidence is not None else self.simulation_requirement
            families = set()
            if evidence is not None:
                for agent in evidence.agents:
                    parsed = parse_identifier(agent.get("name"))
                    if parsed:
                        families.add(parsed[0].upper())
            self.spec = extract_assessment_spec(brief, self.llm, families)
            if evidence is not None:
                evidence.set_requested_ids(self.spec.persona_ids)
                for name in getattr(evidence, "brief_documents", []) or []:
                    evidence.mark_source_used(name)
            self.log.info(
                f"Assessment spec: personas={self.spec.persona_ids or 'all'} "
                f"hypotheses={[h['id'] for h in self.spec.hypotheses]} "
                f"funnel_fields={len(self.spec.funnel_fields)} ({self.spec.funnel_fields_source})"
            )
            if evidence is not None:
                counts = evidence.platform_counts()
                self.log.info(
                    f"Action evidence loaded: twitter={counts.get('twitter', 0)} "
                    f"reddit={counts.get('reddit', 0)}; source files loaded: {len(evidence.sources)}"
                )
            self.structured = StructuredReportBuilder(
                evidence, self.spec, self.llm, get_language_instruction(), self.log
            )

            ReportManager.update_progress(
                report_id, "pending", 0, t("progress.initReport"), completed_sections=[]
            )
            ReportManager.save_report(report)

            # Phase 1: Plan outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id,
                "planning",
                5,
                t("progress.startPlanningOutline"),
                completed_sections=[],
            )

            # Log planning start
            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback("planning", 0, t("progress.startPlanningOutline"))

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: (
                    progress_callback(stage, prog // 5, msg)
                    if progress_callback
                    else None
                )
            )
            # Fixed evidence sections: persona and hypothesis analysis first (the
            # synthesis sections build on them), evidence appendix last
            outline.sections = (
                [ReportSection(title=PERSONA_SECTION_TITLE), ReportSection(title=HYPOTHESIS_SECTION_TITLE)]
                + outline.sections
                + [ReportSection(title=APPENDIX_SECTION_TITLE)]
            )
            report.outline = outline

            # Log planning complete
            self.report_logger.log_planning_complete(outline.to_dict())

            # Save outline to file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id,
                "planning",
                15,
                t("progress.outlineDone", count=len(outline.sections)),
                completed_sections=[],
            )
            ReportManager.save_report(report)

            logger.info(t("report.outlineSavedToFile", reportId=report_id))

            # Phase 2: Generate section by section (save per section)
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # Save content for context

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # Update progress
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress,
                    t(
                        "progress.generatingSection",
                        title=section.title,
                        current=section_num,
                        total=total_sections,
                    ),
                    current_section=section.title,
                    completed_sections=completed_section_titles,
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        t(
                            "progress.generatingSection",
                            title=section.title,
                            current=section_num,
                            total=total_sections,
                        ),
                    )

                # Generate main section content
                if section.title in STRUCTURED_SECTION_TITLES:
                    section_content, context_summary = self._generate_structured_section(
                        section, section_num
                    )
                else:
                    section_content = self._generate_section_react(
                        section=section,
                        outline=outline,
                        previous_sections=generated_sections,
                        progress_callback=lambda stage, prog, msg: (
                            progress_callback(
                                stage, base_progress + int(prog * 0.7 / total_sections), msg
                            )
                            if progress_callback
                            else None
                        ),
                        section_index=section_num,
                    )
                    context_summary = section_content

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{context_summary}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section completion
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip(),
                    )

                logger.info(
                    t(
                        "report.sectionSaved",
                        reportId=report_id,
                        sectionNum=f"{section_num:02d}",
                    )
                )

                # Update progress
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress + int(70 / total_sections),
                    t("progress.sectionDone", title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles,
                )

            # Phase 3: Assemble full report
            if progress_callback:
                progress_callback("generating", 95, t("progress.assemblingReport"))

            ReportManager.update_progress(
                report_id,
                "generating",
                95,
                t("progress.assemblingReport"),
                completed_sections=completed_section_titles,
            )

            # Use ReportManager to assemble full report
            report.markdown_content = ReportManager.assemble_full_report(
                report_id, outline
            )
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()

            # Machine-readable evidence and results next to the Markdown report
            self._save_machine_readable(report_id, report)

            # Calculate total elapsed time
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            self.log.info(
                f"Report completed: {report_id} sections={total_sections} "
                f"graph_calls={self.graph_stats['calls']} graph_hits={self.graph_stats['hits']} "
                f"time={total_time_seconds:.0f}s"
            )

            # Log report completion
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections, total_time_seconds=total_time_seconds
                )

            # Save final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id,
                "completed",
                100,
                t("progress.reportComplete"),
                completed_sections=completed_section_titles,
            )

            if progress_callback:
                progress_callback("completed", 100, t("progress.reportComplete"))

            logger.info(t("report.reportGenDone", reportId=report_id))

            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            self.log.error(t("report.reportGenFailed", error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # Log error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # Save failure state
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id,
                    "failed",
                    -1,
                    t("progress.reportFailed", error=str(e)),
                    completed_sections=completed_section_titles,
                )
            except Exception:
                pass  # Ignore save failure errors

            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def _generate_structured_section(self, section: ReportSection, section_index: int):
        """
        Generate one of the fixed evidence sections

        Returns:
            (section markdown, compact summary passed as context to later sections)
        """
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        builder = self.structured
        if section.title == PERSONA_SECTION_TITLE:
            builder.analyse_personas()
            content = builder.render_personas()
            summary = json.dumps(
                [
                    {
                        "persona": r["canonical_id"],
                        "actions": r["action_counts"].get("total", 0),
                        "observed": [f["field"] for f in r["fields"] if f["status"] == "observed"],
                        "missing": r["missing_stages"],
                        "for": r["strongest_evidence_for"]["summary"][:200],
                        "against": r["strongest_evidence_against"]["summary"][:200],
                    }
                    for r in builder.persona_results
                ],
                ensure_ascii=False,
            )[:6000]
        elif section.title == HYPOTHESIS_SECTION_TITLE:
            builder.assess_hypotheses()
            content = builder.render_hypotheses()
            summary = content[:3000]
        else:
            content = builder.render_appendix(self.graph_stats)
            summary = "(evidence appendix)"
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=content,
                tool_calls_count=0,
            )
        return content, summary

    def _save_machine_readable(self, report_id: str, report: Report) -> None:
        """Write evidence.json (normalized evidence) and report.json (structured results)"""
        folder = ReportManager._ensure_report_folder(report_id)
        evidence = self.evidence
        evidence_data = evidence.to_dict() if evidence is not None else {"error": self.evidence_error}
        evidence_data["graph_retrieval"] = dict(self.graph_stats)
        with open(os.path.join(folder, "evidence.json"), "w", encoding="utf-8") as f:
            json.dump(evidence_data, f, ensure_ascii=False, indent=2)
        structured = self.structured.to_dict() if self.structured else {}
        report_json = {
            "report_id": report_id,
            "simulation_id": self.simulation_id,
            "run_id": self.run_id,
            "title": report.outline.title if report.outline else None,
            "summary": report.outline.summary if report.outline else None,
            "evidence_type": "simulated (synthetic agents); not real-world customer data",
            "assessment_spec": structured.get("assessment_spec"),
            "personas": structured.get("personas", []),
            "hypotheses": structured.get("hypotheses", []),
            "evidence_stats": evidence.stats() if evidence is not None else None,
            "integrity_warnings": evidence.integrity.get("warnings", []) if evidence is not None else [],
            "sources": evidence.source_manifest() if evidence is not None else [],
            "graph_retrieval": dict(self.graph_stats),
            "sections": [s.title for s in report.outline.sections] if report.outline else [],
            "generated_at": datetime.now().isoformat(),
        }
        with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report_json, f, ensure_ascii=False, indent=2)

    def chat(
        self, message: str, chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the Report Agent.

        During the conversation the agent may autonomously call retrieval
        tools to answer questions.

        Args:
            message: User message.
            chat_history: Conversation history.

        Returns:
            {
                "response": "Agent reply",
                "tool_calls": [list of called tools],
                "sources": [information sources]
            }
        """
        logger.info(t("report.agentChat", message=message[:50]))

        chat_history = chat_history or []

        # Get generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id, self.run_id) or ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit report length to avoid excessive context
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(t("report.fetchReportFailed", error=e))

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content
            if report_content
            else "(No report available yet)",
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Build message
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)

        # Add user message
        messages.append({"role": "user", "content": message})

        # ReACT loop (simplified)
        tool_calls_made = []
        max_iterations = 2  # Reduce iterations

        for iteration in range(max_iterations):
            response = self.llm.chat(messages=messages, temperature=0.5)

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No tool calls, return response directly
                clean_response = re.sub(
                    r"<(tool_call|tool)>.*?</\1>", "", response, flags=re.DOTALL
                )
                clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [
                        tc.get("parameters", {}).get("query", "")
                        for tc in tool_calls_made
                    ],
                }

            # Execute tool calls (limit count)
            tool_results = []
            for call in tool_calls[:1]:  # Max 1 tool call per round
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append(
                    {
                        "tool": call["name"],
                        "result": result[:1500],  # Limit result length
                    }
                )
                tool_calls_made.append(call)

            # Add result to message
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join(
                [f"[{r['tool']} result]\n{r['result']}" for r in tool_results]
            )
            messages.append(
                {"role": "user", "content": observation + CHAT_OBSERVATION_SUFFIX}
            )

        # Reached max iterations, get final response
        final_response = self.llm.chat(messages=messages, temperature=0.5)

        # Clean response
        clean_response = re.sub(
            r"<(tool_call|tool)>.*?</\1>", "", final_response, flags=re.DOTALL
        )
        clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [
                tc.get("parameters", {}).get("query", "") for tc in tool_calls_made
            ],
        }


class ReportManager:
    """
    Report manager.

    Responsible for report persistence and retrieval.

    File structure (section-by-section output):
    reports/
      {report_id}/
        meta.json          - report metadata and status
        outline.json       - report outline
        progress.json      - generation progress
        section_01.md      - section 1
        section_02.md      - section 2
        ...
        full_report.md     - full report
    """

    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "reports")

    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure the reports root directory exists."""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Return the report folder path."""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure the report folder exists and return its path."""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Return the report metadata file path."""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Return the full-report Markdown file path."""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Return the outline file path."""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Return the progress file path."""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Return the section Markdown file path."""
        return os.path.join(
            cls._get_report_folder(report_id), f"section_{section_index:02d}.md"
        )

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Return the agent log file path."""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Return the console log file path."""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Return the console log contents.

        These are the console output logs (INFO, WARNING, etc.) produced
        during report generation, distinct from the structured agent_log.jsonl.

        Args:
            report_id: Report ID.
            from_line: Line number to start reading from (for incremental
                retrieval; 0 means from the beginning).

        Returns:
            {
                "logs": [list of log lines],
                "total_lines": total number of lines,
                "from_line": starting line number,
                "has_more": whether more logs remain
            }
        """
        log_path = cls._get_console_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep original log line, strip trailing newline
                    logs.append(line.rstrip("\n\r"))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # Already read to end
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Return the full console log (retrieved all at once).

        Args:
            report_id: Report ID.

        Returns:
            List of log lines.
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Return the agent log contents.

        Args:
            report_id: Report ID.
            from_line: Line number to start reading from (for incremental
                retrieval; 0 means from the beginning).

        Returns:
            {
                "logs": [list of log entries],
                "total_lines": total number of lines,
                "from_line": starting line number,
                "has_more": whether more logs remain
            }
        """
        log_path = cls._get_agent_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that failed to parse
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # Already read to end
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Return the full agent log (retrieved all at once).

        Args:
            report_id: Report ID.

        Returns:
            List of log entries.
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save the report outline.

        Called immediately after the planning stage completes.
        """
        cls._ensure_report_folder(report_id)

        with open(cls._get_outline_path(report_id), "w", encoding="utf-8") as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(t("report.outlineSaved", reportId=report_id))

    @classmethod
    def save_section(
        cls, report_id: str, section_index: int, section: ReportSection
    ) -> str:
        """
        Save a single section.

        Called immediately after each section is generated, enabling
        section-by-section output.

        Args:
            report_id: Report ID.
            section_index: Section index (1-based).
            section: Section object.

        Returns:
            The saved file path.
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content - clean up possible duplicate titles
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(
            t("report.sectionFileSaved", reportId=report_id, fileSuffix=file_suffix)
        )
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean section content.

        1. Remove Markdown heading lines at the start of the content that
           duplicate the section title.
        2. Convert all headings of level ### and below to bold text.

        Args:
            content: Original content.
            section_title: Section title.

        Returns:
            The cleaned content.
        """
        import re

        if not content:
            return content

        content = content.strip()
        lines = content.split("\n")
        cleaned_lines = []
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check if it's a Markdown heading line
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # Check if heading duplicates section title (skip duplicates within first 5 lines)
                if i < 5:
                    if title_text == section_title or title_text.replace(
                        " ", ""
                    ) == section_title.replace(" ", ""):
                        skip_next_empty = True
                        continue

                # Convert all heading levels (#, ##, ###, ####, etc.) to bold
                # Section titles are added by the system, content should not contain any headings
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add blank line
                continue

            # If previous line was a skipped heading and current line is empty, skip it too
            if skip_next_empty and stripped == "":
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # Remove leading blank lines
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)

        # Remove leading separator lines
        while cleaned_lines and cleaned_lines[0].strip() in ["---", "***", "___"]:
            cleaned_lines.pop(0)
            # Also remove blank line after separator
            while cleaned_lines and cleaned_lines[0].strip() == "":
                cleaned_lines.pop(0)

        return "\n".join(cleaned_lines)

    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None,
    ) -> None:
        """
        Update the report generation progress.

        The frontend can read progress.json to obtain real-time progress.
        """
        cls._ensure_report_folder(report_id)

        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat(),
        }

        with open(cls._get_progress_path(report_id), "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Return the report generation progress."""
        path = cls._get_progress_path(report_id)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Return the list of generated sections.

        Returns information about all saved section files.
        """
        folder = cls._get_report_folder(report_id)

        if not os.path.exists(folder):
            return []

        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith("section_") and filename.endswith(".md"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse section index from filename
                parts = filename.replace(".md", "").split("_")
                section_index = int(parts[1])

                sections.append(
                    {
                        "filename": filename,
                        "section_index": section_index,
                        "content": content,
                    }
                )

        return sections

    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble the full report.

        Assemble the full report from the saved section files and clean up
        the headings.
        """
        folder = cls._get_report_folder(report_id)

        # Build report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # Post-process: clean up heading issues in full report
        md_content = cls._post_process_report(md_content, outline)

        # Save full report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(t("report.fullReportAssembled", reportId=report_id))
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process report content.

        1. Remove duplicate headings.
        2. Keep the report main title (#) and section titles (##); remove
           other heading levels (###, ####, etc.).
        3. Clean up extra blank lines and divider lines.

        Args:
            content: Original report content.
            outline: Report outline.

        Returns:
            The processed content.
        """
        import re

        lines = content.split("\n")
        processed_lines = []
        prev_was_heading = False

        # Collect all section titles from outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if it's a heading line
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Check if duplicate heading (same content within 5 consecutive lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r"^(#{1,6})\s+(.+)$", prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:
                    # Skip duplicate heading and following blank line
                    i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    continue

                # Heading level handling:
                # - # (level=1) only keep report main title
                # - ## (level=2) keep section titles
                # - ### and below (level>=3) convert to bold text

                if level == 1:
                    if title == outline.title:
                        # Keep report main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title incorrectly used #, fix to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other level-1 headings converted to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section title
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Non-section level-2 headings converted to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### and below headings converted to bold text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == "---" and prev_was_heading:
                # Skip separator line immediately after heading
                i += 1
                continue

            elif stripped == "" and prev_was_heading:
                # Keep only one blank line after heading
                if processed_lines and processed_lines[-1].strip() != "":
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # Clean up consecutive blank lines (keep max 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and the full report."""
        cls._ensure_report_folder(report.report_id)

        # Save meta info JSON
        with open(cls._get_report_path(report.report_id), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # Save full Markdown report
        if report.markdown_content:
            with open(
                cls._get_report_markdown_path(report.report_id), "w", encoding="utf-8"
            ) as f:
                f.write(report.markdown_content)

        logger.info(t("report.reportSaved", reportId=report.report_id))

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Return the report."""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # Backward compat: check files stored directly in reports directory
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Rebuild Report object
        outline = None
        if data.get("outline"):
            outline_data = data["outline"]
            sections = []
            for s in outline_data.get("sections", []):
                sections.append(
                    ReportSection(title=s["title"], content=s.get("content", ""))
                )
            outline = ReportOutline(
                title=outline_data["title"],
                summary=outline_data["summary"],
                sections=sections,
            )

        # If markdown_content is empty, try reading from full_report.md
        markdown_content = data.get("markdown_content", "")
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

        return Report(
            report_id=data["report_id"],
            simulation_id=data["simulation_id"],
            graph_id=data["graph_id"],
            simulation_requirement=data["simulation_requirement"],
            status=ReportStatus(data["status"]),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error"),
            run_id=data.get("run_id"),
        )

    @classmethod
    def get_report_by_simulation(
        cls, simulation_id: str, run_id: Optional[str] = None
    ) -> Optional[Report]:
        """
        Return the newest report of a simulation (optionally of one run)

        Args:
            simulation_id: Simulation ID
            run_id: Only consider reports generated from this run
        """
        reports = cls.list_reports(simulation_id=simulation_id, limit=10000)
        if run_id:
            reports = [r for r in reports if r.run_id == run_id]
        return reports[0] if reports else None

    @classmethod
    def list_reports(
        cls, simulation_id: Optional[str] = None, limit: int = 50
    ) -> List[Report]:
        """List reports."""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Backward compat: JSON file
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # Sort by creation time descending
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete a report (the whole folder)."""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # New format: delete entire folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t("report.reportFolderDeleted", reportId=report_id))
            return True

        # Backward compat: delete individual file
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")

        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True

        return deleted
