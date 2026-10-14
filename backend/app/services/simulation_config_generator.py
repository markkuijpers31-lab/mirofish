"""
Simulation Config Intelligent Generator
Uses an LLM to automatically generate detailed simulation parameters based on simulation requirements, document content, and graph information
Achieves full automation without manual parameter setting

Adopts a step-by-step generation strategy to avoid failures from generating too much content at once:
1. Generate time configuration
2. Generate event configuration
3. Generate Agent configurations in batches
4. Generate platform configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text
from .entity_reader import EntityNode, EntityReader
from .research_instructions import ORIGIN_SYSTEM_SEED, is_unsafe_public_text, strip_private_text

logger = get_logger("mirofish.simulation_config")

# Activity schedule config (default timezone-based)
CHINA_TIMEZONE_CONFIG = {
    # Late night (almost no activity)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morning (gradually waking up)
    "morning_hours": [6, 7, 8],
    # Work hours
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Evening peak (most active)
    "peak_hours": [19, 20, 21, 22],
    # Night (activity declining)
    "night_hours": [23],
    # Activity coefficients
    "activity_multipliers": {
        "dead": 0.05,  # Late night: almost no one
        "morning": 0.4,  # Morning: gradually active
        "work": 0.7,  # Work hours: moderate
        "peak": 1.5,  # Evening peak
        "night": 0.5,  # Night: declining
    },
}


@dataclass
class AgentActivityConfig:
    """A single Agent's activity configuration"""

    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Activity level config (0.0-1.0)
    activity_level: float = 0.5  # Overall activity level

    # Posting frequency (expected posts per hour)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Active hours (24-hour format, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Response speed (reaction delay to hot events, unit: simulation minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment bias (-1.0 to 1.0, negative to positive)
    sentiment_bias: float = 0.0

    # Stance (attitude toward specific topics)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Influence weight (determines probability of other Agents seeing their posts)
    influence_weight: float = 1.0

    # Population roles chosen in the agent selection (configuration, not behaviour)
    simulation_role: str = ""
    evidence_role: str = ""


@dataclass
class TimeSimulationConfig:
    """Time simulation config (based on Chinese daily routines)"""

    # Total simulation duration (simulated hours)
    total_simulation_hours: int = 72  # Default: 72 hours (3 days)

    # Time per round (simulated minutes) - default 60 min (1 hour), accelerated time flow
    minutes_per_round: int = 60

    # Agent activation range per hour
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Peak hours (evening, most active period)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Off-peak hours (late night, almost no activity)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Late night: extremely low activity

    # Morning hours
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Work hours
    work_hours: List[int] = field(
        default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Event configuration"""

    # Initial events (trigger events at simulation start)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Scheduled events (triggered at specific times)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Hot topic keywords
    hot_topics: List[str] = field(default_factory=list)

    # Opinion guidance direction
    narrative_direction: str = ""

    # Seed posts rejected because they contained research/orchestration content
    # (kept for inspection, never published)
    rejected_seed_posts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlatformConfig:
    """Platform-specific configuration"""

    platform: str  # twitter or reddit

    # Recommendation algorithm weights
    recency_weight: float = 0.4  # Recency
    popularity_weight: float = 0.3  # Popularity
    relevance_weight: float = 0.3  # Relevance

    # Viral threshold (interaction count to trigger spread)
    viral_threshold: int = 10

    # Echo chamber strength (degree of similar opinion clustering)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Complete simulation parameter configuration"""

    # Basic info
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Time config
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Agent config list
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Event config
    event_config: EventConfig = field(default_factory=EventConfig)

    # Platform config
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # LLM config
    llm_model: str = ""
    llm_base_url: str = ""

    # Generation metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM reasoning explanation

    # Population of this preparation (active agents vs. context-only entities)
    population: Dict[str, Any] = field(default_factory=dict)

    # Private research instruction texts known at prepare time. The runner uses
    # them to classify leaked instructions; they are never published.
    private_texts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config)
            if self.twitter_config
            else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
            "population": self.population,
            "private_texts": self.private_texts,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to a JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Simulation Config Intelligent Generator

    Uses an LLM to analyze simulation requirements, document content, and graph entity info,
    and automatically generates the optimal simulation parameter configuration

    Adopts a step-by-step generation strategy:
    1. Generate time configuration and event configuration (lightweight)
    2. Generate Agent configurations in batches (10-20 per batch)
    3. Generate platform configuration
    """

    # Max context characters
    MAX_CONTEXT_LENGTH = 50000
    # Agents per batch
    AGENTS_PER_BATCH = 15

    # Context truncation length per step (characters)
    TIME_CONFIG_CONTEXT_LENGTH = 10000  # Time config
    EVENT_CONFIG_CONTEXT_LENGTH = 8000  # Event config
    ENTITY_SUMMARY_LENGTH = 300  # Entity summary
    AGENT_SUMMARY_LENGTH = 300  # Entity summary in Agent config
    ENTITIES_PER_TYPE_DISPLAY = 20  # Entities displayed per type

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError(t("api.llmApiKeyMissing"))

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        context_entities: Optional[List[EntityNode]] = None,
        roster: Optional[Dict[str, Any]] = None,
        private_texts: Optional[List[str]] = None,
        brief_text: str = "",
    ) -> SimulationParameters:
        """
        Intelligently generate the complete simulation configuration (step-by-step generation)

        Args:
            simulation_id: Simulation ID
            project_id: Project ID
            graph_id: Graph ID
            simulation_requirement: Simulation requirement description
            document_text: Original document content
            entities: Active agent entities (only these become simulation accounts)
            enable_twitter: Whether to enable Twitter
            enable_reddit: Whether to enable Reddit
            progress_callback: Progress callback function(current_step, total_steps, message)
            context_entities: Context-only graph entities (may be mentioned, never accounts)
            roster: Frozen agent roster (roles per active agent)
            private_texts: Private research instructions (never allowed in seed posts)
            brief_text: Assessment brief (verbatim copies are rejected as seed posts)

        Returns:
            SimulationParameters: Complete simulation parameters
        """
        logger.info(
            f"Starting intelligent simulation config generation: simulation_id={simulation_id}, entity_count={len(entities)}"
        )

        # Calculate total steps
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # Time config + Event config + N batches of Agents + Platform config
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        private_texts = list(private_texts or [])
        self._private_texts = private_texts
        # The requirement may embed research instructions; the LLM never sees them
        simulation_requirement = strip_private_text(simulation_requirement, private_texts) or simulation_requirement
        roles_by_uuid = {
            a["entity_uuid"]: a for a in (roster or {}).get("active_agents", [])
        }

        # 1. Build base context info
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
            context_entities=context_entities,
        )

        reasoning_parts = []

        # ========== Step 1: Generate time config ==========
        report_progress(1, t("progress.generatingTimeConfig"))
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(
            f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}"
        )

        # ========== Step 2: Generate event config ==========
        report_progress(2, t("progress.generatingEventConfig"))
        event_config_result = self._generate_event_config(
            context, simulation_requirement, entities
        )
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(
            f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}"
        )

        # ========== Step 3-N: Generate Agent config in batches ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                t(
                    "progress.generatingAgentConfig",
                    start=start_idx + 1,
                    end=end_idx,
                    total=len(entities),
                ),
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement,
            )
            all_agent_configs.extend(batch_configs)

        for cfg in all_agent_configs:
            role = roles_by_uuid.get(cfg.entity_uuid)
            if role:
                if int(role.get("agent_id", cfg.agent_id)) != cfg.agent_id:
                    raise ValueError(
                        f"Agent id mismatch for {cfg.entity_name}: roster {role.get('agent_id')} vs config {cfg.agent_id}"
                    )
                cfg.simulation_role = role.get("simulation_role", "")
                cfg.evidence_role = role.get("evidence_role", "")

        # Seed posts are in-world content only: drop anything that carries research
        # instructions, persona tasks or verbatim brief text
        event_config = self._filter_seed_posts(event_config, private_texts, brief_text)

        reasoning_parts.append(
            t("progress.agentConfigResult", count=len(all_agent_configs))
        )

        # ========== Assign initial post publishers ==========
        logger.info("Assigning suitable Agent publishers for initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len(
            [
                p
                for p in event_config.initial_posts
                if p.get("poster_agent_id") is not None
            ]
        )
        reasoning_parts.append(t("progress.postAssignResult", count=assigned_count))

        # ========== Final step: Generate platform config ==========
        report_progress(total_steps, t("progress.generatingPlatformConfig"))
        twitter_config = None
        reddit_config = None

        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5,
            )

        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6,
            )

        # Build final params
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts),
            population={
                "active_agents": len(all_agent_configs),
                "context_entities": len(context_entities or []),
                "selection_hash": (roster or {}).get("selection_hash"),
            },
            private_texts=private_texts,
        )

        logger.info(f"Simulation config generation complete: {len(params.agent_configs)}  Agent configs")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        context_entities: Optional[List[EntityNode]] = None,
    ) -> str:
        """Build the LLM context, truncate to max length"""

        # Entity summary
        entity_summary = self._summarize_entities(entities)

        # Build context
        context_parts = [
            f"## Simulation requirement\n{simulation_requirement}",
            f"\n## Active simulation accounts ({len(entities)} agents)\n{entity_summary}",
        ]
        if context_entities:
            names = ", ".join(
                f"{e.name} ({e.get_entity_type() or 'Entity'})" for e in context_entities[:60]
            )
            more = f" ... {len(context_entities) - 60} more" if len(context_entities) > 60 else ""
            context_parts.append(
                f"\n## Context-only entities ({len(context_entities)}; part of the world, may be "
                f"mentioned, but they are NOT simulation accounts and never post)\n{names}{more}"
            )

        current_length = sum(len(p) for p in context_parts)
        remaining_length = (
            self.MAX_CONTEXT_LENGTH - current_length - 500
        )  # Leave 500 char margin

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(Document truncated)"
            context_parts.append(f"\n## Original document content\n{doc_text}")

        return "\n".join(context_parts)

    def _public_summary(self, entity: EntityNode) -> str:
        """Entity summary without private research instruction text"""
        return strip_private_text(entity.summary or "", getattr(self, "_private_texts", []))

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generate an entity summary"""
        lines = []

        # Group by type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} entities)")
            # Use configured display count and summary length
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary = self._public_summary(e)
                summary_preview = (
                    (summary[:summary_len] + "...")
                    if len(summary) > summary_len
                    else summary
                )
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... {len(type_entities) - display_count} more")

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM call with retry, including JSON repair logic"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # Lower temperature on each retry
                    # Don't set max_tokens, let LLM decide
                )

                content = extract_chat_completion_text(response)
                finish_reason = response.choices[0].finish_reason

                # Check if truncated
                if finish_reason == "length":
                    logger.warning(f"LLM output truncated (attempt {attempt + 1})")
                    content = self._fix_truncated_json(content)

                # Try parsing JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON parsing failed (attempt {attempt + 1}): {str(e)[:80]}"
                    )

                    # Try repairing JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {str(e)[:80]}")
                last_error = e
                import time

                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM call failed")

    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON"""
        content = content.strip()

        # Count unclosed brackets
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Check for unclosed strings
        if content and content[-1] not in '",}]':
            content += '"'

        # Close brackets
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair config JSON"""
        import re

        # Fix truncation case
        content = self._fix_truncated_json(content)

        # Extract JSON portion
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # Remove newlines in strings
            def fix_string(match):
                s = match.group(0)
                s = s.replace("\n", " ").replace("\r", " ")
                s = re.sub(r"\s+", " ", s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # Try removing all control characters
                json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                json_str = re.sub(r"\s+", " ", json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generate time configuration"""
        # Use configured context truncation length
        context_truncated = context[: self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calculate max allowed (80% of agent count)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Based on the following simulation requirements, generate the time simulation configuration.

{context_truncated}

## Task
Please generate the time configuration JSON.

### Basic Principles (for reference only; adjust flexibly based on the specific event and participating groups):
- Please infer the time zone and daily routine of the target user group from the simulation scenario. The following is a reference example for UTC+8.
- 0-5 AM: almost no activity (activity multiplier 0.05)
- 6-8 AM: gradually active (activity multiplier 0.4)
- 9-18 (9 AM - 6 PM): moderately active (activity multiplier 0.7)
- 19-22 (7 PM - 10 PM): peak hours (activity multiplier 1.5)
- After 23 (11 PM): activity declines (activity multiplier 0.5)
- General pattern: low at dawn, increasing in the morning, moderate during work hours, peak in the evening
- **Important**: The example values below are for reference only. You must adjust the specific time ranges based on the nature of the event and the characteristics of the participating groups.
  - e.g., student groups may peak at 21-23; media may be active all day; official institutions only during work hours
  - e.g., a sudden hot topic may drive discussion even late at night; off_peak_hours can be shortened accordingly

### Return JSON format (no markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Time configuration explanation for this event"
}}

Field descriptions:
- total_simulation_hours (int): Total simulation duration, 24-168 hours. Shorter for sudden events, longer for sustained topics.
- minutes_per_round (int): Duration per round, 30-120 minutes. 60 minutes recommended.
- agents_per_hour_min (int): Minimum number of agents activated per hour (range: 1-{max_agents_allowed})
- agents_per_hour_max (int): Maximum number of agents activated per hour (range: 1-{max_agents_allowed})
- peak_hours (int array): Peak hours, adjust based on the event's participating groups
- off_peak_hours (int array): Off-peak hours, usually late night and early morning
- morning_hours (int array): Morning hours
- work_hours (int array): Work hours
- reasoning (string): Briefly explain why the configuration is set this way"""

        system_prompt = "You are a social media simulation expert. Return pure JSON format. The time configuration must match the daily routine of the target user group in the simulation scenario."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Time config LLM generation failed: {e}, Using default config")
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Get the default time config (Chinese daily routine)"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 1 hour per round, accelerated time flow
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Using default activity schedule config (1 hour per round)",
        }

    def _parse_time_config(
        self, result: Dict[str, Any], num_entities: int
    ) -> TimeSimulationConfig:
        """Parse the time config result and validate that agents_per_hour values do not exceed the total agent count"""
        # Get raw values
        agents_per_hour_min = result.get(
            "agents_per_hour_min", max(1, num_entities // 15)
        )
        agents_per_hour_max = result.get(
            "agents_per_hour_max", max(5, num_entities // 5)
        )

        # Validate and correct: ensure not exceeding total agent count
        if agents_per_hour_min > num_entities:
            logger.warning(
                f"agents_per_hour_min ({agents_per_hour_min}) Exceeds total Agent count ({num_entities}), corrected"
            )
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(
                f"agents_per_hour_max ({agents_per_hour_max}) Exceeds total Agent count ({num_entities}), corrected"
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # Ensure min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(
                f"agents_per_hour_min >= max, corrected to {agents_per_hour_min}"
            )

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # Default: 1 hour per round
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # Late night: almost no one
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5,
        )

    def _generate_event_config(
        self, context: str, simulation_requirement: str, entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generate event configuration"""

        # Get available entity types for LLM reference
        entity_types_available = list(
            set(e.get_entity_type() or "Unknown" for e in entities)
        )

        # List representative entity names per type
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join(
            [f"- {t}: {', '.join(examples)}" for t, examples in type_examples.items()]
        )

        # Use configured context truncation length
        context_truncated = context[: self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Based on the following simulation requirements, generate the event configuration.

Simulation requirement: {simulation_requirement}

{context_truncated}

## Available Entity Types and Examples
{type_info}

## Task
Please generate the event configuration JSON:
- Extract hot topic keywords
- Describe the direction of public opinion development
- Design the content of initial posts; **each post must specify a poster_type (poster entity type)**

**Important**: poster_type must be chosen from the "Available Entity Types" above, so that initial posts can be assigned to the appropriate agent for publishing.
For example: official statements should be posted by Official/University types, news by MediaOutlet, student opinions by Student.

**Initial posts are in-world seed content only.** They are public social-media posts that exist in the simulated world.
They must NEVER contain research or experiment material: no prompts, instructions, test tasks, stage descriptions,
funnel requirements, hypotheses, persona identifiers (such as "C03"), or text copied from the research brief.
Private research instructions are delivered to agents through a separate private channel, not through posts.

Return JSON format (no markdown):
{{
    "hot_topics": ["keyword1", "keyword2", ...],
    "narrative_direction": "<description of public opinion development direction>",
    "initial_posts": [
        {{"content": "post content", "poster_type": "entity type (must be chosen from available types)"}},
        ...
    ],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = "You are a public opinion analysis expert. Return pure JSON format. Note that poster_type must exactly match the available entity types."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Event config LLM generation failed: {e}, Using default config")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Using default config",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parse the event config result"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", ""),
        )

    def _filter_seed_posts(
        self, event_config: EventConfig, private_texts: List[str], brief_text: str
    ) -> EventConfig:
        """
        Remove seed posts that carry research/orchestration content

        Rejected posts are kept in event_config.rejected_seed_posts for inspection
        and are never published.
        """
        kept = []
        for post in event_config.initial_posts or []:
            if not isinstance(post, dict):
                continue
            content = str(post.get("content") or "").strip()
            if not content:
                continue
            unsafe, reason = is_unsafe_public_text(content, private_texts, brief_text)
            if unsafe:
                logger.warning(f"Seed post rejected ({reason}): {content[:80]!r}")
                event_config.rejected_seed_posts.append({"content": content, "reason": reason,
                                                         "poster_type": post.get("poster_type")})
                continue
            kept.append(post)
        event_config.initial_posts = kept
        return event_config

    def _assign_initial_post_agents(
        self, event_config: EventConfig, agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Assign suitable publisher Agents to initial posts

        Matches the most appropriate agent_id based on each post's poster_type.
        Seed content is preferably published by agents that are not target
        consumers, so target-consumer accounts do not carry system content.
        """
        if not event_config.initial_posts:
            return event_config

        non_target = [a for a in agent_configs if a.evidence_role and a.evidence_role != "target_consumer"]
        if non_target:
            agent_configs = non_target

        # Build agent index by entity type
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Type mapping table (handles different formats LLM may output)
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        # Track used agent indices per type to avoid reusing the same agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # Try to find matching agent
            matched_agent_id = None

            # 1. Direct match
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Match by alias
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            # 3. If still not found, use highest-influence agent
            if matched_agent_id is None:
                logger.warning(
                    f"Type not found '{poster_type}'  matching Agent, using highest-influence Agent"
                )
                if agent_configs:
                    # Sort by influence, pick highest
                    sorted_agents = sorted(
                        agent_configs, key=lambda a: a.influence_weight, reverse=True
                    )
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append(
                {
                    "content": content,
                    "poster_type": post.get("poster_type", "Unknown"),
                    "poster_agent_id": matched_agent_id,
                    "origin": ORIGIN_SYSTEM_SEED,
                }
            )

            logger.info(
                f"Initial post assignment: poster_type='{poster_type}' -> agent_id={matched_agent_id}"
            )

        event_config.initial_posts = updated_posts
        return event_config

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str,
    ) -> List[AgentActivityConfig]:
        """Generate Agent configurations in batches"""

        # Build entity info (using configured summary length)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append(
                {
                    "agent_id": start_idx + i,
                    "entity_name": e.name,
                    "entity_type": e.get_entity_type() or "Unknown",
                    "summary": self._public_summary(e)[:summary_len],
                }
            )

        prompt = f"""Based on the following information, generate social media activity configurations for each entity.

Simulation requirement: {simulation_requirement}

## Entity List
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Task
Generate an activity configuration for each entity. Notes:
- **Times should match the daily routine of the target user group**: The following is a reference (UTC+8). Please adjust based on the simulation scenario.
- **Official institutions** (University/GovernmentAgency): low activity (0.1-0.3), active during work hours (9-17), slow response (60-240 minutes), high influence (2.5-3.0)
- **Media** (MediaOutlet): medium activity (0.4-0.6), active all day (8-23), fast response (5-30 minutes), high influence (2.0-2.5)
- **Individuals** (Student/Person/Alumni): high activity (0.6-0.9), mainly active in the evening (18-23), fast response (1-15 minutes), low influence (0.8-1.2)
- **Public figures/experts**: medium activity (0.4-0.6), medium-high influence (1.5-2.0)

Return JSON format (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match the input>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <post frequency>,
            "comments_per_hour": <comment frequency>,
            "active_hours": [<list of active hours, considering the daily routine of the target user group>],
            "response_delay_min": <minimum response delay in minutes>,
            "response_delay_max": <maximum response delay in minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <influence weight>
        }},
        ...
    ]
}}"""

        system_prompt = "You are a social media behavior analysis expert. Return pure JSON. The configuration must match the daily routine of the target user group in the simulation scenario."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {
                cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])
            }
        except Exception as e:
            logger.warning(f"Agent config batch LLM generation failed: {e}, using rule-based generation")
            llm_configs = {}

        # Build AgentActivityConfig object
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # If LLM didn't generate, use rule-based generation
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0),
            )
            configs.append(config)

        return configs

    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Generate a single Agent configuration based on rules (Chinese daily routine)"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Official institution: work hours active, low frequency, high influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0,
            }
        elif entity_type in ["mediaoutlet"]:
            # Media: all-day active, medium frequency, high influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5,
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Expert/Professor: work + evening active, medium frequency
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0,
            }
        elif entity_type in ["student"]:
            # Student: evening-focused, high frequency
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                ],  # Morning + evening
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8,
            }
        elif entity_type in ["alumni"]:
            # Alumni: evening-focused
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Lunch break + evening
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
        else:
            # Average person: evening peak
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [
                    9,
                    10,
                    11,
                    12,
                    13,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                ],  # Daytime + evening
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
