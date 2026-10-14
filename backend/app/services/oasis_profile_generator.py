"""
OASIS Agent Profile Generator
Converts entities in the graph store into the Agent Profile format required by the OASIS simulation platform

Optimization improvements:
1. Calls graph retrieval to further enrich node information
2. Optimized prompts to generate very detailed personas
3. Distinguishes individual entities from abstract group entities
"""

import json
import random
import zlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, get_locale, set_locale, t
from ..utils.openai_chat_compat import create_chat_completion, extract_chat_completion_text
from .entity_reader import EntityNode, EntityReader
from .local_graph_store import LocalGraphStore

logger = get_logger("mirofish.oasis_profile")


def _coerce_to_str(value: Any) -> str:
    """Coerce a value to a plain string.

    Handles dict, list, and other non-string types that may be returned
    by LLM JSON parsing.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('text', 'value', 'description', 'content', 'summary', 'name'):
            if key in value:
                candidate = _coerce_to_str(value[key])
                if candidate:
                    return candidate
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        str_items = [_coerce_to_str(item) for item in value]
        str_items = [item for item in str_items if item]
        return ', '.join(str_items)
    return str(value)


def _coerce_to_str_list(value: Any) -> List[str]:
    """Coerce a value to a list of strings.

    Handles nested structures that may be returned by LLM JSON parsing.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                result.extend(_coerce_to_str_list(item))
            else:
                text = _coerce_to_str(item)
                if text:
                    result.append(text)
        return result
    text = _coerce_to_str(value)
    return [text] if text else []


@dataclass
class OasisAgentProfile:
    """OASIS Agent Profile data structure"""

    # Common fields
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Optional fields - Reddit style
    karma: int = 1000

    # Optional fields - Twitter style
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Extra persona info
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Source entity info
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def __post_init__(self):
        """Normalize structured LLM fields once at the profile boundary."""
        self.bio = _coerce_to_str(self.bio) or self.name
        self.persona = _coerce_to_str(self.persona) or (
            f"{self.name} is a participant in social discussions."
        )
        self.country = _coerce_to_str(self.country) or None
        self.profession = _coerce_to_str(self.profession) or None
        self.gender = _coerce_to_str(self.gender) or None
        self.mbti = _coerce_to_str(self.mbti) or None
        self.interested_topics = _coerce_to_str_list(self.interested_topics)

    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert to Reddit platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS lib requires field name 'username' (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Add extra persona info (if any)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_format(self) -> Dict[str, Any]:
        """Convert to Twitter platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS lib requires field name 'username' (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Add extra persona info
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a complete dict format"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    OASIS Profile Generator

    Converts entities in the graph store into Agent Profiles required by OASIS simulations

    Optimization features:
    1. Calls graph store retrieval to obtain richer context
    2. Generates very detailed personas (including basic info, career history, personality traits, social media behavior, etc.)
    3. Distinguishes individual entities from abstract group entities
    """

    # MBTI type list
    MBTI_TYPES = [
        "INTJ",
        "INTP",
        "ENTJ",
        "ENTP",
        "INFJ",
        "INFP",
        "ENFJ",
        "ENFP",
        "ISTJ",
        "ISFJ",
        "ESTJ",
        "ESFJ",
        "ISTP",
        "ISFP",
        "ESTP",
        "ESFP",
    ]

    # Common country list
    COUNTRIES = [
        "China",
        "US",
        "UK",
        "Japan",
        "Germany",
        "France",
        "Canada",
        "Australia",
        "Brazil",
        "India",
        "South Korea",
    ]

    # Individual-type entities (need detailed persona generation)
    INDIVIDUAL_ENTITY_TYPES = [
        "student",
        "alumni",
        "professor",
        "person",
        "publicfigure",
        "expert",
        "faculty",
        "official",
        "journalist",
        "activist",
    ]

    # Group/institution-type entities (need group representative persona generation)
    GROUP_ENTITY_TYPES = [
        "university",
        "governmentagency",
        "organization",
        "ngo",
        "mediaoutlet",
        "company",
        "institution",
        "group",
        "community",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        graph_api_key: Optional[str] = None,
        graph_id: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError(t("api.llmApiKeyMissing"))

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Local graph store for entity context enrichment
        self.graph_id = graph_id

    def generate_profile_from_entity(
        self, entity: EntityNode, user_id: int, use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Generate an OASIS Agent Profile from a graph entity

        Args:
            entity: graph entity node
            user_id: User ID (for OASIS)
            use_llm: Whether to use an LLM to generate a detailed persona

        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"

        # Basic info
        name = entity.name
        user_name = self._generate_username(name)

        # Build context info
        context = self._build_entity_context(entity)

        if use_llm:
            # Use LLM to generate detailed persona
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context,
            )
        else:
            # Use rule-based basic persona generation
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
            )

        # Deterministic social metrics per entity
        rng = random.Random(zlib.crc32(name.encode("utf-8")))

        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get(
                "persona", entity.summary or f"A {entity_type} named {name}."
            ),
            karma=profile_data.get("karma", rng.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", rng.randint(50, 500)),
            follower_count=profile_data.get(
                "follower_count", rng.randint(100, 1000)
            ),
            statuses_count=profile_data.get(
                "statuses_count", rng.randint(100, 2000)
            ),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )

    def _generate_username(self, name: str) -> str:
        """
        Generate a deterministic username from the entity name

        The same entity name always yields the same username (stable suffix from a
        checksum), so repeated preparations produce comparable profiles.
        """
        # Remove special chars, convert to lowercase
        username = name.lower().replace(" ", "_")
        username = "".join(c for c in username if c.isalnum() or c == "_") or "agent"

        suffix = 100 + zlib.crc32(name.encode("utf-8")) % 900
        return f"{username}_{suffix}"

    def _search_graph_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Search local graph store for entity-related context (facts + node summaries).
        Replaces former Zep hybrid search (now uses local search).
        """
        if not self.graph_id:
            return {"facts": [], "node_summaries": [], "context": ""}

        entity_name = entity.name
        results = {"facts": [], "node_summaries": [], "context": ""}

        try:
            store = LocalGraphStore(self.graph_id)

            # Search edges (facts) by entity name
            edges = store.search_edges(entity_name, limit=30)
            all_facts = set()
            for edge in edges:
                if edge.fact:
                    all_facts.add(edge.fact)
            results["facts"] = list(all_facts)

            # Search nodes by entity name
            nodes = store.search_nodes(entity_name, limit=20)
            all_summaries = set()
            for node in nodes:
                if node.summary:
                    all_summaries.add(node.summary)
                if node.name and node.name != entity_name:
                    all_summaries.add(f"Related entity: {node.name}")
            results["node_summaries"] = list(all_summaries)

            # Build context text
            context_parts = []
            if results["facts"]:
                context_parts.append(
                    "Facts:\n" + "\n".join(f"- {f}" for f in results["facts"][:20])
                )
            if results["node_summaries"]:
                context_parts.append(
                    "Related entities:\n"
                    + "\n".join(f"- {s}" for s in results["node_summaries"][:10])
                )
            results["context"] = "\n\n".join(context_parts)

            logger.info(
                f"Local graph search complete: {entity_name}, {len(results['facts'])} facts, {len(results['node_summaries'])} related nodes"
            )

        except Exception as e:
            logger.warning(f"Graph search failed ({entity_name}): {e}")

        return results

    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Build the entity's complete context information

        Includes:
        1. The entity's own edge information (facts)
        2. Detailed info of related nodes
        3. Rich information retrieved by graph search
        """
        context_parts = []

        # 1. Add entity attribute info
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entity attributes\n" + "\n".join(attrs))

        # 2. Add related edge info (facts/relationships)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # No limit on count
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(
                            f"- {entity.name} --[{edge_name}]--> (related entity)"
                        )
                    else:
                        relationships.append(
                            f"- (related entity) --[{edge_name}]--> {entity.name}"
                        )

            if relationships:
                context_parts.append("### Related facts and relationships\n" + "\n".join(relationships))

        # 3. Add related node details
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # No limit on count
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                # Filter out default labels
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append("### Related entity info\n" + "\n".join(related_info))

        # 4. Use graph search for richer info
        graph_results = self._search_graph_for_entity(entity)

        if graph_results.get("facts"):
            # Deduplicate: exclude existing facts
            new_facts = [f for f in graph_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append(
                    "### Facts retrieved via graph search\n"
                    + "\n".join(f"- {f}" for f in new_facts[:15])
                )

        if graph_results.get("node_summaries"):
            context_parts.append(
                "### Related nodes retrieved via graph search\n"
                + "\n".join(f"- {s}" for s in graph_results["node_summaries"][:10])
            )

        return "\n\n".join(context_parts)

    def _is_individual_entity(self, entity_type: str) -> bool:
        """Determine whether it is an individual-type entity"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Determine whether it is a group/organization-type entity"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES

    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> Dict[str, Any]:
        """
        Use an LLM to generate a very detailed persona

        Differentiates by entity type:
        - Individual entities: generate a concrete person setting
        - Group/organization entities: generate a representative account setting
        """

        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Try multiple times until success or max retries
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = create_chat_completion(
                    self.client,
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(is_individual),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # Lower temperature on each retry
                    # Don't set max_tokens, let LLM decide
                )

                content = extract_chat_completion_text(response)

                # Check if truncated (finish_reason is not 'stop')
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        f"LLM output truncated (attempt {attempt + 1}), attempting repair..."
                    )
                    content = self._fix_truncated_json(content)

                # Try parsing JSON
                try:
                    result = json.loads(content)

                    # Validate required fields
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = (
                            entity_summary[:200]
                            if entity_summary
                            else f"{entity_type}: {entity_name}"
                        )
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = (
                            entity_summary or f"{entity_name}is a{entity_type}。"
                        )

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(
                        f"JSON parsing failed (attempt {attempt + 1}): {str(je)[:80]}"
                    )

                    # Try repairing JSON
                    result = self._try_fix_json(
                        content, entity_name, entity_type, entity_summary
                    )
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {str(e)[:80]}")
                last_error = e
                import time

                time.sleep(1 * (attempt + 1))  # Exponential backoff

        logger.warning(
            f"LLM persona generation failed ({max_attempts} attempts): {last_error}, using rule-based generation"
        )
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )

    def _fix_truncated_json(self, content: str) -> str:
        """Fix truncated JSON (output truncated by max_tokens)"""
        import re

        # If JSON is truncated, try to close it
        content = content.strip()

        # Count unclosed brackets
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Check for unclosed strings
        # Simple check: if last quote is not followed by comma or closing bracket, string may be truncated
        if content and content[-1] not in '",}]':
            # Try to close string
            content += '"'

        # Close brackets
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_json(
        self, content: str, entity_name: str, entity_type: str, entity_summary: str = ""
    ) -> Dict[str, Any]:
        """Attempt to repair broken JSON"""
        import re

        # 1. First try to fix truncation
        content = self._fix_truncated_json(content)

        # 2. Try to extract JSON portion
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # 3. Handle newlines in strings
            # Find all string values and replace newlines within them
            def fix_string_newlines(match):
                s = match.group(0)
                # Replace actual newlines in strings with spaces
                s = s.replace("\n", " ").replace("\r", " ")
                # Replace extra spaces
                s = re.sub(r"\s+", " ", s)
                return s

            # Match JSON string values
            json_str = re.sub(
                r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str
            )

            # 4. Try parsing
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. If still failing, try more aggressive repair
                try:
                    # Remove all control characters
                    json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                    # Replace all consecutive whitespace
                    json_str = re.sub(r"\s+", " ", json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. Try to extract partial info from content
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # May be truncated

        bio = (
            bio_match.group(1)
            if bio_match
            else (
                entity_summary[:200]
                if entity_summary
                else f"{entity_type}: {entity_name}"
            )
        )
        persona = (
            persona_match.group(1)
            if persona_match
            else (entity_summary or f"{entity_name}is a{entity_type}。")
        )

        # If meaningful content extracted, mark as repaired
        if bio_match or persona_match:
            logger.info(f"Extracted partial info from corrupted JSON")
            return {"bio": bio, "persona": persona, "_fixed": True}

        # 7. Complete failure, return basic structure
        logger.warning(f"JSON repair failed, returning basic structure")
        return {
            "bio": entity_summary[:200]
            if entity_summary
            else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name}is a{entity_type}。",
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        """Get the system prompt"""
        base_prompt = "You are an expert in social media user persona generation. Generate detailed, realistic personas for public opinion simulation, faithfully reflecting existing real-world conditions. You must return valid JSON format, and all string values must not contain unescaped newline characters."
        return f"{base_prompt}\n\n{get_language_instruction()}"

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """Build the detailed persona prompt for an individual entity"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "None"
        )
        context_str = context[:3000] if context else "No additional context"

        return f"""Generate a detailed social media user persona for the entity, faithfully reflecting existing real-world conditions.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_str}

Context information:
{context_str}

Please generate JSON containing the following fields:

1. bio: a social media bio of about 200 characters
2. persona: a detailed persona description (about 2000 characters of plain text), which must include:
   - Basic information (age, occupation, educational background, location)
   - Personal background (key life experiences, connection to the event, social relationships)
   - Personality traits (MBTI type, core personality, emotional expression style)
   - Social media behavior (posting frequency, content preferences, interaction style, language characteristics)
   - Stance and viewpoints (attitude toward the topic, what may provoke/move them)
   - Distinctive traits (catchphrases, special experiences, personal hobbies)
   - Personal memories (an important part of the persona; describe this individual's connection to the event, as well as their existing actions and reactions during the event)
3. age: the age as a number (must be an integer)
4. gender: gender, must be in English: "male" or "female"
5. mbti: MBTI type (e.g. INTJ, ENFP, etc.)
6. country: country of residence as stated or clearly implied by the entity information/context; "Unknown" if it is not stated (never invent one)
7. profession: occupation
8. interested_topics: an array of topics of interest

Important:
- All field values must be strings or numbers; do not use newline characters
- persona must be a single coherent text description
- {get_language_instruction()} (the gender field must use English male/female)
- The content must be consistent with the entity information
- age must be a valid integer, and gender must be "male" or "female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """Build the detailed persona prompt for a group/organization entity"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "None"
        )
        context_str = context[:3000] if context else "No additional context"

        return f"""Generate a detailed social media account profile for an organization/group entity, faithfully reflecting existing real-world conditions.

Entity name: {entity_name}
Entity type: {entity_type}
Entity summary: {entity_summary}
Entity attributes: {attrs_str}

Context information:
{context_str}

Please generate JSON containing the following fields:

1. bio: an official account bio of about 200 characters, professional and appropriate
2. persona: a detailed account profile description (about 2000 characters of plain text), which must include:
   - Organizational basic information (official name, nature of the organization, founding background, main functions)
   - Account positioning (account type, target audience, core functions)
   - Speaking style (language characteristics, common expressions, taboo topics)
   - Publishing content characteristics (content types, posting frequency, active time periods)
   - Stance and attitude (official position on core topics, how to handle controversies)
   - Special notes (the audience group represented, operational habits)
   - Organizational memory (an important part of the organization's persona; describe this organization's connection to the event, as well as its existing actions and reactions during the event)
3. age: fixed value 30 (the virtual age of an organizational account)
4. gender: fixed value "other" (organizational accounts use "other" to indicate non-individual)
5. mbti: MBTI type, used to describe the account style, e.g. ISTJ represents rigorous and conservative
6. country: country of residence as stated or clearly implied by the entity information/context; "Unknown" if it is not stated (never invent one)
7. profession: description of the organization's functions
8. interested_topics: an array of focus areas

Important:
- All field values must be strings or numbers; null values are not allowed
- persona must be a single coherent text description; do not use newline characters
- {get_language_instruction()} (the gender field must use English "other")
- age must be the integer 30, and gender must be the string "other"
- The organizational account's speech should match its identity positioning"""

    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a basic persona using rules"""

        # Deterministic per entity: the same entity always gets the same basic persona
        rng = random.Random(zlib.crc32(entity_name.encode("utf-8")))

        # Generate different personas based on entity type
        entity_type_lower = entity_type.lower()

        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": rng.randint(18, 30),
                "gender": rng.choice(["male", "female"]),
                "mbti": rng.choice(self.MBTI_TYPES),
                "country": "Unknown",
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": rng.randint(35, 60),
                "gender": rng.choice(["male", "female"]),
                "mbti": rng.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": "Unknown",
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # Virtual age for institution
                "gender": "other",  # Institution uses 'other'
                "mbti": "ISTJ",  # Institution style: rigorous and conservative
                "country": "Unknown",
                "profession": "Media",
                "interested_topics": [
                    "General News",
                    "Current Events",
                    "Public Affairs",
                ],
            }

        elif entity_type_lower in [
            "university",
            "governmentagency",
            "ngo",
            "organization",
        ]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # Virtual age for institution
                "gender": "other",  # Institution uses 'other'
                "mbti": "ISTJ",  # Institution style: rigorous and conservative
                "country": "Unknown",
                "profession": entity_type,
                "interested_topics": [
                    "Public Policy",
                    "Community",
                    "Official Announcements",
                ],
            }

        else:
            # Default persona
            return {
                "bio": entity_summary[:150]
                if entity_summary
                else f"{entity_type}: {entity_name}",
                "persona": entity_summary
                or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": rng.randint(25, 50),
                "gender": rng.choice(["male", "female"]),
                "mbti": rng.choice(self.MBTI_TYPES),
                "country": "Unknown",
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }

    def set_graph_id(self, graph_id: str):
        """Set the graph ID for graph retrieval"""
        self.graph_id = graph_id

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit",
    ) -> List[OasisAgentProfile]:
        """
        Batch generate Agent Profiles from entities (supports parallel generation)

        Args:
            entities: Entity list
            use_llm: Whether to use an LLM to generate detailed personas
            progress_callback: Progress callback function (current, total, message)
            graph_id: Graph ID, used for graph retrieval to get richer context
            parallel_count: Parallel generation count, default 5
            realtime_output_path: Real-time write file path (if provided, writes once per generated profile)
            output_platform: Output platform format ("reddit" or "twitter")

        Returns:
            Agent Profile list
        """
        import concurrent.futures
        from threading import Lock

        # Set graph_id for graph retrieval
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # Pre-allocate list to maintain order
        completed_count = [0]  # Use list for mutability in closure
        lock = Lock()

        # Helper function for real-time file writing
        def save_profiles_realtime():
            """Real-time save generated profiles to file"""
            if not realtime_output_path:
                return

            with lock:
                # Filter out generated profiles
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Reddit JSON format
                        profiles_data = [
                            p.to_reddit_format() for p in existing_profiles
                        ]
                        with open(realtime_output_path, "w", encoding="utf-8") as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Twitter CSV format
                        import csv

                        profiles_data = [
                            p.to_twitter_format() for p in existing_profiles
                        ]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(
                                realtime_output_path, "w", encoding="utf-8", newline=""
                            ) as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Failed to save profiles in real-time: {e}")

        # Capture locale before spawning thread pool workers
        current_locale = get_locale()

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Worker function to generate a single profile"""
            set_locale(current_locale)
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity, user_id=idx, use_llm=use_llm
                )

                # Output generated persona to console and log in real-time
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"Generating entity {entity.name}  persona generation failed: {str(e)}")
                # Create a basic profile
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)

        logger.info(f"Starting parallel generation of {total}  Agent personas (parallel: {parallel_count})...")
        print(f"\n{'=' * 60}")
        print(f"Starting Agent persona generation - {total} entities total, parallel: {parallel_count}")
        print(f"{'=' * 60}\n")

        # Execute in parallel using thread pool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_count
        ) as executor:
            # Submit all tasks
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # Real-time file write
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Completed {current}/{total}: {entity.name}（{entity_type}）",
                        )

                    if error:
                        logger.warning(
                            f"[{current}/{total}] {entity.name} Using fallback persona: {error}"
                        )
                    else:
                        logger.info(
                            f"[{current}/{total}] Successfully generated persona: {entity.name} ({entity_type})"
                        )

                except Exception as e:
                    logger.error(f"Processing entity {entity.name}  encountered exception: {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary
                        or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Real-time file write (even for fallback persona)
                    save_profiles_realtime()

        print(f"\n{'=' * 60}")
        print(f"Persona generation complete! Generated {len([p for p in profiles if p])} Agents total")
        print(f"{'=' * 60}\n")

        return profiles

    def _print_generated_profile(
        self, entity_name: str, entity_type: str, profile: OasisAgentProfile
    ):
        """Real-time output of the generated persona to the console (full content, not truncated)"""
        separator = "-" * 70

        # Build full output content (no truncation)
        topics_str = (
            ", ".join(profile.interested_topics) if profile.interested_topics else "None"
        )

        output_lines = [
            f"\n{separator}",
            t("progress.profileGenerated", name=entity_name, type=entity_type),
            f"{separator}",
            f"Username: {profile.user_name}",
            f"",
            f"[Bio]",
            f"{profile.bio}",
            f"",
            f"[Detailed Persona]",
            f"{profile.persona}",
            f"",
            f"[Basic Attributes]",
            f"Age: {profile.age} | Gender: {profile.gender} | MBTI: {profile.mbti}",
            f"Profession: {profile.profession} | Country: {profile.country}",
            f"Interested topics: {topics_str}",
            separator,
        ]

        output = "\n".join(output_lines)

        # Output to console only (avoid duplication, logger no longer outputs full content)
        print(output)

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """
        Save Profiles to a file (selects the correct format by platform)

        OASIS platform format requirements:
        - Twitter: CSV format
        - Reddit: JSON format

        Args:
            profiles: Profile list
            file_path: File path
            platform: Platform type ("reddit" or "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Save Twitter Profiles as CSV format (meets OASIS official requirements)

        OASIS Twitter required CSV fields:
        - user_id: User ID (starting from 0 based on CSV order)
        - name: User's real name
        - username: Username in the system
        - user_char: Detailed persona description (injected into the LLM system prompt to guide Agent behavior)
        - description: A short public bio (displayed on the user's profile page)

        Difference between user_char vs description:
        - user_char: For internal use, LLM system prompt, decides how the Agent thinks and acts
        - description: For external display, the bio visible to other users
        """
        import csv

        # Ensure file extension is .csv
        if not file_path.endswith(".csv"):
            file_path = file_path.replace(".json", ".csv")

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write OASIS-required header
            headers = ["user_id", "name", "username", "user_char", "description"]
            writer.writerow(headers)

            # Write data rows
            for idx, profile in enumerate(profiles):
                # user_char: full persona (bio + persona), used for LLM system prompt
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Handle newlines (replace with spaces in CSV)
                user_char = user_char.replace("\n", " ").replace("\r", " ")

                # description: short bio, for external display
                description = profile.bio.replace("\n", " ").replace("\r", " ")

                row = [
                    idx,  # user_id: sequential ID starting from 0
                    profile.name,  # name: real name
                    profile.user_name,  # username: Username
                    user_char,  # user_char: full persona (internal LLM use)
                    description,  # description: short bio (external display)
                ]
                writer.writerow(row)

        logger.info(
            f"Saved {len(profiles)}  Twitter Profiles to {file_path} (OASIS CSV format)"
        )

    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Normalize the gender field to the English format required by OASIS

        OASIS requires: male, female, other
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # Chinese-to-English mapping
        gender_map = {
            "Male": "male",
            "Female": "female",
            "Institution": "other",
            "Other": "other",
            # English already present
            "male": "male",
            "female": "female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Save Reddit Profiles as JSON format

        Uses the same format as to_reddit_format() to ensure OASIS can read them correctly.
        Must contain the user_id field, which is the key for OASIS agent_graph.get_agent() matching!

        Required fields:
        - user_id: User ID (integer, used to match poster_agent_id in initial_posts)
        - username: Username
        - name: Display name
        - bio: Bio
        - persona: Detailed persona
        - age: Age (integer)
        - gender: "male", "female", or "other"
        - mbti: MBTI type
        - country: Country
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Use format consistent with to_reddit_format()
            item = {
                "user_id": profile.user_id
                if profile.user_id is not None
                else idx,  # Critical: must include user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona
                or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # OASIS required fields - ensure all have defaults
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "Unknown",
            }

            # Optional fields
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved {len(profiles)}  Reddit Profiles to {file_path} (JSON format, includes user_id field)"
        )

    # Keep old method name as alias for backward compatibility
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """[Deprecated] Please use the save_profiles() method"""
        logger.warning("save_profiles_to_json is deprecated, please use save_profiles method")
        self.save_profiles(profiles, file_path, platform)
