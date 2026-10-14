"""
Ontology generation service
Endpoint 1: Analyze text content, generate entity and relation type definitions suitable for social simulation
"""

import json
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction
from ..utils.logger import get_logger
from ..utils.file_parser import split_text_into_chunks
from ..utils.ontology import (
    MAX_ONTOLOGY_TYPES,
    normalize_ontology_attributes,
    normalize_ontology_source_targets,
)

logger = get_logger("mirofish.ontology")


def _to_pascal_case(name: str) -> str:
    """Convert a name of any format to PascalCase (e.g. 'works_for' -> 'WorksFor', 'person' -> 'Person')"""
    # Split by non-alphanumeric characters
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    # Then split by camelCase boundaries (e.g. 'camelCase' -> ['camel', 'Case'])
    words = []
    for part in parts:
        words.extend(re.sub(r"([a-z])([A-Z])", r"\1_\2", part).split("_"))
    # Capitalize each word, filter empty strings
    result = "".join(word.capitalize() for word in words if word)
    return result if result else "Unknown"


def _to_upper_snake_case(name: str) -> str:
    """Convert free-form or camelCase names to SCREAMING_SNAKE_CASE (e.g. 'worksFor' -> 'WORKS_FOR')"""
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", separated).strip("_").upper()
    if not normalized:
        return "UNKNOWN"
    if normalized[0].isdigit():
        normalized = f"REL_{normalized}"
    return normalized


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """You are a professional knowledge-graph ontology design expert. Your task is to analyze the given text content and simulation requirements, and design entity types and relation types suitable for **social media public opinion simulation**.

**Important: You must output valid JSON-format data and nothing else.**

## Core Task Background

We are building a **social media public opinion simulation system**. In this system:
- Each entity is an "account" or "actor" that can speak, interact, and spread information on social media
- Entities influence, repost, comment on, and respond to each other
- We need to simulate how various parties react and how information propagates during a public-opinion event

Therefore, **entities must be real-world actors that can speak and interact on social media**:

**Allowed**:
- Specific individuals (public figures, involved parties, opinion leaders, experts/scholars, ordinary people)
- Companies and enterprises (including their official accounts)
- Organizations (universities, associations, NGOs, unions, etc.)
- Government departments and regulatory agencies
- Media organizations (newspapers, TV stations, self-media, websites)
- Social-media platforms themselves
- Representatives of specific groups (e.g., alumni associations, fan clubs, rights-defense groups)

**Not allowed**:
- Abstract concepts (e.g., "public opinion", "emotion", "trend")
- Topics/themes (e.g., "academic integrity", "education reform")
- Viewpoints/attitudes (e.g., "supporters", "opponents")

## Output Format

Output JSON with the following structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name (English, PascalCase)",
            "description": "Short description (English, no more than 100 characters)",
            "attributes": [
                {
                    "name": "Attribute name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["example entity 1", "example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relation type name (English, UPPER_SNAKE_CASE)",
            "description": "Short description (English, no more than 100 characters)",
            "source_targets": [
                {"source": "source entity type", "target": "target entity type"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis of the text content"
}
```

## Design Guidelines (Extremely Important!)

### 1. Entity Type Design - Must Be Strictly Followed

**Quantity requirement: exactly 10 entity types**

**Hierarchy requirement (must include both specific types and fallback types)**:

Your 10 entity types must contain the following hierarchy:

A. **Fallback types (must be included, placed as the last 2 in the list)**:
   - `Person`: Fallback type for any natural person. When an individual does not fit a more specific person type, classify them here.
   - `Organization`: Fallback type for any organization. When an organization does not fit a more specific organization type, classify it here.

B. **Specific types (8 of them, designed based on the text content)**:
   - Design more specific types for the main actors appearing in the text
   - For example: if the text concerns an academic event, you may have `Student`, `Professor`, `University`
   - For example: if the text concerns a business event, you may have `Company`, `CEO`, `Employee`

**Why fallback types are needed**:
- The text will mention various people such as "K-12 teachers", "a passerby", "a certain netizen"
- If no dedicated type matches, they should be classified as `Person`
- Likewise, small organizations, ad-hoc groups, etc. should be classified as `Organization`

**Design principles for specific types**:
- Identify high-frequency or key actor types from the text
- Each specific type should have clear boundaries and avoid overlap
- The description must clearly explain how this type differs from the fallback type

### 2. Relation Type Design

- Quantity: 6-10
- Relations should reflect real connections in social-media interactions
- For each relation, list only the **most representative and meaningful** source-target pairs — typically 1-4 pairs per relation
- Do NOT exhaustively enumerate every possible source×target combination; only include pairs that represent the primary or high-frequency direction of that relation
- As a rule of thumb, the combined `source_targets` across ALL relations should cover your entity types, but no single relation should list more than ~5 pairs

### 3. Attribute Design

- 1-3 key attributes per entity type
- **Note**: Attribute names must not use `name`, `uuid`, `group_id`, `created_at`, `summary` (these are system reserved words)
- Recommended: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Entity Type Reference

**Person types (specific)**:
- Student: Student
- Professor: Professor/Scholar
- Journalist: Journalist
- Celebrity: Celebrity/Influencer
- Executive: Executive
- Official: Government official
- Lawyer: Lawyer
- Doctor: Doctor

**Person types (fallback)**:
- Person: Any natural person (use when not fitting the above specific types)

**Organization types (specific)**:
- University: University/Higher-education institution
- Company: Company/Enterprise
- GovernmentAgency: Government agency
- MediaOutlet: Media organization
- Hospital: Hospital
- School: K-12 school
- NGO: Non-governmental organization

**Organization types (fallback)**:
- Organization: Any organization (use when not fitting the above specific types)

## Relation Type Reference

- WORKS_FOR: Works at
- STUDIES_AT: Studies at
- AFFILIATED_WITH: Affiliated with
- REPRESENTS: Represents
- REGULATES: Regulates
- REPORTS_ON: Reports on
- COMMENTS_ON: Comments on
- RESPONDS_TO: Responds to
- SUPPORTS: Supports
- OPPOSES: Opposes
- COLLABORATES_WITH: Collaborates with
- COMPETES_WITH: Competes with
"""


class OntologyGenerator:
    """
    Ontology generator
    Analyze text content, generate entity and relation type definitions
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate ontology definition

        Args:
            document_texts: List of document texts
            simulation_requirement: Simulation requirement description
            additional_context: Additional context

        Returns:
            Ontology definition (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts, simulation_requirement, additional_context
        )

        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Call LLM
        result = self.llm_client.chat_json(
            messages=messages, temperature=0.3, max_tokens=8192
        )

        # Validate and post-process
        result = self._validate_and_process(result)

        return result

    # Max text length sent to LLM (50k chars)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    # Long documents are sampled across their whole length instead of truncated
    LONG_TEXT_CHUNK_SIZE = 8000
    LONG_TEXT_CHUNK_OVERLAP = 200
    MAX_LONG_TEXT_CHUNKS = 60
    MIN_LONG_TEXT_EXCERPT = 400

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str],
    ) -> str:
        """Build the user message"""

        combined_text = self._build_document_context(document_texts)

        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""

        if additional_context:
            message += f"""
## Additional Notes

{additional_context}
"""

        message += """
Based on the above content, design entity types and relation types suitable for social public-opinion simulation.

**Rules that must be followed**:
1. Output exactly 10 entity types
2. The last 2 must be fallback types: Person (individual fallback) and Organization (organization fallback)
3. The first 8 are specific types designed based on the text content
4. All entity types must be real-world actors that can speak; they must not be abstract concepts
5. Attribute names must not use reserved words such as name, uuid, group_id; use full_name, org_name, etc. instead
"""

        return message

    def _build_document_context(self, document_texts: List[str]) -> str:
        """
        Build the document context for ontology analysis

        Short input is passed unchanged. Long input is split into chunks and a
        representative, evenly spaced sample (beginning, middle and end) is used,
        so the ontology is not designed from the first pages only.
        """
        texts = [t for t in document_texts if isinstance(t, str) and t.strip()]
        combined_text = "\n\n---\n\n".join(texts)
        original_length = len(combined_text)

        if original_length <= self.MAX_TEXT_LENGTH_FOR_LLM:
            return combined_text

        chunks = self._collect_document_chunks(texts)
        if not chunks:
            return ""

        selected_chunks = self._select_representative_chunks(chunks)
        excerpt_budget = self._calculate_excerpt_budget(len(selected_chunks))
        context = self._render_chunked_context(
            selected_chunks, original_length, len(chunks), excerpt_budget
        )

        while (
            len(context) > self.MAX_TEXT_LENGTH_FOR_LLM
            and excerpt_budget > self.MIN_LONG_TEXT_EXCERPT
        ):
            excerpt_budget = max(self.MIN_LONG_TEXT_EXCERPT, int(excerpt_budget * 0.85))
            context = self._render_chunked_context(
                selected_chunks, original_length, len(chunks), excerpt_budget
            )

        if len(context) > self.MAX_TEXT_LENGTH_FOR_LLM:
            marker = "\n\n...(chunked context compressed to the ontology analysis limit)..."
            context = context[: self.MAX_TEXT_LENGTH_FOR_LLM - len(marker)] + marker

        return context

    def _collect_document_chunks(self, document_texts: List[str]) -> List[Dict[str, Any]]:
        """Chunk every document, keeping document and chunk numbers"""
        all_chunks: List[Dict[str, Any]] = []
        for doc_index, text in enumerate(document_texts, 1):
            doc_chunks = split_text_into_chunks(
                text,
                chunk_size=self.LONG_TEXT_CHUNK_SIZE,
                overlap=self.LONG_TEXT_CHUNK_OVERLAP,
            )
            for chunk_index, chunk in enumerate(doc_chunks, 1):
                all_chunks.append({
                    "document_index": doc_index,
                    "chunk_index": chunk_index,
                    "total_document_chunks": len(doc_chunks),
                    "text": chunk,
                })
        return all_chunks

    def _select_representative_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evenly spaced sample covering beginning, middle and end"""
        if len(chunks) <= self.MAX_LONG_TEXT_CHUNKS:
            return chunks
        if self.MAX_LONG_TEXT_CHUNKS <= 1:
            return [chunks[0]]
        last_index = len(chunks) - 1
        selected = {
            round(i * last_index / (self.MAX_LONG_TEXT_CHUNKS - 1))
            for i in range(self.MAX_LONG_TEXT_CHUNKS)
        }
        return [chunks[i] for i in sorted(selected)]

    def _calculate_excerpt_budget(self, selected_count: int) -> int:
        """Character budget per selected chunk"""
        header_budget = 600
        chunk_header_budget = 120 * selected_count
        available = max(
            self.MIN_LONG_TEXT_EXCERPT * selected_count,
            self.MAX_TEXT_LENGTH_FOR_LLM - header_budget - chunk_header_budget,
        )
        return max(self.MIN_LONG_TEXT_EXCERPT, available // max(selected_count, 1))

    def _render_chunked_context(
        self,
        selected_chunks: List[Dict[str, Any]],
        original_length: int,
        total_chunks: int,
        excerpt_limit: int,
    ) -> str:
        """Render the sampled long-text context"""
        lines = [
            (
                f"[Long text, automatically chunked] The original text has {original_length} characters "
                f"and was split into {total_chunks} chunks for full coverage."
            ),
            (
                f"Below are excerpts of {len(selected_chunks)} representative chunks covering the beginning, "
                "middle and end; design the ontology from these cross-document clues, not from the first part only."
            ),
        ]
        for chunk in selected_chunks:
            lines.append(
                f"--- Document {chunk['document_index']} / "
                f"chunk {chunk['chunk_index']}/{chunk['total_document_chunks']} ---\n"
                + self._excerpt_text(chunk["text"], excerpt_limit)
            )
        return "\n\n".join(lines)

    @staticmethod
    def _excerpt_text(text: str, char_limit: int) -> str:
        """Keep head and tail of a long chunk"""
        text = text.strip()
        if len(text) <= char_limit:
            return text
        marker = "\n...(middle of this chunk omitted)...\n"
        if char_limit <= len(marker) + 20:
            return text[:char_limit]
        remaining = char_limit - len(marker)
        head_len = remaining // 2
        tail_len = remaining - head_len
        return f"{text[:head_len].rstrip()}{marker}{text[-tail_len:].lstrip()}"

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the LLM ontology output

        LLM output is untrusted: entity/edge entries may be strings, null or
        malformed objects, attributes may be strings, and source_targets may
        reference renamed or unknown types. Every entry is copied and
        normalized; unusable entries are dropped instead of crashing.
        """
        if not isinstance(result, dict):
            raise ValueError("Ontology result must be a JSON object")

        raw_entities = result.get("entity_types")
        raw_edges = result.get("edge_types")
        if not isinstance(raw_entities, list):
            raw_entities = []
        if not isinstance(raw_edges, list):
            raw_edges = []
        if not isinstance(result.get("analysis_summary"), str):
            result["analysis_summary"] = ""

        entity_name_map: Dict[str, str] = {}
        processed_entities: List[Dict[str, Any]] = []
        seen_entity_names = set()
        for raw_entity in raw_entities:
            if isinstance(raw_entity, str):
                entity = {"name": raw_entity}
            elif isinstance(raw_entity, dict):
                entity = dict(raw_entity)
            else:
                logger.warning("Ignoring non-object ontology entity entry")
                continue

            original_name = entity.get("name")
            if not isinstance(original_name, str) or not original_name.strip():
                logger.warning("Ignoring ontology entity without a usable name")
                continue
            original_name = original_name.strip()
            normalized_name = _to_pascal_case(original_name)
            if normalized_name == "Unknown":
                continue
            entity_name_map[original_name] = normalized_name
            entity_name_map[original_name.lower()] = normalized_name
            if normalized_name in seen_entity_names:
                logger.warning(f"Duplicate entity type '{normalized_name}' removed during validation")
                continue
            if normalized_name != original_name:
                logger.warning(
                    f"Entity type name '{original_name}' auto-converted to '{normalized_name}'"
                )
            entity["name"] = normalized_name
            entity["attributes"] = normalize_ontology_attributes(entity.get("attributes", []))
            if not isinstance(entity.get("examples"), list):
                entity["examples"] = []
            description = entity.get("description")
            if not isinstance(description, str) or not description:
                description = f"A {normalized_name} entity."
            entity["description"] = description[:97] + "..." if len(description) > 100 else description

            seen_entity_names.add(normalized_name)
            processed_entities.append(entity)

        result["entity_types"] = processed_entities

        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"},
            ],
            "examples": ["ordinary citizen", "anonymous netizen"],
        }
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"},
            ],
            "examples": ["small business", "community group"],
        }

        entity_names = {e["name"] for e in result["entity_types"]}
        fallbacks_to_add = []
        if "Person" not in entity_names:
            fallbacks_to_add.append(person_fallback)
        if "Organization" not in entity_names:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            # Keep the fallbacks within the limit by dropping trailing specific types
            overflow = len(result["entity_types"]) + len(fallbacks_to_add) - MAX_ONTOLOGY_TYPES
            if overflow > 0:
                result["entity_types"] = result["entity_types"][:-overflow]
            result["entity_types"].extend(fallbacks_to_add)

        result["entity_types"] = result["entity_types"][:MAX_ONTOLOGY_TYPES]

        valid_entity_names = {entity["name"] for entity in result["entity_types"]}
        for name in valid_entity_names:
            entity_name_map[name] = name
            entity_name_map[name.lower()] = name

        def resolve_entity_name(value: str) -> Optional[str]:
            stripped = value.strip()
            if stripped == "Entity":
                return stripped
            mapped = entity_name_map.get(stripped) or entity_name_map.get(stripped.lower())
            if mapped in valid_entity_names:
                return mapped
            pascal_name = _to_pascal_case(stripped)
            return pascal_name if pascal_name in valid_entity_names else None

        processed_edges: List[Dict[str, Any]] = []
        seen_edge_names = set()
        for raw_edge in raw_edges:
            if isinstance(raw_edge, str):
                logger.warning(f"Ignoring ontology edge without source_targets: {raw_edge}")
                continue
            if not isinstance(raw_edge, dict):
                logger.warning("Ignoring non-object ontology edge entry")
                continue
            edge = dict(raw_edge)

            original_name = edge.get("name")
            if not isinstance(original_name, str) or not original_name.strip():
                logger.warning("Ignoring ontology edge without a usable name")
                continue
            normalized_name = _to_upper_snake_case(original_name)
            if normalized_name == "UNKNOWN":
                continue
            if normalized_name in seen_edge_names:
                logger.warning(f"Duplicate edge type '{normalized_name}' removed during validation")
                continue
            if normalized_name != original_name:
                logger.warning(
                    f"Edge type name '{original_name}' auto-converted to '{normalized_name}'"
                )
            edge["name"] = normalized_name

            # Resolve endpoints first, then dedupe and cap, so invalid pairs
            # never consume the limit
            resolved_targets = []
            for pair in normalize_ontology_source_targets(edge.get("source_targets", []), limit=None):
                source = resolve_entity_name(pair["source"])
                target = resolve_entity_name(pair["target"])
                if source and target:
                    resolved_targets.append({"source": source, "target": target})
            edge["source_targets"] = normalize_ontology_source_targets(resolved_targets)
            edge["attributes"] = normalize_ontology_attributes(edge.get("attributes", []))
            description = edge.get("description")
            if not isinstance(description, str) or not description:
                description = f"A {normalized_name} relationship."
            edge["description"] = description[:97] + "..." if len(description) > 100 else description

            seen_edge_names.add(normalized_name)
            processed_edges.append(edge)
            if len(processed_edges) == MAX_ONTOLOGY_TYPES:
                break

        result["edge_types"] = processed_edges
        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convert ontology definition to Python code (similar to ontology.py)

        Args:
            ontology: Ontology definition

        Returns:
            Python code string
        """
        code_lines = [
            '"""',
            "Custom entity type definitions",
            "Auto-generated by MiroFish for social opinion simulation",
            '"""',
            "",
            "from pydantic import BaseModel, Field",
            "",
            "",
            "# Local base classes (replaces former zep_cloud ontology imports)",
            "class EntityText(BaseModel):",
            "    \"\"\"Text attribute for entity fields\"\"\"",
            "    description: str = \"\"",
            "    default: str | None = None",
            "",
            "class EntityModel(BaseModel):",
            "    \"\"\"Base class for entity types\"\"\"",
            "    pass",
            "",
            "class EdgeModel(BaseModel):",
            "    \"\"\"Base class for edge types\"\"\"",
            "    pass",
            "",
            "",
            "",
            "# ============== Entity Type Definitions ==============",
            "",
        ]

        # Generate entity types
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f"class {name}(EntityModel):")
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f"    {attr_name}: EntityText = Field(")
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f"        default=None")
                    code_lines.append(f"    )")
            else:
                code_lines.append("    pass")

            code_lines.append("")
            code_lines.append("")

        code_lines.append("# ============== Relation Type Definitions ==============")
        code_lines.append("")

        # Generate relation types
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = "".join(word.capitalize() for word in name.split("_"))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f"class {class_name}(EdgeModel):")
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f"    {attr_name}: EntityText = Field(")
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f"        default=None")
                    code_lines.append(f"    )")
            else:
                code_lines.append("    pass")

            code_lines.append("")
            code_lines.append("")

        # Generate type dictionary
        code_lines.append("# ============== Type Config ==============")
        code_lines.append("")
        code_lines.append("ENTITY_TYPES = {")
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append("}")
        code_lines.append("")
        code_lines.append("EDGE_TYPES = {")
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = "".join(word.capitalize() for word in name.split("_"))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append("}")
        code_lines.append("")

        # Generate edge source_targets mapping
        code_lines.append("EDGE_SOURCE_TARGETS = {")
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ", ".join(
                    [
                        f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                        for st in source_targets
                    ]
                )
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append("}")

        return "\n".join(code_lines)
