"""
Local graph extractor — LLM-based entity/relation extraction from text chunks.
Replaces Zep Cloud server-side NLP extraction pipeline (now local LLM-based).
"""

import json
import re
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.embeddings import embed
from ..utils.persona_ids import parse_identifier
from .local_graph_store import LocalGraphStore

logger = get_logger("mirofish.extractor")

EXTRACT_SYSTEM_PROMPT = """You are a knowledge-graph extraction engine. Given a text chunk and an ontology (entity types + relation types), extract entities and relations as valid JSON.

Output ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "EntityName", "type": "EntityType", "summary": "One-sentence description", "attributes": {}}
  ],
  "edges": [
    {"name": "relation_type", "fact": "Natural language fact statement", "source": "SourceEntityName", "target": "TargetEntityName", "attributes": {}}
  ]
}

Rules:
1. Only use entity types and relation types defined in the ontology
2. Entity names should be proper nouns or specific identifiers from the text
3. Each edge fact must be a complete sentence describing the relationship
4. If no entities/edges are found, return {"entities": [], "edges": []}
5. Do not include any text outside the JSON object"""


def _to_pascal_case(name: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    words = []
    for part in parts:
        words.extend(re.sub(r"([a-z])([A-Z])", r"\1_\2", part).split("_"))
    return "".join(word.capitalize() for word in words if word) or "Entity"


def _identifier_key(name: str) -> Optional[str]:
    """
    Identifier key of an entity name ("C01 Dutch family" -> "C1"), or None

    Only short letter prefixes (1-2 letters) count, so ordinary names such as
    "GPT4 fan" are never merged.
    """
    parsed = parse_identifier(name)
    if not parsed or len(parsed[0]) > 2:
        return None
    return f"{parsed[0].upper()}{int(parsed[1])}"


class LocalGraphExtractor:
    """Extracts entities and relations from text using LLM, stores in LocalGraphStore."""

    def __init__(self, graph_id: str, ontology: Dict[str, Any], llm: Any = None):
        self.store = LocalGraphStore(graph_id)
        self.ontology = ontology or {}
        self.llm = llm or LLMClient()
        self._entity_index: Dict[str, str] = {}  # "name:type" -> uuid
        self._identifier_index: Dict[str, str] = {}  # identifier key -> uuid
        self._type_map: Dict[str, str] = {}
        for entity_type in self.ontology.get("entity_types", []) or []:
            name = entity_type.get("name") if isinstance(entity_type, dict) else None
            if isinstance(name, str) and name.strip():
                self._type_map[name.strip().lower()] = name.strip()
                self._type_map[_to_pascal_case(name).lower()] = name.strip()
        for node in self.store.get_all_nodes():
            key = _identifier_key(node.name)
            if key and key not in self._identifier_index:
                self._identifier_index[key] = node.uuid
        self.stats = {
            "chunks_total": 0,
            "chunks_failed": 0,
            "entities_added": 0,
            "entities_merged": 0,
            "edges_added": 0,
            "edges_skipped": 0,
        }

    def extract_and_store(self, chunks: List[str], progress_callback=None) -> int:
        """Extract entities/edges from text chunks, store in graph. Returns total extracted count."""
        entity_types = sorted(set(self._type_map.values()))
        edge_types = [
            e["name"] for e in self.ontology.get("edge_types", []) or []
            if isinstance(e, dict) and e.get("name")
        ]
        ontology_summary = json.dumps(
            {"entity_types": entity_types, "edge_types": edge_types}, ensure_ascii=False
        )

        total = 0
        self.stats["chunks_total"] = len(chunks)
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))

            result = self._extract_from_chunk(chunk, ontology_summary)
            if not isinstance(result, dict):
                self.stats["chunks_failed"] += 1
                continue

            entities = result.get("entities") if isinstance(result.get("entities"), list) else []
            edges = result.get("edges") if isinstance(result.get("edges"), list) else []
            for entity in entities:
                if isinstance(entity, dict):
                    self._add_or_merge_entity(entity)
            for edge in edges:
                if isinstance(edge, dict):
                    self._add_edge(edge)

            total += len(entities) + len(edges)

        logger.info(f"Extraction finished for graph {self.store.graph_id}: {self.stats}")
        return total

    def _extract_from_chunk(
        self, chunk: str, ontology_summary: str
    ) -> Optional[Dict[str, Any]]:
        user_msg = f"Ontology types: {ontology_summary}\n\nText chunk:\n{chunk}"
        try:
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            return self.llm.chat_json(messages, temperature=0.2, max_tokens=8192)
        except Exception as e:
            logger.warning(f"Extraction failed for chunk: {str(e)[:100]}")
            return None

    def _normalize_type(self, raw_type: Any) -> str:
        """Map an extracted type onto the ontology (case/format-insensitive)"""
        if not isinstance(raw_type, str) or not raw_type.strip():
            return "Entity"
        raw_type = raw_type.strip()
        mapped = self._type_map.get(raw_type.lower()) or self._type_map.get(_to_pascal_case(raw_type).lower())
        if mapped:
            return mapped
        if raw_type.lower() == "entity":
            return "Entity"
        logger.debug(f"Entity type '{raw_type}' not in ontology; kept as {_to_pascal_case(raw_type)}")
        return _to_pascal_case(raw_type)

    def _add_or_merge_entity(self, entity: Dict[str, Any]):
        raw_name = entity.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if not name:
            return
        etype = self._normalize_type(entity.get("type"))
        summary = entity.get("summary") if isinstance(entity.get("summary"), str) else ""
        attributes = entity.get("attributes") if isinstance(entity.get("attributes"), dict) else {}

        existing = self.store.find_node_by_name(name, labels=[etype])
        if existing:
            self._entity_index[f"{name}:{etype}"] = existing.uuid
            return

        # "C01" and "C01 Dutch first-time family" are one persona: merge on the identifier
        key = _identifier_key(name)
        if key and key in self._identifier_index:
            node = self.store.get_node(self._identifier_index[key])
            if node:
                aliases = list(node.attributes.get("aliases", [])) if isinstance(node.attributes.get("aliases"), list) else []
                for alias in (node.name, name):
                    if alias not in aliases:
                        aliases.append(alias)
                new_attributes = {**attributes, **node.attributes, "aliases": aliases}
                longer_name = name if len(name) > len(node.name) else node.name
                new_summary = node.summary if len(node.summary or "") >= len(summary) else summary
                self.store.update_node(node.uuid, name=longer_name, summary=new_summary, attributes=new_attributes)
                self._entity_index[f"{name}:{etype}"] = node.uuid
                self.stats["entities_merged"] += 1
                return

        labels = ["Entity", etype] if etype != "Entity" else ["Entity"]
        emb = embed(f"{name} {summary}") if summary else embed(name)
        node_uuid = self.store.add_node(
            name=name,
            labels=labels,
            summary=summary,
            attributes=attributes,
            embedding=emb,
        )
        self._entity_index[f"{name}:{etype}"] = node_uuid
        if key:
            self._identifier_index[key] = node_uuid
        self.stats["entities_added"] += 1

    def _add_edge(self, edge: Dict[str, Any]):
        def _text(value: Any) -> str:
            return value.strip() if isinstance(value, str) else ""

        name = _text(edge.get("name"))
        source_name = _text(edge.get("source"))
        target_name = _text(edge.get("target"))
        fact = _text(edge.get("fact"))

        if not name or not source_name or not target_name or not fact:
            self.stats["edges_skipped"] += 1
            return

        source_uuid = self._find_entity_uuid(source_name)
        target_uuid = self._find_entity_uuid(target_name)

        if not source_uuid or not target_uuid:
            self.stats["edges_skipped"] += 1
            return

        emb = embed(fact)
        self.store.add_edge(
            name=name,
            fact=fact,
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            fact_type=name,
            attributes=edge.get("attributes") if isinstance(edge.get("attributes"), dict) else {},
            embedding=emb,
        )
        self.stats["edges_added"] += 1

    def _find_entity_uuid(self, name: str) -> Optional[str]:
        for key, uuid in self._entity_index.items():
            if key.startswith(f"{name}:"):
                return uuid
        node = self.store.find_node_by_name(name)
        if node:
            self._entity_index[
                f"{node.name}:{node.labels[-1] if len(node.labels) > 1 else 'Entity'}"
            ] = node.uuid
            return node.uuid
        key = _identifier_key(name)
        if key and key in self._identifier_index:
            return self._identifier_index[key]
        return None
