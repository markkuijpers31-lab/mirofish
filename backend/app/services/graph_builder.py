"""
Graph build service — uses local sqlite graph store + LLM-based extraction.
Replaces the former Zep Cloud integration (now uses local SQLite).
"""

import os
import uuid
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor
from .local_graph_store import LocalGraphStore, LocalNode, LocalEdge
from .local_graph_extractor import LocalGraphExtractor
from ..utils.locale import t, get_locale, set_locale
from ..utils.logger import get_logger

logger = get_logger("mirofish.build")


@dataclass
class GraphInfo:
    """Graph info"""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph build service
    Uses local sqlite graph store + LLM-based entity/relation extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.task_manager = TaskManager()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """
        Async graph build

        Args:
            text: input text
            ontology: ontology definition (from ontology generation output)
            graph_name: graph name
            chunk_size: text chunk size
            chunk_overlap: chunk overlap size
            batch_size: chunks per extraction batch

        Returns:
            task_id
        """
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        current_locale = get_locale()

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(
                task_id,
                text,
                ontology,
                graph_name,
                chunk_size,
                chunk_overlap,
                batch_size,
                current_locale,
            ),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = "en",
    ):
        """Graph build worker thread"""
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t("progress.startBuildingGraph"),
            )

            # 1. Create graph
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t("progress.graphCreated", graphId=graph_id),
            )

            # 2. Set ontology (stored in graph meta)
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id, progress=15, message=t("progress.ontologySet")
            )

            # 3. Text chunking
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t("progress.textSplit", count=total_chunks),
            )

            # 4. LLM extraction + store
            extractor = LocalGraphExtractor(graph_id, ontology)
            total_extracted = extractor.extract_and_store(
                chunks,
                lambda done, total: self.task_manager.update_task(
                    task_id,
                    progress=20 + int((done / total) * 60) if total > 0 else 80,
                    message=t("progress.extractingEntities", done=done, total=total),
                ),
            )

            # 5. No polling needed — local store is synchronous
            self.task_manager.update_task(
                task_id,
                progress=85,
                message=t(
                    "progress.processingComplete",
                    completed=total_chunks,
                    total=total_chunks,
                ),
            )

            # 6. Get graph info
            self.task_manager.update_task(
                task_id, progress=90, message=t("progress.fetchingGraphInfo")
            )

            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                    "items_extracted": total_extracted,
                },
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """Create a local graph (returns graph_id)"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        store = LocalGraphStore(graph_id)
        store.set_meta(
            name, "MiroFish Social Simulation Graph", {}, datetime.now().isoformat()
        )
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Store the ontology in the local graph metadata (as JSON)"""
        store = LocalGraphStore(graph_id)
        meta = store.get_ontology()
        meta.update(ontology or {})
        store.set_ontology(meta)

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """Get graph info"""
        store = LocalGraphStore(graph_id)
        stats = store.get_statistics()
        return GraphInfo(
            graph_id=graph_id,
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
            entity_types=stats["entity_types"],
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Get full graph data (nodes + edges with details)

        Args:
            graph_id: graph ID

        Returns:
            dict with nodes and edges, including temporal info and attributes
        """
        store = LocalGraphStore(graph_id)
        nodes = store.get_all_nodes()
        edges = store.get_all_edges()

        node_map = {n.uuid: n.name for n in nodes}

        nodes_data = []
        for node in nodes:
            nodes_data.append(
                {
                    "uuid": node.uuid,
                    "name": node.name,
                    "labels": node.labels,
                    "summary": node.summary,
                    "attributes": node.attributes,
                    "created_at": node.created_at,
                }
            )

        edges_data = []
        for edge in edges:
            edges_data.append(
                {
                    "uuid": edge.uuid,
                    "name": edge.name,
                    "fact": edge.fact,
                    "fact_type": edge.fact_type,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "source_node_name": node_map.get(edge.source_node_uuid, ""),
                    "target_node_name": node_map.get(edge.target_node_uuid, ""),
                    "attributes": edge.attributes,
                    "created_at": edge.created_at,
                    "valid_at": edge.valid_at,
                    "invalid_at": edge.invalid_at,
                    "expired_at": edge.expired_at,
                    "episodes": edge.episodes or [],
                }
            )

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """Delete graph"""
        from .local_graph_store import delete_graph as _delete

        _delete(graph_id)
