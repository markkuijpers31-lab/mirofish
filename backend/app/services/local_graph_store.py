"""
Local graph store — sqlite-backed knowledge graph (replaces Zep Cloud).
Stores nodes, edges, episodes, and embeddings per graph.
"""

import os
import json
import re
import sqlite3
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger("mirofish.graph_store")

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "../../uploads/graphs")


def _ensure_graph_dir(graph_id: str) -> str:
    d = os.path.join(GRAPH_DIR, graph_id)
    os.makedirs(d, exist_ok=True)
    return d


def _db_path(graph_id: str) -> str:
    return os.path.join(_ensure_graph_dir(graph_id), "graph.db")


def _connect(graph_id: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(graph_id))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_meta (
    graph_id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    ontology TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS nodes (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT,
    name TEXT,
    labels TEXT,
    summary TEXT,
    attributes TEXT,
    embedding TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT,
    name TEXT,
    fact TEXT,
    fact_type TEXT,
    source_node_uuid TEXT,
    target_node_uuid TEXT,
    attributes TEXT,
    embedding TEXT,
    created_at TEXT,
    valid_at TEXT,
    invalid_at TEXT,
    expired_at TEXT,
    episodes TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT,
    data TEXT,
    type TEXT,
    processed INTEGER DEFAULT 1,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes(graph_id);
CREATE INDEX IF NOT EXISTS idx_edges_graph ON edges(graph_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_uuid);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_uuid);
"""


@dataclass
class LocalNode:
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None
    embedding: Optional[List[float]] = None

    @property
    def uuid_(self) -> str:
        return self.uuid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


@dataclass
class LocalEdge:
    uuid: str
    name: str
    fact: str
    fact_type: str
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any]
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

    @property
    def uuid_(self) -> str:
        return self.uuid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "fact_type": self.fact_type,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
            "episodes": self.episodes or [],
        }


class LocalGraphStore:
    """Sqlite-backed graph store. One DB file per graph."""

    def __init__(self, graph_id: str):
        self.graph_id = graph_id
        self._init_schema()

    def _init_schema(self):
        with _connect(self.graph_id) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def set_meta(
        self, name: str, description: str, ontology: Dict[str, Any], created_at: str
    ):
        with _connect(self.graph_id) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_meta (graph_id, name, description, ontology, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    self.graph_id,
                    name,
                    description,
                    json.dumps(ontology, ensure_ascii=False),
                    created_at,
                ),
            )
            conn.commit()

    def get_ontology(self) -> Dict[str, Any]:
        with _connect(self.graph_id) as conn:
            row = conn.execute(
                "SELECT ontology FROM graph_meta WHERE graph_id=?", (self.graph_id,)
            ).fetchone()
            if row:
                return json.loads(row["ontology"])
            return {}

    def add_episode(self, data: str, episode_type: str = "text") -> str:
        ep_uuid = str(uuid.uuid4())
        from datetime import datetime

        with _connect(self.graph_id) as conn:
            conn.execute(
                "INSERT INTO episodes (uuid, graph_id, data, type, processed, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                (
                    ep_uuid,
                    self.graph_id,
                    data,
                    episode_type,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        return ep_uuid

    def add_node(
        self,
        name: str,
        labels: List[str],
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        node_uuid = str(uuid.uuid4())
        from datetime import datetime

        with _connect(self.graph_id) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (uuid, graph_id, name, labels, summary, attributes, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node_uuid,
                    self.graph_id,
                    name,
                    json.dumps(labels, ensure_ascii=False),
                    summary,
                    json.dumps(attributes or {}, ensure_ascii=False),
                    json.dumps(embedding) if embedding else None,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        return node_uuid

    def add_edge(
        self,
        name: str,
        fact: str,
        source_node_uuid: str,
        target_node_uuid: str,
        fact_type: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        valid_at: Optional[str] = None,
        invalid_at: Optional[str] = None,
        expired_at: Optional[str] = None,
    ) -> str:
        edge_uuid = str(uuid.uuid4())
        from datetime import datetime

        with _connect(self.graph_id) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO edges
                (uuid, graph_id, name, fact, fact_type, source_node_uuid, target_node_uuid,
                 attributes, embedding, created_at, valid_at, invalid_at, expired_at, episodes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    edge_uuid,
                    self.graph_id,
                    name,
                    fact,
                    fact_type or name,
                    source_node_uuid,
                    target_node_uuid,
                    json.dumps(attributes or {}, ensure_ascii=False),
                    json.dumps(embedding) if embedding else None,
                    datetime.now().isoformat(),
                    valid_at,
                    invalid_at,
                    expired_at,
                    "[]",
                ),
            )
            conn.commit()
        return edge_uuid

    def find_node_by_name(
        self, name: str, labels: Optional[List[str]] = None
    ) -> Optional[LocalNode]:
        with _connect(self.graph_id) as conn:
            if labels:
                rows = conn.execute(
                    "SELECT * FROM nodes WHERE graph_id=? AND name=?",
                    (self.graph_id, name),
                ).fetchall()
                for row in rows:
                    node_labels = json.loads(row["labels"])
                    if any(l in node_labels for l in labels):
                        return self._row_to_node(row)
                if rows:
                    return self._row_to_node(rows[0])
            else:
                row = conn.execute(
                    "SELECT * FROM nodes WHERE graph_id=? AND name=? LIMIT 1",
                    (self.graph_id, name),
                ).fetchone()
                if row:
                    return self._row_to_node(row)
        return None

    def get_node(self, node_uuid: str) -> Optional[LocalNode]:
        with _connect(self.graph_id) as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE uuid=?", (node_uuid,)
            ).fetchone()
            if row:
                return self._row_to_node(row)
        return None

    def get_all_nodes(self) -> List[LocalNode]:
        with _connect(self.graph_id) as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE graph_id=?", (self.graph_id,)
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_all_edges(self) -> List[LocalEdge]:
        with _connect(self.graph_id) as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE graph_id=?", (self.graph_id,)
            ).fetchall()
            return [self._row_to_edge(r) for r in rows]

    def get_node_edges(self, node_uuid: str) -> List[LocalEdge]:
        with _connect(self.graph_id) as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE source_node_uuid=? OR target_node_uuid=?",
                (node_uuid, node_uuid),
            ).fetchall()
            return [self._row_to_edge(r) for r in rows]

    def search_nodes(self, query: str, limit: int = 20) -> List[LocalNode]:
        # ponytail: LIKE keyword search; O(n) scan per query, fine for graphs < 10k nodes;
        # upgrade to embedding cosine sim via embed() if recall matters
        pattern = f"%{query}%"
        with _connect(self.graph_id) as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE graph_id=? AND (name LIKE ? OR summary LIKE ?) LIMIT ?",
                (self.graph_id, pattern, pattern, limit),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def search_edges(self, query: str, limit: int = 20) -> List[LocalEdge]:
        """
        Keyword search over edge facts

        A natural-language query rarely occurs verbatim in a fact, so the query is
        split into terms (3+ chars, common stopwords dropped); edges matching any
        term are ranked by the number of matching terms.
        """
        terms = _query_terms(query)
        if not terms:
            pattern = f"%{query}%"
            with _connect(self.graph_id) as conn:
                rows = conn.execute(
                    "SELECT * FROM edges WHERE graph_id=? AND (fact LIKE ? OR name LIKE ?) LIMIT ?",
                    (self.graph_id, pattern, pattern, limit),
                ).fetchall()
                return [self._row_to_edge(r) for r in rows]

        clause = " OR ".join(["fact LIKE ? OR name LIKE ?"] * len(terms))
        params: List[Any] = [self.graph_id]
        for term in terms:
            params += [f"%{term}%", f"%{term}%"]
        with _connect(self.graph_id) as conn:
            rows = conn.execute(
                f"SELECT * FROM edges WHERE graph_id=? AND ({clause}) LIMIT 2000",
                params,
            ).fetchall()
        edges = [self._row_to_edge(r) for r in rows]

        def score(edge: LocalEdge) -> int:
            text = f"{edge.fact} {edge.name}".lower()
            return sum(1 for term in terms if term in text)

        edges.sort(key=score, reverse=True)
        return edges[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        with _connect(self.graph_id) as conn:
            node_count = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE graph_id=?", (self.graph_id,)
            ).fetchone()[0]
            edge_count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE graph_id=?", (self.graph_id,)
            ).fetchone()[0]
            entity_types = set()
            for row in conn.execute(
                "SELECT labels FROM nodes WHERE graph_id=?", (self.graph_id,)
            ).fetchall():
                labels = json.loads(row["labels"])
                for l in labels:
                    if l not in ("Entity", "Node"):
                        entity_types.add(l)
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_types": list(entity_types),
            }

    def delete(self):
        import shutil

        d = os.path.join(GRAPH_DIR, self.graph_id)
        if os.path.exists(d):
            shutil.rmtree(d)

    def _row_to_node(self, row: sqlite3.Row) -> LocalNode:
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except Exception:
                pass
        return LocalNode(
            uuid=row["uuid"],
            name=row["name"],
            labels=json.loads(row["labels"]) if row["labels"] else [],
            summary=row["summary"] or "",
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            created_at=row["created_at"],
            embedding=embedding,
        )

    def _row_to_edge(self, row: sqlite3.Row) -> LocalEdge:
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except Exception:
                pass
        episodes = []
        if row["episodes"]:
            try:
                episodes = json.loads(row["episodes"])
            except Exception:
                pass
        return LocalEdge(
            uuid=row["uuid"],
            name=row["name"] or "",
            fact=row["fact"] or "",
            fact_type=row["fact_type"] or "",
            source_node_uuid=row["source_node_uuid"],
            target_node_uuid=row["target_node_uuid"],
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            created_at=row["created_at"],
            valid_at=row["valid_at"],
            invalid_at=row["invalid_at"],
            expired_at=row["expired_at"],
            episodes=episodes,
            embedding=embedding,
        )


def delete_graph(graph_id: str):
    store = LocalGraphStore(graph_id)
    store.delete()


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "how", "why", "who", "are", "was",
    "were", "from", "into", "about", "does", "did", "have", "has", "will", "would", "their",
    "they", "them", "which", "when", "where", "there", "been", "between", "over", "under",
    "een", "het", "van", "voor", "met", "die", "dat", "wat", "hoe", "zijn", "wordt", "niet",
}


def _query_terms(query: str) -> List[str]:
    """Search terms of a free-text query (lower-case, 3+ chars, no stopwords)"""
    terms = []
    for term in re.findall(r"\w+", (query or "").lower()):
        if len(term) >= 3 and term not in _STOPWORDS and term not in terms:
            terms.append(term)
    return terms[:12]
