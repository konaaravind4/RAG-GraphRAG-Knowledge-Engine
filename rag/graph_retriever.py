"""
Neo4j graph retriever for multi-hop entity-relationship traversal.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain.schema import Document

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class GraphRetriever:
    """
    Retrieves related context from a Neo4j knowledge graph.
    Performs entity-centric multi-hop traversal.
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Neo4j connected at %s", uri)
        except Exception as e:
            logger.warning("Neo4j unavailable (%s). Graph retrieval disabled.", e)
            self.driver = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, max_hops: int = 2, limit: int = 10) -> list[Document]:
        """
        Retrieve related documents via entity graph traversal.
        Extracts entities from query, then does k-hop BFS.
        """
        if self.driver is None:
            return []

        entities = self._extract_keywords(query)
        if not entities:
            return []

        docs = []
        for entity in entities[:3]:  # cap at 3 entity seeds
            results = self._traverse(entity, max_hops=max_hops, limit=limit)
            docs.extend(results)

        # Deduplicate by page_content
        seen: set[str] = set()
        unique = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return unique[:limit]

    def ingest_document(self, doc: Document, entity_key: str = "title") -> None:
        """Ingest a document node and its entity relationships into Neo4j."""
        if self.driver is None:
            return
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.content = $content, d.source = $source
                """,
                id=doc.metadata.get(entity_key, doc.page_content[:50]),
                content=doc.page_content,
                source=doc.metadata.get("source", ""),
            )

    def add_relationship(self, from_id: str, to_id: str, rel_type: str = "RELATED_TO") -> None:
        """Add a directed relationship between two document nodes."""
        if self.driver is None:
            return
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a:Document {{id: $from_id}})
                MATCH (b:Document {{id: $to_id}})
                MERGE (a)-[:{rel_type}]->(b)
                """,
                from_id=from_id,
                to_id=to_id,
            )

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _traverse(self, entity: str, max_hops: int, limit: int) -> list[Document]:
        query = f"""
        MATCH (n:Document)
        WHERE toLower(n.content) CONTAINS toLower($entity)
           OR toLower(n.id) CONTAINS toLower($entity)
        WITH n
        CALL apoc.path.subgraphNodes(n, {{
            maxLevel: {max_hops},
            relationshipFilter: "RELATED_TO>"
        }}) YIELD node
        RETURN DISTINCT node.content AS content, node.id AS id, node.source AS source
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                records = session.run(query, entity=entity, limit=limit)
                docs = []
                for rec in records:
                    content = rec["content"] or ""
                    if content:
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": rec["source"] or "", "graph_id": rec["id"]},
                        ))
                return docs
        except Exception as e:
            logger.warning("Graph traversal error: %s", e)
            return []

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Simple keyword extraction — split by stop words."""
        stop = {"what", "is", "the", "a", "an", "of", "in", "for", "to", "how", "why", "when"}
        words = [w.strip("?.,!") for w in text.lower().split()]
        return [w for w in words if w not in stop and len(w) > 3]
