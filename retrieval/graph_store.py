"""
retrieval/graph_store.py — Neo4j graph retriever for multi-hop reasoning.

Optional dependency — system works fully without Neo4j.
Provides entity-based traversal and Cypher query generation.
"""

from __future__ import annotations

from typing import Optional

from retrieval.vector_store import ChunkMetadata, RetrievedChunk
from observability.logger import get_logger

logger = get_logger(__name__)


class GraphStore:
    """
    Neo4j-backed graph retriever for relational/multi-hop queries.

    Gracefully degrades to no-op if Neo4j is not available or configured.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        username: str = "neo4j",
        password: str = "password",
    ):
        self._url = url
        self._driver = None
        self._available = False

        if url:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(url, auth=(username, password))
                # Verify connectivity
                self._driver.verify_connectivity()
                self._available = True
                logger.info("Neo4j connected", url=url)
            except Exception as exc:
                logger.warning("Neo4j unavailable, using vector-only mode", error=str(exc))
                self._driver = None

    @property
    def is_available(self) -> bool:
        return self._available

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """
        Search the graph for entities and relationships matching the query.

        Uses a simple keyword-based Cypher query. In production, this could
        be enhanced with entity extraction + LLM-generated Cypher.
        """
        if not self._available or not self._driver:
            return []

        try:
            # Extract key terms for graph search
            terms = [t.strip() for t in query.split() if len(t.strip()) > 3]
            if not terms:
                return []

            # Build a flexible Cypher query
            where_clauses = " OR ".join(
                f"toLower(n.text) CONTAINS toLower('{term}')" for term in terms[:5]
            )

            cypher = f"""
                MATCH (n)-[r]->(m)
                WHERE {where_clauses}
                RETURN
                    n.text AS source_text,
                    type(r) AS relationship,
                    m.text AS target_text,
                    n.source AS source
                LIMIT $k
            """

            with self._driver.session() as session:
                result = session.run(cypher, k=k)
                chunks = []
                for record in result:
                    text_parts = []
                    if record.get("source_text"):
                        text_parts.append(record["source_text"])
                    if record.get("relationship") and record.get("target_text"):
                        text_parts.append(
                            f"[{record['relationship']}] → {record['target_text']}"
                        )

                    if text_parts:
                        chunks.append(RetrievedChunk(
                            text=" ".join(text_parts),
                            score=0.5,  # Graph results get a fixed base score
                            source=record.get("source", "graph"),
                            retrieval_method="graph",
                            metadata=ChunkMetadata(source="neo4j"),
                        ))

                logger.info("Graph search completed", results=len(chunks))
                return chunks

        except Exception as exc:
            logger.warning("Graph search failed", error=str(exc))
            return []

    def add_entities(
        self,
        entities: list[dict],
        relationships: list[dict],
    ) -> int:
        """
        Add entities and relationships to the graph.

        Args:
            entities: [{"id": "...", "text": "...", "type": "...", "source": "..."}]
            relationships: [{"from": "id1", "to": "id2", "type": "RELATES_TO"}]

        Returns:
            Number of entities added.
        """
        if not self._available or not self._driver:
            return 0

        try:
            with self._driver.session() as session:
                # Create entities
                for entity in entities:
                    session.run(
                        """
                        MERGE (n:Entity {id: $id})
                        SET n.text = $text, n.type = $type, n.source = $source
                        """,
                        **entity,
                    )

                # Create relationships
                for rel in relationships:
                    session.run(
                        f"""
                        MATCH (a:Entity {{id: $from_id}})
                        MATCH (b:Entity {{id: $to_id}})
                        MERGE (a)-[:{rel['type']}]->(b)
                        """,
                        from_id=rel["from"],
                        to_id=rel["to"],
                    )

            logger.info(
                "Added graph data",
                entities=len(entities),
                relationships=len(relationships),
            )
            return len(entities)

        except Exception as exc:
            logger.error("Failed to add graph data", error=str(exc))
            return 0

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")
