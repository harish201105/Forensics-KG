import re
from typing import Dict, Any, Optional, List
from loguru import logger
from app.services.query.nl_to_cypher import NLToCypherEngine
from app.services.query.graph_retriever import GraphRetriever
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations

# Patterns that indicate destructive intent
_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\bDETACH\s+DELETE\b", re.IGNORECASE),
    re.compile(r"\bDELETE\b", re.IGNORECASE),
    re.compile(r"\bDROP\b", re.IGNORECASE),
    re.compile(r"\bCALL\s+\{?\s*(?:db|apoc)\.\w+\.drop", re.IGNORECASE),
]


def _is_destructive(query: str) -> bool:
    """Check if a Cypher query contains destructive operations."""
    # Strip string literals to avoid false positives
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", query)
    return any(p.search(stripped) for p in _DESTRUCTIVE_PATTERNS)


class QueryService:
    """Unified query service for NL and Cypher queries."""

    def __init__(
        self,
        nl_engine: Optional[NLToCypherEngine],
        neo4j_client: Neo4jClient,
        graph_ops: Optional[GraphOperations],
    ):
        self._nl = nl_engine
        self._neo4j = neo4j_client
        self._graph_ops = graph_ops
        self._retriever = GraphRetriever(neo4j_client) if neo4j_client else None

    async def natural_language_query(
        self,
        question: str,
        case_id: Optional[str] = None,
        max_results: int = 10,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """NL -> RAG Retrieval -> Cypher -> Execute -> Interpret."""
        # Step 0: Retrieve graph context (RAG)
        graph_context = None
        if self._retriever:
            try:
                graph_context = await self._retriever.retrieve_context(question, case_id)
                if graph_context:
                    logger.info(f"RAG context retrieved ({len(graph_context)} chars)")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        # Step 1: Generate Cypher (with graph context)
        if not self._nl:
            raise RuntimeError("NLToCypherEngine not configured")
        cypher_result = await self._nl.generate_cypher(
            question, case_id,
            graph_context=graph_context,
            model_override=model_override,
        )
        cypher_query = cypher_result["cypher_query"]

        # Validate generated Cypher
        if not cypher_query or not cypher_query.strip():
            return {
                "answer": "Could not generate a valid query for this question.",
                "confidence": 0.0,
                "sources": [],
                "cypher_query": "",
                "reasoning": cypher_result.get("explanation", ""),
            }
        upper = cypher_query.upper()
        if "RETURN" not in upper:
            cypher_query += " RETURN *"
        if "LIMIT" not in upper:
            cypher_query += " LIMIT 50"

        logger.info(f"Generated Cypher: {cypher_query}")

        # Step 2: Execute
        try:
            raw_results = await self._neo4j.execute_query(cypher_query)
        except Exception as e:
            logger.error(f"Cypher execution failed: {e}")
            return {
                "answer": f"Query execution failed: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "cypher_query": cypher_query,
                "reasoning": cypher_result.get("explanation", ""),
            }

        # Step 3: Interpret (self._nl already checked above)
        interpretation = await self._nl.interpret_results(  # type: ignore[union-attr]
            question, cypher_query, raw_results, model_override=model_override
        )

        return {
            "answer": interpretation["answer"],
            "confidence": interpretation["confidence"],
            "sources": raw_results[:max_results],
            "cypher_query": cypher_query,
            "reasoning": interpretation.get("reasoning", ""),
            "graph_context": graph_context,
        }

    async def cypher_query(
        self, query: str, parameters: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Execute a raw Cypher query with safety checks."""
        if _is_destructive(query):
            raise ValueError(
                "Destructive queries (DELETE/DROP) are not allowed "
                "through this endpoint. Use the graph/clear endpoint instead."
            )

        results = await self._neo4j.execute_query(query, parameters or {})
        return {"results": results, "count": len(results)}
