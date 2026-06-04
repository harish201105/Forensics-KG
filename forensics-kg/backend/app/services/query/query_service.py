import re
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from loguru import logger
from app.services.query.nl_to_cypher import NLToCypherEngine
from app.services.query.graph_retriever import GraphRetriever
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations

if TYPE_CHECKING:
    from app.services.graph.embedding_service import EmbeddingService

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


_INTERNAL_PROPS = ("embedding", "embed_text")


def _strip_embeddings(rows: List[Dict]) -> List[Dict]:
    """Remove embedding vectors from query rows (one level deep) to keep
    sources and LLM prompts compact."""
    cleaned = []
    for row in rows:
        new_row = {}
        for key, val in row.items():
            if isinstance(val, dict):
                new_row[key] = {k: v for k, v in val.items() if k not in _INTERNAL_PROPS}
            elif key in _INTERNAL_PROPS:
                continue
            else:
                new_row[key] = val
        cleaned.append(new_row)
    return cleaned


class QueryService:
    """Unified query service for NL and Cypher queries."""

    def __init__(
        self,
        nl_engine: Optional[NLToCypherEngine],
        neo4j_client: Neo4jClient,
        graph_ops: Optional[GraphOperations],
        embedding_service: "Optional[EmbeddingService]" = None,
    ):
        self._nl = nl_engine
        self._neo4j = neo4j_client
        self._graph_ops = graph_ops
        self._retriever = (
            GraphRetriever(neo4j_client, embedding_service) if neo4j_client else None
        )

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
        cypher_query = self._finalize_cypher(cypher_query)
        logger.info(f"Generated Cypher: {cypher_query}")

        # Step 2: Execute with a self-correction loop. If Cypher errors, feed the
        # Neo4j error back to the LLM to repair it (up to `max_attempts` times).
        raw_results: Optional[List[Dict]] = None
        last_error = ""
        attempts: List[str] = [cypher_query]
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw_results = await self._neo4j.execute_query(cypher_query)
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Cypher attempt {attempt + 1}/{max_attempts} failed: {last_error}"
                )
                if attempt == max_attempts - 1:
                    break
                fix = await self._nl.fix_cypher(
                    question, cypher_query, last_error, model_override=model_override
                )
                fixed = (fix.get("cypher_query") or "").strip()
                if not fixed or fixed == cypher_query:
                    break
                cypher_query = self._finalize_cypher(fixed)
                attempts.append(cypher_query)
                logger.info(f"Retrying with corrected Cypher: {cypher_query}")

        if raw_results is None:
            return {
                "answer": (
                    "I couldn't construct a working query for that question after "
                    f"{len(attempts)} attempt(s). Last error: {last_error}"
                ),
                "confidence": 0.0,
                "sources": [],
                "cypher_query": cypher_query,
                "reasoning": cypher_result.get("explanation", ""),
                "attempts": len(attempts),
            }

        raw_results = _strip_embeddings(raw_results)

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
            "key_entities": interpretation.get("key_entities", []),
            "follow_up_questions": interpretation.get("follow_up_questions", []),
            "attempts": len(attempts),
        }

    @staticmethod
    def _finalize_cypher(cypher_query: str) -> str:
        """Ensure a generated Cypher query has RETURN and a bounded LIMIT."""
        upper = cypher_query.upper()
        if "RETURN" not in upper:
            cypher_query += " RETURN *"
        if "LIMIT" not in upper:
            cypher_query += " LIMIT 50"
        return cypher_query

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
