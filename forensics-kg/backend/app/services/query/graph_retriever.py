"""RAG-style graph context retriever for query augmentation."""

import re
from typing import Optional, TYPE_CHECKING
from loguru import logger
from app.services.graph.neo4j_client import Neo4jClient

if TYPE_CHECKING:
    from app.services.graph.embedding_service import EmbeddingService


# Common English stop words
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "what", "which", "who", "whom",
    "this", "that", "these", "those", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "show", "find", "get", "tell", "list", "give", "many",
})


class GraphRetriever:
    """Retrieves relevant graph context before Cypher generation."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        embedding_service: "Optional[EmbeddingService]" = None,
    ):
        self._neo4j = neo4j_client
        self._embeddings = embedding_service

    async def retrieve_context(
        self, question: str, case_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve graph context relevant to the question.
        If case_id is provided, get that case's subgraph summary.
        Otherwise, prefer semantic (embedding) search, falling back to keywords.
        """
        try:
            if case_id:
                return await self._get_case_summary(case_id)
            # Semantic retrieval first (much more accurate than substring matching).
            if self._embeddings is not None:
                semantic = await self._semantic_context(question)
                if semantic:
                    return semantic
            keywords = self._extract_keywords(question)
            if not keywords:
                return None
            return await self._search_by_keywords(keywords)
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")
            return None

    async def _semantic_context(self, question: str) -> Optional[str]:
        """Use embedding KNN to find the most relevant nodes for the question."""
        try:
            hits = await self._embeddings.semantic_search(question, k=12)
        except Exception as e:
            logger.info(f"Semantic retrieval unavailable, falling back to keywords: {e}")
            return None
        if not hits:
            return None
        lines = [
            "Most relevant nodes (semantic search). Use these EXACT property values "
            "when writing filters:"
        ]
        ids = []
        for h in hits:
            labels = ":".join(l for l in h.get("labels", []) if l != "Embedded") or "Node"
            props = h.get("properties", {})
            name = (
                props.get("name")
                or props.get("title")
                or props.get("case_id")
                or props.get("experiment_id")
                or props.get("description", "")
            )
            # Surface a few distinguishing property values so the LLM filters on
            # real stored values (e.g. 'impact_spatter', not 'impact spatter').
            detail_keys = ("pattern_type", "type", "mechanism", "crime_type", "status",
                           "weapon_type", "category", "verdict")
            details = [
                f"{k}={props[k]}"
                for k in detail_keys
                if props.get(k) not in (None, "")
            ]
            score = h.get("score")
            score_str = f" [sim {score:.2f}]" if isinstance(score, (int, float)) else ""
            detail_str = f" {{{', '.join(details[:4])}}}" if details else ""
            lines.append(f"  ({labels}) {str(name)[:100]}{detail_str}{score_str}")
            if h.get("id"):
                ids.append(h["id"])

        # Append the ACTUAL relationship patterns around the retrieved nodes so the
        # LLM traverses real edges (e.g. (InjuryPattern)-[:LED_TO_DEATH]->(CauseOfDeath))
        # instead of guessing relationship names.
        patterns = await self._relationship_patterns(ids)
        if patterns:
            lines.append("")
            lines.append(
                "Actual relationship patterns connecting these nodes "
                "(use these exact paths/types):"
            )
            lines.extend(f"  {p}" for p in patterns)
        return "\n".join(lines) if len(lines) > 1 else None

    async def _relationship_patterns(self, ids: list[str]) -> list[str]:
        """Fetch distinct schema-level relationship patterns around the given nodes."""
        if not ids:
            return []
        try:
            rows = await self._neo4j.execute_query(
                """
                MATCH (n)-[r]-(m)
                WHERE elementId(n) IN $ids
                WITH DISTINCT
                  CASE WHEN startNode(r) = n THEN labels(n)[0] ELSE labels(m)[0] END AS src,
                  type(r) AS rel,
                  CASE WHEN startNode(r) = n THEN labels(m)[0] ELSE labels(n)[0] END AS dst
                RETURN src, rel, dst LIMIT 40
                """,
                {"ids": ids},
            )
        except Exception as e:
            logger.info(f"Relationship-pattern lookup skipped: {e}")
            return []
        seen, patterns = set(), []
        for r in rows:
            src, rel, dst = r.get("src"), r.get("rel"), r.get("dst")
            if not (src and rel and dst):
                continue
            pat = f"(:{src})-[:{rel}]->(:{dst})"
            if pat not in seen:
                seen.add(pat)
                patterns.append(pat)
        return patterns

    async def _get_case_summary(self, case_id: str) -> Optional[str]:
        """Get a summary of a case and its connected entities."""
        query = """
        MATCH (c:Case {case_id: $case_id})
        OPTIONAL MATCH (c)-[r]-(connected)
        RETURN c, type(r) as rel_type, labels(connected) as labels,
               properties(connected) as props
        LIMIT 50
        """
        results = await self._neo4j.execute_query(query, {"case_id": case_id})
        if not results:
            return None

        lines = [f"Case: {case_id}"]
        for record in results:
            rel = record.get("rel_type", "?")
            labels = record.get("labels", [])
            props = record.get("props", {})
            name = props.get("name", props.get("case_id", "?"))
            label_str = ":".join(labels) if labels else "Unknown"
            lines.append(f"  -[{rel}]-> ({label_str}) {name}")

        return "\n".join(lines)

    async def _search_by_keywords(self, keywords: list[str]) -> Optional[str]:
        """Search nodes by keywords and return a context summary."""
        # Build a Cypher query that searches across node names
        conditions = " OR ".join(
            f"toLower(n.name) CONTAINS toLower('{kw}')" for kw in keywords[:5]
        )
        query = f"""
        MATCH (n)
        WHERE ({conditions}) AND n.name IS NOT NULL
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN labels(n) as labels, properties(n) as props,
               type(r) as rel_type, labels(m) as m_labels,
               m.name as m_name
        LIMIT 30
        """
        results = await self._neo4j.execute_query(query)
        if not results:
            return None

        lines = [f"Relevant nodes for keywords: {', '.join(keywords)}"]
        seen = set()
        for record in results:
            labels = record.get("labels", [])
            props = record.get("props", {})
            name = props.get("name", "?")
            key = f"{':'.join(labels)}:{name}"
            if key not in seen:
                seen.add(key)
                lines.append(f"  ({':'.join(labels)}) {name}")
                rel = record.get("rel_type")
                m_name = record.get("m_name")
                if rel and m_name:
                    m_labels = record.get("m_labels", [])
                    lines.append(
                        f"    -[{rel}]-> ({':'.join(m_labels)}) {m_name}"
                    )

        return "\n".join(lines) if len(lines) > 1 else None

    @staticmethod
    def _extract_keywords(question: str) -> list[str]:
        """Extract meaningful keywords from a question."""
        # Remove punctuation, lowercase, split
        words = re.findall(r"[a-zA-Z0-9\-]+", question.lower())
        keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        return keywords[:8]
