"""RAG-style graph context retriever for query augmentation."""

import re
from typing import Optional
from loguru import logger
from app.services.graph.neo4j_client import Neo4jClient


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

    def __init__(self, neo4j_client: Neo4jClient):
        self._neo4j = neo4j_client

    async def retrieve_context(
        self, question: str, case_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve graph context relevant to the question.
        If case_id is provided, get that case's subgraph summary.
        Otherwise, do keyword-based node search.
        """
        try:
            if case_id:
                return await self._get_case_summary(case_id)
            else:
                keywords = self._extract_keywords(question)
                if not keywords:
                    return None
                return await self._search_by_keywords(keywords)
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")
            return None

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
