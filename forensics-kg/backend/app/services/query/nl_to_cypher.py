from typing import Dict, Any, Optional, List
from loguru import logger
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.schemas import (
    CYPHER_GENERATION_SCHEMA,
    ANSWER_GENERATION_SCHEMA,
)
from app.services.ontology.schema import ForensicsOntologySchema


class NLToCypherEngine:
    """Converts natural language questions to Cypher queries and interprets results."""

    def __init__(self, openai_client: OpenAIClient, ontology: ForensicsOntologySchema):
        self._openai = openai_client
        self._ontology = ontology

    async def generate_cypher(
        self,
        question: str,
        context_case_id: Optional[str] = None,
        graph_context: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        schema_desc = self._ontology.get_schema_description()
        system_prompt = (
            "You are a Neo4j Cypher query expert for a forensics knowledge graph.\n\n"
            f"The graph has the following schema:\n{schema_desc}\n\n"
        )
        if graph_context:
            system_prompt += f"Relevant graph context:\n{graph_context}\n\n"
        system_prompt += (
            "Generate a valid Cypher query to answer the user's question.\n"
            "Rules:\n"
            "- Use MATCH, OPTIONAL MATCH, WHERE, RETURN, ORDER BY, LIMIT\n"
            "- Always include LIMIT (max 100) to prevent unbounded results\n"
            "- Use parameterized queries with $param syntax where appropriate\n"
            "- Return meaningful properties, not just nodes\n"
            "- For text search, use CONTAINS (case sensitive) or toLower() for case insensitive\n"
            "- IMPORTANT: When using aggregation functions (collect, count, sum, avg) in RETURN, "
            "ORDER BY must reference only aliases defined in the RETURN clause, not raw variables. "
            "For example, use 'RETURN c.date AS date, collect(x) AS items ORDER BY date' "
            "instead of 'ORDER BY c.date'"
        )
        user_prompt = question
        if context_case_id:
            user_prompt += f"\n\nContext: Focus on case with case_id = '{context_case_id}'"

        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=CYPHER_GENERATION_SCHEMA,
            temperature=0.0,
            model_override=model_override,
        )

    async def interpret_results(
        self,
        question: str,
        cypher_query: str,
        raw_results: List[Dict],
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a forensic analysis expert. Interpret knowledge graph query "
            "results and provide clear, evidence-based answers. Cite specific "
            "entities and relationships from the results."
        )
        # Truncate results to avoid token limits
        results_str = str(raw_results[:30])
        if len(results_str) > 4000:
            results_str = results_str[:4000] + "... (truncated)"

        user_prompt = (
            f"Question: {question}\n\n"
            f"Cypher Query: {cypher_query}\n\n"
            f"Results ({len(raw_results)} records):\n{results_str}\n\n"
            f"Provide a comprehensive answer. If no results, explain what "
            f"data might be missing."
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=ANSWER_GENERATION_SCHEMA,
            model_override=model_override,
        )
