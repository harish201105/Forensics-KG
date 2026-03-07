from fastapi import APIRouter, Depends

from app.config import get_settings
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
from app.services.query.nl_to_cypher import NLToCypherEngine
from app.services.query.query_service import QueryService
from app.models.requests import NaturalLanguageQueryRequest, CypherQueryRequest
from app.models.responses import QueryResponse

router = APIRouter()


def _build_query_service(
    client: Neo4jClient, schema: ForensicsOntologySchema
) -> QueryService:
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    nl_engine = NLToCypherEngine(openai_client, schema)
    graph_ops = GraphOperations(client, schema)
    return QueryService(nl_engine, client, graph_ops)


@router.post("/natural", response_model=QueryResponse)
async def natural_language_query(
    request: NaturalLanguageQueryRequest,
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    service = _build_query_service(client, schema)
    result = await service.natural_language_query(
        question=request.question,
        case_id=request.context_case_id,
        max_results=request.max_results,
        model_override=request.model,
    )
    return QueryResponse(**result)


@router.post("/cypher")
async def cypher_query(
    request: CypherQueryRequest,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    service = QueryService(None, client, None)
    result = await service.cypher_query(request.query, request.parameters)
    return result
