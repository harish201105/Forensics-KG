from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
from app.services.graph.embedding_service import EmbeddingService
from app.services.reasoning.engine import ForensicReasoningEngine
from app.models.responses import AnalysisResponse

router = APIRouter()


@router.post("/hypothesis/{case_id}", response_model=AnalysisResponse)
async def generate_hypothesis(
    case_id: str,
    model: str | None = None,
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    # Verify case exists
    check = await client.execute_query(
        "MATCH (c:Case {case_id: $cid}) RETURN c LIMIT 1", {"cid": case_id}
    )
    if not check:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    settings = get_settings()
    openai_client = OpenAIClient(settings)
    embedding_service = EmbeddingService(client, openai_client, settings.embedding_dimensions)
    engine = ForensicReasoningEngine(openai_client, client, embedding_service)
    result = await engine.generate_hypothesis(case_id, model_override=model)
    return AnalysisResponse(**result)


@router.get("/statistics/{entity_id}")
async def get_experiment_statistics(
    entity_id: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    # Try as Experiment first, then as Case
    stains: list = []
    for query in [
        """
        MATCH (e:Experiment {experiment_id: $eid})
        OPTIONAL MATCH (e)<-[:CAUSED_BY]-(p:BloodstainPattern)-[:CONTAINS_STAIN]->(s:Stain)
        RETURN collect(s {.*, embedding: NULL, embed_text: NULL}) as stains
        """,
        """
        MATCH (c:Case {case_id: $eid})
        OPTIONAL MATCH (c)-[:HAS_PATTERN]->(p:BloodstainPattern)-[:CONTAINS_STAIN]->(s:Stain)
        RETURN collect(s {.*, embedding: NULL, embed_text: NULL}) as stains
        """,
    ]:
        results = await client.execute_query(query, {"eid": entity_id})
        if results and results[0].get("stains"):
            stains = results[0]["stains"]
            break

    if not stains:
        return {"error": f"No stain data found for '{entity_id}'"}

    engine = ForensicReasoningEngine(None, client)  # type: ignore[arg-type]
    return engine._perform_statistical_analysis(stains)
