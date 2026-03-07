from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
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
    engine = ForensicReasoningEngine(openai_client, client)
    result = await engine.generate_hypothesis(case_id, model_override=model)
    return AnalysisResponse(**result)


@router.get("/statistics/{experiment_id}")
async def get_experiment_statistics(
    experiment_id: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    query = """
    MATCH (e:Experiment {experiment_id: $exp_id})
    OPTIONAL MATCH (e)<-[:CAUSED_BY]-(p:BloodstainPattern)-[:CONTAINS_STAIN]->(s:Stain)
    RETURN e, collect(properties(s)) as stains, properties(p) as pattern
    """
    results = await client.execute_query(query, {"exp_id": experiment_id})
    if not results:
        return {"error": "Experiment not found"}
    return {"experiment": results[0]}
