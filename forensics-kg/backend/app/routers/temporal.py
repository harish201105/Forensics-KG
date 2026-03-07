"""Temporal reasoning endpoints: timeline reconstruction and inconsistency detection."""

from fastapi import APIRouter, Depends

from app.dependencies import get_neo4j_client
from app.services.graph.neo4j_client import Neo4jClient
from app.services.temporal.timeline import TemporalAnalyzer

router = APIRouter()


@router.get("/timeline/{case_id}")
async def get_timeline(
    case_id: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Get reconstructed timeline for a case."""
    analyzer = TemporalAnalyzer(client)
    return await analyzer.reconstruct_timeline(case_id)


@router.get("/inconsistencies/{case_id}")
async def get_inconsistencies(
    case_id: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Detect temporal inconsistencies in a case timeline."""
    analyzer = TemporalAnalyzer(client)
    return await analyzer.detect_inconsistencies(case_id)
