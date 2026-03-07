from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, field_validator
from typing import Optional

from app.dependencies import get_neo4j_client
from app.services.graph.neo4j_client import Neo4jClient
from app.services.collaboration.annotations import AnnotationService

router = APIRouter()

_VALID_CASE_STATUSES = {"open", "investigating", "closed", "cold"}


class AnnotationRequest(BaseModel):
    entity_type: str
    entity_id: str
    text: str
    author: str = "anonymous"


class CaseStatusRequest(BaseModel):
    status: str
    user: str = "anonymous"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in _VALID_CASE_STATUSES:
            raise ValueError(f"Invalid status '{v}'. Must be one of: {', '.join(sorted(_VALID_CASE_STATUSES))}")
        return v


@router.post("/annotate")
async def add_annotation(
    request: AnnotationRequest,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    service = AnnotationService(client)
    return await service.add_annotation(
        entity_type=request.entity_type,
        entity_id=request.entity_id,
        text=request.text,
        author=request.author,
    )


@router.get("/annotations/{entity_type}/{entity_id}")
async def get_annotations(
    entity_type: str,
    entity_id: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    service = AnnotationService(client)
    return await service.get_annotations(entity_type, entity_id)


@router.patch("/case/{case_id}/status")
async def update_case_status(
    case_id: str,
    request: CaseStatusRequest,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    service = AnnotationService(client)
    return await service.update_case_status(case_id, request.status, request.user)


@router.get("/activity-log")
async def get_activity_log(
    limit: int = Query(50, le=200),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    service = AnnotationService(client)
    return await service.get_activity_log(limit)
