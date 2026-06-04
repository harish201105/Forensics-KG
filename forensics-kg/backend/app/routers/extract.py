import time
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional

from app.config import get_settings
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.text_extractor import TextExtractor
from app.services.extraction.image_extractor import ImageExtractor
from app.services.extraction.pipeline import ExtractionPipeline
from app.services.graph.deduplication import EntityDeduplicator
from app.services.graph.embedding_service import EmbeddingService
from app.services.graph.entity_resolution import EntityResolutionService
from app.models.requests import TextExtractionRequest
from app.models.responses import ExtractionResponse

router = APIRouter()

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}
_MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
_SUPPORTED_FORENSIC_IMAGE_TYPES = {
    "bloodstain", "fingerprint", "wound",
    "ballistics", "document_forensics", "tool_marks",
}


def _build_pipeline(
    client: Neo4jClient, schema: ForensicsOntologySchema
) -> ExtractionPipeline:
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    text_extractor = TextExtractor(openai_client, schema)
    image_extractor = ImageExtractor(openai_client, settings)
    graph_ops = GraphOperations(client, schema)
    dedup = EntityDeduplicator(client, schema)
    embedding_service = EmbeddingService(
        client, openai_client, settings.embedding_dimensions
    )
    resolution_service = EntityResolutionService(client)
    return ExtractionPipeline(
        text_extractor, image_extractor, graph_ops, dedup,
        openai_client=openai_client,
        embedding_service=embedding_service,
        resolution_service=resolution_service,
    )


@router.post("/text", response_model=ExtractionResponse)
async def extract_from_text(
    request: TextExtractionRequest,
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    pipeline = _build_pipeline(client, schema)
    start = time.time()
    result = await pipeline.process_text(
        text=request.text,
        source_type=request.source_type,
        store=request.store_in_graph,
        model_override=request.model,
        # When a case_id is supplied, use it as the document id so entity ids are
        # stable/idempotent (enables faithful re-processing of a known document).
        doc_id=request.case_id,
    )
    elapsed_ms = (time.time() - start) * 1000
    return ExtractionResponse(
        entities=result["entities"],
        relationships=result["relationships"],
        metadata=result.get("metadata", {}),
        processing_time_ms=elapsed_ms,
    )


@router.post("/image", response_model=ExtractionResponse)
async def extract_from_image(
    file: UploadFile = File(...),
    experiment_id: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
    image_type: str = Form("bloodstain"),
    store_in_graph: bool = Form(True),
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    # Validate image_type
    if image_type not in _SUPPORTED_FORENSIC_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image_type}'. "
                   f"Supported: {', '.join(sorted(_SUPPORTED_FORENSIC_IMAGE_TYPES))}",
        )

    # Validate file type
    if file.content_type and file.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {', '.join(_ALLOWED_IMAGE_TYPES)}",
        )

    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > _MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). Max: 50 MB.",
        )

    # Validate non-empty
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pipeline = _build_pipeline(client, schema)
    metadata = {}
    if experiment_id:
        metadata["experiment_id"] = experiment_id
    if case_id:
        metadata["case_id"] = case_id

    start = time.time()
    result = await pipeline.process_image(
        image_bytes=image_bytes,
        metadata=metadata,
        store=store_in_graph,
        image_type=image_type,
    )
    elapsed_ms = (time.time() - start) * 1000
    return ExtractionResponse(
        entities=result.get("entities", []),
        relationships=result.get("relationships", []),
        metadata=result.get("metadata", {}),
        processing_time_ms=elapsed_ms,
    )


@router.post("/combined", response_model=ExtractionResponse)
async def extract_combined(
    file: UploadFile = File(...),
    text: str = Form(...),
    source_type: str = Form("fir"),
    image_type: str = Form("bloodstain"),
    experiment_id: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
    store_in_graph: bool = Form(True),
    model: Optional[str] = Form(None),
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Combined text + image extraction with multi-modal synthesis."""
    # Validate image_type
    if image_type not in _SUPPORTED_FORENSIC_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image_type}'. "
                   f"Supported: {', '.join(sorted(_SUPPORTED_FORENSIC_IMAGE_TYPES))}",
        )

    # Validate file type
    if file.content_type and file.content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {', '.join(_ALLOWED_IMAGE_TYPES)}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > _MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). Max: 50 MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content is required.")

    pipeline = _build_pipeline(client, schema)
    metadata = {}
    if experiment_id:
        metadata["experiment_id"] = experiment_id
    if case_id:
        metadata["case_id"] = case_id

    start = time.time()
    result = await pipeline.process_combined(
        text=text,
        image_bytes=image_bytes,
        source_type=source_type,
        image_type=image_type,
        store=store_in_graph,
        model_override=model,
        image_metadata=metadata or None,
    )
    elapsed_ms = (time.time() - start) * 1000
    return ExtractionResponse(
        entities=result.get("entities", []),
        relationships=result.get("relationships", []),
        metadata=result.get("metadata", {}),
        processing_time_ms=elapsed_ms,
    )
