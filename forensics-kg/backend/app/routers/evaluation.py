"""Evaluation endpoints: compare extraction against gold standard."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import get_ontology_schema
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.evaluator import ExtractionEvaluator
from app.services.evaluation.metrics import EvaluationResult, AggregateEvaluationResult

router = APIRouter()

# Cache last evaluation results
_last_results: Optional[AggregateEvaluationResult] = None

_TEXT_SOURCE_TYPES = {"fir", "court_judgment", "postmortem", "lab_report", "witness_deposition", "soco_report"}


def _validate_source_type(source_type: str) -> None:
    if source_type not in _TEXT_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation is only supported for text-based source types: "
                   f"{', '.join(sorted(_TEXT_SOURCE_TYPES))}. "
                   f"'{source_type}' is an image-only type.",
        )


def _build_evaluator(schema: ForensicsOntologySchema) -> ExtractionEvaluator:
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    text_extractor = TextExtractor(openai_client, schema)
    return ExtractionEvaluator(settings.data_dir, text_extractor)


@router.post("/run-single/{doc_id}", response_model=EvaluationResult)
async def evaluate_single(
    doc_id: str,
    source_type: str = "fir",
    model: Optional[str] = None,
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Evaluate extraction on a single document against its gold standard."""
    _validate_source_type(source_type)
    evaluator = _build_evaluator(schema)
    try:
        return await evaluator.evaluate_single(doc_id, source_type=source_type, model_override=model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/run-all", response_model=AggregateEvaluationResult)
async def evaluate_all(
    source_type: str = "fir",
    model: Optional[str] = None,
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Evaluate extraction on all gold standard documents of a given source type."""
    _validate_source_type(source_type)
    global _last_results
    evaluator = _build_evaluator(schema)
    result = await evaluator.evaluate_all(source_type=source_type, model_override=model)
    _last_results = result
    return result


@router.get("/results")
async def get_cached_results():
    """Return the last evaluation results (if any)."""
    if _last_results is None:
        return {"message": "No evaluation results yet. Run /run-all first."}
    return _last_results
