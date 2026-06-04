"""Evaluation endpoints: compare extraction against gold standard."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.dependencies import get_ontology_schema
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.text_extractor import TextExtractor
from app.services.extraction.image_extractor import ImageExtractor
from app.services.evaluation.evaluator import ExtractionEvaluator
from app.services.evaluation.image_evaluator import ImageEvaluator
from app.services.evaluation.real_case_evaluator import RealCaseEvaluator
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


# ── Image-analysis evaluation (quantitative, per image type) ────────────────
_IMAGE_TYPES = {"bloodstain", "fingerprint", "wound", "ballistics",
                "document_forensics", "tool_marks"}
_last_image_results: dict = {}


@router.get("/image/types")
async def image_eval_types():
    """List image types that support quantitative evaluation + their ground-truth basis."""
    from app.services.evaluation.image_gold import ADAPTERS
    return {
        "types": [
            {"image_type": t, "attributes": [
                {"name": s.name, "kind": s.kind, "note": s.note}
                for s in cls.specs
            ]}
            for t, cls in ADAPTERS.items()
        ]
    }


@router.post("/image/run")
async def evaluate_image(
    image_type: str,
    limit: int = 5,
    model: Optional[str] = None,
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Run the real image pipeline on labelled images and score predictions vs
    ground truth (accuracy / precision-recall-F1 / MAE per attribute)."""
    if image_type not in _IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown image type '{image_type}'. One of: {', '.join(sorted(_IMAGE_TYPES))}",
        )
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    image_extractor = ImageExtractor(openai_client, settings)
    evaluator = ImageEvaluator(image_extractor, settings)
    try:
        result = await evaluator.evaluate(image_type, limit=limit, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _last_image_results[image_type] = result
    return result


@router.get("/image/results")
async def image_eval_results():
    """Return the last image evaluation result per type."""
    return _last_image_results or {"message": "No image evaluations yet. Run /image/run."}


# ── Real-case validation (documented-fact gold, not LLM-generated) ──────────
_last_real_case_results: Optional[AggregateEvaluationResult] = None


def _build_real_case_evaluator(schema: ForensicsOntologySchema) -> RealCaseEvaluator:
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    text_extractor = TextExtractor(openai_client, schema)
    return RealCaseEvaluator(settings.data_dir, text_extractor)


@router.get("/real-cases/list")
async def list_real_cases(schema: ForensicsOntologySchema = Depends(get_ontology_schema)):
    """List the curated real cases (documented-fact gold)."""
    ev = _build_real_case_evaluator(schema)
    return {"cases": [
        {"case_id": c["case_id"], "title": c.get("title"), "source": c.get("source"),
         "gold_persons": len(c.get("gold", {}).get("persons", []))}
        for c in ev.list_cases()
    ]}


@router.post("/real-cases/run", response_model=AggregateEvaluationResult)
async def run_real_case_validation(
    model: Optional[str] = None,
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Score extraction against documented-fact gold for all real cases (P/R/F1)."""
    global _last_real_case_results
    ev = _build_real_case_evaluator(schema)
    result = await ev.evaluate_all(model_override=model)
    _last_real_case_results = result
    return result


@router.get("/real-cases/results")
async def real_case_results():
    if _last_real_case_results is None:
        return {"message": "No real-case validation yet. Run /real-cases/run."}
    return _last_real_case_results
