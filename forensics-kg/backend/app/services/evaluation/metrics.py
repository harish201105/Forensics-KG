"""Pydantic models for evaluation metrics."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class MatchedPair(BaseModel):
    gold: str
    extracted: str
    similarity: float


class EntityTypeMetrics(BaseModel):
    entity_type: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    matched_pairs: List[MatchedPair] = []


class EvaluationResult(BaseModel):
    fir_id: str
    entity_metrics: Dict[str, EntityTypeMetrics]
    aggregate_precision: float
    aggregate_recall: float
    aggregate_f1: float
    extraction_time_ms: float
    gold_entity_count: int
    extracted_entity_count: int
    model_used: str


class AggregateEvaluationResult(BaseModel):
    total_cases: int
    per_case: List[EvaluationResult]
    per_entity_type: Dict[str, EntityTypeMetrics]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    model_used: str
