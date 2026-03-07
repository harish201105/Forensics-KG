from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class SourceType(str, Enum):
    fir = "fir"
    court_judgment = "court_judgment"
    postmortem = "postmortem"
    lab_report = "lab_report"
    witness_deposition = "witness_deposition"
    soco_report = "soco_report"
    report = "report"
    evidence = "evidence"
    witness = "witness"
    generic = "generic"


class TextExtractionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)
    source_type: SourceType = SourceType.fir
    case_id: Optional[str] = None
    store_in_graph: bool = True
    model: Optional[str] = None


class NaturalLanguageQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    context_case_id: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=100)
    model: Optional[str] = None


class CypherQueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=5000)
    parameters: dict = {}


class SyntheticFIRRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=100)
    crime_types: Optional[List[str]] = None
    include_bloodstain: bool = True

    @field_validator("crime_types")
    @classmethod
    def validate_crime_types(cls, v):
        if v is not None and len(v) > 20:
            raise ValueError("Too many crime types (max 20)")
        return v
