from pydantic import BaseModel, Field
from typing import Dict, Any


class RelationshipBase(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any] = {}
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
