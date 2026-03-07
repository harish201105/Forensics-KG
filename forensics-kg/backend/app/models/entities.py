from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.services.ontology.schema import (
    PatternType, ImpactMechanismType, PersonRole, EvidenceType, CrimeCategory,
)


class EntityBase(BaseModel):
    entity_type: str
    entity_id: str
    properties: Dict[str, Any] = {}
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CaseEntity(BaseModel):
    case_id: str
    title: str
    description: Optional[str] = None
    status: str = "open"
    date_filed: Optional[str] = None
    jurisdiction: Optional[str] = None
    crime_type: Optional[str] = None
    fir_number: Optional[str] = None
    police_station: Optional[str] = None


class PersonEntity(BaseModel):
    person_id: str
    name: str
    role: PersonRole
    age: Optional[int] = None
    gender: Optional[str] = None
    description: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None


class LocationEntity(BaseModel):
    location_id: str
    name: str
    type: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None


class EvidenceEntity(BaseModel):
    evidence_id: str
    type: EvidenceType
    description: str
    collected_by: Optional[str] = None
    collection_date: Optional[str] = None


class WeaponEntity(BaseModel):
    weapon_id: str
    type: str
    description: Optional[str] = None
    caliber: Optional[str] = None
    serial_number: Optional[str] = None


class VehicleEntity(BaseModel):
    vehicle_id: str
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    color: Optional[str] = None
    license_plate: Optional[str] = None


class BloodstainPatternEntity(BaseModel):
    pattern_id: str
    pattern_type: PatternType
    confidence: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    key_features: List[str] = []
    uncertainties: List[str] = []
    spatial_distribution: Optional[str] = None
    estimated_stain_count: Optional[int] = None


class StainEntity(BaseModel):
    stain_id: str
    area: float = 0.0
    perimeter: float = 0.0
    aspect_ratio: float = 1.0
    circularity: float = 1.0
    solidity: Optional[float] = None
    convexity: Optional[float] = None
    orientation: float = 0.0
    eccentricity: Optional[float] = None
    center_x: float = 0.0
    center_y: float = 0.0
    density: Optional[float] = None
    nearest_neighbor_distance: Optional[float] = None


class ExperimentEntity(BaseModel):
    experiment_id: str
    category: str
    date: Optional[str] = None
    designed_by: Optional[str] = None
    performed_by: Optional[str] = None
    image_scale: Optional[str] = None
    room_temp: Optional[float] = None
    room_humidity: Optional[float] = None
    blood_type: Optional[str] = None
    hematocrit: Optional[float] = None
    blood_volume: Optional[float] = None
    blood_supply: Optional[str] = None
    height: Optional[float] = None
    origin_x: Optional[float] = None
    origin_y: Optional[float] = None
    origin_z: Optional[float] = None
    target_material: Optional[str] = None
    target_position: Optional[str] = None
    description: Optional[str] = None


class HypothesisEntity(BaseModel):
    hypothesis_id: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    mechanism: Optional[str] = None
    likelihood_ratio: Optional[float] = None
    supporting_evidence: List[str] = []
    contradicting_evidence: List[str] = []
