from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from pathlib import Path
import yaml
from loguru import logger


# --- Enums (salvaged from existing src/ontology.py:18-56, expanded) ---

class PatternType(str, Enum):
    PASSIVE_DROP = "passive_drop"
    TRANSFER = "transfer"
    PROJECTED = "projected"
    CAST_OFF = "cast_off"
    IMPACT_SPATTER = "impact_spatter"
    ARTERIAL_SPURT = "arterial_spurt"
    VOID = "void"
    WIPE = "wipe"
    SWIPE = "swipe"
    FLOW = "flow"
    POOL = "pool"
    SATELLITE = "satellite"


class ImpactMechanismType(str, Enum):
    BLUNT_FORCE = "blunt_force"
    SHARP_FORCE = "sharp_force"
    GUNSHOT = "gunshot"
    STABBING = "stabbing"
    BEATING = "beating"
    FALL = "fall"
    UNKNOWN = "unknown"


class SubstrateType(str, Enum):
    PAPER = "paper"
    CARDBOARD = "cardboard"
    FABRIC = "fabric"
    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"
    PLASTIC = "plastic"
    CONCRETE = "concrete"
    TILE = "tile"
    OTHER = "other"


class PersonRole(str, Enum):
    VICTIM = "victim"
    SUSPECT = "suspect"
    WITNESS = "witness"
    OFFICER = "officer"
    FORENSIC_ANALYST = "forensic_analyst"


class EvidenceType(str, Enum):
    PHYSICAL = "physical"
    DIGITAL = "digital"
    BIOLOGICAL = "biological"
    DOCUMENTARY = "documentary"
    TESTIMONIAL = "testimonial"


class CrimeCategory(str, Enum):
    HOMICIDE = "homicide"
    ASSAULT = "assault"
    ROBBERY = "robbery"
    BURGLARY = "burglary"
    DOMESTIC_VIOLENCE = "domestic_violence"
    HIT_AND_RUN = "hit_and_run"
    KIDNAPPING = "kidnapping"
    CYBERCRIME = "cybercrime"
    OTHER = "other"


# --- Schema models ---

class NodeTypeDefinition(BaseModel):
    name: str
    properties: Dict[str, str]
    required_properties: List[str]
    unique_key: str


class RelationshipTypeDefinition(BaseModel):
    name: str
    source_types: List[str]
    target_types: List[str]
    properties: Dict[str, str]


class ForensicsOntologySchema:
    """Loads and manages the forensics ontology from YAML."""

    def __init__(self, yaml_path: Path):
        self._yaml_path = yaml_path
        self._raw: Dict[str, Any] = {}
        self._node_types: Dict[str, NodeTypeDefinition] = {}
        self._relationship_types: Dict[str, RelationshipTypeDefinition] = {}
        self._enums: Dict[str, List[str]] = {}

    def load(self) -> None:
        with open(self._yaml_path, "r") as f:
            self._raw = yaml.safe_load(f)

        # Parse enums
        for enum_name, values in self._raw.get("enums", {}).items():
            self._enums[enum_name] = values

        # Parse node types
        for name, definition in self._raw.get("node_types", {}).items():
            self._node_types[name] = NodeTypeDefinition(
                name=name,
                properties=definition.get("properties", {}),
                required_properties=definition.get("required", []),
                unique_key=definition.get("unique_key", ""),
            )

        # Parse relationship types
        for name, definition in self._raw.get("relationship_types", {}).items():
            self._relationship_types[name] = RelationshipTypeDefinition(
                name=name,
                source_types=definition.get("source", []),
                target_types=definition.get("target", []),
                properties=definition.get("properties", {}),
            )

        logger.info(
            f"Ontology loaded: {len(self._node_types)} node types, "
            f"{len(self._relationship_types)} relationship types, "
            f"{len(self._enums)} enums"
        )

    def get_node_type(self, name: str) -> Optional[NodeTypeDefinition]:
        return self._node_types.get(name)

    def get_relationship_type(self, name: str) -> Optional[RelationshipTypeDefinition]:
        return self._relationship_types.get(name)

    def get_all_node_types(self) -> List[NodeTypeDefinition]:
        return list(self._node_types.values())

    def get_all_relationship_types(self) -> List[RelationshipTypeDefinition]:
        return list(self._relationship_types.values())

    def get_node_type_names(self) -> List[str]:
        return list(self._node_types.keys())

    def get_relationship_type_names(self) -> List[str]:
        return list(self._relationship_types.keys())

    def get_enum(self, name: str) -> List[str]:
        return self._enums.get(name, [])

    def validate_node(self, node_type: str, properties: Dict[str, Any]) -> bool:
        nt = self._node_types.get(node_type)
        if not nt:
            return False
        for req in nt.required_properties:
            if req not in properties:
                return False
        return True

    def validate_relationship(
        self, rel_type: str, source_type: str, target_type: str
    ) -> bool:
        rt = self._relationship_types.get(rel_type)
        if not rt:
            return False
        return source_type in rt.source_types and target_type in rt.target_types

    def generate_neo4j_constraints(self) -> List[str]:
        constraints = []
        for name, nt in self._node_types.items():
            if nt.unique_key:
                constraints.append(
                    f"CREATE CONSTRAINT {name.lower()}_{nt.unique_key}_unique "
                    f"IF NOT EXISTS FOR (n:{name}) REQUIRE n.{nt.unique_key} IS UNIQUE"
                )
        return constraints

    def generate_neo4j_indexes(self) -> List[str]:
        indexes = []
        # Index commonly queried properties
        index_targets = {
            "Case": ["status", "crime_type"],
            "Person": ["role", "name"],
            "BloodstainPattern": ["pattern_type", "confidence"],
            "Evidence": ["type"],
            "Experiment": ["category"],
            "Hypothesis": ["confidence"],
            "Stain": ["area", "circularity"],
        }
        for label, props in index_targets.items():
            if label in self._node_types:
                for prop in props:
                    indexes.append(
                        f"CREATE INDEX {label.lower()}_{prop}_idx "
                        f"IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
                    )
        return indexes

    def get_schema_description(self) -> str:
        """Build a text description for LLM prompts."""
        lines = ["Node Types:"]
        for nt in self._node_types.values():
            props = ", ".join(nt.properties.keys())
            lines.append(f"  (:{nt.name} {{{props}}})")
        lines.append("\nRelationship Types:")
        for rt in self._relationship_types.values():
            sources = "|".join(rt.source_types)
            targets = "|".join(rt.target_types)
            lines.append(f"  (:{sources})-[:{rt.name}]->(:{targets})")
        return "\n".join(lines)
