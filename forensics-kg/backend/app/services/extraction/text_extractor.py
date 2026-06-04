from typing import Dict, List, Any, Optional, Tuple, Callable
from loguru import logger
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.schemas import (
    FIR_ENTITY_EXTRACTION_SCHEMA,
    FIR_RELATIONSHIP_EXTRACTION_SCHEMA,
    COURT_JUDGMENT_ENTITY_SCHEMA,
    COURT_JUDGMENT_RELATIONSHIP_SCHEMA,
    POSTMORTEM_ENTITY_SCHEMA,
    POSTMORTEM_RELATIONSHIP_SCHEMA,
    LAB_REPORT_ENTITY_SCHEMA,
    LAB_REPORT_RELATIONSHIP_SCHEMA,
    DEPOSITION_ENTITY_SCHEMA,
    DEPOSITION_RELATIONSHIP_SCHEMA,
    SOCO_ENTITY_SCHEMA,
    SOCO_RELATIONSHIP_SCHEMA,
)
from app.services.ontology.schema import ForensicsOntologySchema


# --- Prompt builders for each source type ---

def _fir_entity_prompt(source_type: str, type_descriptions: str) -> str:
    return (
        f"You are a forensic data extraction expert. Extract structured entities "
        f"from {source_type} reports.\n\n"
        f"Available entity types:\n{type_descriptions}\n\n"
        f"Rules:\n"
        f"- Generate unique IDs: person_1, loc_1, evidence_1, weapon_1, etc.\n"
        f"- Only extract information explicitly stated in the text.\n"
        f"- Do NOT infer or hallucinate entities not mentioned.\n"
        f"- For Person entities, always include 'name' and 'role' "
        f"(victim/suspect/witness/officer) in properties.\n"
        f"- For Case entities, include 'title' and 'case_id' in properties.\n"
        f"- Extract ALL persons, locations, evidence, weapons, vehicles, "
        f"and timeline events mentioned."
    )


def _court_judgment_entity_prompt(source_type: str, type_descriptions: str) -> str:
    return (
        f"You are a legal forensic data extraction expert. Extract structured entities "
        f"from Indian court judgment texts.\n\n"
        f"Available entity types:\n{type_descriptions}\n\n"
        f"Rules:\n"
        f"- Generate unique IDs: person_1, judge_1, lawyer_1, verdict_1, section_1, etc.\n"
        f"- Only extract information explicitly stated in the judgment.\n"
        f"- For Person entities, include 'name' and 'role' "
        f"(judge/accused/victim/witness/lawyer/prosecutor) in properties.\n"
        f"- For LegalSection entities, include the IPC/CrPC section number and description.\n"
        f"- For Verdict entities, include the outcome (convicted/acquitted/remanded) and sentence.\n"
        f"- Extract ALL persons, locations, evidence, weapons, legal sections, "
        f"timeline events, and court orders mentioned.\n"
        f"- Pay attention to the 'Facts of the Case' section for incident details."
    )


def _postmortem_entity_prompt(source_type: str, type_descriptions: str) -> str:
    return (
        f"You are a forensic pathology data extraction expert. Extract structured entities "
        f"from post-mortem / autopsy examination reports.\n\n"
        f"Available entity types:\n{type_descriptions}\n\n"
        f"Rules:\n"
        f"- Generate unique IDs: person_1, injury_1, cod_1, tox_1, organ_1, etc.\n"
        f"- Only extract information explicitly stated in the report.\n"
        f"- For InjuryPattern entities, include injury type, anatomical location, "
        f"dimensions, and estimated age of injury.\n"
        f"- For CauseOfDeath, include primary and contributing causes.\n"
        f"- For ToxicologyResult, include substance, concentration, and method.\n"
        f"- Extract the estimated time of death as a TimeEvent.\n"
        f"- The deceased subject of the report is a Person with role 'victim' "
        f"(use 'victim', NOT 'deceased', for consistency).\n"
        f"- Include the examining pathologist as a Person with role 'forensic_analyst'."
    )


def _lab_report_entity_prompt(source_type: str, type_descriptions: str) -> str:
    return (
        f"You are a forensic laboratory data extraction expert. Extract structured entities "
        f"from forensic laboratory analysis reports.\n\n"
        f"Available entity types:\n{type_descriptions}\n\n"
        f"Rules:\n"
        f"- Generate unique IDs: sample_1, test_1, dna_1, compound_1, etc.\n"
        f"- Only extract information explicitly stated in the report.\n"
        f"- For Sample entities, include sample type, collection date, and source.\n"
        f"- For TestResult entities, include the test name, result, and analytical method.\n"
        f"- For DNAProfile entities, include loci, alleles, and match probability.\n"
        f"- For ChemicalCompound entities, include compound name and concentration.\n"
        f"- Link samples to their test results and any matching evidence."
    )


def _deposition_entity_prompt(source_type: str, type_descriptions: str) -> str:
    return (
        f"You are a forensic investigative data extraction expert. Extract structured entities "
        f"from witness deposition / statement records.\n\n"
        f"Available entity types:\n{type_descriptions}\n\n"
        f"Rules:\n"
        f"- Generate unique IDs: person_1, statement_1, event_1, loc_1, etc.\n"
        f"- Only extract information explicitly stated in the deposition.\n"
        f"- For Person entities, include 'name' and 'role' (witness/accused/victim).\n"
        f"- For Statement entities, include the key claim content and context.\n"
        f"- Extract timeline events mentioned by the witness.\n"
        f"- Pay attention to potential contradictions with other statements.\n"
        f"- Extract alibis as separate Statement entities with type 'alibi'."
    )


def _soco_entity_prompt(source_type: str, type_descriptions: str) -> str:
    return (
        f"You are a crime scene investigation data extraction expert. Extract structured entities "
        f"from Scene of Crime Officer (SOCO) reports.\n\n"
        f"Available entity types:\n{type_descriptions}\n\n"
        f"Rules:\n"
        f"- Generate unique IDs: scene_1, evidence_1, custody_1, person_1, etc.\n"
        f"- Only extract information explicitly stated in the report.\n"
        f"- For CrimeScene entities, include scene type, dimensions, and conditions.\n"
        f"- For PhysicalEvidence entities, include item description, exact location "
        f"in scene, condition, and collection method.\n"
        f"- For ChainOfCustody entities, include handler name, timestamp, and action.\n"
        f"- Carefully track who collected what evidence and when.\n"
        f"- Include weather/environmental conditions if mentioned."
    )


# Schema registry: source_type -> (entity_schema, relationship_schema, entity_prompt_builder)
# The prompt builder receives (source_type, type_descriptions) and returns the system prompt.
SchemaRegistryEntry = Tuple[dict, dict, Callable[[str, str], str]]

SCHEMA_REGISTRY: Dict[str, SchemaRegistryEntry] = {
    "fir": (FIR_ENTITY_EXTRACTION_SCHEMA, FIR_RELATIONSHIP_EXTRACTION_SCHEMA, _fir_entity_prompt),
    "court_judgment": (COURT_JUDGMENT_ENTITY_SCHEMA, COURT_JUDGMENT_RELATIONSHIP_SCHEMA, _court_judgment_entity_prompt),
    "postmortem": (POSTMORTEM_ENTITY_SCHEMA, POSTMORTEM_RELATIONSHIP_SCHEMA, _postmortem_entity_prompt),
    "lab_report": (LAB_REPORT_ENTITY_SCHEMA, LAB_REPORT_RELATIONSHIP_SCHEMA, _lab_report_entity_prompt),
    "witness_deposition": (DEPOSITION_ENTITY_SCHEMA, DEPOSITION_RELATIONSHIP_SCHEMA, _deposition_entity_prompt),
    "soco_report": (SOCO_ENTITY_SCHEMA, SOCO_RELATIONSHIP_SCHEMA, _soco_entity_prompt),
}

# Image-related node types excluded from text extraction prompts
_IMAGE_ONLY_TYPES = {"BloodstainPattern", "Stain", "Experiment", "ImpactMechanismNode",
                     "FingerprintPattern", "MinutiaePoint", "RidgeDetail",
                     "WoundPattern", "TissueAnalysis", "BallisticsPattern",
                     "HandwritingFeature", "InkAnalysis", "ForgeryIndicator", "ToolMarkPattern"}


class TextExtractor:
    """Two-stage entity and relationship extraction from text using GPT-4o."""

    def __init__(self, openai_client: OpenAIClient, ontology: ForensicsOntologySchema):
        self._openai = openai_client
        self._ontology = ontology

    def _get_registry_entry(self, source_type: str) -> SchemaRegistryEntry:
        """Look up schema + prompt builder for a source type, fallback to FIR."""
        entry = SCHEMA_REGISTRY.get(source_type)
        if entry is None:
            logger.warning(f"Unknown source_type '{source_type}', falling back to 'fir'")
            entry = SCHEMA_REGISTRY["fir"]
        return entry

    async def extract(
        self, text: str, source_type: str = "fir", model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Two-stage extraction:
        1. Extract all entities from text
        2. Extract relationships between the identified entities
        """
        # Stage 1: Entity extraction
        entities = await self._extract_entities(text, source_type, model_override)
        logger.info(f"Stage 1: Extracted {len(entities)} entities")

        # Stage 2: Relationship extraction
        relationships = await self._extract_relationships(text, entities, source_type, model_override)
        logger.info(f"Stage 2: Extracted {len(relationships)} relationships")

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "source_type": source_type,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        }

    async def _extract_entities(
        self, text: str, source_type: str, model_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        entity_schema, _, prompt_builder = self._get_registry_entry(source_type)
        system_prompt = self._build_entity_system_prompt(source_type, prompt_builder)
        user_prompt = (
            f"Extract all forensic entities from the following {source_type} report:\n\n"
            f"{text}"
        )
        result = await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=entity_schema,
            model_override=model_override,
        )
        return result.get("entities", [])

    async def _extract_relationships(
        self, text: str, entities: List[Dict], source_type: str, model_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        _, rel_schema, _ = self._get_registry_entry(source_type)
        system_prompt = self._build_relationship_system_prompt(source_type)
        entity_summary = "\n".join(
            f"- {e['entity_type']}: {e['entity_id']} "
            f"(name={e.get('properties', {}).get('name', 'N/A')})"
            for e in entities
        )
        user_prompt = (
            f"Given these entities extracted from a {source_type} report:\n"
            f"{entity_summary}\n\n"
            f"Original text:\n{text}\n\n"
            f"Extract all relationships between these entities. "
            f"Include the exact text span that supports each relationship."
        )
        result = await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=rel_schema,
            model_override=model_override,
        )
        return result.get("relationships", [])

    def _build_entity_system_prompt(
        self, source_type: str, prompt_builder: Callable[[str, str], str]
    ) -> str:
        node_types = self._ontology.get_all_node_types()
        type_descriptions = "\n".join(
            f"- {nt.name}: required={nt.required_properties}, "
            f"optional={[p for p in nt.properties if p not in nt.required_properties]}"
            for nt in node_types
            if nt.name not in _IMAGE_ONLY_TYPES
        )
        return prompt_builder(source_type, type_descriptions)

    def _build_relationship_system_prompt(self, source_type: str) -> str:
        rel_types = self._ontology.get_all_relationship_types()
        rel_descriptions = "\n".join(
            f"- {rt.name}: ({', '.join(rt.source_types)}) -> ({', '.join(rt.target_types)})"
            for rt in rel_types
        )
        death_hint = ""
        if source_type in ("postmortem", "lab_report"):
            death_hint = (
                "- IMPORTANT: when a CauseOfDeath is identified, link it to the "
                "deceased Person with CAUSE_OF_DEATH_OF (CauseOfDeath -> Person), "
                "so the cause of death is directly attributable to the victim.\n"
            )
        return (
            f"You are a forensic data extraction expert. Extract relationships "
            f"between entities from {source_type} reports.\n\n"
            f"Available relationship types:\n{rel_descriptions}\n\n"
            f"Rules:\n"
            f"- Only create relationships between entities already extracted.\n"
            f"- Use the exact entity_id values from the entity list.\n"
            f"{death_hint}"
            f"- Assign confidence 0.0-1.0 (1.0 = explicitly stated, "
            f"0.7+ = strongly implied, 0.5 = inferred).\n"
            f"- Include the exact text span that supports the relationship "
            f"in 'evidence_span'.\n"
            f"- Do NOT create relationships between entity types that don't "
            f"match the allowed source/target types."
        )
