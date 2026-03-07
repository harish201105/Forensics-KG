"""Multi-modal extractor: combines text extraction + image analysis in a single reasoning pass."""

import asyncio
import json
from typing import Dict, Any, Optional
from loguru import logger
from app.services.extraction.text_extractor import TextExtractor
from app.services.extraction.image_extractor import ImageExtractor
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.schemas import MULTIMODAL_SYNTHESIS_SCHEMA


class MultimodalExtractor:
    """Combines text and image extraction with a GPT-4o synthesis pass."""

    def __init__(
        self,
        text_extractor: TextExtractor,
        image_extractor: ImageExtractor,
        openai_client: OpenAIClient,
    ):
        self._text = text_extractor
        self._image = image_extractor
        self._openai = openai_client

    async def extract_combined(
        self,
        text: str,
        image_bytes: bytes,
        source_type: str = "fir",
        image_type: str = "bloodstain",
        model_override: Optional[str] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run text extraction + image analysis in parallel, then synthesize.

        Returns unified entities + relationships from both modalities.
        """
        # Step 1: Run both extractions in parallel
        text_task = self._text.extract(text, source_type, model_override=model_override)
        image_task = self._image.analyze(image_bytes, image_metadata, image_type=image_type)

        text_raw, image_raw = await asyncio.gather(
            text_task, image_task, return_exceptions=True
        )

        # Handle partial failures
        if isinstance(text_raw, BaseException):
            logger.error(f"Text extraction failed: {text_raw}")
            text_result: Dict[str, Any] = {"entities": [], "relationships": [], "metadata": {}}
        else:
            text_result = text_raw
        if isinstance(image_raw, BaseException):
            logger.error(f"Image extraction failed: {image_raw}")
            image_result: Dict[str, Any] = {"entities": [], "relationships": [], "metadata": {}}
        else:
            image_result = image_raw

        text_entities = text_result.get("entities", [])
        text_relationships = text_result.get("relationships", [])
        image_entities = image_result.get("entities", [])
        image_relationships = image_result.get("relationships", [])

        logger.info(
            f"Multimodal: {len(text_entities)} text entities, "
            f"{len(image_entities)} image entities"
        )

        # Step 2: GPT-4o synthesis pass to cross-reference findings
        synthesis = await self._synthesize(
            text_entities, text_relationships,
            image_entities, image_relationships,
            model_override=model_override,
        )

        # Step 3: Merge all entities and relationships
        all_entities = text_entities + image_entities
        all_relationships = text_relationships + image_relationships

        # Add synthesis-derived entities
        for ent in synthesis.get("additional_entities", []):
            all_entities.append({
                "entity_type": ent["entity_type"],
                "entity_id": ent["entity_id"],
                "properties": {
                    "name": ent["name"],
                    "description": ent["description"],
                    "source": "multimodal_synthesis",
                },
            })

        # Add synthesis-derived relationships
        for rel in synthesis.get("additional_relationships", []):
            all_relationships.append({
                "source_entity_id": rel["source_id"],
                "target_entity_id": rel["target_id"],
                "relationship_type": rel["relationship_type"],
                "confidence": rel["confidence"],
                "evidence_span": rel["evidence"],
            })

        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "metadata": {
                "text_entities": len(text_entities),
                "image_entities": len(image_entities),
                "synthesis_entities": len(synthesis.get("additional_entities", [])),
                "cross_references": len(synthesis.get("cross_references", [])),
                "unified_assessment": synthesis.get("unified_assessment", ""),
                "synthesis_confidence": synthesis.get("confidence", 0),
                "reasoning": synthesis.get("reasoning", ""),
            },
        }

    async def _synthesize(
        self,
        text_entities: list,
        text_relationships: list,
        image_entities: list,
        image_relationships: list,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GPT-4o synthesis pass combining text + image findings."""
        system_prompt = (
            "You are a forensic analysis expert. You have been given entities and "
            "relationships extracted from both a text report and a forensic image "
            "analysis. Your task is to:\n"
            "1. Identify cross-references between text mentions and image findings\n"
            "2. Generate additional entities or relationships discovered by combining both\n"
            "3. Provide a unified assessment of the forensic evidence\n"
            "Be precise and evidence-based."
        )

        # Summarize text entities
        text_summary = []
        for e in text_entities[:20]:
            text_summary.append(
                f"  [{e.get('entity_type')}] {e.get('entity_id')}: "
                f"{e.get('properties', {}).get('name', 'N/A')}"
            )

        # Summarize image entities
        image_summary = []
        for e in image_entities[:10]:
            props = e.get("properties", {})
            image_summary.append(
                f"  [{e.get('entity_type')}] {e.get('entity_id')}: "
                f"{props.get('pattern_type', props.get('mechanism_type', 'N/A'))}"
            )

        user_prompt = (
            f"TEXT EXTRACTION ({len(text_entities)} entities):\n"
            + "\n".join(text_summary[:20])
            + f"\n\nText relationships: {len(text_relationships)}\n"
            + f"\nIMAGE ANALYSIS ({len(image_entities)} entities):\n"
            + "\n".join(image_summary[:10])
            + f"\n\nImage relationships: {len(image_relationships)}\n"
            + "\nSynthesize the findings. Identify cross-references between "
            "text evidence mentions and image findings. Generate any additional "
            "entities or relationships that emerge from the combination."
        )

        try:
            return await self._openai.extract_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=MULTIMODAL_SYNTHESIS_SCHEMA,
                model_override=model_override,
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "unified_assessment": "Synthesis unavailable",
                "cross_references": [],
                "additional_entities": [],
                "additional_relationships": [],
                "confidence": 0.0,
                "reasoning": str(e),
            }
