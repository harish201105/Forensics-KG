import uuid
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from loguru import logger
from app.services.extraction.text_extractor import TextExtractor
from app.services.extraction.image_extractor import ImageExtractor
from app.services.extraction.multimodal_extractor import MultimodalExtractor
from app.services.extraction.openai_client import OpenAIClient
from app.services.graph.operations import GraphOperations
from app.services.graph.deduplication import EntityDeduplicator

if TYPE_CHECKING:
    from app.services.graph.embedding_service import EmbeddingService
    from app.services.graph.entity_resolution import EntityResolutionService


class ExtractionPipeline:
    """Orchestrates extraction -> deduplication -> validation -> storage."""

    def __init__(
        self,
        text_extractor: Optional[TextExtractor],
        image_extractor: Optional[ImageExtractor],
        graph_ops: GraphOperations,
        deduplicator: Optional[EntityDeduplicator] = None,
        openai_client: Optional[OpenAIClient] = None,
        embedding_service: "Optional[EmbeddingService]" = None,
        resolution_service: "Optional[EntityResolutionService]" = None,
    ):
        self._text = text_extractor
        self._image = image_extractor
        self._graph = graph_ops
        self._dedup = deduplicator
        self._openai = openai_client
        self._embeddings = embedding_service
        self._resolution = resolution_service

    async def _auto_embed(self, result: Dict[str, Any]) -> None:
        """Embed newly stored nodes so semantic search/projector stay current.

        Uses only_missing so only the just-added nodes are embedded. Never lets
        an embedding failure break the extraction response.
        """
        if self._embeddings is None:
            return
        try:
            stats = await self._embeddings.backfill(only_missing=True)
            result.setdefault("metadata", {})["embedded_nodes"] = stats.get("processed", 0)
        except Exception as e:
            logger.warning(f"Auto-embedding skipped (extraction unaffected): {e}")

    async def _auto_resolve(self, prefix: str, result: Dict[str, Any]) -> None:
        """Link the just-added entities to matching entities in other cases so the
        graph stays connected (cross-case). Runs incrementally; never breaks extraction."""
        if self._resolution is None:
            return
        try:
            stats = await self._resolution.resolve_incremental(prefix)
            result.setdefault("metadata", {})["cross_case_links"] = stats.get("links_created", 0)
        except Exception as e:
            logger.warning(f"Auto entity-resolution skipped (extraction unaffected): {e}")

    @staticmethod
    def _prefix_ids(
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        doc_id: str,
    ) -> None:
        """Prefix all generic entity IDs with a document identifier so
        entities from different documents don't collide on IDs like
        ``person_1``.  Also updates matching property values (e.g.
        ``stain_id``, ``pattern_id``) so Neo4j uniqueness constraints
        don't collide across documents.  Mutates the lists in-place."""
        id_map: Dict[str, str] = {}
        for ent in entities:
            old_id = ent.get("entity_id", "")
            if old_id and not old_id.startswith(doc_id):
                new_id = f"{doc_id}_{old_id}"
                id_map[old_id] = new_id
                ent["entity_id"] = new_id
            # Also prefix any property whose value matches the old entity_id
            props = ent.get("properties", {})
            for key, val in props.items():
                if isinstance(val, str) and val in id_map:
                    props[key] = id_map[val]
        for rel in relationships:
            src = rel.get("source_entity_id", "")
            tgt = rel.get("target_entity_id", "")
            if src in id_map:
                rel["source_entity_id"] = id_map[src]
            if tgt in id_map:
                rel["target_entity_id"] = id_map[tgt]

    async def process_text(
        self, text: str, source_type: str = "fir", store: bool = True,
        model_override: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract from text and optionally store in Neo4j."""
        if not self._text:
            raise RuntimeError("TextExtractor not configured")
        result = await self._text.extract(text, source_type, model_override=model_override)

        # Make entity IDs unique per document
        prefix = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
        self._prefix_ids(result["entities"], result["relationships"], prefix)

        if store:
            entities = result["entities"]
            if self._dedup:
                entities = await self._dedup.deduplicate(entities)
            node_ids = await self._graph.store_extracted_entities(entities)
            rel_count = await self._graph.store_extracted_relationships(
                result["relationships"], entities
            )
            result["metadata"]["stored_nodes"] = len(node_ids)
            result["metadata"]["stored_relationships"] = rel_count
            await self._auto_embed(result)
            await self._auto_resolve(prefix, result)

        return result

    async def process_image(
        self,
        image_bytes: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        store: bool = True,
        image_type: str = "bloodstain",
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze image and optionally store results in Neo4j."""
        if not self._image:
            raise RuntimeError("ImageExtractor not configured")
        result = await self._image.analyze(image_bytes, metadata, image_type=image_type)

        prefix = doc_id or f"img_{uuid.uuid4().hex[:8]}"
        self._prefix_ids(result["entities"], result["relationships"], prefix)

        if store:
            entities = result["entities"]
            if self._dedup:
                entities = await self._dedup.deduplicate(entities)
            node_ids = await self._graph.store_extracted_entities(entities)
            rel_count = await self._graph.store_extracted_relationships(
                result["relationships"], entities
            )
            result["metadata"]["stored_nodes"] = len(node_ids)
            result["metadata"]["stored_relationships"] = rel_count
            await self._auto_embed(result)
            await self._auto_resolve(prefix, result)

        return result

    async def process_combined(
        self,
        text: str,
        image_bytes: bytes,
        source_type: str = "fir",
        image_type: str = "bloodstain",
        store: bool = True,
        model_override: Optional[str] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Combined text + image extraction with synthesis pass."""
        if not self._text:
            raise RuntimeError("TextExtractor not configured")
        if not self._image:
            raise RuntimeError("ImageExtractor not configured")
        if not self._openai:
            raise RuntimeError("OpenAIClient not configured")
        multimodal = MultimodalExtractor(self._text, self._image, self._openai)
        result = await multimodal.extract_combined(
            text, image_bytes, source_type, image_type=image_type,
            model_override=model_override, image_metadata=image_metadata,
        )

        prefix = doc_id or f"combined_{uuid.uuid4().hex[:8]}"
        self._prefix_ids(result["entities"], result["relationships"], prefix)

        if store:
            entities = result["entities"]
            if self._dedup:
                entities = await self._dedup.deduplicate(entities)
            node_ids = await self._graph.store_extracted_entities(entities)
            rel_count = await self._graph.store_extracted_relationships(
                result["relationships"], entities
            )
            result["metadata"]["stored_nodes"] = len(node_ids)
            result["metadata"]["stored_relationships"] = rel_count
            await self._auto_embed(result)
            await self._auto_resolve(prefix, result)

        return result
