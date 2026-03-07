from typing import Dict, List, Any, Optional
from loguru import logger
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema


class EntityDeduplicator:
    """Fuzzy match entities against existing graph to prevent duplicates."""

    def __init__(self, client: Neo4jClient, schema: ForensicsOntologySchema):
        self._client = client
        self._schema = schema

    async def deduplicate(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check entities against existing graph, merge IDs where duplicates found."""
        deduped = []
        seen_ids = set()

        for entity in entities:
            eid = entity.get("entity_id", "")
            etype = entity.get("entity_type", "")

            if not eid or not etype:
                deduped.append(entity)
                continue

            # Skip within-batch duplicates
            batch_key = f"{etype}:{eid}"
            if batch_key in seen_ids:
                logger.debug(f"Skipping in-batch duplicate: {batch_key}")
                continue
            seen_ids.add(batch_key)

            # Check for exact ID match in graph
            id_key = self._get_id_key(etype)
            existing = await self._client.get_node_by_property(etype, id_key, eid)
            if existing:
                logger.debug(f"Entity {batch_key} already exists, will merge")

            # Check for fuzzy name match
            name = entity.get("properties", {}).get("name", "")
            if name and not existing:
                match = await self._find_by_name(etype, name)
                if match:
                    logger.info(
                        f"Fuzzy match: '{name}' -> existing entity in graph"
                    )
                    # Use existing ID instead
                    existing_id = match.get(id_key, "")
                    if existing_id:
                        entity = {**entity, "entity_id": existing_id}

            deduped.append(entity)

        removed = len(entities) - len(deduped)
        if removed > 0:
            logger.info(f"Deduplication removed {removed} duplicate entities")
        return deduped

    async def _find_by_name(
        self, entity_type: str, name: str
    ) -> Optional[Dict[str, Any]]:
        """Search for an entity with a similar name."""
        query = (
            f"MATCH (n:{entity_type}) "
            f"WHERE toLower(n.name) = toLower($name) "
            f"RETURN properties(n) as props LIMIT 1"
        )
        results = await self._client.execute_query(query, {"name": name})
        return results[0]["props"] if results else None

    def _get_id_key(self, entity_type: str) -> str:
        nt = self._schema.get_node_type(entity_type)
        if nt and nt.unique_key:
            return nt.unique_key
        return f"{entity_type.lower()}_id"
