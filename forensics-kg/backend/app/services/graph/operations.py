import re
from typing import Dict, List, Any, Optional
from loguru import logger
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class GraphOperations:
    """Domain-specific graph operations for forensics KG."""

    def __init__(self, client: Neo4jClient, schema: ForensicsOntologySchema):
        self._client = client
        self._schema = schema

    async def store_extracted_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Store a batch of extracted entities in a single transaction."""
        import json as _json

        queries = []
        stored_ids = []
        for entity in entities:
            entity_type = entity.get("entity_type", "")
            entity_id_key = self._get_id_key(entity_type)
            entity_id = entity.get("entity_id", "")
            properties = entity.get("properties", {})

            if not entity_type or not entity_id:
                logger.warning(f"Skipping entity with missing type/id: {entity}")
                continue

            # Sanitize entity_type to prevent Cypher injection
            if not _SAFE_IDENTIFIER.fullmatch(entity_type):
                logger.warning(f"Skipping entity with unsafe type name: '{entity_type}'")
                continue

            # Validate against ontology (warn but don't block)
            known_type = self._schema.get_node_type(entity_type)
            if not known_type:
                logger.warning(f"Entity type '{entity_type}' not in ontology, storing anyway")

            # Clean properties: convert lists to JSON strings for Neo4j
            clean_props = {}
            for k, v in properties.items():
                # Sanitize property keys to prevent Cypher injection
                if not _SAFE_IDENTIFIER.fullmatch(k):
                    continue
                if isinstance(v, list):
                    clean_props[k] = _json.dumps(v)
                elif v is not None:
                    clean_props[k] = v

            match_clause = f"{entity_id_key}: $match_{entity_id_key}"
            params = {f"match_{entity_id_key}": entity_id}

            set_clause = ""
            if clean_props:
                set_items = ", ".join(f"n.{k} = $set_{k}" for k in clean_props)
                set_clause = f" ON CREATE SET {set_items} ON MATCH SET {set_items}"
                params.update({f"set_{k}": v for k, v in clean_props.items()})

            query = f"MERGE (n:{entity_type} {{{match_clause}}}){set_clause} RETURN n"
            queries.append({"query": query, "parameters": params})
            stored_ids.append(entity_id)

        if queries:
            try:
                await self._client.execute_transaction(queries)
            except Exception as e:
                logger.error(f"Batched entity storage failed, falling back to individual: {e}")
                # Fallback to individual merges
                stored_ids = []
                for entity in entities:
                    entity_type = entity.get("entity_type", "")
                    entity_id_key = self._get_id_key(entity_type)
                    entity_id = entity.get("entity_id", "")
                    properties = entity.get("properties", {})
                    if not entity_type or not entity_id:
                        continue
                    clean_props = {}
                    for k, v in properties.items():
                        if isinstance(v, list):
                            clean_props[k] = _json.dumps(v)
                        elif v is not None:
                            clean_props[k] = v
                    await self._client.merge_node(entity_type, {entity_id_key: entity_id}, clean_props)
                    stored_ids.append(entity_id)

        logger.info(f"Stored {len(stored_ids)} entities")
        return stored_ids

    async def store_extracted_relationships(
        self, relationships: List[Dict[str, Any]], entities: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Store a batch of extracted relationships."""
        # Build entity lookup for type resolution
        entity_map = {}
        if entities:
            for e in entities:
                entity_map[e.get("entity_id", "")] = e

        count = 0
        for rel in relationships:
            source_id = rel.get("source_entity_id", "")
            target_id = rel.get("target_entity_id", "")
            rel_type = rel.get("relationship_type", "")

            if not source_id or not target_id or not rel_type:
                continue

            source_entity = entity_map.get(source_id, {})
            target_entity = entity_map.get(target_id, {})
            source_type = source_entity.get("entity_type", "")
            target_type = target_entity.get("entity_type", "")

            if not source_type or not target_type:
                # Try to find in graph
                source_type, target_type = await self._resolve_entity_types(
                    source_id, target_id
                )
                if not source_type or not target_type:
                    logger.warning(
                        f"Cannot resolve types for {source_id} -> {target_id}"
                    )
                    continue

            source_key = self._get_id_key(source_type)
            target_key = self._get_id_key(target_type)

            props = {}
            for k, v in rel.get("properties", {}).items():
                if v is None:
                    continue
                if isinstance(v, list):
                    props[k] = ", ".join(str(item) for item in v)
                else:
                    props[k] = v
            if "confidence" in rel:
                props["confidence"] = rel["confidence"]

            try:
                await self._client.merge_relationship(
                    source_label=source_type,
                    source_key=source_key,
                    source_value=source_id,
                    target_label=target_type,
                    target_key=target_key,
                    target_value=target_id,
                    rel_type=rel_type,
                    properties=props if props else None,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to create relationship: {e}")

        logger.info(f"Stored {count} relationships")
        return count

    async def get_case_graph(self, case_id: str) -> Dict[str, Any]:
        """Get complete subgraph for a specific case."""
        return await self._client.get_subgraph(
            center_node_id=case_id, label="Case", key="case_id", depth=3
        )

    async def get_experiment_graph(self, experiment_id: str) -> Dict[str, Any]:
        """Get subgraph for an experiment."""
        return await self._client.get_subgraph(
            center_node_id=experiment_id,
            label="Experiment",
            key="experiment_id",
            depth=2,
        )

    def _get_id_key(self, entity_type: str) -> str:
        """Get the unique ID property name for a given entity type."""
        nt = self._schema.get_node_type(entity_type)
        if nt and nt.unique_key:
            return nt.unique_key
        # Fallback convention
        return f"{entity_type.lower()}_id"

    async def _resolve_entity_types(
        self, source_id: str, target_id: str
    ) -> tuple:
        """Try to find entity types by searching the graph."""
        source_type = ""
        target_type = ""

        for nt in self._schema.get_all_node_types():
            key = nt.unique_key or f"{nt.name.lower()}_id"
            if not source_type:
                node = await self._client.get_node_by_property(nt.name, key, source_id)
                if node:
                    source_type = nt.name
            if not target_type:
                node = await self._client.get_node_by_property(nt.name, key, target_id)
                if node:
                    target_type = nt.name
            if source_type and target_type:
                break

        return source_type, target_type
