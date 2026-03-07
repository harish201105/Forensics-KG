"""Lightweight collaboration: annotations, case status, activity logging."""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from loguru import logger
from app.services.graph.neo4j_client import Neo4jClient


class AnnotationService:
    """Manages annotations, case status, and activity logging."""

    def __init__(self, neo4j_client: Neo4jClient):
        self._neo4j = neo4j_client

    async def add_annotation(
        self,
        entity_type: str,
        entity_id: str,
        text: str,
        author: str = "anonymous",
    ) -> Dict[str, Any]:
        """Create an Annotation node linked to an entity via HAS_ANNOTATION."""
        annotation_id = f"ann-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        query = """
        MATCH (e {entity_id: $entity_id})
        WHERE $entity_type IN labels(e) OR e.case_id = $entity_id
        CREATE (a:Annotation {
            annotation_id: $annotation_id,
            text: $text,
            author: $author,
            created_at: $created_at
        })
        CREATE (e)-[:HAS_ANNOTATION]->(a)
        RETURN a {.annotation_id, .text, .author, .created_at} as annotation
        """
        # Fallback: if entity uses case_id as key instead of entity_id
        results = await self._neo4j.execute_query(query, {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "annotation_id": annotation_id,
            "text": text,
            "author": author,
            "created_at": now,
        })

        if not results:
            # Try matching by name or case_id property
            fallback_query = """
            MATCH (e)
            WHERE (e.name = $entity_id OR e.case_id = $entity_id)
              AND $entity_type IN labels(e)
            CREATE (a:Annotation {
                annotation_id: $annotation_id,
                text: $text,
                author: $author,
                created_at: $created_at
            })
            CREATE (e)-[:HAS_ANNOTATION]->(a)
            RETURN a {.annotation_id, .text, .author, .created_at} as annotation
            """
            results = await self._neo4j.execute_query(fallback_query, {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "annotation_id": annotation_id,
                "text": text,
                "author": author,
                "created_at": now,
            })

        await self._log_activity(
            "add_annotation",
            f"Annotation on {entity_type}/{entity_id}: {text[:50]}",
            author,
        )

        if results:
            return results[0].get("annotation", {"annotation_id": annotation_id})
        return {"annotation_id": annotation_id, "text": text, "author": author, "created_at": now}

    async def get_annotations(
        self, entity_type: str, entity_id: str
    ) -> List[Dict[str, Any]]:
        """Get all annotations for an entity."""
        query = """
        MATCH (e)-[:HAS_ANNOTATION]->(a:Annotation)
        WHERE (e.entity_id = $entity_id OR e.case_id = $entity_id OR e.name = $entity_id)
          AND $entity_type IN labels(e)
        RETURN a {.annotation_id, .text, .author, .created_at} as annotation
        ORDER BY a.created_at DESC
        """
        results = await self._neo4j.execute_query(query, {
            "entity_type": entity_type,
            "entity_id": entity_id,
        })
        return [r["annotation"] for r in results if "annotation" in r]

    async def update_case_status(
        self, case_id: str, status: str, user: str = "anonymous"
    ) -> Dict[str, Any]:
        """Update a Case node's status property."""
        query = """
        MATCH (c:Case {case_id: $case_id})
        SET c.status = $status, c.status_updated_at = $updated_at
        RETURN c.case_id as case_id, c.status as status
        """
        now = datetime.now(timezone.utc).isoformat()
        results = await self._neo4j.execute_query(query, {
            "case_id": case_id,
            "status": status,
            "updated_at": now,
        })

        await self._log_activity(
            "update_status",
            f"Case {case_id} status -> {status}",
            user,
        )

        if results:
            return results[0]
        return {"case_id": case_id, "status": status}

    async def get_activity_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity log entries."""
        query = """
        MATCH (a:ActivityLog)
        RETURN a {.log_id, .action, .details, .user, .timestamp} as entry
        ORDER BY a.timestamp DESC
        LIMIT $limit
        """
        results = await self._neo4j.execute_query(query, {"limit": limit})
        return [r["entry"] for r in results if "entry" in r]

    async def _log_activity(
        self, action: str, details: str, user: str
    ) -> None:
        """Create an ActivityLog node."""
        log_id = f"log-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        query = """
        CREATE (a:ActivityLog {
            log_id: $log_id,
            action: $action,
            details: $details,
            user: $user,
            timestamp: $timestamp
        })
        """
        try:
            await self._neo4j.execute_query(query, {
                "log_id": log_id,
                "action": action,
                "details": details,
                "user": user,
                "timestamp": now,
            })
        except Exception as e:
            logger.warning(f"Failed to log activity: {e}")
