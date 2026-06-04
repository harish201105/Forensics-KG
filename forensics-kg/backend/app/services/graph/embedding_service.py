"""Semantic search over the knowledge graph using OpenAI embeddings + Neo4j vector index.

Every embeddable node is tagged with a shared `:Embedded` label and given an
`embedding` vector property, so a single vector index covers all node types.
Retrieval is cosine-KNN via `db.index.vector.queryNodes` — the same nearest-
neighbour idea as the TensorFlow Embedding Projector, but server-side.
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from app.services.graph.neo4j_client import Neo4jClient
from app.services.extraction.openai_client import OpenAIClient

INDEX_NAME = "node_embeddings"
EMBEDDED_LABEL = "Embedded"
# Internal/operational node types that should never appear in semantic search.
SKIP_LABELS = ("ActivityLog",)

# Properties that should never be folded into the embedding text.
_SKIP_PROPS = frozenset({
    "embedding", "embed_text", "embedding_model",
})
# Substrings marking properties that are binary/base64/serialized blobs.
_SKIP_SUBSTRINGS = ("base64", "image_data", "_blob", "vector")


def _is_meaningful(key: str, value: Any) -> bool:
    if key in _SKIP_PROPS:
        return False
    if any(s in key.lower() for s in _SKIP_SUBSTRINGS):
        return False
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return False
    text = str(value).strip()
    if not text or len(text) > 600:
        return False
    return True


def node_to_text(labels: List[str], props: Dict[str, Any]) -> str:
    """Build a compact natural-language representation of a node for embedding."""
    label = labels[0] if labels else "Node"
    # Prefer a human-friendly identifier first.
    name = (
        props.get("name")
        or props.get("title")
        or props.get("description")
        or props.get("case_id")
        or props.get("experiment_id")
        or ""
    )
    parts = [f"{label}"]
    if name:
        parts.append(f": {name}")
    fields = []
    for key, value in props.items():
        if key in ("name", "title") or not _is_meaningful(key, value):
            continue
        fields.append(f"{key.replace('_', ' ')}={value}")
    text = "".join(parts)
    if fields:
        text += " | " + "; ".join(fields[:25])
    return text[:2000]


class EmbeddingService:
    """Embeds graph nodes and serves semantic KNN search over them."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        openai_client: OpenAIClient,
        dimensions: int = 1536,
    ):
        self._neo4j = neo4j_client
        self._openai = openai_client
        self._dimensions = dimensions

    async def ensure_index(self) -> None:
        """Create the vector index if it doesn't already exist."""
        query = f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (n:{EMBEDDED_LABEL}) ON (n.embedding)
        OPTIONS {{ indexConfig: {{
            `vector.dimensions`: {self._dimensions},
            `vector.similarity_function`: 'cosine'
        }} }}
        """
        await self._neo4j.execute_write(query)
        logger.info(f"Vector index '{INDEX_NAME}' ensured ({self._dimensions} dims, cosine)")

    async def backfill(
        self, batch_size: int = 64, only_missing: bool = True
    ) -> Dict[str, Any]:
        """Embed nodes and store vectors. Returns counts of processed/skipped."""
        await self.ensure_index()

        skip = " AND ".join(f"NOT n:{lbl}" for lbl in SKIP_LABELS)
        conds = []
        if only_missing:
            conds.append("n.embedding IS NULL")
        if skip:
            conds.append(skip)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        fetch = f"""
        MATCH (n)
        {where}
        RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props
        """
        rows = await self._neo4j.execute_query(fetch)
        total = len(rows)
        if total == 0:
            logger.info("Embedding backfill: nothing to do.")
            return {"processed": 0, "total_candidates": 0, "skipped": 0}

        logger.info(f"Embedding backfill: {total} nodes to process...")
        processed = 0
        skipped = 0
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            texts = [node_to_text(r["labels"], r["props"]) for r in batch]
            try:
                vectors = await self._openai.embed(texts)
            except Exception as e:
                logger.error(f"Embedding batch failed ({start}-{start+len(batch)}): {e}")
                skipped += len(batch)
                continue
            updates = [
                {"id": r["id"], "vec": vec, "text": text}
                for r, vec, text in zip(batch, vectors, texts)
            ]
            await self._neo4j.execute_query(
                f"""
                UNWIND $updates AS u
                MATCH (n) WHERE elementId(n) = u.id
                SET n:{EMBEDDED_LABEL}, n.embedding = u.vec, n.embed_text = u.text
                """,
                {"updates": updates},
            )
            processed += len(updates)
            logger.info(f"  embedded {processed}/{total}")

        return {"processed": processed, "total_candidates": total, "skipped": skipped}

    async def semantic_search(
        self,
        query_text: str,
        k: int = 10,
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the k nearest nodes to the query text by cosine similarity."""
        vector = await self._openai.embed_one(query_text)
        if not vector:
            return []
        # Over-fetch when filtering by label, then trim.
        fetch_k = k * 4 if labels else k
        cypher = f"""
        CALL db.index.vector.queryNodes('{INDEX_NAME}', $k, $vec)
        YIELD node, score
        RETURN elementId(node) AS id, labels(node) AS labels,
               properties(node) AS props, score
        ORDER BY score DESC
        """
        rows = await self._neo4j.execute_query(cypher, {"k": fetch_k, "vec": vector})
        results = []
        label_set = set(labels) if labels else None
        for r in rows:
            node_labels = r.get("labels", [])
            if label_set and not (label_set & set(node_labels)):
                continue
            props = {k: v for k, v in (r.get("props") or {}).items() if k != "embedding"}
            results.append({
                "id": r["id"],
                "labels": node_labels,
                "properties": props,
                "score": r.get("score"),
            })
            if len(results) >= k:
                break
        return results

    async def fetch_all_embeddings(
        self, limit: int = 2000
    ) -> List[Dict[str, Any]]:
        """Fetch stored embeddings for dimensionality-reduction / projection."""
        cypher = f"""
        MATCH (n:{EMBEDDED_LABEL})
        WHERE n.embedding IS NOT NULL
        RETURN elementId(n) AS id, labels(n) AS labels,
               n.embedding AS embedding, coalesce(n.embed_text, '') AS text
        LIMIT $limit
        """
        return await self._neo4j.execute_query(cypher, {"limit": limit})

    async def count_embedded(self) -> Dict[str, int]:
        """Return how many embeddable nodes are embedded vs total."""
        skip = " AND ".join(f"NOT n:{lbl}" for lbl in SKIP_LABELS)
        where = f"WHERE {skip}" if skip else ""
        rows = await self._neo4j.execute_query(
            f"""
            MATCH (n)
            {where}
            RETURN count(n) AS total,
                   count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS embedded
            """
        )
        r = rows[0] if rows else {"total": 0, "embedded": 0}
        return {"total": r.get("total", 0), "embedded": r.get("embedded", 0)}
