from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from typing import Optional, List
from app.config import get_settings
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations
from app.services.graph.embedding_service import EmbeddingService
from app.services.graph.entity_resolution import EntityResolutionService, RESOLVABLE
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.models.responses import GraphResponse, GraphStatsResponse

router = APIRouter()


def _build_embedding_service(client: Neo4jClient) -> EmbeddingService:
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    return EmbeddingService(client, openai_client, settings.embedding_dimensions)


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    client: Neo4jClient = Depends(get_neo4j_client),
):
    stats = await client.get_graph_stats()
    return GraphStatsResponse(**stats)


@router.get("/full", response_model=GraphResponse)
async def get_full_graph(
    limit: int = Query(default=500, le=2000),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    data = await client.get_full_graph(limit=limit)
    return GraphResponse(
        nodes=data["nodes"],
        edges=data["edges"],
        total_nodes=len(data["nodes"]),
        total_edges=len(data["edges"]),
    )


@router.get("/subgraph/{node_id}")
async def get_subgraph(
    node_id: str,
    label: str = "Case",
    key: str = "case_id",
    depth: int = Query(default=2, le=5),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    data = await client.get_subgraph(
        center_node_id=node_id, label=label, key=key, depth=depth
    )
    return GraphResponse(
        nodes=data["nodes"],
        edges=data["edges"],
        total_nodes=len(data["nodes"]),
        total_edges=len(data["edges"]),
    )


@router.get("/search")
async def search_nodes(
    q: str,
    labels: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    label_list = labels.split(",") if labels else None
    results = await client.search_nodes(q, label_list, limit)
    return {"results": results, "count": len(results)}


@router.get("/semantic-search")
async def semantic_search(
    q: str,
    k: int = Query(default=10, le=100),
    labels: Optional[str] = None,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Embedding-based KNN search: finds nodes semantically closest to the query."""
    service = _build_embedding_service(client)
    label_list = labels.split(",") if labels else None
    try:
        results = await service.semantic_search(q, k=k, labels=label_list)
    except Exception as e:
        # Most likely the vector index / embeddings haven't been built yet.
        return {
            "results": [],
            "count": 0,
            "error": f"Semantic search unavailable: {e}. "
                     "Run POST /api/graph/embeddings/backfill first.",
        }
    return {"results": results, "count": len(results)}


@router.get("/embeddings/status")
async def embeddings_status(
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Report how many nodes have embeddings."""
    service = _build_embedding_service(client)
    counts = await service.count_embedded()
    counts["coverage"] = (
        round(counts["embedded"] / counts["total"], 3) if counts["total"] else 0.0
    )
    return counts


@router.post("/embeddings/backfill")
async def embeddings_backfill(
    only_missing: bool = Query(default=True),
    batch_size: int = Query(default=64, ge=1, le=256),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Embed nodes (default: only those missing embeddings) and build the vector index."""
    service = _build_embedding_service(client)
    result = await service.backfill(batch_size=batch_size, only_missing=only_missing)
    return result


@router.get("/embeddings/projection")
async def embeddings_projection(
    dim: int = Query(default=2, ge=2, le=3),
    limit: int = Query(default=1500, le=5000),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Project node embeddings to 2D/3D (PCA) for the embedding-projector UI."""
    import numpy as np

    service = _build_embedding_service(client)
    rows = await service.fetch_all_embeddings(limit=limit)
    if not rows:
        return {"points": [], "dim": dim, "count": 0}

    matrix = np.array([r["embedding"] for r in rows], dtype=float)
    # PCA via SVD on mean-centered data.
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:dim].T
    except np.linalg.LinAlgError:
        coords = centered[:, :dim]

    # Normalize coords to a stable [-1, 1] range for rendering.
    span = np.abs(coords).max(axis=0)
    span[span == 0] = 1.0
    coords = coords / span

    points = []
    for r, c in zip(rows, coords.tolist()):
        labels = r.get("labels", [])
        point = {
            "id": r["id"],
            "label": labels[0] if labels else "Node",
            "labels": labels,
            "text": r.get("text", ""),
            "x": c[0],
            "y": c[1],
        }
        if dim == 3:
            point["z"] = c[2]
        points.append(point)
    return {"points": points, "dim": dim, "count": len(points)}


@router.post("/resolve-entities")
async def resolve_entities(
    labels: Optional[str] = None,
    name_threshold: float = Query(default=0.88, ge=0.5, le=1.0),
    cross_case_only: bool = Query(default=True),
    dry_run: bool = Query(default=False),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Link duplicate entities across cases via SAME_AS (embedding-blocked,
    name-confirmed). Requires embeddings (run embeddings/backfill first)."""
    service = EntityResolutionService(client)
    label_list = labels.split(",") if labels else None
    return await service.resolve(
        labels=label_list,
        name_threshold=name_threshold,
        cross_case_only=cross_case_only,
        dry_run=dry_run,
    )


@router.get("/cross-case")
async def cross_case(
    label: str = Query(..., description="Entity label, e.g. Person/Location/Weapon"),
    value: str = Query(..., description="Entity name/type value"),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """All cases connected to an entity, following SAME_AS links across cases."""
    if label not in RESOLVABLE:
        return {"error": f"Unsupported label '{label}'. One of: {', '.join(RESOLVABLE)}"}
    service = EntityResolutionService(client)
    return await service.cross_case(label, value)


@router.get("/node/{label}/{key}/{value}")
async def get_node(
    label: str,
    key: str,
    value: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    node = await client.get_node_by_property(label, key, value)
    if not node:
        return {"error": "Node not found"}
    return {"node": node}


@router.delete("/clear")
async def clear_graph(
    confirm: str = Query(default=""),
    client: Neo4jClient = Depends(get_neo4j_client),
):
    if confirm != "yes-delete-all-data":
        return {
            "error": "This will delete ALL data. Pass ?confirm=yes-delete-all-data to proceed."
        }
    result = await client.clear_database()
    return {"message": "Database cleared", "result": result}


# WebSocket for real-time status
_connected_clients: list = []


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connected_clients.remove(websocket)


async def broadcast_status(message: dict):
    for client in _connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            _connected_clients.remove(client)
