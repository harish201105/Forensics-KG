from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from typing import Optional, List
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations
from app.services.ontology.schema import ForensicsOntologySchema
from app.models.responses import GraphResponse, GraphStatsResponse

router = APIRouter()


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
