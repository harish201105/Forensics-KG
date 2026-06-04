import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from io import StringIO

from app.dependencies import get_neo4j_client
from app.services.graph.neo4j_client import Neo4jClient

router = APIRouter()


@router.get("/cases/csv")
async def export_cases_csv(
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Export all cases as CSV."""
    results = await client.execute_query(
        "MATCH (c:Case) RETURN properties(c) as props ORDER BY c.case_id"
    )
    if not results:
        return StreamingResponse(
            iter(["No cases found"]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=cases.csv"},
        )

    # Collect all keys, excluding internal embedding properties
    _internal = {"embedding", "embed_text"}
    all_keys = set()
    for r in results:
        all_keys.update(k for k in r["props"].keys() if k not in _internal)
    keys = sorted(all_keys)

    buf = StringIO()
    buf.write(",".join(keys) + "\n")
    for r in results:
        row = []
        for k in keys:
            val = str(r["props"].get(k, "")).replace('"', '""')
            # Prevent CSV formula injection
            if val and val[0] in ("=", "+", "-", "@", "\t", "\r"):
                val = "'" + val
            row.append(f'"{val}"')
        buf.write(",".join(row) + "\n")

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cases.csv"},
    )


@router.get("/graph/json")
async def export_graph_json(
    limit: int = 2000,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Export full graph as JSON."""
    data = await client.get_full_graph(limit=limit)
    content = json.dumps(data, indent=2, default=str)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=graph.json"},
    )


@router.get("/case/{case_id}/report")
async def export_case_report(
    case_id: str,
    client: Neo4jClient = Depends(get_neo4j_client),
):
    """Export a case report as JSON with its subgraph."""
    # Get case properties
    case_node = await client.get_node_by_property("Case", "case_id", case_id)
    if not case_node:
        return {"error": f"Case '{case_id}' not found"}

    # Get subgraph
    subgraph = await client.get_subgraph(
        center_node_id=case_id, label="Case", key="case_id", depth=3
    )

    report = {
        "case": case_node,
        "subgraph": subgraph,
        "summary": {
            "total_nodes": len(subgraph.get("nodes", [])),
            "total_edges": len(subgraph.get("edges", [])),
        },
    }

    content = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=case_{case_id}_report.json"
        },
    )
