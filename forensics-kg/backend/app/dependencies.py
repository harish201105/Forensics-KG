from fastapi import Request, HTTPException
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema


def get_neo4j_client(request: Request) -> Neo4jClient:
    client = request.app.state.neo4j_client
    if client is None:
        raise HTTPException(status_code=503, detail="Neo4j is not connected")
    return client


def get_ontology_schema(request: Request) -> ForensicsOntologySchema:
    return request.app.state.ontology_schema
