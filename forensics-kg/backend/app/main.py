import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger

from app.config import get_settings
from app.services.graph.neo4j_client import Neo4jClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.routers import extract, graph, query, datasets, analysis, export, evaluation, temporal, collaboration


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Connect Neo4j
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        await neo4j_client.connect()
        app.state.neo4j_client = neo4j_client
    except Exception as e:
        logger.warning(f"Neo4j connection failed: {e}. Running without graph DB.")
        app.state.neo4j_client = None

    # Load ontology
    schema = ForensicsOntologySchema(settings.ontology_path)
    schema.load()
    app.state.ontology_schema = schema

    logger.info("Forensics KG API started")
    yield

    # Shutdown
    if app.state.neo4j_client:
        await neo4j_client.disconnect()
    logger.info("Forensics KG API stopped")


app = FastAPI(
    title="Forensics Knowledge Graph API",
    description="LLM-powered forensic analysis with knowledge graph backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.0f}ms)"
    )
    return response


# Global exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


# Routers
app.include_router(extract.router, prefix="/api/extract", tags=["extraction"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
app.include_router(temporal.router, prefix="/api/temporal", tags=["temporal"])
app.include_router(collaboration.router, prefix="/api/collaboration", tags=["collaboration"])


@app.get("/api/health")
async def health_check():
    neo4j_status = "connected" if app.state.neo4j_client else "disconnected"
    return {"status": "ok", "neo4j": neo4j_status}
