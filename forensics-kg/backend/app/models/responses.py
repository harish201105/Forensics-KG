from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ExtractionResponse(BaseModel):
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    processing_time_ms: float = 0.0


class GraphNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    total_nodes: int = 0
    total_edges: int = 0


class GraphStatsResponse(BaseModel):
    node_counts: Dict[str, int]
    relationship_counts: Dict[str, int]
    total_nodes: int
    total_relationships: int


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = []
    cypher_query: Optional[str] = None
    graph_context: Optional[GraphResponse] = None
    reasoning: Optional[str] = None


class AnalysisResponse(BaseModel):
    primary_hypothesis: Dict[str, Any]
    alternative_hypotheses: List[Dict[str, Any]]
    statistical_analysis: Dict[str, Any] = {}
    recommendations: List[str] = []
    reasoning_chain: List[str] = []
    confidence: float = 0.0


class DatasetInfoResponse(BaseModel):
    dataset_id: str
    name: str
    type: str
    record_count: int
    loaded: bool = False
    metadata: Dict[str, Any] = {}


class ProcessingStatusUpdate(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    data: Optional[Dict[str, Any]] = None
