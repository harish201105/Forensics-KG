import axios from 'axios';
import type { GraphData, GraphStats, ExtractionResult, QueryResponse, DatasetInfo, AnalysisResult, EvaluationResult, AggregateEvaluationResult, SemanticHit, ProjectionResult, EmbeddingStatus, CrossCaseResult, ImageEvalResult } from '@/types';

// Use direct backend URL to avoid Next.js proxy timeout on long-running requests
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${BACKEND_URL}/api`,
  timeout: 300000, // 5 minutes for long-running LLM operations
});

export const graphApi = {
  getStats: () => api.get<GraphStats>('/graph/stats').then(r => r.data),
  getFullGraph: (limit = 500) => api.get<GraphData>(`/graph/full?limit=${limit}`).then(r => r.data),
  getSubgraph: (nodeId: string, label: string, key: string, depth = 2) =>
    api.get<GraphData>(`/graph/subgraph/${nodeId}?label=${label}&key=${key}&depth=${depth}`).then(r => r.data),
  search: (q: string, labels?: string) =>
    api.get(`/graph/search?q=${q}${labels ? `&labels=${labels}` : ''}`).then(r => r.data),
  semanticSearch: (q: string, k = 10, labels?: string) =>
    api.get<{ results: SemanticHit[]; count: number; error?: string }>(
      `/graph/semantic-search?q=${encodeURIComponent(q)}&k=${k}${labels ? `&labels=${labels}` : ''}`,
    ).then(r => r.data),
  embeddingsStatus: () =>
    api.get<EmbeddingStatus>('/graph/embeddings/status').then(r => r.data),
  embeddingsBackfill: (onlyMissing = true) =>
    api.post(`/graph/embeddings/backfill?only_missing=${onlyMissing}`).then(r => r.data),
  projection: (dim = 2, limit = 2000) =>
    api.get<ProjectionResult>(`/graph/embeddings/projection?dim=${dim}&limit=${limit}`).then(r => r.data),
  resolveEntities: (dryRun = false, labels?: string) =>
    api.post(`/graph/resolve-entities?dry_run=${dryRun}${labels ? `&labels=${labels}` : ''}`).then(r => r.data),
  crossCase: (label: string, value: string) =>
    api.get<CrossCaseResult>(`/graph/cross-case?label=${encodeURIComponent(label)}&value=${encodeURIComponent(value)}`).then(r => r.data),
  clear: () => api.delete('/graph/clear?confirm=yes-delete-all-data').then(r => r.data),
};

export const extractApi = {
  fromText: (text: string, sourceType = 'fir', storeInGraph = true, model?: string) =>
    api.post<ExtractionResult>('/extract/text', { text, source_type: sourceType, store_in_graph: storeInGraph, model: model || undefined }).then(r => r.data),
  fromImage: (formData: FormData) =>
    api.post<ExtractionResult>('/extract/image', formData, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data),
  combined: (formData: FormData) =>
    api.post<ExtractionResult>('/extract/combined', formData, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data),
};

export const queryApi = {
  natural: (question: string, caseId?: string, model?: string) =>
    api.post<QueryResponse>('/query/natural', { question, context_case_id: caseId, model: model || undefined }).then(r => r.data),
  cypher: (query: string, parameters?: Record<string, any>) =>
    api.post('/query/cypher', { query, parameters }).then(r => r.data),
};

export const datasetApi = {
  list: () => api.get<DatasetInfo[]>('/datasets/').then(r => r.data),
  generateFIR: (count: number, crimeTypes?: string[], includeBloodstain = true) =>
    api.post('/datasets/generate-fir', { count, crime_types: crimeTypes, include_bloodstain: includeBloodstain }).then(r => r.data),
  processBloodstain: (limit = 5) => api.post(`/datasets/process-bloodstain?limit=${limit}`).then(r => r.data),
  upload: (formData: FormData) => api.post('/datasets/upload', formData).then(r => r.data),
  ingestJudgments: (source = 'huggingface', limit = 20, query = 'murder FIR IPC', csvFilename?: string) =>
    api.post(`/datasets/ingest-judgments?source=${source}&limit=${limit}&query=${encodeURIComponent(query)}${csvFilename ? `&csv_filename=${csvFilename}` : ''}`).then(r => r.data),
  listJudgments: () => api.get('/datasets/judgments').then(r => r.data),
  generateGold: (caseId: string, model?: string) =>
    api.post(`/datasets/generate-gold/${caseId}${model ? `?model=${model}` : ''}`).then(r => r.data),
  generateGoldBatch: (limit = 20, model?: string) =>
    api.post(`/datasets/generate-gold-batch?limit=${limit}${model ? `&model=${model}` : ''}`).then(r => r.data),
  listGold: () => api.get('/datasets/gold').then(r => r.data),
  generateDocuments: (docType: string, count = 10) =>
    api.post(`/datasets/generate/${docType}?count=${count}`).then(r => r.data),
  getSources: () => api.get('/datasets/sources').then(r => r.data),
  downloadForensicData: (datasetType = 'all', limit = 20) =>
    api.post(`/datasets/download-forensic-data?dataset_type=${datasetType}&limit=${limit}`).then(r => r.data),
  getForensicDataStatus: () => api.get('/datasets/forensic-data-status').then(r => r.data),
  generateImages: (imageType: string, count = 20) =>
    api.post(`/datasets/generate-images/${imageType}?count=${count}`).then(r => r.data),
  processImages: (imageType: string, limit = 10) =>
    api.post(`/datasets/process-images?image_type=${imageType}&limit=${limit}`).then(r => r.data),
  getImageCounts: () =>
    api.get('/datasets/image-counts').then(r => r.data),
  processJudgments: (limit = 20) =>
    api.post(`/datasets/process-judgments?limit=${limit}`).then(r => r.data),
};

export const analysisApi = {
  hypothesis: (caseId: string, model?: string) =>
    api.post<AnalysisResult>(`/analysis/hypothesis/${caseId}${model ? `?model=${model}` : ''}`).then(r => r.data),
  statistics: (experimentId: string) => api.get(`/analysis/statistics/${experimentId}`).then(r => r.data),
};

export const evaluationApi = {
  runSingle: (docId: string, sourceType = 'fir', model?: string) =>
    api.post<EvaluationResult>(`/evaluation/run-single/${docId}?source_type=${sourceType}${model ? `&model=${model}` : ''}`).then(r => r.data),
  runAll: (sourceType = 'fir', model?: string) =>
    api.post<AggregateEvaluationResult>(`/evaluation/run-all?source_type=${sourceType}${model ? `&model=${model}` : ''}`).then(r => r.data),
  getResults: () =>
    api.get('/evaluation/results').then(r => r.data),
  imageTypes: () =>
    api.get<{ types: { image_type: string; attributes: { name: string; kind: string; note: string }[] }[] }>('/evaluation/image/types').then(r => r.data),
  runImage: (imageType: string, limit = 5, model?: string) =>
    api.post<ImageEvalResult>(`/evaluation/image/run?image_type=${imageType}&limit=${limit}${model ? `&model=${model}` : ''}`).then(r => r.data),
  listRealCases: () =>
    api.get<{ cases: { case_id: string; title: string; source: string; gold_persons: number }[] }>('/evaluation/real-cases/list').then(r => r.data),
  runRealCases: (model?: string) =>
    api.post<AggregateEvaluationResult>(`/evaluation/real-cases/run${model ? `?model=${model}` : ''}`).then(r => r.data),
};

export const collaborationApi = {
  addAnnotation: (entityType: string, entityId: string, text: string, author = 'anonymous') =>
    api.post('/collaboration/annotate', { entity_type: entityType, entity_id: entityId, text, author }).then(r => r.data),
  getAnnotations: (entityType: string, entityId: string) =>
    api.get(`/collaboration/annotations/${entityType}/${entityId}`).then(r => r.data),
  updateCaseStatus: (caseId: string, status: string, user = 'anonymous') =>
    api.patch(`/collaboration/case/${caseId}/status`, { status, user }).then(r => r.data),
  getActivityLog: (limit = 50) =>
    api.get(`/collaboration/activity-log?limit=${limit}`).then(r => r.data),
};

export const exportApi = {
  casesCsv: () => `${BACKEND_URL}/api/export/cases/csv`,
  graphJson: (limit = 2000) => `${BACKEND_URL}/api/export/graph/json?limit=${limit}`,
  caseReport: (caseId: string) => `${BACKEND_URL}/api/export/case/${caseId}/report`,
};
