export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
}
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
}
export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
}
export interface ExtractionResult {
  entities: EntityExtraction[];
  relationships: RelationshipExtraction[];
  metadata: Record<string, any>;
  processing_time_ms: number;
}
export interface EntityExtraction {
  entity_type: string;
  entity_id: string;
  properties: Record<string, any>;
  confidence?: number;
}
export interface RelationshipExtraction {
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  properties: Record<string, any>;
  confidence: number;
  evidence_span?: string;
}
export interface QueryResponse {
  answer: string;
  confidence: number;
  sources: Record<string, any>[];
  cypher_query?: string;
  graph_context?: GraphData;
  reasoning?: string;
}
export interface GraphStats {
  node_counts: Record<string, number>;
  relationship_counts: Record<string, number>;
  total_nodes: number;
  total_relationships: number;
}
export interface DatasetInfo {
  dataset_id: string;
  name: string;
  type: string;
  record_count: number;
  loaded: boolean;
  metadata: Record<string, any>;
}
export interface AnalysisResult {
  primary_hypothesis: Hypothesis;
  alternative_hypotheses: Hypothesis[];
  statistical_analysis: Record<string, any>;
  recommendations: string[];
  reasoning_chain: string[];
  confidence: number;
}
export interface Hypothesis {
  hypothesis_id: string;
  description: string;
  mechanism: string;
  confidence: number;
  supporting_evidence?: string[];
  contradicting_evidence?: string[];
  likelihood_ratio?: number;
}
// Evaluation types
export interface MatchedPair {
  gold: string;
  extracted: string;
  similarity: number;
}
export interface EntityTypeMetrics {
  entity_type: string;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  precision: number;
  recall: number;
  f1: number;
  matched_pairs: MatchedPair[];
}
export interface EvaluationResult {
  fir_id: string;
  entity_metrics: Record<string, EntityTypeMetrics>;
  aggregate_precision: number;
  aggregate_recall: number;
  aggregate_f1: number;
  extraction_time_ms: number;
  gold_entity_count: number;
  extracted_entity_count: number;
  model_used: string;
}
export interface AggregateEvaluationResult {
  total_cases: number;
  per_case: EvaluationResult[];
  per_entity_type: Record<string, EntityTypeMetrics>;
  overall_precision: number;
  overall_recall: number;
  overall_f1: number;
  model_used: string;
}
// Collaboration types
export interface Annotation {
  annotation_id: string;
  text: string;
  author: string;
  created_at: string;
}
export interface ActivityLogEntry {
  log_id: string;
  action: string;
  details: string;
  user: string;
  timestamp: string;
}
// Node type color mapping
export const NODE_COLORS: Record<string, string> = {
  Case: '#3b82f6',
  Person: '#22c55e',
  Location: '#a855f7',
  Evidence: '#14b8a6',
  Weapon: '#dc2626',
  Vehicle: '#f97316',
  CrimeType: '#eab308',
  TimeEvent: '#06b6d4',
  BloodstainPattern: '#be123c',
  Stain: '#ec4899',
  Experiment: '#6b7280',
  Hypothesis: '#f59e0b',
  ImpactMechanismNode: '#92400e',
  // Court judgment types
  LegalSection: '#7c3aed',
  Verdict: '#059669',
  CourtOrder: '#0284c7',
  // Post-mortem types
  InjuryPattern: '#dc2626',
  CauseOfDeath: '#991b1b',
  ToxicologyResult: '#ca8a04',
  OrganFinding: '#b45309',
  // Lab report types
  Sample: '#0d9488',
  TestResult: '#0891b2',
  DNAProfile: '#6d28d9',
  ChemicalCompound: '#a16207',
  // Deposition types
  Statement: '#4f46e5',
  // SOCO types
  CrimeScene: '#9333ea',
  PhysicalEvidence: '#0e7490',
  ChainOfCustody: '#64748b',
  // Image analysis types
  FingerprintPattern: '#2563eb',
  MinutiaePoint: '#3b82f6',
  RidgeDetail: '#60a5fa',
  WoundPattern: '#e11d48',
  TissueAnalysis: '#f43f5e',
  BallisticsPattern: '#78350f',
  HandwritingFeature: '#854d0e',
  InkAnalysis: '#713f12',
  ForgeryIndicator: '#ef4444',
  ToolMarkPattern: '#57534e',
};
