import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { graphApi, extractApi, queryApi, datasetApi, analysisApi, evaluationApi, collaborationApi } from './api';

// ─── Graph Queries ──────────────────────────────────────────────────
export function useGraphStats() {
  return useQuery({
    queryKey: ['graphStats'],
    queryFn: graphApi.getStats,
  });
}

export function useFullGraph(limit = 500) {
  return useQuery({
    queryKey: ['fullGraph', limit],
    queryFn: () => graphApi.getFullGraph(limit),
  });
}

export function useCases() {
  return useQuery({
    queryKey: ['cases'],
    queryFn: () =>
      queryApi.cypher(
        'MATCH (c:Case) RETURN properties(c) as props ORDER BY c.case_id LIMIT 200'
      ),
  });
}

export function useExperiments() {
  return useQuery({
    queryKey: ['experiments'],
    queryFn: () =>
      queryApi.cypher(
        'MATCH (e:Experiment) RETURN properties(e) as props ORDER BY e.experiment_id LIMIT 200'
      ),
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: ['datasets'],
    queryFn: datasetApi.list,
  });
}

// ─── Mutations ──────────────────────────────────────────────────────
export function useExtractText() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ text, sourceType, store, model }: { text: string; sourceType?: string; store?: boolean; model?: string }) =>
      extractApi.fromText(text, sourceType, store, model),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['graphStats'] });
      qc.invalidateQueries({ queryKey: ['fullGraph'] });
    },
  });
}

export function useExtractImage() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (formData: FormData) => extractApi.fromImage(formData),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['graphStats'] });
      qc.invalidateQueries({ queryKey: ['fullGraph'] });
    },
  });
}

export function useGenerateFIR() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ count, crimeTypes, includeBloodstain }: {
      count: number;
      crimeTypes?: string[];
      includeBloodstain?: boolean;
    }) => datasetApi.generateFIR(count, crimeTypes, includeBloodstain),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
      qc.invalidateQueries({ queryKey: ['graphStats'] });
    },
  });
}

export function useProcessBloodstain() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (limit: number) => datasetApi.processBloodstain(limit),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
      qc.invalidateQueries({ queryKey: ['graphStats'] });
    },
  });
}

export function useIngestJudgments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ source, limit, query, csvFilename }: {
      source?: string; limit?: number; query?: string; csvFilename?: string;
    }) => datasetApi.ingestJudgments(source, limit, query, csvFilename),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

export function useGenerateGold() {
  return useMutation({
    mutationFn: ({ caseId, model }: { caseId: string; model?: string }) =>
      datasetApi.generateGold(caseId, model),
  });
}

export function useGenerateGoldBatch() {
  return useMutation({
    mutationFn: ({ limit, model }: { limit?: number; model?: string } = {}) =>
      datasetApi.generateGoldBatch(limit, model),
  });
}

export function useGenerateDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ docType, count }: { docType: string; count?: number }) =>
      datasetApi.generateDocuments(docType, count),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

export function useDownloadForensicData() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ datasetType, limit }: { datasetType?: string; limit?: number } = {}) =>
      datasetApi.downloadForensicData(datasetType, limit),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

export function useGenerateImages() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ imageType, count }: { imageType: string; count?: number }) =>
      datasetApi.generateImages(imageType, count),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
    },
  });
}

export function useProcessImages() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ imageType, limit }: { imageType: string; limit?: number }) =>
      datasetApi.processImages(imageType, limit),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
      qc.invalidateQueries({ queryKey: ['graphStats'] });
      qc.invalidateQueries({ queryKey: ['fullGraph'] });
      qc.invalidateQueries({ queryKey: ['imageCounts'] });
    },
  });
}

export function useImageCounts() {
  return useQuery({
    queryKey: ['imageCounts'],
    queryFn: datasetApi.getImageCounts,
  });
}

export function useProcessJudgments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (limit: number) => datasetApi.processJudgments(limit),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['datasets'] });
      qc.invalidateQueries({ queryKey: ['graphStats'] });
      qc.invalidateQueries({ queryKey: ['fullGraph'] });
    },
  });
}

export function useNaturalQuery() {
  return useMutation({
    mutationFn: ({ question, caseId, model }: { question: string; caseId?: string; model?: string }) =>
      queryApi.natural(question, caseId, model),
  });
}

export function useCypherQuery() {
  return useMutation({
    mutationFn: ({ query, parameters }: { query: string; parameters?: Record<string, any> }) =>
      queryApi.cypher(query, parameters),
  });
}

export function useAnalyzeCase() {
  return useMutation({
    mutationFn: ({ caseId, model }: { caseId: string; model?: string }) =>
      analysisApi.hypothesis(caseId, model),
  });
}

export function useStatistics() {
  return useMutation({
    mutationFn: (experimentId: string) => analysisApi.statistics(experimentId),
  });
}

// ─── Evaluation ────────────────────────────────────────────────────
export function useRunEvaluationAll() {
  return useMutation({
    mutationFn: ({ sourceType, model }: { sourceType?: string; model?: string } = {}) =>
      evaluationApi.runAll(sourceType, model),
  });
}

export function useRunEvaluationSingle() {
  return useMutation({
    mutationFn: ({ docId, sourceType, model }: { docId: string; sourceType?: string; model?: string }) =>
      evaluationApi.runSingle(docId, sourceType, model),
  });
}

export function useEvaluationResults() {
  return useQuery({
    queryKey: ['evaluationResults'],
    queryFn: evaluationApi.getResults,
    enabled: false, // only fetch on demand
  });
}

// ─── Combined Extraction ────────────────────────────────────────────
export function useExtractCombined() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (formData: FormData) => extractApi.combined(formData),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['graphStats'] });
      qc.invalidateQueries({ queryKey: ['fullGraph'] });
    },
  });
}

// ─── Collaboration ──────────────────────────────────────────────────
export function useAddAnnotation() {
  return useMutation({
    mutationFn: ({ entityType, entityId, text, author }: {
      entityType: string; entityId: string; text: string; author?: string;
    }) => collaborationApi.addAnnotation(entityType, entityId, text, author),
  });
}

export function useAnnotations(entityType: string, entityId: string) {
  return useQuery({
    queryKey: ['annotations', entityType, entityId],
    queryFn: () => collaborationApi.getAnnotations(entityType, entityId),
    enabled: !!entityType && !!entityId,
  });
}

export function useUpdateCaseStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ caseId, status, user }: { caseId: string; status: string; user?: string }) =>
      collaborationApi.updateCaseStatus(caseId, status, user),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['cases'] });
    },
  });
}

export function useActivityLog(limit = 50) {
  return useQuery({
    queryKey: ['activityLog', limit],
    queryFn: () => collaborationApi.getActivityLog(limit),
  });
}
