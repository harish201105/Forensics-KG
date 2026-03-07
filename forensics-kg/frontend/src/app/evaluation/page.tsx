'use client';

import { useState } from 'react';
import { useRunEvaluationAll, useRunEvaluationSingle } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { AggregateEvaluationResult, EvaluationResult, EntityTypeMetrics } from '@/types';

function MetricCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="bg-surface-inset rounded-lg p-4 text-center">
      <p className="text-xs text-text-faint mb-1">{label}</p>
      <p className="text-3xl font-bold" style={{ color }}>
        {(value * 100).toFixed(1)}%
      </p>
    </div>
  );
}

const SOURCE_TYPES = [
  { value: 'fir', label: 'FIR Reports' },
  { value: 'court_judgment', label: 'Court Judgments' },
  { value: 'postmortem', label: 'Post-Mortem Reports' },
  { value: 'lab_report', label: 'Lab Reports' },
  { value: 'witness_deposition', label: 'Witness Depositions' },
  { value: 'soco_report', label: 'SOCO Reports' },
];

export default function EvaluationPage() {
  const [results, setResults] = useState<AggregateEvaluationResult | null>(null);
  const [singleResult, setSingleResult] = useState<EvaluationResult | null>(null);
  const [singleDocId, setSingleDocId] = useState('FIR-2024-0001');
  const [sourceType, setSourceType] = useState('fir');
  const selectedModel = useAppStore((s) => s.selectedModel);
  const { toast } = useToast();

  const runAllMutation = useRunEvaluationAll();
  const runSingleMutation = useRunEvaluationSingle();

  const handleRunAll = () => {
    runAllMutation.mutate({ sourceType, model: selectedModel }, {
      onSuccess: (data) => {
        setResults(data);
        setSingleResult(null);
        toast(`Evaluation complete: F1=${(data.overall_f1 * 100).toFixed(1)}%`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || 'Evaluation failed', 'error'),
    });
  };

  const handleRunSingle = () => {
    if (!singleDocId.trim()) return;
    runSingleMutation.mutate({ docId: singleDocId, sourceType, model: selectedModel }, {
      onSuccess: (data) => {
        setSingleResult(data);
        toast(`${data.fir_id}: F1=${(data.aggregate_f1 * 100).toFixed(1)}%`, 'success');
      },
      onError: (e: any) => toast(e.response?.data?.detail || 'Evaluation failed', 'error'),
    });
  };

  const isLoading = runAllMutation.isPending || runSingleMutation.isPending;

  // Prepare chart data from per_entity_type
  const chartData = results
    ? Object.values(results.per_entity_type).map((m: EntityTypeMetrics) => ({
        name: m.entity_type,
        Precision: +(m.precision * 100).toFixed(1),
        Recall: +(m.recall * 100).toFixed(1),
        F1: +(m.f1 * 100).toFixed(1),
      }))
    : [];

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Evaluation Framework</h1>
      <p className="text-text-muted mb-6">
        Compare LLM extraction against gold standard annotations (precision / recall / F1)
      </p>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-8">
        <select
          value={sourceType}
          onChange={(e) => setSourceType(e.target.value)}
          className="bg-[var(--card)] border border-border-default rounded-lg px-3 py-2 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50"
          aria-label="Source type for evaluation"
        >
          {SOURCE_TYPES.map((st) => (
            <option key={st.value} value={st.value}>{st.label}</option>
          ))}
        </select>
        <ModelSelector />
        <button
          type="button"
          onClick={handleRunAll}
          disabled={isLoading}
          className="px-6 py-2.5 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-[var(--primary)]/80 flex items-center gap-2"
        >
          {runAllMutation.isPending && <Spinner size="sm" />}
          {runAllMutation.isPending ? 'Evaluating all...' : `Run All (${SOURCE_TYPES.find(s => s.value === sourceType)?.label || 'Documents'})`}
        </button>
        <div className="flex gap-2 items-center">
          <input
            value={singleDocId}
            onChange={(e) => setSingleDocId(e.target.value)}
            placeholder="Document ID"
            aria-label="Document ID for evaluation"
            className="bg-[var(--card)] border border-border-default rounded-lg px-3 py-2 text-sm text-text-body w-44 focus:outline-none focus:border-[var(--primary)]/50"
          />
          <button
            type="button"
            onClick={handleRunSingle}
            disabled={isLoading || !singleDocId.trim()}
            className="px-4 py-2.5 bg-surface-active text-text-body rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-surface-elevated flex items-center gap-2"
          >
            {runSingleMutation.isPending && <Spinner size="sm" />}
            Run Single
          </button>
        </div>
      </div>

      {(runAllMutation.error || runSingleMutation.error) && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400 mb-6">
          {(runAllMutation.error as any)?.response?.data?.detail || (runSingleMutation.error as any)?.response?.data?.detail || 'Evaluation failed'}
        </div>
      )}

      {/* Aggregate Results */}
      {results && (
        <div className="space-y-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <MetricCard label="Overall Precision" value={results.overall_precision} color="#3b82f6" />
            <MetricCard label="Overall Recall" value={results.overall_recall} color="#22c55e" />
            <MetricCard label="Overall F1" value={results.overall_f1} color="#f59e0b" />
            <div className="bg-surface-inset rounded-lg p-4 text-center">
              <p className="text-xs text-text-faint mb-1">Cases Evaluated</p>
              <p className="text-3xl font-bold text-text-body">{results.total_cases}</p>
              <p className="text-xs text-text-ghost mt-1">Model: {results.model_used}</p>
            </div>
          </div>

          {/* Per Entity Type Chart */}
          <Card>
            <CardTitle>Per-Entity-Type Metrics</CardTitle>
            <div className="h-80 mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
                  <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 12 }} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'var(--card)', border: '1px solid var(--border-default)', borderRadius: '8px' }}
                    labelStyle={{ color: 'var(--text-muted)' }}
                  />
                  <Legend />
                  <Bar dataKey="Precision" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="Recall" fill="#22c55e" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="F1" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Per Entity Type Table */}
          <Card>
            <CardTitle>Detailed Metrics by Entity Type</CardTitle>
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border-default text-left text-text-faint text-xs">
                    <th className="pb-2 pr-4">Entity Type</th>
                    <th className="pb-2 pr-4">TP</th>
                    <th className="pb-2 pr-4">FP</th>
                    <th className="pb-2 pr-4">FN</th>
                    <th className="pb-2 pr-4">Precision</th>
                    <th className="pb-2 pr-4">Recall</th>
                    <th className="pb-2">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.values(results.per_entity_type).map((m: EntityTypeMetrics) => (
                    <tr key={m.entity_type} className="border-b border-border-subtle text-text-secondary">
                      <td className="py-2 pr-4 font-medium">{m.entity_type}</td>
                      <td className="py-2 pr-4 text-green-400">{m.true_positives}</td>
                      <td className="py-2 pr-4 text-red-400">{m.false_positives}</td>
                      <td className="py-2 pr-4 text-yellow-400">{m.false_negatives}</td>
                      <td className="py-2 pr-4">{(m.precision * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-4">{(m.recall * 100).toFixed(1)}%</td>
                      <td className="py-2 font-semibold">{(m.f1 * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Per Case Breakdown */}
          <Card>
            <CardTitle>Per-Case Results</CardTitle>
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border-default text-left text-text-faint text-xs">
                    <th className="pb-2 pr-4">Document ID</th>
                    <th className="pb-2 pr-4">Gold Entities</th>
                    <th className="pb-2 pr-4">Extracted</th>
                    <th className="pb-2 pr-4">Precision</th>
                    <th className="pb-2 pr-4">Recall</th>
                    <th className="pb-2 pr-4">F1</th>
                    <th className="pb-2">Time (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {results.per_case.map((c: EvaluationResult) => (
                    <tr key={c.fir_id} className="border-b border-border-subtle text-text-secondary">
                      <td className="py-2 pr-4 font-mono text-xs text-[var(--primary)]">{c.fir_id}</td>
                      <td className="py-2 pr-4">{c.gold_entity_count}</td>
                      <td className="py-2 pr-4">{c.extracted_entity_count}</td>
                      <td className="py-2 pr-4">{(c.aggregate_precision * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-4">{(c.aggregate_recall * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-4 font-semibold">
                        <Badge variant={c.aggregate_f1 > 0.7 ? 'success' : c.aggregate_f1 > 0.4 ? 'warning' : 'error'}>
                          {(c.aggregate_f1 * 100).toFixed(1)}%
                        </Badge>
                      </td>
                      <td className="py-2 text-text-faint">{c.extraction_time_ms.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Single FIR Result */}
      {singleResult && !results && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <MetricCard label="Precision" value={singleResult.aggregate_precision} color="#3b82f6" />
            <MetricCard label="Recall" value={singleResult.aggregate_recall} color="#22c55e" />
            <MetricCard label="F1" value={singleResult.aggregate_f1} color="#f59e0b" />
            <div className="bg-surface-inset rounded-lg p-4 text-center">
              <p className="text-xs text-text-faint mb-1">{singleResult.fir_id}</p>
              <p className="text-lg font-bold text-text-body">
                {singleResult.gold_entity_count} gold / {singleResult.extracted_entity_count} extracted
              </p>
              <p className="text-xs text-text-ghost mt-1">{singleResult.extraction_time_ms.toFixed(0)}ms</p>
            </div>
          </div>

          <Card>
            <CardTitle>Entity Metrics for {singleResult.fir_id}</CardTitle>
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border-default text-left text-text-faint text-xs">
                    <th className="pb-2 pr-4">Type</th>
                    <th className="pb-2 pr-4">TP</th>
                    <th className="pb-2 pr-4">FP</th>
                    <th className="pb-2 pr-4">FN</th>
                    <th className="pb-2 pr-4">P</th>
                    <th className="pb-2 pr-4">R</th>
                    <th className="pb-2 pr-4">F1</th>
                    <th className="pb-2">Matches</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.values(singleResult.entity_metrics).map((m: EntityTypeMetrics) => (
                    <tr key={m.entity_type} className="border-b border-border-subtle text-text-secondary">
                      <td className="py-2 pr-4 font-medium">{m.entity_type}</td>
                      <td className="py-2 pr-4 text-green-400">{m.true_positives}</td>
                      <td className="py-2 pr-4 text-red-400">{m.false_positives}</td>
                      <td className="py-2 pr-4 text-yellow-400">{m.false_negatives}</td>
                      <td className="py-2 pr-4">{(m.precision * 100).toFixed(0)}%</td>
                      <td className="py-2 pr-4">{(m.recall * 100).toFixed(0)}%</td>
                      <td className="py-2 pr-4 font-semibold">{(m.f1 * 100).toFixed(0)}%</td>
                      <td className="py-2">
                        {m.matched_pairs.map((p, i) => (
                          <div key={i} className="text-xs text-text-faint">
                            &quot;{p.gold.slice(0, 25)}&quot; ↔ &quot;{p.extracted.slice(0, 25)}&quot;
                            <span className="text-text-whisper ml-1">({(p.similarity * 100).toFixed(0)}%)</span>
                          </div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {!results && !singleResult && !isLoading && (
        <Card>
          <p className="text-sm text-text-faint text-center py-12">
            Select a source type and model, then click &quot;Run All&quot; to evaluate extraction across gold standard documents,
            or enter a specific document ID and click &quot;Run Single&quot; for detailed analysis.
          </p>
        </Card>
      )}
    </div>
  );
}
