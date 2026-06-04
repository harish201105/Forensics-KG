'use client';

import { useState } from 'react';
import { useAnalyzeCase, useStatistics, useCases, useExperiments } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';
import type { AnalysisResult } from '@/types';

function StatTable({ title, data }: { title: string; data: Record<string, Record<string, number | boolean | string>> }) {
  const features = Object.keys(data);
  if (features.length === 0) return null;
  const columns = Object.keys(data[features[0]]);

  return (
    <div className="mb-4">
      <h4 className="text-xs font-semibold text-text-secondary mb-2">{title}</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="text-left py-1.5 px-2 text-text-faint font-medium">Feature</th>
              {columns.map((col) => (
                <th key={col} className="text-right py-1.5 px-2 text-text-faint font-medium">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feat) => (
              <tr key={feat} className="border-b border-border-subtle/50">
                <td className="py-1.5 px-2 text-text-tertiary font-mono">{feat}</td>
                {columns.map((col) => {
                  const val = data[feat][col];
                  const display = typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(4)) :
                    typeof val === 'boolean' ? (val ? 'Yes' : 'No') : String(val);
                  return (
                    <td key={col} className={`text-right py-1.5 px-2 font-mono ${typeof val === 'boolean' ? (val ? 'text-green-400' : 'text-red-400') : 'text-text-body'}`}>
                      {display}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CorrelationMatrix({ data }: { data: { features: string[]; values: number[][] } }) {
  const { features, values } = data;
  return (
    <div className="mb-4">
      <h4 className="text-xs font-semibold text-text-secondary mb-2">Correlation Matrix</h4>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="py-1.5 px-2" />
              {features.map((f) => (
                <th key={f} className="text-center py-1.5 px-2 text-text-faint font-mono">{f}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((rowFeat, ri) => (
              <tr key={rowFeat}>
                <td className="py-1.5 px-2 text-text-faint font-mono">{rowFeat}</td>
                {values[ri].map((val, ci) => {
                  const abs = Math.abs(val);
                  const color = ri === ci ? 'text-text-ghost' :
                    abs > 0.7 ? 'text-green-400' :
                    abs > 0.4 ? 'text-yellow-400' : 'text-text-muted';
                  return (
                    <td key={ci} className={`text-center py-1.5 px-2 font-mono ${color}`}>
                      {val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatisticalAnalysisCard({ data, maxHeight = 'max-h-none' }: { data: Record<string, any>; maxHeight?: string }) {
  if (!data || Object.keys(data).length === 0) return null;

  const { pattern_statistics, feature_distributions, confidence_intervals, correlation_matrix, ...rest } = data;
  const hasExtra = Object.keys(rest).length > 0;

  return (
    <>
      {pattern_statistics && Object.keys(pattern_statistics).length > 0 && (
        <StatTable title="Pattern Statistics" data={pattern_statistics} />
      )}
      {feature_distributions && Object.keys(feature_distributions).length > 0 && (
        <StatTable title="Feature Distributions" data={feature_distributions} />
      )}
      {confidence_intervals && Object.keys(confidence_intervals).length > 0 && (
        <StatTable title="95% Confidence Intervals" data={confidence_intervals} />
      )}
      {correlation_matrix && correlation_matrix.features?.length > 0 && (
        <CorrelationMatrix data={correlation_matrix} />
      )}
      {hasExtra && (
        <details className="mt-2">
          <summary className="text-xs text-text-ghost cursor-pointer">Raw data</summary>
          <pre className="text-xs bg-surface-input rounded p-3 overflow-auto max-h-48 text-text-muted mt-1">
            {JSON.stringify(rest, null, 2)}
          </pre>
        </details>
      )}
    </>
  );
}

export default function AnalysisPage() {
  const [selectedId, setSelectedId] = useState('');
  const [mode, setMode] = useState<'hypothesis' | 'statistics'>('hypothesis');
  const { toast } = useToast();
  const selectedModel = useAppStore((s) => s.selectedModel);

  const { data: casesData } = useCases();
  const { data: experimentsData } = useExperiments();
  const analyzeMutation = useAnalyzeCase();
  const statsMutation = useStatistics();

  const cases = (casesData?.results || []).map((r: any) => ({
    case_id: r.props?.case_id || '',
    title: r.props?.title || r.props?.case_id || 'Untitled',
  }));

  const experiments = (experimentsData?.results || []).map((r: any) => ({
    experiment_id: r.props?.experiment_id || '',
    description: r.props?.description || r.props?.experiment_id || 'Unnamed',
  }));

  const handleAnalyze = () => {
    if (!selectedId.trim()) return;
    if (mode === 'hypothesis') {
      analyzeMutation.mutate({ caseId: selectedId, model: selectedModel }, {
        onSuccess: () => toast('Hypothesis generated', 'success'),
        onError: (e: any) => toast(e.response?.data?.detail || 'Analysis failed', 'error'),
      });
    } else {
      statsMutation.mutate(selectedId, {
        onSuccess: () => toast('Statistics computed', 'success'),
        onError: (e: any) => toast(e.response?.data?.detail || 'Statistics failed', 'error'),
      });
    }
  };

  const isLoading = analyzeMutation.isPending || statsMutation.isPending;
  const result: AnalysisResult | undefined = analyzeMutation.data;
  const statsResult = statsMutation.data;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Forensic Analysis</h1>
      <p className="text-text-muted mb-6">Generate hypotheses and perform statistical analysis on case evidence</p>

      {/* Input */}
      <div className="flex flex-wrap gap-3 mb-8">
        <select
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          aria-label={mode === 'hypothesis' ? 'Select case' : 'Select experiment'}
          className="bg-[var(--card)] border border-border-default rounded-lg px-4 py-3 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50 min-w-[250px]"
        >
          {mode === 'hypothesis' ? (
            <>
              <option value="">Select a case...</option>
              {cases.map((c: any) => (
                <option key={c.case_id} value={c.case_id}>{c.title} ({c.case_id})</option>
              ))}
            </>
          ) : (
            <>
              <option value="">Select an experiment or case...</option>
              {experiments.map((e: any) => (
                <option key={e.experiment_id} value={e.experiment_id}>{e.description} ({e.experiment_id})</option>
              ))}
              {cases.length > 0 && <option disabled>--- Cases ---</option>}
              {cases.map((c: any) => (
                <option key={`stat-${c.case_id}`} value={c.case_id}>{c.title} ({c.case_id})</option>
              ))}
            </>
          )}
        </select>
        <input
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
          placeholder={mode === 'hypothesis' ? '...or enter Case ID' : '...or enter Experiment/Case ID'}
          aria-label={mode === 'hypothesis' ? 'Case ID' : 'Experiment ID'}
          className="flex-1 max-w-xs bg-[var(--card)] border border-border-default rounded-lg px-4 py-3 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50"
        />
        <div className="flex gap-1">
          {(['hypothesis', 'statistics'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => { setMode(m); setSelectedId(''); }}
              className={`px-3 py-3 rounded-lg text-sm ${mode === m ? 'bg-surface-active text-text-body' : 'text-text-faint'}`}
            >
              {m === 'hypothesis' ? 'Hypothesis' : 'Statistics'}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={isLoading || !selectedId.trim()}
          className="px-6 py-3 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
        >
          {isLoading && <Spinner size="sm" />}
          {isLoading ? 'Analyzing...' : mode === 'hypothesis' ? 'Generate Hypothesis' : 'Compute Statistics'}
        </button>
        {mode === 'hypothesis' && <ModelSelector />}
      </div>

      {(analyzeMutation.error || statsMutation.error) && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400 mb-6">
          {(analyzeMutation.error as any)?.response?.data?.detail || (statsMutation.error as any)?.response?.data?.detail || 'Analysis failed'}
        </div>
      )}

      {/* Hypothesis Results */}
      {result && (
        <div className="space-y-4">
          {result.primary_hypothesis && (
            <Card>
              <CardTitle className="text-[#f59e0b] mb-3">Primary Hypothesis</CardTitle>
              <p className="text-sm text-text-body mb-3">{result.primary_hypothesis.description}</p>
              <div className="flex flex-wrap gap-3 mb-3">
                <Badge variant="info">{result.primary_hypothesis.mechanism}</Badge>
                <Badge variant={(result.primary_hypothesis.confidence ?? 0) > 0.7 ? 'success' : (result.primary_hypothesis.confidence ?? 0) > 0.4 ? 'warning' : 'error'}>
                  {((result.primary_hypothesis.confidence ?? 0) * 100).toFixed(0)}% confidence
                </Badge>
                {result.primary_hypothesis.likelihood_ratio && (
                  <Badge>LR: {result.primary_hypothesis.likelihood_ratio.toFixed(2)}</Badge>
                )}
              </div>
              {result.primary_hypothesis.supporting_evidence && result.primary_hypothesis.supporting_evidence.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs text-green-400/70 mb-1">Supporting Evidence:</p>
                  <ul className="text-xs text-text-muted list-disc list-inside">
                    {result.primary_hypothesis.supporting_evidence.map((e, i) => <li key={i}>{e}</li>)}
                  </ul>
                </div>
              )}
              {result.primary_hypothesis.contradicting_evidence && result.primary_hypothesis.contradicting_evidence.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-red-400/70 mb-1">Contradicting Evidence:</p>
                  <ul className="text-xs text-text-muted list-disc list-inside">
                    {result.primary_hypothesis.contradicting_evidence.map((e, i) => <li key={i}>{e}</li>)}
                  </ul>
                </div>
              )}
            </Card>
          )}

          {result.alternative_hypotheses?.length > 0 && (
            <Card>
              <CardTitle>Alternative Hypotheses</CardTitle>
              <div className="space-y-3 mt-3">
                {result.alternative_hypotheses.map((h, i) => (
                  <div key={i} className="bg-surface-inset rounded p-3">
                    <p className="text-sm text-text-secondary">{h.description}</p>
                    <div className="flex gap-2 mt-2">
                      <Badge>{h.mechanism}</Badge>
                      <Badge variant={(h.confidence ?? 0) > 0.5 ? 'warning' : 'default'}>
                        {((h.confidence ?? 0) * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {result.reasoning_chain?.length > 0 && (
            <Card>
              <CardTitle>Reasoning Chain</CardTitle>
              <ol className="list-decimal list-inside space-y-1.5 mt-3">
                {result.reasoning_chain.map((step, i) => (
                  <li key={i} className="text-xs text-text-tertiary">{step}</li>
                ))}
              </ol>
            </Card>
          )}

          {result.recommendations?.length > 0 && (
            <Card>
              <CardTitle>Recommendations</CardTitle>
              <ul className="space-y-1.5 mt-3">
                {result.recommendations.map((r, i) => (
                  <li key={i} className="text-xs text-text-tertiary flex gap-2">
                    <span className="text-[var(--primary)]">&bull;</span> {r}
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {result.statistical_analysis && Object.keys(result.statistical_analysis).length > 0 && (
            <Card>
              <CardTitle>Statistical Analysis</CardTitle>
              <div className="mt-3">
                <StatisticalAnalysisCard data={result.statistical_analysis} />
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Statistics Results */}
      {statsResult && !result && (
        <Card>
          <CardTitle>Statistical Analysis</CardTitle>
          <div className="mt-3">
            <StatisticalAnalysisCard data={statsResult} />
          </div>
        </Card>
      )}
    </div>
  );
}
