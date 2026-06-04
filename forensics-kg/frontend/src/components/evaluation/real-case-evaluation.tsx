'use client';

import { useState } from 'react';
import { useRealCaseList, useRunRealCaseEval } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';
import type { AggregateEvaluationResult } from '@/types';

function pct(v: number) { return `${(v * 100).toFixed(0)}%`; }
function color(v: number) {
  return v >= 0.8 ? 'text-green-400' : v >= 0.6 ? 'text-yellow-400' : 'text-red-400';
}

export function RealCaseEvaluation() {
  const { toast } = useToast();
  const list = useRealCaseList();
  const run = useRunRealCaseEval();
  const selectedModel = useAppStore((s) => s.selectedModel);
  const [result, setResult] = useState<AggregateEvaluationResult | null>(null);

  const go = () => {
    run.mutate({ model: selectedModel }, {
      onSuccess: (d) => { setResult(d); toast(`Validated ${d.total_cases} real cases`, 'success'); },
      onError: (e: any) => toast(e.response?.data?.detail || 'Validation failed', 'error'),
    });
  };

  return (
    <div className="space-y-5">
      <p className="text-sm text-text-muted max-w-3xl">
        Scores extraction against <strong>human-curated gold from authoritative public sources</strong>
        {' '}(court records / Wikipedia) for well-documented real Indian criminal cases — not
        LLM-generated gold. This breaks the circular-evaluation problem and validates extraction on real data.
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-text-muted">{list.data?.cases.length ?? '…'} documented cases</span>
        <ModelSelector />
        <button
          type="button"
          onClick={go}
          disabled={run.isPending}
          className="px-5 py-2 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
        >
          {run.isPending && <Spinner size="sm" />}
          {run.isPending ? 'Extracting & scoring…' : 'Run real-case validation'}
        </button>
      </div>

      {run.isPending && (
        <p className="text-xs text-text-faint">Running extraction on each case narrative — calls the LLM per case, ~a minute.</p>
      )}

      {list.data && (
        <div className="flex flex-wrap gap-1.5">
          {list.data.cases.map((c) => (
            <a key={c.case_id} href={c.source} target="_blank" rel="noopener noreferrer"
               className="text-[11px] px-2 py-1 rounded-full bg-surface-hover text-text-muted hover:text-text-secondary">
              {c.title}
            </a>
          ))}
        </div>
      )}

      {result && (
        <>
          <div className="grid grid-cols-3 gap-3">
            {([['Precision', result.overall_precision], ['Recall', result.overall_recall], ['F1', result.overall_f1]] as const).map(([label, v]) => (
              <Card key={label} className="text-center">
                <div className="text-xs text-text-faint mb-1">Overall {label}</div>
                <div className={`text-3xl font-bold ${color(v)}`}>{pct(v)}</div>
              </Card>
            ))}
          </div>

          <Card>
            <h3 className="text-sm font-semibold text-text-secondary mb-3">By entity type (micro-averaged across cases)</h3>
            <table className="w-full text-sm">
              <thead><tr className="text-text-faint text-xs text-left">
                <th className="py-1">Entity</th><th>P</th><th>R</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th>
              </tr></thead>
              <tbody>
                {Object.values(result.per_entity_type).map((m) => (
                  <tr key={m.entity_type} className="border-t border-border-default/40">
                    <td className="py-1.5 text-text-secondary">{m.entity_type}</td>
                    <td>{pct(m.precision)}</td><td>{pct(m.recall)}</td>
                    <td className={color(m.f1)}>{pct(m.f1)}</td>
                    <td className="text-text-faint">{m.true_positives}</td>
                    <td className="text-text-faint">{m.false_positives}</td>
                    <td className="text-text-faint">{m.false_negatives}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[11px] text-text-faint mt-2">
              Persons / locations / weapons use fuzzy string matching; crime type uses canonical-keyword
              matching against the case title; timeline events use date-aware matching (year/month) so
              descriptive event names are fairly compared to the gold&apos;s documented dates.
            </p>
          </Card>

          <Card>
            <h3 className="text-sm font-semibold text-text-secondary mb-3">Per case</h3>
            <div className="space-y-1.5">
              {[...result.per_case].sort((a, b) => b.aggregate_f1 - a.aggregate_f1).map((r) => (
                <div key={r.fir_id} className="flex items-center gap-3 text-xs">
                  <span className="font-mono text-text-faint w-36 truncate">{r.fir_id}</span>
                  <div className="flex-1 bg-surface-input rounded-full h-2 overflow-hidden">
                    <div className="h-full bg-[var(--primary)]" style={{ width: `${r.aggregate_f1 * 100}%` }} />
                  </div>
                  <span className={`w-10 text-right ${color(r.aggregate_f1)}`}>{pct(r.aggregate_f1)}</span>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
