'use client';

import { useState } from 'react';
import { useImageEvalTypes, useRunImageEval } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';
import type { ImageEvalResult, ImageEvalAttr } from '@/types';

const LABELS: Record<string, string> = {
  bloodstain: 'Bloodstain', fingerprint: 'Fingerprint', wound: 'Wound',
  ballistics: 'Ballistics', document_forensics: 'Document Forensics', tool_marks: 'Tool Marks',
};

function pct(v?: number | null) {
  return v == null ? '—' : `${(v * 100).toFixed(0)}%`;
}
function scoreColor(v?: number | null) {
  if (v == null) return 'text-text-faint';
  if (v >= 0.7) return 'text-green-400';
  if (v >= 0.4) return 'text-yellow-400';
  return 'text-red-400';
}

function AttrMetric({ name, m }: { name: string; m: ImageEvalAttr }) {
  return (
    <div className="bg-surface-input rounded-lg p-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-text-secondary">{name}</span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-hover text-text-faint">{m.kind}</span>
      </div>
      {m.note && <p className="text-[11px] text-text-faint mb-2">{m.note}</p>}
      {m.kind === 'numeric' ? (
        <div className="text-xs text-text-muted">
          MAE <span className="font-mono text-text-secondary">{m.mae ?? '—'}</span> ·
          within ±{m.tolerance}: <span className={scoreColor(m.within_tolerance)}>{pct(m.within_tolerance)}</span>
        </div>
      ) : m.kind === 'binary' ? (
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
          <span>acc <span className={scoreColor(m.accuracy)}>{pct(m.accuracy)}</span></span>
          <span>P <span className="text-text-secondary">{pct(m.precision)}</span></span>
          <span>R <span className="text-text-secondary">{pct(m.recall)}</span></span>
          <span>F1 <span className={scoreColor(m.f1)}>{pct(m.f1)}</span></span>
        </div>
      ) : (
        <div className="text-xs">accuracy <span className={`text-base font-bold ${scoreColor(m.accuracy)}`}>{pct(m.accuracy)}</span></div>
      )}
      <div className="text-[10px] text-text-faint mt-1">n={m.samples}</div>
    </div>
  );
}

export function ImageEvaluation() {
  const { toast } = useToast();
  const types = useImageEvalTypes();
  const runEval = useRunImageEval();
  const selectedModel = useAppStore((s) => s.selectedModel);

  const [imageType, setImageType] = useState('bloodstain');
  const [limit, setLimit] = useState(5);
  const [result, setResult] = useState<ImageEvalResult | null>(null);

  const run = () => {
    runEval.mutate(
      { imageType, limit, model: selectedModel },
      {
        onSuccess: (d) => { setResult(d); toast(`Evaluated ${d.samples} ${imageType} image(s)`, 'success'); },
        onError: (e: any) => toast(e.response?.data?.detail || 'Image evaluation failed', 'error'),
      },
    );
  };

  return (
    <div className="space-y-5">
      <p className="text-sm text-text-muted max-w-3xl">
        Runs the real CV + GPT-4o vision pipeline on labelled images and scores predictions
        against ground truth (accuracy / precision-recall-F1). Ground truth: synthetic gold
        (ballistics, tool marks), CEDAR genuine/forged, AZH wound presence, Attinger
        impact-spatter, SOCOFing hand/finger labels.
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <select
          value={imageType}
          onChange={(e) => setImageType(e.target.value)}
          className="bg-card border border-border-default rounded-md px-3 py-2 text-sm text-foreground"
        >
          {(types.data?.types ?? []).map((t) => (
            <option key={t.image_type} value={t.image_type}>{LABELS[t.image_type] || t.image_type}</option>
          ))}
        </select>
        <label className="text-sm text-text-muted flex items-center gap-2">
          images
          <input
            type="number" min={1} max={20} value={limit}
            onChange={(e) => setLimit(Math.max(1, Math.min(20, Number(e.target.value))))}
            className="w-16 bg-card border border-border-default rounded-md px-2 py-2 text-sm"
          />
        </label>
        <ModelSelector />
        <button
          type="button"
          onClick={run}
          disabled={runEval.isPending}
          className="px-5 py-2 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
        >
          {runEval.isPending && <Spinner size="sm" />}
          {runEval.isPending ? 'Running pipeline…' : 'Run image evaluation'}
        </button>
      </div>

      {runEval.isPending && (
        <p className="text-xs text-text-faint">Running CV + vision on {limit} image(s) — this calls the LLM per image and may take a minute.</p>
      )}

      {result && !result.error && (
        <Card className="space-y-4">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold text-text-secondary">
              {LABELS[result.image_type] || result.image_type} — {result.samples} images
            </h3>
            <span className="text-xs text-text-faint">{result.model}</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(result.attributes).map(([name, m]) => (
              <AttrMetric key={name} name={name} m={m} />
            ))}
          </div>
          {result.per_image.length > 0 && (
            <details>
              <summary className="text-xs text-text-faint cursor-pointer hover:text-text-tertiary">per-image predictions</summary>
              <div className="mt-2 space-y-1 max-h-72 overflow-auto">
                {result.per_image.map((row) => (
                  <div key={row.image} className="text-[11px] flex flex-wrap gap-x-3 items-center bg-surface-input rounded px-2 py-1">
                    <span className="font-mono text-text-faint truncate max-w-[180px]">{row.image}</span>
                    {Object.entries(row.attrs).map(([a, v]) => (
                      <span key={a} className={v.correct ? 'text-green-400' : 'text-red-400'}>
                        {a}: {String(v.pred)}{v.correct ? ' ✓' : ` ✗ (gold: ${String(v.gold)})`}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </details>
          )}
        </Card>
      )}

      {result?.error && (
        <Card><p className="text-sm text-text-muted">{result.error}</p></Card>
      )}
    </div>
  );
}
