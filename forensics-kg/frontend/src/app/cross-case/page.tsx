'use client';

import { useState } from 'react';
import { useResolveEntities, useCrossCase } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { NODE_COLORS, type ResolveResult, type CrossCaseResult } from '@/types';

const LABELS = ['Person', 'Location', 'Weapon', 'Vehicle', 'CrimeType'];

export default function CrossCasePage() {
  const { toast } = useToast();
  const resolve = useResolveEntities();
  const lookup = useCrossCase();

  const [label, setLabel] = useState('Person');
  const [value, setValue] = useState('');
  const [preview, setPreview] = useState<ResolveResult | null>(null);

  const result: CrossCaseResult | undefined = lookup.data;

  const runResolve = (dryRun: boolean) => {
    resolve.mutate(
      { dryRun },
      {
        onSuccess: (d: ResolveResult) => {
          setPreview(d);
          toast(
            dryRun
              ? `Dry run: ${d.links_created} cross-case links would be created`
              : `Linked ${d.links_created} cross-case entity pairs`,
            'success',
          );
        },
        onError: (e: any) => toast(e.response?.data?.detail || 'Resolution failed', 'error'),
      },
    );
  };

  const runLookup = () => {
    if (!value.trim()) return;
    lookup.mutate({ label, value }, {
      onError: (e: any) => toast(e.response?.data?.detail || 'Lookup failed', 'error'),
    });
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Cross-Case Reasoning</h1>
      <p className="text-text-muted mb-6 max-w-3xl">
        Each case is extracted into its own entities, so the same real-world person,
        location, or weapon appears as separate nodes per case. Entity resolution links
        these duplicates (<span className="font-mono text-xs">SAME_AS</span>) using
        embeddings + name matching — so you can ask <em>“show all cases involving X.”</em>
      </p>

      {/* Resolution controls */}
      <Card className="mb-6">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-sm font-semibold text-text-secondary">Entity resolution</h3>
          <button
            type="button"
            onClick={() => runResolve(true)}
            disabled={resolve.isPending}
            className="px-3 py-1.5 text-xs rounded-lg bg-surface-hover text-text-secondary hover:bg-surface-active disabled:opacity-50 flex items-center gap-2"
          >
            {resolve.isPending && <Spinner size="sm" />}Preview (dry run)
          </button>
          <button
            type="button"
            onClick={() => runResolve(false)}
            disabled={resolve.isPending}
            className="px-3 py-1.5 text-xs rounded-lg bg-[var(--primary)] text-white hover:bg-[var(--primary)]/80 disabled:opacity-50 flex items-center gap-2"
          >
            {resolve.isPending && <Spinner size="sm" />}Resolve &amp; link
          </button>
          <span className="text-xs text-text-faint">
            Runs automatically after each ingestion; use this to (re)link the whole graph.
          </span>
        </div>

        {preview && (
          <div className="mt-4 space-y-2">
            <p className="text-xs text-text-muted">
              {preview.dry_run ? 'Would create' : 'Created'}{' '}
              <span className="text-[var(--primary)] font-semibold">{preview.links_created}</span>{' '}
              SAME_AS links:
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
              {Object.entries(preview.labels).map(([lbl, s]) => (
                <div key={lbl} className="text-[11px] bg-surface-input rounded p-2">
                  <span style={{ color: NODE_COLORS[lbl] || '#6b7280' }} className="font-semibold">{lbl}</span>
                  <div className="text-text-faint">{s.nodes} nodes</div>
                  <div className="text-text-muted">{s.pairs_linked} links · {s.clusters} clusters</div>
                </div>
              ))}
            </div>
            {Object.values(preview.labels).flatMap((s) => s.examples || []).length > 0 && (
              <details className="mt-1">
                <summary className="text-xs text-text-faint cursor-pointer">example matches</summary>
                <ul className="mt-1 space-y-0.5">
                  {Object.values(preview.labels).flatMap((s) => s.examples || []).slice(0, 12).map((ex, i) => (
                    <li key={i} className="text-[11px] text-text-muted font-mono">{ex}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </Card>

      {/* Cross-case lookup */}
      <Card>
        <h3 className="text-sm font-semibold text-text-secondary mb-3">
          Show all cases involving an entity
        </h3>
        <div className="flex flex-wrap gap-3">
          <select
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            className="bg-[var(--card)] border border-border-default rounded-lg px-3 py-2.5 text-sm text-text-body"
          >
            {LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runLookup()}
            placeholder="e.g. Chandrasekhara Aiyar"
            className="flex-1 min-w-[240px] bg-[var(--card)] border border-border-default rounded-lg px-4 py-2.5 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50"
          />
          <button
            type="button"
            onClick={runLookup}
            disabled={lookup.isPending || !value.trim()}
            className="px-6 py-2.5 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {lookup.isPending && <Spinner size="sm" />}Find cases
          </button>
        </div>

        <div className="flex flex-wrap gap-2 mt-3">
          {['Chandrasekhara Aiyar', 'Bose', 'Fazl Ali', 'Mohinder Singh'].map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => { setLabel('Person'); setValue(s); }}
              className="text-xs px-3 py-1 rounded-full bg-surface-hover text-text-muted hover:text-text-secondary"
            >
              {s}
            </button>
          ))}
        </div>

        {result && (
          <div className="mt-5">
            {result.case_count > 0 ? (
              <>
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-3xl font-bold text-[var(--primary)]">{result.case_count}</span>
                  <span className="text-sm text-text-muted">case{result.case_count !== 1 ? 's' : ''} involve “{result.entity}”</span>
                </div>
                {result.aliases.length > 1 && (
                  <p className="text-xs text-text-faint mb-3">
                    Resolved aliases: {result.aliases.map((a) => `“${a}”`).join(', ')}
                  </p>
                )}
                <div className="space-y-1.5">
                  {result.cases.map((cid, i) => (
                    <div key={cid} className="flex items-center gap-3 text-sm bg-surface-input rounded px-3 py-2">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: NODE_COLORS['Case'] || '#6b7280' }} />
                      <span className="font-mono text-xs text-text-faint">{cid}</span>
                      <span className="text-text-muted">{result.case_titles[i] || ''}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-text-muted">
                No cases found for “{result.entity}”. Try running resolution first, or check the spelling.
              </p>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
