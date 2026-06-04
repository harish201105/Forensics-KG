'use client';

import { useMemo, useState } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { useProjection, useEmbeddingStatus, useEmbeddingBackfill, useSemanticSearch } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { NODE_COLORS, type ProjectionPoint, type SemanticHit } from '@/types';

const FALLBACK_COLOR = '#6b7280';

export default function ProjectorPage() {
  const { toast } = useToast();
  const status = useEmbeddingStatus();
  const projection = useProjection(2, 2000);
  const backfill = useEmbeddingBackfill();
  const semantic = useSemanticSearch();

  const [query, setQuery] = useState('');
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<ProjectionPoint | null>(null);

  const points = projection.data?.points ?? [];
  const coverage = status.data ? Math.round(status.data.coverage * 100) : 0;

  // Highlighted node ids from the latest semantic search.
  const matchIds = useMemo(
    () => new Set((semantic.data?.results ?? []).map((r: SemanticHit) => r.id)),
    [semantic.data],
  );

  // Group points by primary label into recharts series.
  const series = useMemo(() => {
    const groups: Record<string, ProjectionPoint[]> = {};
    for (const p of points) {
      if (hidden.has(p.label)) continue;
      (groups[p.label] ||= []).push(p);
    }
    return Object.entries(groups)
      .map(([label, pts]) => ({ label, pts, color: NODE_COLORS[label] || FALLBACK_COLOR }))
      .sort((a, b) => b.pts.length - a.pts.length);
  }, [points, hidden]);

  const matchPoints = useMemo(
    () => points.filter((p) => matchIds.has(p.id)),
    [points, matchIds],
  );

  const allLabels = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const p of points) counts[p.label] = (counts[p.label] || 0) + 1;
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [points]);

  const runSearch = () => {
    if (!query.trim()) return;
    semantic.mutate(
      { q: query, k: 15 },
      { onError: (e: any) => toast(e.response?.data?.detail || 'Search failed', 'error') },
    );
  };

  const toggleLabel = (label: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      next.has(label) ? next.delete(label) : next.add(label);
      return next;
    });

  const renderTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null;
    const p: ProjectionPoint = payload[0].payload;
    return (
      <div className="bg-card border border-border-default rounded-lg p-2 max-w-xs shadow-lg">
        <div className="text-xs font-semibold" style={{ color: NODE_COLORS[p.label] || FALLBACK_COLOR }}>
          {p.label}
        </div>
        <div className="text-[11px] text-text-muted line-clamp-3">{p.text}</div>
      </div>
    );
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Embedding Projector</h1>
      <p className="text-text-muted mb-6">
        Each point is a graph node placed by semantic similarity (PCA of OpenAI embeddings).
        Nearby points are conceptually related — search to find nearest neighbours (KNN).
      </p>

      {/* Status / controls bar */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="text-sm text-text-muted">
          Embeddings:{' '}
          <span className={coverage === 100 ? 'text-green-400' : 'text-yellow-400'}>
            {status.data ? `${status.data.embedded}/${status.data.total} (${coverage}%)` : '…'}
          </span>
        </div>
        <button
          type="button"
          onClick={() =>
            backfill.mutate(true, {
              onSuccess: (r: any) => toast(`Embedded ${r.processed} new node(s)`, 'success'),
              onError: () => toast('Backfill failed', 'error'),
            })
          }
          disabled={backfill.isPending}
          className="px-3 py-1.5 text-xs rounded-lg bg-surface-hover text-text-secondary hover:bg-surface-active flex items-center gap-2 disabled:opacity-50"
        >
          {backfill.isPending && <Spinner size="sm" />}
          {backfill.isPending ? 'Embedding…' : 'Embed new nodes'}
        </button>
      </div>

      {/* Semantic search */}
      <div className="flex gap-3 mb-6">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && runSearch()}
          placeholder="Search semantically, e.g. 'wound from a sharp weapon' — highlights nearest nodes"
          className="flex-1 bg-[var(--card)] border border-border-default rounded-lg px-4 py-2.5 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50"
        />
        <button
          type="button"
          onClick={runSearch}
          disabled={semantic.isPending || !query.trim()}
          className="px-6 py-2.5 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
        >
          {semantic.isPending && <Spinner size="sm" />}
          Search
        </button>
      </div>

      {coverage === 0 && !projection.isLoading ? (
        <Card>
          <p className="text-sm text-text-muted">
            No embeddings yet. Click <strong>Embed new nodes</strong> above to build the
            semantic index, then the projector will populate.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Plot */}
          <Card className="lg:col-span-3">
            {projection.isLoading ? (
              <div className="h-[520px] flex items-center justify-center"><Spinner /></div>
            ) : (
              <ResponsiveContainer width="100%" height={520}>
                <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeOpacity={0.1} />
                  <XAxis type="number" dataKey="x" hide domain={[-1.05, 1.05]} />
                  <YAxis type="number" dataKey="y" hide domain={[-1.05, 1.05]} />
                  <ZAxis range={[18, 18]} />
                  <Tooltip content={renderTooltip} cursor={{ strokeOpacity: 0.2 }} />
                  {series.map((s) => (
                    <Scatter
                      key={s.label}
                      name={s.label}
                      data={s.pts}
                      fill={s.color}
                      fillOpacity={matchIds.size ? 0.25 : 0.75}
                      onClick={(d: any) => setSelected(d?.payload ?? null)}
                      isAnimationActive={false}
                    />
                  ))}
                  {matchPoints.length > 0 && (
                    <Scatter
                      name="matches"
                      data={matchPoints}
                      fill="#ffffff"
                      shape="star"
                      onClick={(d: any) => setSelected(d?.payload ?? null)}
                      isAnimationActive={false}
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Side panel: legend + selection + search results */}
          <div className="space-y-4">
            {selected && (
              <Card>
                <div className="text-xs font-semibold mb-1" style={{ color: NODE_COLORS[selected.label] || FALLBACK_COLOR }}>
                  {selected.label}
                </div>
                <p className="text-[11px] text-text-muted">{selected.text}</p>
              </Card>
            )}

            {semantic.data?.results?.length ? (
              <Card>
                <h3 className="text-xs font-semibold text-text-secondary mb-2">
                  Nearest neighbours
                </h3>
                <div className="space-y-1.5 max-h-60 overflow-auto">
                  {semantic.data.results.map((r: SemanticHit) => {
                    const label = r.labels.find((l) => l !== 'Embedded') || 'Node';
                    const p = r.properties;
                    const name = p.name || p.case_id || p.description || p.pattern_type || r.id;
                    return (
                      <div key={r.id} className="text-[11px] flex items-start gap-2">
                        <span className="px-1.5 py-0.5 rounded shrink-0" style={{ backgroundColor: (NODE_COLORS[label] || FALLBACK_COLOR) + '33', color: NODE_COLORS[label] || FALLBACK_COLOR }}>
                          {r.score.toFixed(2)}
                        </span>
                        <span className="text-text-muted truncate">
                          <span className="text-text-faint">{label}</span> {String(name).slice(0, 50)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </Card>
            ) : null}

            <Card>
              <h3 className="text-xs font-semibold text-text-secondary mb-2">
                Node types <span className="text-text-faint">(click to toggle)</span>
              </h3>
              <div className="space-y-1 max-h-72 overflow-auto">
                {allLabels.map(([label, count]) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => toggleLabel(label)}
                    className={`flex items-center gap-2 w-full text-left text-[11px] ${hidden.has(label) ? 'opacity-40' : ''}`}
                  >
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: NODE_COLORS[label] || FALLBACK_COLOR }} />
                    <span className="text-text-muted flex-1 truncate">{label}</span>
                    <span className="text-text-faint">{count}</span>
                  </button>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
