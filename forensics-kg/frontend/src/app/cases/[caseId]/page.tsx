'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { graphApi, analysisApi, collaborationApi } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { PageSpinner, Spinner } from '@/components/ui/spinner';
import { Card, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/components/ui/toast';
import { CaseTimeline } from '@/components/timeline/case-timeline';
import type { GraphData, Annotation } from '@/types';
import { NODE_COLORS } from '@/types';
import { useTheme } from 'next-themes';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

interface CaseProps {
  case_id: string;
  title?: string;
  crime_type?: string;
  status?: string;
  date_filed?: string;
  description?: string;
  location?: string;
  [key: string]: any;
}

export default function CaseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { toast } = useToast();
  const caseId = params.caseId as string;
  const { setLastVisitedCaseId } = useAppStore();

  const [caseData, setCaseData] = useState<CaseProps | null>(null);
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [entities, setEntities] = useState<Record<string, any[]>>({});
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('graph');
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [annotationText, setAnnotationText] = useState('');
  const [caseStatus, setCaseStatus] = useState('');
  const graphRef = useRef<any>(null);
  const { resolvedTheme } = useTheme();
  const [themeColors, setThemeColors] = useState({ graphBg: '#060610', labelColor: 'rgba(255,255,255,0.7)', edgeColor: 'rgba(255,255,255,0.12)' });

  useEffect(() => {
    const style = getComputedStyle(document.documentElement);
    setThemeColors({
      graphBg: style.getPropertyValue('--graph-bg').trim() || '#060610',
      labelColor: style.getPropertyValue('--text-secondary').trim() || 'rgba(255,255,255,0.7)',
      edgeColor: style.getPropertyValue('--border-default').trim() || 'rgba(255,255,255,0.12)',
    });
  }, [resolvedTheme]);

  useEffect(() => {
    setLastVisitedCaseId(caseId);
    loadCaseData();
    loadAnnotations();
  }, [caseId]);

  const loadCaseData = async () => {
    setLoading(true);
    try {
      // Fetch case node
      const caseNode = await graphApi.search(caseId, 'Case');
      if (caseNode.results?.[0]) {
        const props = caseNode.results[0].properties;
        setCaseData({ case_id: caseId, ...props });
        setCaseStatus(props?.status || '');
      }

      // Fetch subgraph
      const sub = await graphApi.getSubgraph(caseId, 'Case', 'case_id', 3);
      processGraph(sub);
    } catch {
      toast('Failed to load case data', 'error');
    } finally {
      setLoading(false);
    }
  };

  const processGraph = (data: GraphData) => {
    const nodeMap = new Map<string, any>();
    const grouped: Record<string, any[]> = {};

    for (const n of data.nodes) {
      const type = n.labels?.[0] || 'Unknown';
      const name = n.properties?.name || n.properties?.case_id ||
        n.properties?.experiment_id || n.properties?.pattern_id ||
        n.properties?.stain_id || n.id;
      const node = {
        id: n.id,
        label: String(name).slice(0, 30),
        type,
        color: NODE_COLORS[type] || '#6b7280',
        properties: n.properties || {},
      };
      nodeMap.set(n.id, node);
      if (!grouped[type]) grouped[type] = [];
      grouped[type].push(n.properties || {});
    }

    const links = data.edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({ source: e.source, target: e.target, type: e.type }));

    setGraphData({ nodes: Array.from(nodeMap.values()), links });
    setEntities(grouped);
  };

  const loadAnnotations = async () => {
    try {
      const anns = await collaborationApi.getAnnotations('Case', caseId);
      setAnnotations(anns || []);
    } catch { /* ignore */ }
  };

  const handleAddAnnotation = async () => {
    if (!annotationText.trim()) return;
    try {
      await collaborationApi.addAnnotation('Case', caseId, annotationText);
      setAnnotationText('');
      toast('Annotation added', 'success');
      loadAnnotations();
    } catch {
      toast('Failed to add annotation', 'error');
    }
  };

  const handleStatusChange = async (newStatus: string) => {
    try {
      await collaborationApi.updateCaseStatus(caseId, newStatus);
      setCaseStatus(newStatus);
      toast(`Status updated to ${newStatus}`, 'success');
    } catch {
      toast('Failed to update status', 'error');
    }
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    try {
      const model = useAppStore.getState().selectedModel;
      const res = await analysisApi.hypothesis(caseId, model);
      setAnalysis(res);
      setActiveTab('analysis');
      toast('Analysis complete', 'success');
    } catch {
      toast('Analysis failed', 'error');
    } finally {
      setAnalyzing(false);
    }
  };

  if (loading) return <PageSpinner message={`Loading case ${caseId}...`} />;

  const tabs = [
    { id: 'graph', label: 'Graph' },
    { id: 'entities', label: `Entities (${graphData.nodes.length})` },
    { id: 'timeline', label: 'Timeline' },
    { id: 'analysis', label: 'Analysis' },
    { id: 'annotations', label: `Notes (${annotations.length})` },
  ];

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <button
            type="button"
            onClick={() => router.push('/cases')}
            className="text-xs text-text-faint hover:text-text-tertiary mb-2 block"
          >
            &larr; Back to Cases
          </button>
          <h1 className="text-2xl font-bold">{caseData?.title || caseId}</h1>
          <div className="flex items-center gap-3 mt-2">
            <span className="text-sm font-mono text-[var(--primary)]">{caseId}</span>
            {caseData?.crime_type && <Badge variant="info">{caseData.crime_type}</Badge>}
            <select
              value={caseStatus}
              onChange={(e) => handleStatusChange(e.target.value)}
              aria-label="Case status"
              className="bg-[var(--card)] border border-border-default rounded px-2 py-1 text-xs text-text-secondary focus:outline-none"
            >
              <option value="">No status</option>
              <option value="open">Open</option>
              <option value="investigating">Investigating</option>
              <option value="closed">Closed</option>
              <option value="cold">Cold</option>
            </select>
            {caseData?.date_filed && (
              <span className="text-xs text-text-faint">{caseData.date_filed}</span>
            )}
          </div>
          {caseData?.description && (
            <p className="text-sm text-text-muted mt-2 max-w-2xl">{caseData.description}</p>
          )}
        </div>
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={analyzing}
          className="px-4 py-2 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
        >
          {analyzing && <Spinner size="sm" />}
          {analyzing ? 'Analyzing...' : 'Generate Hypothesis'}
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActiveTab(t.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === t.id ? 'bg-[var(--primary)] text-white' : 'bg-surface-hover text-text-muted'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'graph' && (
        <Card padding={false} className="overflow-hidden">
          <div className="h-[500px] relative bg-graph-bg">
            {graphData.nodes.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center text-text-faint">
                No graph data for this case
              </div>
            ) : (
              <ForceGraph2D
                ref={graphRef}
                graphData={graphData}
                nodeLabel="label"
                nodeColor="color"
                nodeRelSize={6}
                linkColor={() => themeColors.edgeColor}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                onNodeClick={(node: any) => {
                  if (graphRef.current) {
                    graphRef.current.centerAt(node.x, node.y, 500);
                    graphRef.current.zoom(3, 500);
                  }
                }}
                nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                  const fontSize = Math.max(10 / globalScale, 2);
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI);
                  ctx.fillStyle = node.color;
                  ctx.fill();
                  if (globalScale > 1.2) {
                    ctx.font = `${fontSize}px Inter, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillStyle = themeColors.labelColor;
                    ctx.fillText(node.label || '', node.x, node.y + 7);
                  }
                }}
                backgroundColor={themeColors.graphBg}
                height={500}
              />
            )}
          </div>
          {/* Legend */}
          <div className="flex flex-wrap gap-3 p-3 border-t border-border-default">
            {Object.keys(entities).map((type) => (
              <div key={type} className="flex items-center gap-1.5 text-xs text-text-muted">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS[type] || '#6b7280' }} />
                {type} ({entities[type].length})
              </div>
            ))}
          </div>
        </Card>
      )}

      {activeTab === 'entities' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Object.entries(entities).map(([type, items]) => (
            <Card key={type}>
              <h3 className="text-sm font-semibold mb-3" style={{ color: NODE_COLORS[type] || '#6b7280' }}>
                {type} ({items.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {items.map((item, i) => (
                  <div key={i} className="bg-surface-inset rounded p-2 text-xs">
                    {Object.entries(item).slice(0, 5).map(([k, v]) => (
                      <div key={k} className="flex gap-2">
                        <span className="text-text-faint min-w-[80px]">{k}:</span>
                        <span className="text-text-secondary">{String(v).slice(0, 80)}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}

      {activeTab === 'timeline' && (
        <CaseTimeline caseId={caseId} />
      )}

      {activeTab === 'analysis' && (
        <div className="space-y-4">
          {!analysis ? (
            <Card>
              <p className="text-sm text-text-faint text-center py-8">
                Click &quot;Generate Hypothesis&quot; to run forensic analysis on this case.
              </p>
            </Card>
          ) : (
            <>
              {analysis.primary_hypothesis && (
                <Card>
                  <CardTitle className="text-[#f59e0b] mb-3">Primary Hypothesis</CardTitle>
                  <p className="text-sm text-text-body mb-3">{analysis.primary_hypothesis.description}</p>
                  <div className="flex gap-4 text-xs text-text-faint">
                    <span>Mechanism: {analysis.primary_hypothesis.mechanism}</span>
                    <span>Confidence: {((analysis.primary_hypothesis.confidence ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                  {analysis.primary_hypothesis.supporting_evidence?.length > 0 && (
                    <div className="mt-3">
                      <p className="text-xs text-green-400/70 mb-1">Supporting Evidence:</p>
                      <ul className="text-xs text-text-muted list-disc list-inside">
                        {analysis.primary_hypothesis.supporting_evidence.map((e: string, i: number) => (
                          <li key={i}>{e}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </Card>
              )}

              {analysis.alternative_hypotheses?.length > 0 && (
                <Card>
                  <CardTitle>Alternative Hypotheses</CardTitle>
                  <div className="space-y-3 mt-3">
                    {analysis.alternative_hypotheses.map((h: any, i: number) => (
                      <div key={i} className="bg-surface-inset rounded p-3">
                        <p className="text-sm text-text-secondary">{h.description}</p>
                        <div className="flex gap-3 mt-1 text-xs text-text-faint">
                          <span>{h.mechanism}</span>
                          <span>{((h.confidence ?? 0) * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {analysis.recommendations?.length > 0 && (
                <Card>
                  <CardTitle>Recommendations</CardTitle>
                  <ul className="space-y-1.5 mt-3">
                    {analysis.recommendations.map((r: string, i: number) => (
                      <li key={i} className="text-xs text-text-tertiary flex gap-2">
                        <span className="text-[var(--primary)]">&bull;</span> {r}
                      </li>
                    ))}
                  </ul>
                </Card>
              )}
            </>
          )}
        </div>
      )}

      {activeTab === 'annotations' && (
        <div className="space-y-4">
          <Card>
            <CardTitle>Add Note</CardTitle>
            <div className="flex gap-2 mt-3">
              <input
                value={annotationText}
                onChange={(e) => setAnnotationText(e.target.value)}
                placeholder="Add a note about this case..."
                aria-label="Case annotation"
                className="flex-1 bg-surface-input border border-border-default rounded-lg px-3 py-2 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50"
                onKeyDown={(e) => e.key === 'Enter' && handleAddAnnotation()}
              />
              <button
                type="button"
                onClick={handleAddAnnotation}
                disabled={!annotationText.trim()}
                className="px-4 py-2 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40"
              >
                Add
              </button>
            </div>
          </Card>
          {annotations.length > 0 ? (
            <Card>
              <CardTitle>Case Notes</CardTitle>
              <div className="space-y-3 mt-3">
                {annotations.map((ann) => (
                  <div key={ann.annotation_id} className="bg-surface-inset rounded-lg p-3">
                    <p className="text-sm text-text-body">{ann.text}</p>
                    <div className="flex gap-3 mt-1.5 text-xs text-text-ghost">
                      <span>{ann.author}</span>
                      <span>{new Date(ann.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          ) : (
            <Card>
              <p className="text-sm text-text-faint text-center py-8">
                No notes yet. Add one above.
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
