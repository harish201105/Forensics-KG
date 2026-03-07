'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import { useFullGraph } from '@/lib/hooks';
import { graphApi } from '@/lib/api';
import { PageSpinner } from '@/components/ui/spinner';
import { useToast } from '@/components/ui/toast';
import type { GraphData } from '@/types';
import { NODE_COLORS } from '@/types';
import { useTheme } from 'next-themes';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

interface GraphNode2D {
  id: string;
  label: string;
  type: string;
  color: string;
  properties: Record<string, any>;
  x?: number;
  y?: number;
}

interface GraphLink2D {
  source: string;
  target: string;
  type: string;
}

export default function GraphPage() {
  const { data: rawData, isLoading, error: graphError } = useFullGraph(500);
  const { toast } = useToast();
  const [graphData, setGraphData] = useState<{ nodes: GraphNode2D[]; links: GraphLink2D[] }>({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState<GraphNode2D | null>(null);
  const [visibleTypes, setVisibleTypes] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
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

  // ResizeObserver for proper sizing
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Process raw data into graph format
  useEffect(() => {
    if (!rawData) return;
    const nodeMap = new Map<string, GraphNode2D>();
    const types = new Set<string>();

    for (const n of rawData.nodes) {
      const type = n.labels?.[0] || 'Unknown';
      types.add(type);
      const name = n.properties?.name || n.properties?.case_id ||
        n.properties?.experiment_id || n.properties?.pattern_id ||
        n.properties?.stain_id || n.properties?.hypothesis_id || n.id;
      nodeMap.set(n.id, {
        id: n.id,
        label: String(name).slice(0, 30),
        type,
        color: NODE_COLORS[type] || '#6b7280',
        properties: n.properties || {},
      });
    }

    const links: GraphLink2D[] = rawData.edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({ source: e.source, target: e.target, type: e.type }));

    setGraphData({ nodes: Array.from(nodeMap.values()), links });
    setVisibleTypes(types);
  }, [rawData]);

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 500);
      graphRef.current.zoom(3, 500);
    }
  }, []);

  const toggleType = (type: string) => {
    setVisibleTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const res = await graphApi.search(searchQuery);
      setSearchResults(res.results || []);
      // Center on first result
      if (res.results?.length > 0) {
        const found = graphData.nodes.find((n) => n.id === res.results[0].id);
        if (found && graphRef.current) {
          graphRef.current.centerAt(found.x, found.y, 500);
          graphRef.current.zoom(3, 500);
          setSelectedNode(found);
        }
        toast(`Found ${res.results.length} results`, 'info');
      } else {
        toast('No results found', 'info');
      }
    } catch {
      toast('Search failed', 'error');
    }
  };

  const handleZoomIn = () => graphRef.current?.zoom(graphRef.current.zoom() * 1.5, 300);
  const handleZoomOut = () => graphRef.current?.zoom(graphRef.current.zoom() / 1.5, 300);
  const handleFit = () => graphRef.current?.zoomToFit(400);

  const filteredData = {
    nodes: graphData.nodes.filter((n) => visibleTypes.has(n.type)),
    links: graphData.links.filter((l) => {
      const sId = typeof l.source === 'string' ? l.source : (l.source as any).id;
      const tId = typeof l.target === 'string' ? l.target : (l.target as any).id;
      const sNode = graphData.nodes.find((n) => n.id === sId);
      const tNode = graphData.nodes.find((n) => n.id === tId);
      return sNode && tNode && visibleTypes.has(sNode.type) && visibleTypes.has(tNode.type);
    }),
  };

  if (isLoading) return <PageSpinner message="Loading graph..." />;

  return (
    <div className="h-full flex">
      {/* Graph Canvas */}
      <div className="flex-1 relative bg-graph-bg" ref={containerRef}>
        {graphError ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <p className="text-red-400 text-sm mb-2">Failed to load graph data</p>
              <p className="text-text-ghost text-xs">{(graphError as any)?.response?.data?.detail || 'Check backend connection'}</p>
            </div>
          </div>
        ) : graphData.nodes.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-text-faint">
            No data in graph. Generate or upload data first.
          </div>
        ) : (
          <ForceGraph2D
            ref={graphRef}
            graphData={filteredData}
            nodeLabel="label"
            nodeColor="color"
            nodeRelSize={6}
            linkColor={() => themeColors.edgeColor}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            onNodeClick={handleNodeClick}
            nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
              const fontSize = Math.max(10 / globalScale, 2);
              ctx.beginPath();
              ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI);
              ctx.fillStyle = node.color;
              ctx.fill();
              if (globalScale > 1.5) {
                ctx.font = `${fontSize}px Inter, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillStyle = themeColors.labelColor;
                ctx.fillText(node.label || '', node.x, node.y + 7);
              }
            }}
            backgroundColor={themeColors.graphBg}
            width={dimensions.width}
            height={dimensions.height}
          />
        )}

        {/* Controls overlay */}
        <div className="absolute top-3 left-3 flex gap-2">
          <div className="flex items-center bg-[var(--card)]/90 border border-border-default rounded-lg overflow-hidden">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search nodes..."
              aria-label="Search graph nodes"
              className="px-3 py-2 bg-transparent text-sm text-text-body placeholder:text-text-ghost focus:outline-none w-48"
            />
            <button
              type="button"
              onClick={handleSearch}
              className="px-3 py-2 text-sm text-text-muted hover:text-text-body border-l border-border-default"
            >
              Search
            </button>
          </div>
        </div>

        <div className="absolute bottom-3 left-3 flex gap-1">
          <button type="button" onClick={handleZoomIn} className="p-2 bg-[var(--card)]/90 border border-border-default rounded-lg text-text-tertiary hover:text-foreground text-sm">+</button>
          <button type="button" onClick={handleZoomOut} className="p-2 bg-[var(--card)]/90 border border-border-default rounded-lg text-text-tertiary hover:text-foreground text-sm">-</button>
          <button type="button" onClick={handleFit} className="p-2 bg-[var(--card)]/90 border border-border-default rounded-lg text-text-tertiary hover:text-foreground text-xs">Fit</button>
        </div>
      </div>

      {/* Right Panel */}
      <div className="w-70 bg-[var(--card)] border-l border-border-default overflow-y-auto">
        {/* Legend */}
        <div className="p-4 border-b border-border-default">
          <h3 className="text-xs uppercase tracking-wider text-text-faint mb-3">Node Types</h3>
          <div className="space-y-1.5">
            {Array.from(new Set(graphData.nodes.map((n) => n.type))).sort().map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => toggleType(type)}
                className={`flex items-center gap-2 text-xs w-full px-2 py-1 rounded transition-opacity ${
                  visibleTypes.has(type) ? 'opacity-100' : 'opacity-30'
                }`}
              >
                <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: NODE_COLORS[type] || '#6b7280' }} />
                <span className="text-text-secondary">{type}</span>
                <span className="ml-auto text-text-ghost">{graphData.nodes.filter((n) => n.type === type).length}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="p-4 border-b border-border-default">
            <h3 className="text-xs uppercase tracking-wider text-text-faint mb-2">Search Results</h3>
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {searchResults.slice(0, 10).map((r: any) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => {
                    const found = graphData.nodes.find((n) => n.id === r.id);
                    if (found) handleNodeClick(found);
                  }}
                  className="block w-full text-left px-2 py-1 rounded text-xs text-text-tertiary hover:bg-surface-hover"
                >
                  <span style={{ color: NODE_COLORS[r.labels?.[0]] || '#6b7280' }}>{r.labels?.[0]}</span>
                  {' '}{r.properties?.name || r.properties?.case_id || r.id}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Selected Node Details */}
        {selectedNode && (
          <div className="p-4">
            <h3 className="text-xs uppercase tracking-wider text-text-faint mb-2">Selected Node</h3>
            <div className="text-sm font-semibold mb-1" style={{ color: selectedNode.color }}>
              {selectedNode.type}
            </div>
            <p className="text-sm text-text-body mb-3">{selectedNode.label}</p>
            <div className="space-y-1">
              {Object.entries(selectedNode.properties || {}).map(([key, value]) => (
                <div key={key} className="text-xs">
                  <span className="text-text-faint">{key}:</span>{' '}
                  <span className="text-text-secondary">{String(value).slice(0, 100)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="p-4 border-t border-border-default">
          <p className="text-xs text-text-ghost">
            {filteredData.nodes.length} nodes, {filteredData.links.length} edges
          </p>
        </div>
      </div>
    </div>
  );
}
