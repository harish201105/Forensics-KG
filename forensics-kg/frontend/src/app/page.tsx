'use client';

import Link from 'next/link';
import { useGraphStats } from '@/lib/hooks';
import { PageSpinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { NODE_COLORS } from '@/types';

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <Card>
      <p className="text-xs uppercase tracking-wider text-text-faint mb-1">{label}</p>
      <p className="text-3xl font-bold" style={{ color }}>{value.toLocaleString()}</p>
    </Card>
  );
}

const quickActions = [
  { href: '/extract', label: 'Extract Entities', desc: 'Upload FIR text or images', icon: '⬡', color: '#3b82f6' },
  { href: '/datasets', label: 'Generate Data', desc: 'Create synthetic FIRs', icon: '▦', color: '#22c55e' },
  { href: '/query', label: 'Query Graph', desc: 'Ask questions in natural language', icon: '⬢', color: '#a855f7' },
  { href: '/graph', label: 'Visualize', desc: 'Explore the knowledge graph', icon: '◈', color: '#f59e0b' },
];

export default function Dashboard() {
  const { data: stats, isLoading, error, refetch } = useGraphStats();

  if (isLoading) return <PageSpinner message="Loading dashboard..." />;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Dashboard</h1>
      <p className="text-text-muted mb-8">Forensics Knowledge Graph Overview</p>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 text-red-400 text-sm flex items-center justify-between">
          <span>{(error as any).response?.data?.detail || 'Failed to connect to backend'}</span>
          <button
            type="button"
            onClick={() => refetch()}
            className="px-3 py-1 bg-red-500/20 rounded text-xs hover:bg-red-500/30 transition"
          >
            Retry
          </button>
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {quickActions.map((a) => (
          <Link key={a.href} href={a.href}>
            <Card className="hover:bg-surface-hover transition cursor-pointer h-full">
              <span className="text-2xl" style={{ color: a.color }}>{a.icon}</span>
              <p className="text-sm font-semibold text-text-body mt-2">{a.label}</p>
              <p className="text-xs text-text-faint mt-0.5">{a.desc}</p>
            </Card>
          </Link>
        ))}
      </div>

      {stats && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <StatCard label="Total Nodes" value={stats.total_nodes} color="var(--primary)" />
            <StatCard label="Total Relationships" value={stats.total_relationships} color="#22c55e" />
            <StatCard label="Node Types" value={Object.keys(stats.node_counts).length} color="#a855f7" />
            <StatCard label="Relationship Types" value={Object.keys(stats.relationship_counts).length} color="#f59e0b" />
          </div>

          {Object.keys(stats.node_counts).length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <h2 className="text-sm font-semibold mb-4 text-text-secondary">Nodes by Type</h2>
                {Object.entries(stats.node_counts)
                  .sort(([, a], [, b]) => b - a)
                  .map(([type, count]) => (
                  <div key={type} className="flex items-center gap-2 py-1.5 border-b border-border-subtle last:border-0">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: NODE_COLORS[type] || '#6b7280' }} />
                    <span className="text-sm text-text-tertiary flex-1">{type}</span>
                    <span className="text-sm font-mono font-medium">{count}</span>
                  </div>
                ))}
              </Card>
              <Card>
                <h2 className="text-sm font-semibold mb-4 text-text-secondary">Relationships by Type</h2>
                {Object.entries(stats.relationship_counts)
                  .sort(([, a], [, b]) => b - a)
                  .map(([type, count]) => (
                  <div key={type} className="flex justify-between py-1.5 border-b border-border-subtle last:border-0">
                    <span className="text-sm text-text-tertiary">{type}</span>
                    <span className="text-sm font-mono font-medium">{count}</span>
                  </div>
                ))}
              </Card>
            </div>
          ) : (
            <EmptyState
              icon="📊"
              title="No data yet"
              message="Generate synthetic FIR reports or extract from documents to populate the knowledge graph."
              actionLabel="Go to Datasets"
              actionHref="/datasets"
            />
          )}
        </>
      )}
    </div>
  );
}
