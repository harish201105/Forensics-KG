'use client';

import { useRouter } from 'next/navigation';
import { useCases } from '@/lib/hooks';
import { PageSpinner } from '@/components/ui/spinner';
import { DataTable } from '@/components/ui/data-table';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/empty-state';

interface CaseRecord {
  case_id: string;
  title: string;
  crime_type: string;
  status: string;
  date_filed: string;
}

export default function CasesPage() {
  const router = useRouter();
  const { data, isLoading, error } = useCases();

  const cases: CaseRecord[] = (data?.results || []).map((r: any) => ({
    case_id: r.props?.case_id || 'unknown',
    title: r.props?.title || r.props?.case_id || 'Untitled',
    crime_type: r.props?.crime_type || '-',
    status: r.props?.status || 'open',
    date_filed: r.props?.date_filed || '-',
  }));

  if (isLoading) return <PageSpinner message="Loading cases..." />;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Cases</h1>
      <p className="text-text-muted mb-6">Browse and manage forensic cases in the knowledge graph</p>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 text-red-400 text-sm">
          {(error as any)?.response?.data?.detail || 'Failed to load cases'}
        </div>
      )}

      {cases.length === 0 && !error ? (
        <EmptyState
          icon="▣"
          title="No cases found"
          message="Generate synthetic FIR data or extract from reports to create cases."
          actionLabel="Go to Datasets"
          actionHref="/datasets"
        />
      ) : (
        <DataTable
          data={cases}
          columns={[
            { key: 'case_id', header: 'Case ID', render: (r) => (
              <span className="font-mono text-[var(--primary)]">{r.case_id}</span>
            )},
            { key: 'title', header: 'Title' },
            { key: 'crime_type', header: 'Crime Type' },
            { key: 'status', header: 'Status', render: (r) => (
              <Badge variant={r.status === 'closed' ? 'success' : 'warning'}>{r.status}</Badge>
            )},
            { key: 'date_filed', header: 'Date' },
          ]}
          searchable
          searchKeys={['case_id', 'title', 'crime_type']}
          onRowClick={(row) => router.push(`/cases/${row.case_id}`)}
          pageSize={15}
        />
      )}
    </div>
  );
}
