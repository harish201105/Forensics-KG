'use client';

import { useState } from 'react';
import { useNaturalQuery, useCypherQuery } from '@/lib/hooks';
import { useToast } from '@/components/ui/toast';
import { Spinner } from '@/components/ui/spinner';
import { Card } from '@/components/ui/card';
import { DataTable } from '@/components/ui/data-table';
import { ModelSelector } from '@/components/ui/model-selector';
import { useAppStore } from '@/lib/store';
import type { QueryResponse } from '@/types';

export default function QueryPage() {
  const [tab, setTab] = useState<'natural' | 'cypher'>('natural');
  const [question, setQuestion] = useState('');
  const [cypherQuery, setCypherQuery] = useState('MATCH (n) RETURN labels(n) as type, count(n) as count');
  const [resultView, setResultView] = useState<'table' | 'json'>('table');
  const { toast } = useToast();
  const selectedModel = useAppStore((s) => s.selectedModel);

  const nlMutation = useNaturalQuery();
  const cypherMutation = useCypherQuery();

  const suggestedQueries = [
    'What bloodstain patterns are associated with blunt force trauma?',
    'Show all cases with their suspects and victims',
    'Which experiments used hockey puck impact?',
    'Find evidence linked to homicide cases',
    'What are the most common crime types?',
  ];

  const handleNLQuery = () => {
    if (!question.trim()) return;
    nlMutation.mutate({ question, model: selectedModel }, {
      onError: (e: any) => toast(e.response?.data?.detail || 'Query failed', 'error'),
    });
  };

  const handleCypherQuery = () => {
    if (!cypherQuery.trim()) return;
    cypherMutation.mutate({ query: cypherQuery }, {
      onError: (e: any) => toast(e.response?.data?.detail || e.message || 'Query failed', 'error'),
    });
  };

  const nlResult: QueryResponse | undefined = nlMutation.data;
  const cypherResult = cypherMutation.data;

  // Auto-detect columns from cypher results
  const cypherColumns = cypherResult?.results?.length
    ? Object.keys(cypherResult.results[0]).map((key) => ({
        key,
        header: key,
        render: (row: any) => {
          const val = row[key];
          return typeof val === 'object' ? JSON.stringify(val) : String(val ?? '-');
        },
      }))
    : [];

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Query Knowledge Graph</h1>
      <p className="text-text-muted mb-6">Ask questions in natural language or write Cypher queries directly</p>

      <div className="flex items-center justify-between mb-6">
        <div className="flex gap-2">
        {(['natural', 'cypher'] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
              tab === t ? 'bg-[var(--primary)] text-white' : 'bg-surface-hover text-text-muted'
            }`}
          >
            {t === 'natural' ? 'Natural Language' : 'Cypher Query'}
          </button>
        ))}
        </div>
        {tab === 'natural' && <ModelSelector />}
      </div>

      {tab === 'natural' ? (
        <div className="space-y-6">
          <div className="flex flex-wrap gap-2">
            {suggestedQueries.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => setQuestion(q)}
                className="text-xs px-3 py-1.5 rounded-full bg-surface-hover text-text-muted hover:bg-surface-active hover:text-text-secondary transition"
              >
                {q}
              </button>
            ))}
          </div>

          <div className="flex gap-3">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleNLQuery()}
              placeholder="Ask a question about the forensic data..."
              aria-label="Natural language query"
              className="flex-1 bg-[var(--card)] border border-border-default rounded-lg px-4 py-3 text-sm text-text-body focus:outline-none focus:border-[var(--primary)]/50"
            />
            <button
              type="button"
              onClick={handleNLQuery}
              disabled={nlMutation.isPending || !question.trim()}
              className="px-6 py-3 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 hover:bg-[var(--primary)]/80 flex items-center gap-2"
            >
              {nlMutation.isPending && <Spinner size="sm" />}
              {nlMutation.isPending ? 'Querying...' : 'Ask'}
            </button>
          </div>

          {nlResult && (
            <Card className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-text-secondary mb-2">Answer</h3>
                <p className="text-sm text-text-body leading-relaxed">{nlResult.answer}</p>
              </div>
              <div className="flex gap-4 text-xs text-text-faint">
                <span>Confidence: {(nlResult.confidence * 100).toFixed(0)}%</span>
                {nlResult.sources.length > 0 && <span>{nlResult.sources.length} sources</span>}
              </div>
              {nlResult.cypher_query && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="text-xs text-text-faint">Generated Cypher</h4>
                    <button
                      type="button"
                      onClick={() => { navigator.clipboard.writeText(nlResult.cypher_query || ''); toast('Copied to clipboard', 'info'); }}
                      className="text-xs text-text-ghost hover:text-text-tertiary"
                    >
                      Copy
                    </button>
                  </div>
                  <pre className="text-xs bg-surface-input rounded p-2 text-green-400/70 overflow-x-auto">{nlResult.cypher_query}</pre>
                </div>
              )}
              {nlResult.reasoning && (
                <div>
                  <h4 className="text-xs text-text-faint mb-1">Reasoning</h4>
                  <p className="text-xs text-text-muted">{nlResult.reasoning}</p>
                </div>
              )}
            </Card>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <textarea
            value={cypherQuery}
            onChange={(e) => setCypherQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleCypherQuery(); }}
            aria-label="Cypher query"
            className="w-full h-32 bg-[var(--card)] border border-border-default rounded-lg p-3 font-mono text-sm text-green-400/70 focus:outline-none focus:border-[var(--primary)]/50"
            placeholder="Enter Cypher query... (Ctrl+Enter to execute)"
          />
          <button
            type="button"
            onClick={handleCypherQuery}
            disabled={cypherMutation.isPending || !cypherQuery.trim()}
            className="px-6 py-2.5 bg-[var(--primary)] text-white rounded-lg text-sm font-medium disabled:opacity-40 flex items-center gap-2"
          >
            {cypherMutation.isPending && <Spinner size="sm" />}
            {cypherMutation.isPending ? 'Executing...' : 'Execute'}
          </button>

          {cypherResult && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs text-text-faint">{cypherResult.count} results</p>
                <div className="flex gap-1">
                  {(['table', 'json'] as const).map((v) => (
                    <button
                      key={v}
                      type="button"
                      onClick={() => setResultView(v)}
                      className={`px-2 py-1 text-xs rounded ${resultView === v ? 'bg-surface-active text-text-secondary' : 'text-text-ghost'}`}
                    >
                      {v === 'table' ? 'Table' : 'JSON'}
                    </button>
                  ))}
                </div>
              </div>
              {resultView === 'table' && cypherColumns.length > 0 ? (
                <DataTable data={cypherResult.results} columns={cypherColumns} pageSize={20} />
              ) : (
                <pre className="text-xs bg-surface-input rounded p-3 overflow-auto max-h-96 text-text-secondary">
                  {JSON.stringify(cypherResult.results, null, 2)}
                </pre>
              )}
            </Card>
          )}
        </div>
      )}

      {(nlMutation.error || cypherMutation.error) && (
        <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400">
          {(nlMutation.error as any)?.response?.data?.detail || (cypherMutation.error as any)?.response?.data?.detail || 'Query failed'}
        </div>
      )}
    </div>
  );
}
