'use client';

import { useEffect, useState } from 'react';
import { Card, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Spinner } from '@/components/ui/spinner';

interface TimelineEvent {
  event_id: string;
  timestamp: string;
  parsed_timestamp: string | null;
  description: string;
  precision: string;
}

interface TimelineData {
  case_id: string;
  events: TimelineEvent[];
  total_events: number;
  time_span: { start: string; end: string; duration_hours: number } | null;
  gaps: { after_event: string; before_event: string; gap_hours: number }[];
  inconsistencies: { type: string; severity: string; description: string; between: string[] }[];
}

export function CaseTimeline({ caseId }: { caseId: string }) {
  const [data, setData] = useState<TimelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTimeline = async () => {
      setLoading(true);
      setError(null);
      try {
        const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
        const res = await fetch(`${BACKEND}/api/temporal/timeline/${caseId}`);
        if (!res.ok) throw new Error('Failed to fetch timeline');
        const json = await res.json();
        setData(json);
      } catch (e: any) {
        setError(e.message || 'Failed to load timeline');
      } finally {
        setLoading(false);
      }
    };
    fetchTimeline();
  }, [caseId]);

  if (loading) return <div className="flex justify-center py-12"><Spinner size="md" /></div>;
  if (error) return <Card><p className="text-sm text-red-400 text-center py-8">{error}</p></Card>;
  if (!data || data.total_events === 0) {
    return (
      <Card>
        <p className="text-sm text-text-faint text-center py-8">
          No timeline events found for this case. Extract FIR data first.
        </p>
      </Card>
    );
  }

  const formatTime = (ts: string) => {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: true
      });
    } catch {
      return ts;
    }
  };

  const gapSet = new Set(data.gaps.map(g => g.after_event));

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex flex-wrap gap-4 mb-4">
        <Badge variant="info">{data.total_events} events</Badge>
        {data.time_span && (
          <Badge>{data.time_span.duration_hours.toFixed(1)} hours span</Badge>
        )}
        {data.gaps.length > 0 && (
          <Badge variant="warning">{data.gaps.length} gaps detected</Badge>
        )}
        {data.inconsistencies.length > 0 && (
          <Badge variant="error">{data.inconsistencies.length} inconsistencies</Badge>
        )}
      </div>

      {/* Timeline */}
      <Card>
        <CardTitle>Event Timeline</CardTitle>
        <div className="mt-4 relative ml-4">
          {/* Vertical line */}
          <div className="absolute left-2 top-0 bottom-0 w-0.5 bg-surface-active" />

          {data.events.map((event, i) => (
            <div key={event.event_id || i}>
              {/* Event dot + content */}
              <div className="relative flex items-start gap-4 pb-6">
                <div className="relative z-10 w-4 h-4 rounded-full bg-[var(--primary)] border-2 border-[var(--card)] mt-1 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-mono text-[var(--primary)]">
                      {formatTime(event.timestamp)}
                    </span>
                    {event.event_id && (
                      <span className="text-xs text-text-whisper">{event.event_id}</span>
                    )}
                  </div>
                  <p className="text-sm text-text-secondary">{event.description}</p>
                </div>
              </div>

              {/* Gap indicator */}
              {gapSet.has(event.event_id) && (
                <div className="relative flex items-start gap-4 pb-4 ml-0.5">
                  <div className="relative z-10 w-3 h-3 rounded-full bg-yellow-500/50 mt-1 flex-shrink-0 ml-0.5" />
                  <div className="text-xs text-yellow-400/70 italic">
                    Gap: {data.gaps.find(g => g.after_event === event.event_id)?.gap_hours.toFixed(1)}h
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Inconsistencies */}
      {data.inconsistencies.length > 0 && (
        <Card>
          <CardTitle>Inconsistencies</CardTitle>
          <div className="space-y-2 mt-3">
            {data.inconsistencies.map((inc, i) => (
              <div key={i} className="bg-surface-inset rounded p-3 text-sm">
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant={inc.severity === 'high' ? 'error' : 'warning'}>
                    {inc.type}
                  </Badge>
                </div>
                <p className="text-text-tertiary text-xs">{inc.description}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
