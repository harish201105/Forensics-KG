'use client';

import { useAppStore } from '@/lib/store';

const AVAILABLE_MODELS = [
  { id: 'gpt-4.1', label: 'GPT-4.1', description: 'Latest, best reasoning (default)' },
  { id: 'gpt-4.1-mini', label: 'GPT-4.1 Mini', description: 'Fast + cheap' },
  { id: 'gpt-4o', label: 'GPT-4o', description: 'Multimodal, moderate speed' },
  { id: 'gpt-4o-mini', label: 'GPT-4o Mini', description: 'Fast, good for simple tasks' },
];

export function ModelSelector({ className = '' }: { className?: string }) {
  const selectedModel = useAppStore((s) => s.selectedModel);
  const setSelectedModel = useAppStore((s) => s.setSelectedModel);

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <label htmlFor="model-select" className="text-sm text-text-muted whitespace-nowrap">Model:</label>
      <select
        id="model-select"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        className="bg-card border border-border-default rounded-md px-2 py-1.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
      >
        {AVAILABLE_MODELS.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label} — {m.description}
          </option>
        ))}
      </select>
    </div>
  );
}
