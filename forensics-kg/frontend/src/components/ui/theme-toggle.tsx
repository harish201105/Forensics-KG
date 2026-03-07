'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-8 w-20" />;

  const options = [
    { value: 'light', icon: '\u2600', label: 'Light' },
    { value: 'dark', icon: '\u263E', label: 'Dark' },
    { value: 'system', icon: '\u2699', label: 'System' },
  ] as const;

  return (
    <div className="flex items-center gap-0.5 bg-surface-inset rounded-md p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => setTheme(opt.value)}
          title={opt.label}
          className={`px-1.5 py-1 rounded text-xs transition-colors ${
            theme === opt.value
              ? 'bg-surface-active text-text-body'
              : 'text-text-faint hover:text-text-secondary'
          }`}
        >
          {opt.icon}
        </button>
      ))}
    </div>
  );
}
