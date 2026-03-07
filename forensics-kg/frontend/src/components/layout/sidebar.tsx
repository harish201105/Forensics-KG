'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAppStore } from '@/lib/store';
import { ThemeToggle } from '@/components/ui/theme-toggle';

const navItems = [
  { href: '/', label: 'Dashboard', icon: '\u25C9' },
  { href: '/graph', label: 'Knowledge Graph', icon: '\u25C8' },
  { href: '/extract', label: 'Extract', icon: '\u2B21' },
  { href: '/query', label: 'Query', icon: '\u2B22' },
  { href: '/datasets', label: 'Datasets', icon: '\u25A6' },
  { href: '/cases', label: 'Cases', icon: '\u25A3' },
  { href: '/analysis', label: 'Analysis', icon: '\u25B3' },
  { href: '/evaluation', label: 'Evaluation', icon: '\u25CE' },
];

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarOpen, setSidebarOpen } = useAppStore();

  return (
    <>
      {/* Mobile hamburger */}
      <button
        type="button"
        onClick={() => setSidebarOpen(true)}
        className="fixed top-4 left-4 z-50 md:hidden p-2 rounded-lg bg-card border border-border-default"
        aria-label="Open menu"
      >
        <span className="text-lg">{'\u2630'}</span>
      </button>

      {/* Backdrop (mobile) */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-overlay-backdrop z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 h-full w-56 bg-card border-r border-border-default flex flex-col z-50 transition-transform duration-200 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}
      >
        <div className="p-4 border-b border-border-default">
          <h1 className="text-lg font-bold text-primary">Forensics KG</h1>
          <p className="text-xs text-text-faint mt-1">Knowledge Graph System</p>
        </div>
        <nav className="flex-1 py-4">
          {navItems.map((item) => {
            const isActive = pathname === item.href ||
              (item.href !== '/' && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setSidebarOpen(false)}
                className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-primary/10 text-primary border-r-2 border-primary'
                    : 'text-text-tertiary hover:text-foreground hover:bg-surface-hover'
                }`}
              >
                <span className="text-base">{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="p-4 border-t border-border-default flex items-center justify-between">
          <span className="text-xs text-text-ghost">UROP</span>
          <ThemeToggle />
        </div>
      </aside>
    </>
  );
}
