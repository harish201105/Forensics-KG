'use client';

import { Component, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <div className="text-4xl">⚠</div>
          <h2 className="text-lg font-semibold text-text-body">Something went wrong</h2>
          <p className="text-sm text-text-faint max-w-md text-center">
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-2 bg-[var(--primary)] text-white rounded-lg text-sm hover:opacity-90"
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
