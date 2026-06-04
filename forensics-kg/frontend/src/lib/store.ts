import { create } from 'zustand';
import type { GraphData, GraphStats } from '@/types';

interface AppState {
  graphData: GraphData | null;
  graphStats: GraphStats | null;
  selectedNode: any | null;
  isLoading: boolean;
  sidebarOpen: boolean;
  visibleNodeTypes: Set<string>;
  lastVisitedCaseId: string | null;
  selectedModel: string;
  setGraphData: (data: GraphData) => void;
  setGraphStats: (stats: GraphStats) => void;
  setSelectedNode: (node: any | null) => void;
  setLoading: (loading: boolean) => void;
  setSidebarOpen: (open: boolean) => void;
  toggleNodeType: (type: string) => void;
  setLastVisitedCaseId: (id: string | null) => void;
  setSelectedModel: (model: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  graphData: null,
  graphStats: null,
  selectedNode: null,
  isLoading: false,
  sidebarOpen: false,
  visibleNodeTypes: new Set(),
  lastVisitedCaseId: null,
  selectedModel: 'gpt-4.1',
  setGraphData: (data) => set({ graphData: data }),
  setGraphStats: (stats) => set({ graphStats: stats }),
  setSelectedNode: (node) => set({ selectedNode: node }),
  setLoading: (loading) => set({ isLoading: loading }),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleNodeType: (type) =>
    set((state) => {
      const next = new Set(state.visibleNodeTypes);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return { visibleNodeTypes: next };
    }),
  setLastVisitedCaseId: (id) => set({ lastVisitedCaseId: id }),
  setSelectedModel: (model) => set({ selectedModel: model }),
}));
