import React from 'react';
import {
  LayoutDashboard,
  Database,
  Network,
  Cpu,
  FlaskConical,
  Crosshair,
  BookOpenCheck,
  ChevronRight
} from 'lucide-react';

export type NavTab =
  | 'overview'
  | 'dataset'
  | 'graph'
  | 'models'
  | 'experiments'
  | 'prediction'
  | 'findings';

interface SidebarProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
}

interface NavItem {
  id: NavTab;
  label: string;
  subtitle: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  {
    id: 'overview',
    label: 'OVERVIEW',
    subtitle: 'Research Executive Summary',
    icon: LayoutDashboard,
  },
  {
    id: 'dataset',
    label: 'DATASET',
    subtitle: 'Temporal Dynamics & Splits',
    icon: Database,
  },
  {
    id: 'graph',
    label: 'GRAPH EXPLORER',
    subtitle: 'Local Neighborhood Graph',
    icon: Network,
  },
  {
    id: 'models',
    label: 'MODEL LAB',
    subtitle: '7-Model Performance Matrix',
    icon: Cpu,
  },
  {
    id: 'experiments',
    label: 'EXPERIMENTS',
    subtitle: 'Ablations & Significance',
    icon: FlaskConical,
  },
  {
    id: 'prediction',
    label: 'PREDICTION',
    subtitle: 'Live Multi-Model Inference',
    icon: Crosshair,
  },
  {
    id: 'findings',
    label: 'RESEARCH FINDINGS',
    subtitle: 'Insights & Figure Gallery',
    icon: BookOpenCheck,
  },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab }) => {
  return (
    <aside className="w-64 border-r border-lab-border bg-lab-surface flex flex-col justify-between shrink-0 h-[calc(100vh-4rem)]">
      {/* Navigation Links */}
      <div className="p-3 space-y-1 overflow-y-auto">
        <div className="px-3 py-2 text-[10px] font-mono tracking-widest text-lab-muted uppercase">
          Research Navigation
        </div>

        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center justify-between group transition-all ${
                isActive
                  ? 'bg-indigo-600/15 text-white border border-indigo-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-lab-card border border-transparent'
              }`}
            >
              <div className="flex items-center space-x-3 min-w-0">
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    isActive ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-300'
                  }`}
                />
                <div className="truncate">
                  <div className={`text-xs font-semibold tracking-wider ${isActive ? 'text-white' : 'text-slate-300'}`}>
                    {item.label}
                  </div>
                  <div className="text-[10px] text-lab-muted truncate">
                    {item.subtitle}
                  </div>
                </div>
              </div>

              {isActive && (
                <ChevronRight className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              )}
            </button>
          );
        })}
      </div>

      {/* Footer System Status */}
      <div className="p-4 border-t border-lab-border bg-lab-bg/50">
        <div className="rounded-lg bg-lab-card border border-lab-border p-3 text-xs space-y-1.5 font-mono">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400">Environment</span>
            <span className="text-emerald-400 font-medium">PyTorch + PyG</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400">Partition</span>
            <span className="text-indigo-300 font-medium">Chronological</span>
          </div>
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400">Seeds</span>
            <span className="text-cyan-400 font-medium">N = 5 (42..999)</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
