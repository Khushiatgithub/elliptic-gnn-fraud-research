import React from 'react';
import { Database, Activity, Play, ShieldAlert, Cpu } from 'lucide-react';

interface NavbarProps {
  isPresentationMode: boolean;
  onTogglePresentation: () => void;
  backendOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  isPresentationMode,
  onTogglePresentation,
  backendOnline
}) => {
  return (
    <header className="h-16 border-b border-lab-border bg-lab-surface/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Title & Branding */}
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-sm font-semibold tracking-wide text-white">
              Bitcoin Illicit Transaction Detection
            </h1>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              IEEE Benchmark
            </span>
          </div>
          <p className="text-xs text-lab-muted font-mono">
            Graph ML × Tabular ML Research Platform
          </p>
        </div>
      </div>

      {/* Center Status Indicators */}
      <div className="hidden md:flex items-center space-x-4 text-xs font-mono">
        <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-lab-card border border-lab-border">
          <Database className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-slate-300">Dataset:</span>
          <span className="text-emerald-400 font-medium">203,769 Txs</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-lab-card border border-lab-border">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-300">Models:</span>
          <span className="text-cyan-400 font-medium">7 Architectures</span>
        </div>

        <div className="flex items-center space-x-2 px-3 py-1 rounded-md bg-lab-card border border-lab-border">
          <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
          <span className="text-slate-300">Backend:</span>
          <span className={backendOnline ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
            {backendOnline ? 'FastAPI Connected' : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-3">
        <button
          onClick={onTogglePresentation}
          className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isPresentationMode
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 ring-1 ring-indigo-400'
              : 'bg-lab-card text-slate-300 border border-lab-border hover:border-lab-borderLight hover:text-white'
          }`}
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>{isPresentationMode ? 'Exit Presentation' : 'Presentation Mode'}</span>
        </button>
      </div>
    </header>
  );
};
