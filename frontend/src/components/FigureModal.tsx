import React from 'react';
import { X, ExternalLink, ZoomIn } from 'lucide-react';
import { FigureInfo } from '../types/api';
import { api } from '../services/api';

interface FigureModalProps {
  figure: FigureInfo | null;
  onClose: () => void;
}

export const FigureModal: React.FC<FigureModalProps> = ({ figure, onClose }) => {
  if (!figure) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-5xl bg-lab-surface border border-lab-border rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-lab-border flex items-center justify-between bg-lab-card/50">
          <div className="min-w-0 pr-4">
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              {figure.category} • {figure.paper_section}
            </span>
            <h3 className="text-sm font-semibold text-white mt-1 truncate">
              {figure.title}
            </h3>
          </div>

          <div className="flex items-center space-x-2">
            <a
              href={api.getFigureUrl(figure.filename)}
              target="_blank"
              rel="noreferrer"
              className="p-1.5 rounded-lg bg-lab-card border border-lab-border text-slate-300 hover:text-white hover:bg-lab-border transition-colors"
              title="Open full resolution in new tab"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg bg-lab-card border border-lab-border text-slate-300 hover:text-white hover:bg-lab-border transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Image Display */}
        <div className="flex-1 overflow-auto p-6 flex items-center justify-center bg-[#07090e]">
          <img
            src={api.getFigureUrl(figure.filename)}
            alt={figure.title}
            className="max-h-[65vh] w-auto object-contain rounded-lg border border-lab-border shadow-lg"
          />
        </div>

        {/* Footer Caption */}
        <div className="px-6 py-3 border-t border-lab-border bg-lab-card/30 text-xs font-mono text-slate-400">
          <span className="text-slate-200 font-medium">Source Experiment Description: </span>
          {figure.description}
        </div>
      </div>
    </div>
  );
};
