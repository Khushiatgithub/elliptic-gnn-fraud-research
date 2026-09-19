import React, { useEffect, useState } from 'react';
import {
  BookOpenCheck,
  ZoomIn,
  ExternalLink,
  ShieldCheck,
  AlertTriangle,
  Layers,
  BarChart3,
  Network,
  Scale,
  Sparkles,
  Info
} from 'lucide-react';
import { FigureInfo } from '../types/api';
import { FigureModal } from '../components/FigureModal';
import { api } from '../services/api';

export const ResearchFindingsPage: React.FC = () => {
  const [figures, setFigures] = useState<FigureInfo[]>([]);
  const [selectedFigure, setSelectedFigure] = useState<FigureInfo | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    api.getFiguresList()
      .then((data) => setFigures(data))
      .catch((err) => console.error('Failed to load figures:', err));
  }, []);

  const categories = ['all', ...new Set(figures.map((f) => f.category))];
  const filteredFigures = figures.filter((f) =>
    selectedCategory === 'all' ? true : f.category === selectedCategory
  );

  return (
    <div className="p-8 space-y-10 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
          <BookOpenCheck className="w-3.5 h-3.5" />
          <span>RESEARCH SYNTHESIS & EVIDENCE ARCHIVE</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Comprehensive Research Findings & Publication Gallery
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Scientific conclusions, empirical insights, and publication-grade artifacts from the IEEE study.
        </p>
      </div>

      {/* Findings Narrative Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 1. Research Question & Core Findings */}
        <div className="lab-card p-6 space-y-3">
          <div className="flex items-center space-x-2 text-indigo-400">
            <span className="text-xs font-mono font-bold uppercase">01. Central Research Question</span>
          </div>
          <h3 className="text-base font-bold text-white">
            Can graph-based learning improve illicit transaction detection?
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-light">
            Under strict chronological partitioning, conventional tree ensembles (Random Forest F1: <strong>0.7247</strong>, XGBoost F1: <strong>0.7131</strong>) demonstrated stronger performance than evaluated Graph Neural Networks (GAT F1: <strong>0.3308</strong>, GraphSAGE: <strong>0.2822</strong>, GCN: <strong>0.2589</strong>).
            The superior performance of flat baselines is explained by the 72 pre-aggregated 1-hop statistical features that capture local graph context without the vulnerability to noise propagation across ~77% unknown nodes.
          </p>
        </div>

        {/* 2. GNN Architecture Comparison */}
        <div className="lab-card p-6 space-y-3">
          <div className="flex items-center space-x-2 text-cyan-400">
            <span className="text-xs font-mono font-bold uppercase">02. Graph Neural Network Dynamics</span>
          </div>
          <h3 className="text-base font-bold text-white">
            GAT Outperforms Isotropic Message Passing
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-light">
            Among GNN architectures, <strong>GAT (Graph Attention Network)</strong> was the strongest performer under the primary inductive setup (Test F1: 0.3308 vs 0.2822 for GraphSAGE and 0.2589 for GCN).
            GAT's 8 multi-head attention coefficients enable it to adaptively downweight uninformative and noisy unlabeled neighbors, mitigating the signal dilution that severely degrades isotropic models.
          </p>
        </div>

        {/* 3. Temporal Generalization & Darknet Disruption */}
        <div className="lab-card p-6 space-y-3">
          <div className="flex items-center space-x-2 text-rose-400">
            <span className="text-xs font-mono font-bold uppercase">03. Temporal Generalization</span>
          </div>
          <h3 className="text-base font-bold text-white">
            Chronological Degradation & Regime Shift
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-light">
            All 7 model families experienced substantial performance degradation between the validation split (timesteps 35–39) and the chronological test split (timesteps 40–49).
            This degradation was exacerbated by a dramatic darknet market shutdown at timesteps 43–45, where the illicit base rate collapsed from ~10% to <strong>0.28%–1.75%</strong>, causing severe precision collapse across all models.
          </p>
        </div>

        {/* 4. Statistical Rigor & Significance Limits */}
        <div className="lab-card p-6 space-y-3">
          <div className="flex items-center space-x-2 text-amber-400">
            <span className="text-xs font-mono font-bold uppercase">04. Statistical Significance Rigor</span>
          </div>
          <h3 className="text-base font-bold text-white">
            Conservative Reporting Post Holm-Bonferroni
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-light">
            Across 5 random seeds (N=5), raw paired Student's t-tests between GraphSAGE and tree baselines yielded p ≈ 0.02–0.03.
            However, after rigorous <strong>Holm-Bonferroni correction</strong> for multiple hypothesis testing, adjusted p-values ranged from <strong>0.0816 to 0.4760</strong>.
            Therefore, in accordance with scientific integrity, no performance differential is claimed as statistically significant after multi-testing correction.
          </p>
        </div>
      </div>

      {/* Publication Figure Gallery */}
      <div className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white">
              Publication Figure Gallery (300 DPI High-Resolution)
            </h2>
            <p className="text-xs text-lab-muted font-mono">
              Click any figure to view in full resolution with experimental captions.
            </p>
          </div>

          {/* Category Filter */}
          <div className="flex items-center space-x-2 bg-lab-card border border-lab-border px-3 py-1.5 rounded-lg text-xs font-mono">
            <span className="text-lab-muted">Category:</span>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-transparent text-white focus:outline-none cursor-pointer"
            >
              {categories.map((c) => (
                <option key={c} value={c} className="bg-lab-surface text-white">
                  {c === 'all' ? 'All Figure Categories' : c}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Figure Card Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredFigures.map((fig) => (
            <div
              key={fig.filename}
              onClick={() => setSelectedFigure(fig)}
              className="lab-card overflow-hidden cursor-pointer group flex flex-col justify-between lab-card-hover"
            >
              {/* Image Preview */}
              <div className="h-48 w-full bg-[#07090e] p-3 flex items-center justify-center relative overflow-hidden border-b border-lab-border">
                <img
                  src={api.getFigureUrl(fig.filename)}
                  alt={fig.title}
                  className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-300"
                />
                <div className="absolute inset-0 bg-indigo-950/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
                  <div className="px-3 py-1.5 rounded-full bg-lab-surface/90 border border-lab-border text-white text-xs font-mono flex items-center space-x-1.5 shadow-xl">
                    <ZoomIn className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Click to Enlarge</span>
                  </div>
                </div>
              </div>

              {/* Card Meta */}
              <div className="p-4 space-y-2">
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    {fig.category}
                  </span>
                  <span className="text-slate-500">{fig.paper_section}</span>
                </div>

                <h3 className="text-xs font-bold text-white line-clamp-1 group-hover:text-indigo-300 transition-colors">
                  {fig.title}
                </h3>
                <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed font-sans">
                  {fig.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Modal Zoom Viewer */}
      <FigureModal figure={selectedFigure} onClose={() => setSelectedFigure(null)} />
    </div>
  );
};
