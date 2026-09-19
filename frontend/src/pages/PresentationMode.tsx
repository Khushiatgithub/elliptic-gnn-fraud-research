import React, { useState, useEffect } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  X,
  Layers,
  Database,
  Network,
  Cpu,
  Calendar,
  BarChart3,
  FlaskConical,
  TrendingDown,
  Crosshair,
  CheckCircle2,
  ShieldAlert,
  HelpCircle,
  Clock
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetSummary, MainResultRecord } from '../types/api';

interface PresentationModeProps {
  onExit: () => void;
}

export const PresentationMode: React.FC<PresentationModeProps> = ({ onExit }) => {
  const [currentSlide, setCurrentSlide] = useState<number>(0);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [mainResults, setMainResults] = useState<MainResultRecord[]>([]);

  useEffect(() => {
    Promise.all([api.getSummary(), api.getMainResults()])
      .then(([s, r]) => {
        setSummary(s);
        setMainResults(r);
      })
      .catch((err) => console.error(err));
  }, []);

  const totalSlides = 10;

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'Space') {
        setCurrentSlide((prev) => Math.min(prev + 1, totalSlides - 1));
      } else if (e.key === 'ArrowLeft') {
        setCurrentSlide((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Escape') {
        onExit();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onExit]);

  const slides = [
    // Slide 1: Problem
    {
      id: 'problem',
      step: '01 / 10',
      tag: 'RESEARCH PROBLEM',
      icon: ShieldAlert,
      title: 'Illicit Transaction Detection in Bitcoin',
      subtitle: 'Graph Neural Networks versus Tabular Machine Learning',
      content: (
        <div className="space-y-6 max-w-4xl">
          <p className="text-xl text-slate-200 leading-relaxed font-light">
            Bitcoin's pseudonymous public ledger enables decentralized value transfer, but is also exploited for ransomware, darknet markets, money laundering, and scams.
          </p>
          <div className="p-6 rounded-2xl bg-lab-card border border-lab-border space-y-4">
            <h4 className="text-sm font-mono uppercase text-indigo-400 font-bold">
              The Central Empirical Question
            </h4>
            <p className="text-2xl font-semibold text-white leading-snug">
              "Do Graph Neural Networks genuinely outperform strong tabular machine learning baselines when evaluated under realistic chronological partitioning?"
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs font-mono text-slate-400">
            <div className="p-4 rounded-xl bg-lab-surface border border-lab-border">
              <span className="text-slate-200 font-bold block text-sm mb-1">Prior Literature Gap</span>
              Many prior GNN studies relied on random node splits, introducing severe future-to-past data leakage and exaggerating graph learning gains.
            </div>
            <div className="p-4 rounded-xl bg-lab-surface border border-lab-border">
              <span className="text-slate-200 font-bold block text-sm mb-1">Our Methodology</span>
              Strict temporal partitioning across 49 timesteps, multi-seed evaluation (N=5), and validation-locked decision thresholds.
            </div>
          </div>
        </div>
      ),
    },

    // Slide 2: Dataset
    {
      id: 'dataset',
      step: '02 / 10',
      tag: 'DATASET ARCHITECTURE',
      icon: Database,
      title: 'The Elliptic Bitcoin Benchmark Dataset',
      subtitle: '203,769 Transactions • 234,355 Directed Edges • 49 Discrete Timesteps',
      content: (
        <div className="space-y-6 max-w-5xl">
          <div className="grid grid-cols-4 gap-4">
            <div className="p-5 rounded-xl bg-lab-card border border-lab-border text-center">
              <div className="text-3xl font-mono font-bold text-white">203,769</div>
              <div className="text-xs font-mono text-slate-400 mt-1">Total Transactions</div>
            </div>
            <div className="p-5 rounded-xl bg-lab-card border border-lab-border text-center">
              <div className="text-3xl font-mono font-bold text-indigo-400">234,355</div>
              <div className="text-xs font-mono text-slate-400 mt-1">Directed Edges</div>
            </div>
            <div className="p-5 rounded-xl bg-lab-card border border-lab-border text-center">
              <div className="text-3xl font-mono font-bold text-cyan-400">4,545</div>
              <div className="text-xs font-mono text-slate-400 mt-1">Illicit (Class 1)</div>
            </div>
            <div className="p-5 rounded-xl bg-lab-card border border-lab-border text-center">
              <div className="text-3xl font-mono font-bold text-emerald-400">42,019</div>
              <div className="text-xs font-mono text-slate-400 mt-1">Licit (Class 2)</div>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-lab-card border border-lab-border space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-amber-400 uppercase">
                The Dominant Structural Challenge
              </span>
              <span className="text-xs font-mono text-slate-400">157,205 Unknown Nodes</span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed font-light">
              <strong className="text-white">77.15% of all transactions</strong> in the network have no ground-truth label.
              Furthermore, <strong className="text-white">84.37% of all directed edges</strong> are incident to at least one unknown transaction, creating massive noise propagation challenges for message-passing GNNs.
            </p>
          </div>
        </div>
      ),
    },

    // Slide 3: Graph Representation
    {
      id: 'representation',
      step: '03 / 10',
      tag: 'FEATURE REPRESENTATION',
      icon: Network,
      title: 'Graph Representation & Feature Engineering',
      subtitle: '165 Continuous Features = 93 Local Moments + 72 Aggregated Statistics',
      content: (
        <div className="grid grid-cols-2 gap-6 max-w-5xl">
          <div className="p-6 rounded-2xl bg-lab-card border border-indigo-500/30 space-y-4">
            <span className="px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-400 text-xs font-mono font-bold">
              93 Local Features (cols 0–92)
            </span>
            <h4 className="text-lg font-bold text-white">Intrinsic Transaction Moments</h4>
            <ul className="text-xs text-slate-300 space-y-2 font-mono list-disc pl-4">
              <li>Input and output address counts</li>
              <li>Transaction transaction fee (Satoshi / byte)</li>
              <li>Output volume total and standard deviation</li>
              <li>Timestep-level activity moments</li>
            </ul>
          </div>

          <div className="p-6 rounded-2xl bg-lab-card border border-cyan-500/30 space-y-4">
            <span className="px-2.5 py-1 rounded bg-cyan-500/10 text-cyan-400 text-xs font-mono font-bold">
              72 Aggregated Features (cols 93–164)
            </span>
            <h4 className="text-lg font-bold text-white">1-Hop Topological Neighborhood</h4>
            <ul className="text-xs text-slate-300 space-y-2 font-mono list-disc pl-4">
              <li>Min, Max, Mean, Std across input neighbors</li>
              <li>Min, Max, Mean, Std across output neighbors</li>
              <li>Captures local graph context without GNN complexity</li>
              <li>Crucial reason why flat tree models excel</li>
            </ul>
          </div>
        </div>
      ),
    },

    // Slide 4: Models
    {
      id: 'models',
      step: '04 / 10',
      tag: 'EVALUATED ARCHITECTURES',
      icon: Cpu,
      title: 'Seven Evaluated Model Architectures',
      subtitle: '4 Conventional Machine Learning Baselines + 3 Graph Neural Networks',
      content: (
        <div className="grid grid-cols-2 gap-6 max-w-5xl">
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-slate-400 font-bold block">
              Conventional Tabular Models
            </span>
            <div className="space-y-2 text-xs font-mono">
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-white font-bold">Random Forest</span>
                <span className="text-slate-400">100 Trees • Balanced Bootstrap</span>
              </div>
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-white font-bold">XGBoost</span>
                <span className="text-slate-400">100 Trees • Histogram Bins</span>
              </div>
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-white font-bold">MLP Baseline</span>
                <span className="text-slate-400">2 Hidden (128-64) • BatchNorm</span>
              </div>
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-white font-bold">Logistic Regression</span>
                <span className="text-slate-400">L2 Regularized Linear Model</span>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-slate-400 font-bold block">
              Graph Neural Networks
            </span>
            <div className="space-y-2 text-xs font-mono">
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-cyan-300 font-bold">GAT</span>
                <span className="text-slate-400">8 Multi-Head Attention Heads</span>
              </div>
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-cyan-300 font-bold">GraphSAGE</span>
                <span className="text-slate-400">Inductive Mean Aggregation</span>
              </div>
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border flex justify-between items-center">
                <span className="text-cyan-300 font-bold">GCN</span>
                <span className="text-slate-400">Spectral Symmetric Convolution</span>
              </div>
            </div>
          </div>
        </div>
      ),
    },

    // Slide 5: Experimental Design
    {
      id: 'design',
      step: '05 / 10',
      tag: 'METHODOLOGICAL RIGOR',
      icon: Calendar,
      title: 'Chronological Splitting & Validation Locking',
      subtitle: 'Strict temporal evaluation preventing future information leakage',
      content: (
        <div className="space-y-6 max-w-5xl font-mono text-xs">
          <div className="grid grid-cols-3 gap-4">
            <div className="p-5 rounded-xl bg-lab-card border border-indigo-500/40 space-y-2">
              <div className="text-indigo-400 font-bold text-sm">TRAIN SPLIT</div>
              <div className="text-white text-lg font-bold">Timesteps 1–34</div>
              <div className="text-slate-400">29,894 labeled transactions (3,462 illicit)</div>
              <div className="text-[11px] text-indigo-300 pt-2 border-t border-lab-border">
                StandardScaler fit strictly on train split
              </div>
            </div>

            <div className="p-5 rounded-xl bg-lab-card border border-cyan-500/40 space-y-2">
              <div className="text-cyan-400 font-bold text-sm">VALIDATION SPLIT</div>
              <div className="text-white text-lg font-bold">Timesteps 35–39</div>
              <div className="text-slate-400">5,486 labeled transactions (447 illicit)</div>
              <div className="text-[11px] text-cyan-300 pt-2 border-t border-lab-border">
                Decision threshold τ optimized for F1
              </div>
            </div>

            <div className="p-5 rounded-xl bg-lab-card border border-rose-500/40 space-y-2">
              <div className="text-rose-400 font-bold text-sm">TEST SPLIT (BLIND)</div>
              <div className="text-white text-lg font-bold">Timesteps 40–49</div>
              <div className="text-slate-400">11,184 labeled transactions (636 illicit)</div>
              <div className="text-[11px] text-rose-300 pt-2 border-t border-lab-border">
                Evaluated with LOCKED validation threshold
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-lab-card border border-lab-border text-slate-300 font-sans text-xs leading-relaxed">
            <strong className="text-white">Validation-Locked Threshold Rule: </strong>
            Decision thresholds are never fitted on the test set. All metrics are computed strictly using the optimal threshold discovered on the validation split.
          </div>
        </div>
      ),
    },

    // Slide 6: Results
    {
      id: 'results',
      step: '06 / 10',
      tag: 'PRIMARY RESULTS',
      icon: BarChart3,
      title: 'Primary Empirical Results (N=5 Seeds)',
      subtitle: 'Observed Test Performance under Full 165 Feature Space',
      content: (
        <div className="space-y-4 max-w-5xl font-mono text-xs">
          <div className="lab-card overflow-hidden">
            <table className="w-full text-left">
              <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                <tr>
                  <th className="py-3 px-4">Architecture</th>
                  <th className="py-3 px-3">Family</th>
                  <th className="py-3 px-3 text-white font-bold">Test F1 (Mean ± Std)</th>
                  <th className="py-3 px-3 text-cyan-300 font-bold">Test PR-AUC</th>
                  <th className="py-3 px-3">Test Precision</th>
                  <th className="py-3 px-3">Test Recall</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-lab-border/40">
                {mainResults.map((r) => (
                  <tr key={r.model} className="hover:bg-lab-card/30">
                    <td className="py-3 px-4 font-bold text-white">{r.model}</td>
                    <td className="py-3 px-3 text-slate-400">{r.family}</td>
                    <td className="py-3 px-3 text-white font-bold">{r.test_f1_mean.toFixed(4)} ± {r.test_f1_std.toFixed(4)}</td>
                    <td className="py-3 px-3 text-cyan-300 font-bold">{r.test_prauc_mean.toFixed(4)} ± {r.test_prauc_std.toFixed(4)}</td>
                    <td className="py-3 px-3 text-slate-300">{r.test_precision_mean.toFixed(4)}</td>
                    <td className="py-3 px-3 text-slate-300">{r.test_recall_mean.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-sans">
            <strong>Key Finding: </strong> Random Forest (F1: 0.7247) and XGBoost (F1: 0.7131) substantially outperformed GAT (0.3308), GraphSAGE (0.2822), and GCN (0.2589).
          </div>
        </div>
      ),
    },

    // Slide 7: Ablations
    {
      id: 'ablations',
      step: '07 / 10',
      tag: 'GNN ABLATION INSIGHTS',
      icon: FlaskConical,
      title: 'Topological Ablations & Mechanistic Drivers',
      subtitle: 'Directionality • Unknown Nodes • Layer Depth',
      content: (
        <div className="grid grid-cols-3 gap-4 max-w-5xl font-mono text-xs">
          <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-3">
            <span className="text-cyan-400 font-bold uppercase text-[11px] block">
              1. Unknown Nodes Context
            </span>
            <p className="text-slate-300 font-sans text-xs leading-relaxed">
              Retaining ~77% unknown nodes dilutes isotropic message passing (GCN F1 drops from 0.4555 to 0.2589). GAT attention mitigates this dilution.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-3">
            <span className="text-emerald-400 font-bold uppercase text-[11px] block">
              2. Edge Directionality
            </span>
            <p className="text-slate-300 font-sans text-xs leading-relaxed">
              Forward direction (u→v) preserves causal payment flows. Backward retrospective aggregation (v→u) slightly improves F1 by tracking upstream funds.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-3">
            <span className="text-rose-400 font-bold uppercase text-[11px] block">
              3. Layer Depth Decay
            </span>
            <p className="text-slate-300 font-sans text-xs leading-relaxed">
              3-layer GNNs suffer from severe over-smoothing in isotropic models (GCN: 0.2862 at 1-layer → 0.2089 at 3-layers).
            </p>
          </div>
        </div>
      ),
    },

    // Slide 8: Temporal Generalization
    {
      id: 'generalization',
      step: '08 / 10',
      tag: 'TEMPORAL DRIFT',
      icon: TrendingDown,
      title: 'Temporal Generalization & Darknet Disruption',
      subtitle: 'The Darknet Market Shutdown at Timesteps 43–45',
      content: (
        <div className="space-y-6 max-w-5xl">
          <div className="p-6 rounded-2xl bg-lab-card border border-rose-500/30 space-y-4">
            <h4 className="text-base font-bold text-white">
              The Sudden Illicit Base Rate Collapse
            </h4>
            <p className="text-xs text-slate-300 font-light leading-relaxed">
              At timestep 43, international law enforcement seized major darknet marketplaces (e.g. Wall Street Market).
              The proportion of illicit transactions collapsed from <strong>11.10% (t=42)</strong> down to <strong>0.28% (t=46)</strong>.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4 text-xs font-mono text-center">
            <div className="p-4 rounded-xl bg-lab-card border border-lab-border">
              <div className="text-slate-400">Timestep 42 (Pre-Shift)</div>
              <div className="text-lg font-bold text-emerald-400 mt-1">F1 = 0.5234</div>
              <div className="text-[10px] text-slate-500">239 Illicit Txs (11.10%)</div>
            </div>

            <div className="p-4 rounded-xl bg-lab-card border border-rose-500/30 bg-rose-500/5">
              <div className="text-rose-400 font-bold">Timestep 45 (Shutdown)</div>
              <div className="text-lg font-bold text-rose-400 mt-1">F1 = 0.0070</div>
              <div className="text-[10px] text-slate-400">5 Illicit Txs (0.41%)</div>
            </div>

            <div className="p-4 rounded-xl bg-lab-card border border-lab-border">
              <div className="text-slate-400">Timestep 49 (Recovery)</div>
              <div className="text-lg font-bold text-cyan-400 mt-1">F1 = 0.1642</div>
              <div className="text-[10px] text-slate-500">56 Illicit Txs (11.76%)</div>
            </div>
          </div>
        </div>
      ),
    },

    // Slide 9: Prediction Demo
    {
      id: 'demo',
      step: '09 / 10',
      tag: 'INTERACTIVE DEMONSTRATION',
      icon: Crosshair,
      title: 'Live Multi-Model Prediction Engine',
      subtitle: 'Real-time inference over authentic serialized model checkpoints',
      content: (
        <div className="p-6 rounded-2xl bg-lab-card border border-lab-border space-y-4 max-w-4xl text-xs font-mono">
          <h4 className="text-sm font-bold text-white uppercase text-indigo-300">
            Interactive Dashboard Capabilities
          </h4>
          <ul className="space-y-3 text-slate-300 font-sans list-disc pl-5">
            <li><strong>Live Transaction Inference:</strong> Look up any of the 203,769 transactions and evaluate against 7 trained models.</li>
            <li><strong>Feature Importance Extraction:</strong> Displays Gini feature importances for tree models and linear coefficients.</li>
            <li><strong>Local Graph Inspection:</strong> Renders 1-hop / 2-hop neighborhood topology with degree connectivity.</li>
            <li><strong>Zero Fabrication:</strong> Connects directly to serialized `.joblib` and `.pt` checkpoints with strict training scalers.</li>
          </ul>
        </div>
      ),
    },

    // Slide 10: Conclusion
    {
      id: 'conclusion',
      step: '10 / 10',
      tag: 'CONCLUSION & RECOMMENDATIONS',
      icon: CheckCircle2,
      title: 'Conclusions & Takeaways for Financial ML',
      subtitle: 'Key Lessons for Graph Neural Networks in High-Noise Regimes',
      content: (
        <div className="space-y-4 max-w-5xl font-sans text-xs">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-2">
              <span className="text-indigo-400 font-bold font-mono text-[11px] block">
                1. Feature Aggregation &gt; Raw Message Passing
              </span>
              <p className="text-slate-300 font-light leading-relaxed">
                Aggregated statistical moments capture essential topological signals for tree models without the optimization volatility and noise propagation of GNNs.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-2">
              <span className="text-cyan-400 font-bold font-mono text-[11px] block">
                2. Attention is Essential for Unknown Context
              </span>
              <p className="text-slate-300 font-light leading-relaxed">
                When graphs have ~77% unknown nodes, isotropic GNNs fail due to signal dilution. Dynamic attention mechanisms (GAT) are required.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-2">
              <span className="text-rose-400 font-bold font-mono text-[11px] block">
                3. Temporal Evaluation is Non-Negotiable
              </span>
              <p className="text-slate-300 font-light leading-relaxed">
                Random splitting produces overly optimistic estimates. Chronological evaluation reveals real-world domain shifts and regime vulnerabilities.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-lab-card border border-lab-border space-y-2">
              <span className="text-amber-400 font-bold font-mono text-[11px] block">
                4. Multi-Seed Statistical Rigor
              </span>
              <p className="text-slate-300 font-light leading-relaxed">
                Always report multi-seed variance and apply multi-testing corrections (Holm-Bonferroni) before claiming statistical superiority.
              </p>
            </div>
          </div>
        </div>
      ),
    },
  ];

  const current = slides[currentSlide];
  const Icon = current.icon;

  return (
    <div className="fixed inset-0 z-50 bg-[#07090e] text-white flex flex-col justify-between select-none">
      {/* Top Slide Header */}
      <div className="px-8 py-4 border-b border-lab-border flex items-center justify-between bg-lab-surface/80 backdrop-blur-md">
        <div className="flex items-center space-x-3 font-mono text-xs">
          <span className="px-2.5 py-1 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-bold">
            {current.step}
          </span>
          <span className="text-slate-400 uppercase tracking-wider">{current.tag}</span>
        </div>

        <div className="flex items-center space-x-3">
          <span className="text-xs font-mono text-slate-500">
            Use <kbd className="px-1.5 py-0.5 rounded bg-lab-card border border-lab-border text-slate-300">←</kbd>{' '}
            <kbd className="px-1.5 py-0.5 rounded bg-lab-card border border-lab-border text-slate-300">→</kbd> or{' '}
            <kbd className="px-1.5 py-0.5 rounded bg-lab-card border border-lab-border text-slate-300">Space</kbd>
          </span>
          <button
            onClick={onExit}
            className="p-1.5 rounded-lg bg-lab-card border border-lab-border text-slate-300 hover:text-white hover:bg-lab-border transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Slide Content */}
      <div className="flex-1 overflow-y-auto px-12 py-10 flex flex-col justify-center items-center">
        <div className="w-full max-w-5xl space-y-6">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
              <Icon className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
                {current.title}
              </h2>
              <p className="text-sm text-cyan-300 font-mono mt-0.5">{current.subtitle}</p>
            </div>
          </div>

          <div className="pt-2">{current.content}</div>
        </div>
      </div>

      {/* Slide Navigation Footer */}
      <div className="px-8 py-4 border-t border-lab-border flex items-center justify-between bg-lab-surface/80 backdrop-blur-md">
        <button
          onClick={() => setCurrentSlide((p) => Math.max(0, p - 1))}
          disabled={currentSlide === 0}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-lab-card border border-lab-border disabled:opacity-30 hover:border-lab-borderLight text-xs font-mono font-medium transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Previous</span>
        </button>

        {/* Slide Dots */}
        <div className="flex items-center space-x-2">
          {slides.map((s, idx) => (
            <button
              key={s.id}
              onClick={() => setCurrentSlide(idx)}
              className={`h-2 rounded-full transition-all ${
                currentSlide === idx
                  ? 'w-8 bg-indigo-500'
                  : 'w-2 bg-lab-border hover:bg-slate-500'
              }`}
            />
          ))}
        </div>

        <button
          onClick={() => setCurrentSlide((p) => Math.min(totalSlides - 1, p + 1))}
          disabled={currentSlide === totalSlides - 1}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 text-white text-xs font-mono font-medium transition-colors shadow-sm"
        >
          <span>Next</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
