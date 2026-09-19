import React, { useEffect, useState } from 'react';
import {
  ShieldAlert,
  ArrowRight,
  Database,
  Layers,
  Cpu,
  GitFork,
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Network
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { DatasetSummary } from '../types/api';
import { api } from '../services/api';

export const OverviewPage: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    api.getSummary()
      .then((data) => setSummary(data))
      .catch((err) => console.error('Failed to load dataset summary:', err))
      .finally(() => setLoading(false));
  }, []);

  const modelFamilies = [
    {
      name: 'Random Forest',
      family: 'Tree Ensemble',
      badge: 'Balanced Bootstrap',
      desc: 'Ensemble of 100 decision trees fitted with balanced bootstrap samples and max depth 15.',
      features: 'Full (165)',
      f1: '0.7247 ± 0.0027',
      prauc: '0.6599 ± 0.0025',
    },
    {
      name: 'XGBoost',
      family: 'Gradient Boosting',
      badge: 'Histogram Bins',
      desc: '100 gradient boosted trees (lr=0.1, max_depth=6) with subsampling and feature fractioning.',
      features: 'Full (165)',
      f1: '0.7131 ± 0.0111',
      prauc: '0.6744 ± 0.0028',
    },
    {
      name: 'MLP',
      family: 'Neural Network',
      badge: 'BatchNorm + Dropout',
      desc: 'Deep feed-forward network with 2 hidden layers (128-64), ReLU activations, and BatchNorm.',
      features: 'Full (165)',
      f1: '0.5848 ± 0.0194',
      prauc: '0.5115 ± 0.0187',
    },
    {
      name: 'Logistic Regression',
      family: 'Linear Model',
      badge: 'L2 Regularized',
      desc: 'Linear probability model over standardized continuous moments with L2 regularization.',
      features: 'Full (165)',
      f1: '0.4015 ± 0.0000',
      prauc: '0.2754 ± 0.0000',
    },
    {
      name: 'GAT',
      family: 'Graph Neural Network',
      badge: '8 Multi-Head Attention',
      desc: '2-layer Graph Attention Network with 8 attention heads per edge to dynamically weight neighbor relevance.',
      features: 'Full (165)',
      f1: '0.3308 ± 0.0602',
      prauc: '0.2667 ± 0.0695',
    },
    {
      name: 'GraphSAGE',
      family: 'Graph Neural Network',
      badge: 'Inductive Mean-Aggr',
      desc: 'Inductive message-passing architecture using mean-pooling neighbor aggregation and residual skip.',
      features: 'Full (165)',
      f1: '0.2822 ± 0.1248',
      prauc: '0.2366 ± 0.1386',
    },
    {
      name: 'GCN',
      family: 'Graph Neural Network',
      badge: 'Spectral Convolution',
      desc: 'Isotropic graph convolutional network with symmetric normalized Laplacian neighborhood aggregation.',
      features: 'Full (165)',
      f1: '0.2589 ± 0.0639',
      prauc: '0.2202 ± 0.0703',
    },
  ];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Hero Research Banner */}
      <div className="lab-card p-8 relative overflow-hidden bg-gradient-to-r from-lab-surface via-lab-surface to-lab-card border-lab-borderLight">
        <div className="absolute right-0 top-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-3xl space-y-3 relative z-10">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-mono text-indigo-400">
            <Layers className="w-3.5 h-3.5" />
            <span>Empirical ML Research Benchmark</span>
          </div>

          <h1 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
            Graph Neural Networks vs Tabular Machine Learning
          </h1>

          <p className="text-base text-slate-300 leading-relaxed font-light">
            Illicit transaction detection under chronological partitioning in the Bitcoin transaction network.
          </p>

          <div className="pt-2 flex items-center space-x-2 text-xs font-mono text-slate-400">
            <span className="text-slate-200 font-semibold">Central Research Question:</span>
            <span className="italic text-cyan-300">
              "Can graph-based learning improve illicit transaction detection by exploiting transaction relationships?"
            </span>
          </div>
        </div>
      </div>

      {/* Verified Dataset Facts (203,769 Txs, 234,355 Edges, 49 Timesteps, etc.) */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-indigo-400" />
            <h2 className="text-sm font-mono uppercase tracking-wider text-slate-300 font-semibold">
              Verified Dataset Facts
            </h2>
          </div>
          <span className="text-xs font-mono text-lab-muted">
            Elliptic Bitcoin Dataset
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <MetricCard
            label="Transactions"
            value="203,769"
            subValue="Graph Nodes"
            badge="Total"
            badgeColor="indigo"
          />
          <MetricCard
            label="Directed Edges"
            value="234,355"
            subValue="Payment Flows"
            badge="Edges"
            badgeColor="indigo"
          />
          <MetricCard
            label="Timesteps"
            value="49"
            subValue="2-Week Intervals"
            badge="Chronological"
            badgeColor="cyan"
          />
          <MetricCard
            label="ML Features"
            value="165"
            subValue="93 Local + 72 Agg"
            badge="Continuous"
            badgeColor="cyan"
          />
          <MetricCard
            label="Illicit Txs"
            value="4,545"
            subValue="Class 1 (9.76% labeled)"
            badge="Positive"
            badgeColor="rose"
          />
          <MetricCard
            label="Licit Txs"
            value="42,019"
            subValue="Class 2 (90.24% labeled)"
            badge="Negative"
            badgeColor="emerald"
          />
          <MetricCard
            label="Unknown"
            value="157,205"
            subValue="77.15% of all nodes"
            badge="Unlabeled"
            badgeColor="slate"
          />
        </div>
      </div>

      {/* Research Methodology Pipeline */}
      <div className="lab-card p-6">
        <div className="flex items-center space-x-2 mb-6">
          <Network className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-mono uppercase tracking-wider text-slate-300 font-semibold">
            Experimental Evaluation Pipeline
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-indigo-400 font-semibold">STEP 01</span>
              <Database className="w-3.5 h-3.5 text-slate-500" />
            </div>
            <div className="text-xs font-semibold text-white">DATASET & GRAPH</div>
            <p className="text-[11px] text-lab-muted">
              203,769 transactions, 234,355 directed edges, 165 features across 49 timesteps.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-lab-card border border-lab-border flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-indigo-400 font-semibold">STEP 02</span>
              <Layers className="w-3.5 h-3.5 text-slate-500" />
            </div>
            <div className="text-xs font-semibold text-white">TEMPORAL PARTITION</div>
            <p className="text-[11px] text-lab-muted">
              Chronological split: Train (t=1–34), Val (t=35–39), Test (t=40–49). No future leakage.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-lab-card border border-lab-border flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-indigo-400 font-semibold">STEP 03</span>
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
            </div>
            <div className="text-xs font-semibold text-white">TABULAR ML + GRAPH ML</div>
            <p className="text-[11px] text-lab-muted">
              7 model families evaluated with standard and cost-sensitive class weighting across 5 seeds.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-lab-card border border-lab-border flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-indigo-400 font-semibold">STEP 04</span>
              <BarChart3 className="w-3.5 h-3.5 text-slate-500" />
            </div>
            <div className="text-xs font-semibold text-white">CHRONOLOGICAL EVAL</div>
            <p className="text-[11px] text-lab-muted">
              Thresholds locked on validation split. Test evaluation over darknet regime shift.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-lab-card border border-lab-border flex flex-col justify-between space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-indigo-400 font-semibold">STEP 05</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-slate-500" />
            </div>
            <div className="text-xs font-semibold text-white">ABLATION & STATS</div>
            <p className="text-[11px] text-lab-muted">
              Unknown-node context, edge direction, layer depth, and Holm-corrected significance tests.
            </p>
          </div>
        </div>
      </div>

      {/* Evaluated Model Families */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <h2 className="text-sm font-mono uppercase tracking-wider text-slate-300 font-semibold">
              Seven Evaluated Model Architectures
            </h2>
          </div>
          <span className="text-xs font-mono text-lab-muted">
            Test Performance (Observed)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {modelFamilies.map((m) => (
            <div
              key={m.name}
              className="lab-card p-5 flex flex-col justify-between space-y-3 lab-card-hover"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-lab-border text-slate-300">
                    {m.family}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400">
                    {m.badge}
                  </span>
                </div>

                <h3 className="text-base font-bold text-white mt-2">
                  {m.name}
                </h3>
                <p className="text-xs text-lab-muted mt-1 leading-relaxed">
                  {m.desc}
                </p>
              </div>

              <div className="pt-3 border-t border-lab-border/70 space-y-1.5 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Test Illicit F1:</span>
                  <span className="text-white font-semibold">{m.f1}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Test PR-AUC:</span>
                  <span className="text-cyan-300 font-semibold">{m.prauc}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
