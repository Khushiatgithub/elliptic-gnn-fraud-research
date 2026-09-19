import React, { useEffect, useState } from 'react';
import {
  FlaskConical,
  Layers,
  HelpCircle,
  Compass,
  GitCommit,
  Dice5,
  Scale,
  Info,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import {
  AblationRecord,
  FeatureComparisonRecord,
  SeedStabilityRecord,
  StatisticalTestRecord,
  WeightingComparisonRecord
} from '../types/api';
import { api } from '../services/api';

type ExperimentTab = 'features' | 'unknown' | 'direction' | 'depth' | 'seeds' | 'stats';

export const ExperimentLabPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ExperimentTab>('features');

  const [featuresData, setFeaturesData] = useState<FeatureComparisonRecord[]>([]);
  const [ablationsData, setAblationsData] = useState<AblationRecord[]>([]);
  const [seedsData, setSeedsData] = useState<SeedStabilityRecord[]>([]);
  const [statsData, setStatsData] = useState<StatisticalTestRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([
      api.getFeatureComparison(),
      api.getAblations(),
      api.getSeedStability(),
      api.getStatisticalTests(),
    ])
      .then(([feats, abls, seeds, stats]) => {
        setFeaturesData(feats);
        setAblationsData(abls);
        setSeedsData(seeds);
        setStatsData(stats);
      })
      .catch((err) => console.error('Failed to load experiment data:', err))
      .finally(() => setLoading(false));
  }, []);

  const unknownAblations = ablationsData.filter((a) =>
    a.ablation.toLowerCase().includes('unknown')
  );
  const directionAblations = ablationsData.filter((a) =>
    a.ablation.toLowerCase().includes('direction')
  );
  const depthAblations = ablationsData.filter((a) =>
    a.ablation.toLowerCase().includes('depth')
  );

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
          <FlaskConical className="w-3.5 h-3.5" />
          <span>CONTROLLED EXPERIMENT LAB</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Ablation Studies & Statistical Significance Testing
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Controlled parameter sweeps isolating feature representations, topological dynamics, layer depth, seed stability, and paired significance tests.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex items-center space-x-2 border-b border-lab-border pb-2 overflow-x-auto text-xs font-mono">
        <button
          onClick={() => setActiveTab('features')}
          className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-all ${
            activeTab === 'features'
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500 font-bold'
              : 'text-slate-400 hover:text-white hover:bg-lab-card'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>1. FEATURE SPACE</span>
        </button>

        <button
          onClick={() => setActiveTab('unknown')}
          className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-all ${
            activeTab === 'unknown'
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500 font-bold'
              : 'text-slate-400 hover:text-white hover:bg-lab-card'
          }`}
        >
          <HelpCircle className="w-3.5 h-3.5" />
          <span>2. UNKNOWN NODES</span>
        </button>

        <button
          onClick={() => setActiveTab('direction')}
          className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-all ${
            activeTab === 'direction'
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500 font-bold'
              : 'text-slate-400 hover:text-white hover:bg-lab-card'
          }`}
        >
          <Compass className="w-3.5 h-3.5" />
          <span>3. EDGE DIRECTION</span>
        </button>

        <button
          onClick={() => setActiveTab('depth')}
          className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-all ${
            activeTab === 'depth'
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500 font-bold'
              : 'text-slate-400 hover:text-white hover:bg-lab-card'
          }`}
        >
          <GitCommit className="w-3.5 h-3.5" />
          <span>4. LAYER DEPTH</span>
        </button>

        <button
          onClick={() => setActiveTab('seeds')}
          className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-all ${
            activeTab === 'seeds'
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500 font-bold'
              : 'text-slate-400 hover:text-white hover:bg-lab-card'
          }`}
        >
          <Dice5 className="w-3.5 h-3.5" />
          <span>5. SEED STABILITY</span>
        </button>

        <button
          onClick={() => setActiveTab('stats')}
          className={`px-3 py-2 rounded-lg flex items-center space-x-2 transition-all ${
            activeTab === 'stats'
              ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500 font-bold'
              : 'text-slate-400 hover:text-white hover:bg-lab-card'
          }`}
        >
          <Scale className="w-3.5 h-3.5" />
          <span>6. STATISTICAL TESTS</span>
        </button>
      </div>

      {/* Tab 1: Feature Space Comparison */}
      {activeTab === 'features' && (
        <div className="space-y-4 animate-in fade-in duration-150">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border text-xs text-slate-300 font-mono flex items-start space-x-3">
            <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Feature Space Ablation: </strong>
              Evaluates Local Features (93 continuous moments) versus Full Features (165 features, including 72 aggregated 1-hop statistics).
              Aggregated features improve tree ensembles and GAT, whereas isotropic models can suffer from variance inflation.
            </div>
          </div>

          <div className="lab-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">Model</th>
                    <th className="py-3 px-3">Feature Space</th>
                    <th className="py-3 px-3">Count</th>
                    <th className="py-3 px-3">Test F1</th>
                    <th className="py-3 px-3">Test PR-AUC</th>
                    <th className="py-3 px-3">Test MCC</th>
                    <th className="py-3 px-3 text-cyan-300">Δ F1 (Full - Local)</th>
                    <th className="py-3 px-3 text-cyan-300">Δ PR-AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lab-border/40">
                  {featuresData.map((r, i) => (
                    <tr key={i} className="hover:bg-lab-card/40 transition-colors">
                      <td className="py-3 px-4 font-semibold text-white">{r.model}</td>
                      <td className="py-3 px-3 text-slate-300">{r.feature_space}</td>
                      <td className="py-3 px-3 text-slate-400">{r.feature_count}</td>
                      <td className="py-3 px-3 text-slate-200">{r.test_f1}</td>
                      <td className="py-3 px-3 text-slate-200">{r.test_prauc}</td>
                      <td className="py-3 px-3 text-slate-200">{r.test_mcc}</td>
                      <td className="py-3 px-3 text-cyan-300 font-semibold">{r.delta_f1}</td>
                      <td className="py-3 px-3 text-cyan-300 font-semibold">{r.delta_prauc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Unknown Nodes Context */}
      {activeTab === 'unknown' && (
        <div className="space-y-4 animate-in fade-in duration-150">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border text-xs text-slate-300 font-mono flex items-start space-x-3">
            <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Unknown-Node Structural Context Ablation: </strong>
              Compares GNN performance when message passing occurs over all 203,769 nodes (100% graph with ~77% unknown nodes) versus filtering the topology to labeled nodes only.
              Unweighted isotropic aggregation over unknown nodes dilutes the illicit signal in GCN/GraphSAGE, whereas GAT attention adaptively downweights uninformative unknown neighbors.
            </div>
          </div>

          <div className="lab-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">GNN Architecture</th>
                    <th className="py-3 px-3">Setting</th>
                    <th className="py-3 px-3">Test F1</th>
                    <th className="py-3 px-3">Test PR-AUC</th>
                    <th className="py-3 px-3">Test Precision</th>
                    <th className="py-3 px-3">Test Recall</th>
                    <th className="py-3 px-4">Mechanistic Interpretation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lab-border/40">
                  {unknownAblations.map((r, i) => (
                    <tr key={i} className="hover:bg-lab-card/40 transition-colors">
                      <td className="py-3 px-4 font-semibold text-white">{r.model}</td>
                      <td className="py-3 px-3 text-cyan-300">{r.setting}</td>
                      <td className="py-3 px-3 text-slate-200 font-semibold">{r.test_f1_formatted}</td>
                      <td className="py-3 px-3 text-slate-200">{r.test_prauc_formatted}</td>
                      <td className="py-3 px-3 text-slate-300">{r.test_precision_formatted}</td>
                      <td className="py-3 px-3 text-slate-300">{r.test_recall_formatted}</td>
                      <td className="py-3 px-4 text-lab-muted text-[11px]">{r.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Edge Directionality */}
      {activeTab === 'direction' && (
        <div className="space-y-4 animate-in fade-in duration-150">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border text-xs text-slate-300 font-mono flex items-start space-x-3">
            <Compass className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Edge Directionality Sensitivity: </strong>
              <span className="text-emerald-400 font-bold">Forward (u → v)</span> is the primary inductive temporal setting preserving causal Bitcoin payment flows (inputs → outputs).
              Backward (v → u) evaluates retrospective upstream aggregation, and Bidirectional (u ↔ v) is an exploratory setting.
            </div>
          </div>

          <div className="lab-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">GNN Model</th>
                    <th className="py-3 px-3">Direction</th>
                    <th className="py-3 px-3">Test F1</th>
                    <th className="py-3 px-3">Test PR-AUC</th>
                    <th className="py-3 px-3">Test Precision</th>
                    <th className="py-3 px-3">Test Recall</th>
                    <th className="py-3 px-4">Context Note</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lab-border/40">
                  {directionAblations.map((r, i) => (
                    <tr key={i} className="hover:bg-lab-card/40 transition-colors">
                      <td className="py-3 px-4 font-semibold text-white">{r.model}</td>
                      <td className="py-3 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            r.setting.startsWith('Forward')
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              : 'bg-lab-border text-slate-300'
                          }`}
                        >
                          {r.setting}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-slate-200 font-semibold">{r.test_f1_formatted}</td>
                      <td className="py-3 px-3 text-slate-200">{r.test_prauc_formatted}</td>
                      <td className="py-3 px-3 text-slate-300">{r.test_precision_formatted}</td>
                      <td className="py-3 px-3 text-slate-300">{r.test_recall_formatted}</td>
                      <td className="py-3 px-4 text-lab-muted text-[11px]">{r.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Layer Depth */}
      {activeTab === 'depth' && (
        <div className="space-y-4 animate-in fade-in duration-150">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border text-xs text-slate-300 font-mono flex items-start space-x-3">
            <GitCommit className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">GNN Depth & Over-Smoothing Study: </strong>
              Evaluates 1-layer (1-hop localized), 2-layer (2-hop structural hierarchy), and 3-layer message passing.
              Isotropic models (GCN: 0.2862 → 0.2589 → 0.2089) show monotonic performance decay due to over-smoothing across dense unknown neighborhoods.
            </div>
          </div>

          <div className="lab-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">Model</th>
                    <th className="py-3 px-3">Depth</th>
                    <th className="py-3 px-3">Test F1</th>
                    <th className="py-3 px-3">Test PR-AUC</th>
                    <th className="py-3 px-3">Test Precision</th>
                    <th className="py-3 px-3">Test Recall</th>
                    <th className="py-3 px-4">Observation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lab-border/40">
                  {depthAblations.map((r, i) => (
                    <tr key={i} className="hover:bg-lab-card/40 transition-colors">
                      <td className="py-3 px-4 font-semibold text-white">{r.model}</td>
                      <td className="py-3 px-3 text-cyan-300 font-bold">{r.setting}</td>
                      <td className="py-3 px-3 text-slate-200 font-semibold">{r.test_f1_formatted}</td>
                      <td className="py-3 px-3 text-slate-200">{r.test_prauc_formatted}</td>
                      <td className="py-3 px-3 text-slate-300">{r.test_precision_formatted}</td>
                      <td className="py-3 px-3 text-slate-300">{r.test_recall_formatted}</td>
                      <td className="py-3 px-4 text-lab-muted text-[11px]">{r.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 5: Seed Stability */}
      {activeTab === 'seeds' && (
        <div className="space-y-4 animate-in fade-in duration-150">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border text-xs text-slate-300 font-mono flex items-start space-x-3">
            <Dice5 className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Multi-Seed Reproducibility & Variance Analysis: </strong>
              Evaluation across random seeds 42, 123, 456, 789, 999.
              Tree ensembles exhibit near-zero variance across seeds (RF F1 range: 0.0062, XGB: 0.0239), whereas GNNs exhibit wider optimization landscape variability (GraphSAGE F1 range: 0.2445).
            </div>
          </div>

          <div className="lab-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">Model Config</th>
                    <th className="py-3 px-3">Seed</th>
                    <th className="py-3 px-3">Val F1</th>
                    <th className="py-3 px-3">Test F1</th>
                    <th className="py-3 px-3">Test PR-AUC</th>
                    <th className="py-3 px-3">Mean ± Std</th>
                    <th className="py-3 px-3 text-amber-400">Range (Max - Min)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lab-border/40">
                  {seedsData.map((r, i) => (
                    <tr key={i} className="hover:bg-lab-card/40 transition-colors">
                      <td className="py-2.5 px-4 font-semibold text-white">{r.model}</td>
                      <td className="py-2.5 px-3 text-cyan-400 font-bold">{r.seed}</td>
                      <td className="py-2.5 px-3 text-slate-400">{r.val_f1.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-slate-200">{r.test_f1.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-slate-200">{r.test_prauc.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-slate-300">
                        {r.test_f1_mean.toFixed(4)} ± {r.test_f1_std.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-amber-400 font-semibold">{r.test_f1_range.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 6: Statistical Significance Tests */}
      {activeTab === 'stats' && (
        <div className="space-y-4 animate-in fade-in duration-150">
          <div className="p-4 rounded-lg bg-lab-card border border-lab-border text-xs text-slate-300 font-mono flex items-start space-x-3">
            <Scale className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Paired Statistical Significance Analysis: </strong>
              Paired Student's t-test and Wilcoxon signed-rank test across 5 random seeds (N=5).
              <div className="mt-1 text-slate-400">
                Notice: At N=5 seeds, the theoretical minimum achievable Wilcoxon p-value is <strong>p = 0.0625</strong> (2⁻⁴), which cannot drop below α = 0.05.
                Furthermore, after Holm-Bonferroni correction for multiple hypothesis testing, paired differences do not retain statistical significance at α = 0.05.
              </div>
            </div>
          </div>

          <div className="lab-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-lab-card text-slate-400 border-b border-lab-border text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">Model A (GNN) vs Model B</th>
                    <th className="py-3 px-3">Metric</th>
                    <th className="py-3 px-3">Mean Difference (Δ)</th>
                    <th className="py-3 px-3">Paired t (p raw)</th>
                    <th className="py-3 px-3 text-amber-400">Holm Adjusted p</th>
                    <th className="py-3 px-3">Wilcoxon W (p)</th>
                    <th className="py-3 px-3">Cohen's dz</th>
                    <th className="py-3 px-4">Methodological Note</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-lab-border/40">
                  {statsData.map((r, i) => (
                    <tr key={i} className="hover:bg-lab-card/40 transition-colors">
                      <td className="py-3 px-4 font-semibold text-white">
                        <div>{r.model_a}</div>
                        <div className="text-[10px] text-slate-400 font-normal">vs {r.model_b}</div>
                      </td>
                      <td className="py-3 px-3 text-slate-300">{r.metric}</td>
                      <td className="py-3 px-3 text-cyan-300 font-bold">{r.mean_difference.toFixed(4)}</td>
                      <td className="py-3 px-3 text-slate-300">
                        t = {r.paired_t_statistic.toFixed(2)} (p = {r.paired_t_test_p_raw.toFixed(4)})
                      </td>
                      <td className="py-3 px-3 text-amber-400 font-bold">
                        {r.holm_adjusted_p.toFixed(4)}
                      </td>
                      <td className="py-3 px-3 text-slate-300">
                        W = {r.wilcoxon_w_statistic.toFixed(1)} (p = {r.wilcoxon_p_value.toFixed(4)})
                      </td>
                      <td className="py-3 px-3 text-slate-300">{r.paired_cohen_dz.toFixed(2)}</td>
                      <td className="py-3 px-4 text-lab-muted text-[10px]">{r.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
