import React, { useState, useEffect } from 'react';
import {
  Crosshair,
  Cpu,
  ArrowRight,
  ShieldCheck,
  ShieldAlert,
  HelpCircle,
  BarChart2,
  Network,
  Sparkles,
  Info,
  Clock,
  Layers,
  ChevronRight,
  Gauge
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid
} from 'recharts';
import { ModelInfo, PredictionResponse, SampleTransaction } from '../types/api';
import { api } from '../services/api';

export const PredictionPage: React.FC = () => {
  const [txIdInput, setTxIdInput] = useState<string>('232629023');
  const [selectedModel, setSelectedModel] = useState<string>('random_forest');
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [samples, setSamples] = useState<SampleTransaction[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load models and samples on mount
  useEffect(() => {
    Promise.all([api.getAvailableModels(), api.getSampleTransactions()])
      .then(([models, sampleList]) => {
        setAvailableModels(models);
        setSamples(sampleList);
      })
      .catch((err) => console.error('Failed to load initial prediction metadata:', err));
  }, []);

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const idNum = parseInt(txIdInput.trim(), 10);
    if (isNaN(idNum)) {
      setErrorMsg('Please enter a valid numeric Transaction ID.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await api.predict(idNum, selectedModel);
      setPrediction(res);
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail || 'Inference error occurred.');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  // Run initial prediction on load
  useEffect(() => {
    handlePredict();
  }, []);

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
          <Crosshair className="w-3.5 h-3.5" />
          <span>LIVE MODEL INFERENCE & EXPLAINABILITY ENGINE</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Real-Time Transaction Analysis & Multi-Model Inference
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Select an authentic transaction and model to execute real-time inference with exact training preprocessing.
        </p>
      </div>

      {/* Input Form & Model Selector Bar */}
      <div className="lab-card p-6">
        <form onSubmit={handlePredict} className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-end">
          {/* Transaction ID Input */}
          <div className="lg:col-span-4 space-y-1.5">
            <label className="text-xs font-mono text-slate-300 font-semibold block">
              Transaction ID
            </label>
            <div className="relative">
              <input
                type="text"
                value={txIdInput}
                onChange={(e) => setTxIdInput(e.target.value)}
                placeholder="e.g. 232629023"
                className="w-full px-3.5 py-2 rounded-lg bg-[#0b0e17] border border-lab-border text-xs text-white font-mono placeholder:text-slate-600 focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          {/* Model Selector */}
          <div className="lg:col-span-5 space-y-1.5">
            <label className="text-xs font-mono text-slate-300 font-semibold block">
              Trained Model Architecture
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3.5 py-2 rounded-lg bg-[#0b0e17] border border-lab-border text-xs text-white font-mono focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              {availableModels.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.name} ({m.family}) — {m.is_gnn ? 'Graph ML' : 'Tabular ML'}
                </option>
              ))}
            </select>
          </div>

          {/* Analyze Button */}
          <div className="lg:col-span-3">
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900 text-white text-xs font-mono font-bold uppercase tracking-wider flex items-center justify-center space-x-2 transition-all shadow-lg shadow-indigo-600/20"
            >
              {loading ? (
                <span>Executing Forward Pass...</span>
              ) : (
                <>
                  <Crosshair className="w-4 h-4" />
                  <span>Analyze Transaction</span>
                </>
              )}
            </button>
          </div>
        </form>

        {/* Quick Sample Selector Chips */}
        <div className="mt-4 pt-4 border-t border-lab-border flex items-center space-x-2 overflow-x-auto text-xs font-mono">
          <span className="text-lab-muted shrink-0 text-[11px]">Select Sample:</span>
          {samples.map((s) => (
            <button
              key={s.tx_id}
              onClick={() => {
                setTxIdInput(s.tx_id.toString());
                api.predict(s.tx_id, selectedModel).then(setPrediction);
              }}
              className={`px-2.5 py-1 rounded-md border text-[11px] shrink-0 transition-all ${
                prediction?.tx_id === s.tx_id
                  ? 'bg-indigo-600/30 text-indigo-300 border-indigo-500 font-bold'
                  : 'bg-lab-card border-lab-border text-slate-400 hover:text-slate-200 hover:border-lab-borderLight'
              }`}
            >
              <span
                className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${
                  s.label === 'illicit'
                    ? 'bg-rose-500'
                    : s.label === 'licit'
                    ? 'bg-emerald-500'
                    : 'bg-slate-500'
                }`}
              />
              {s.tx_id} ({s.label})
            </button>
          ))}
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs font-mono">
          {errorMsg}
        </div>
      )}

      {/* Prediction Output Results */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-in fade-in duration-200">
          {/* Main Decision & Confidence Card */}
          <div className="lg:col-span-5 lab-card p-6 flex flex-col justify-between space-y-6">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase text-lab-muted">
                  Model Output Assessment
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-lab-card border border-lab-border text-slate-300">
                  {prediction.model_name}
                </span>
              </div>

              {/* Prediction Banner */}
              <div
                className={`mt-4 p-5 rounded-xl border flex items-center space-x-4 ${
                  prediction.predicted_class === 1
                    ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                    : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                }`}
              >
                {prediction.predicted_class === 1 ? (
                  <ShieldAlert className="w-8 h-8 shrink-0 text-rose-400" />
                ) : (
                  <ShieldCheck className="w-8 h-8 shrink-0 text-emerald-400" />
                )}
                <div>
                  <div className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300">
                    Classification Result
                  </div>
                  <div className="text-lg font-bold">
                    {prediction.predicted_label}
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                    (Model prediction, not verified ground truth)
                  </div>
                </div>
              </div>

              {/* Probability Meter */}
              <div className="mt-6 space-y-2 font-mono">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Predicted Probability (P[Illicit]):</span>
                  <span className="text-white font-bold text-sm">
                    {(prediction.predicted_probability * 100).toFixed(2)}%
                  </span>
                </div>

                <div className="relative w-full h-3.5 bg-lab-card rounded-full overflow-hidden border border-lab-border">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      prediction.predicted_probability >= prediction.classification_threshold
                        ? 'bg-gradient-to-r from-rose-600 to-rose-400'
                        : 'bg-gradient-to-r from-emerald-600 to-emerald-400'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, prediction.predicted_probability * 100))}%` }}
                  />
                  {/* Decision Threshold Marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_4px_#ffffff]"
                    style={{ left: `${prediction.classification_threshold * 100}%` }}
                    title={`Decision Threshold: ${(prediction.classification_threshold * 100).toFixed(0)}%`}
                  />
                </div>

                <div className="flex items-center justify-between text-[10px] text-slate-500">
                  <span>0.0 (Licit)</span>
                  <span className="text-cyan-400 font-semibold">
                    Optimal Threshold: τ = {prediction.classification_threshold.toFixed(2)}
                  </span>
                  <span>1.0 (Illicit)</span>
                </div>
              </div>
            </div>

            {/* Metadata Grid */}
            <div className="pt-4 border-t border-lab-border grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-2.5 rounded bg-lab-card border border-lab-border">
                <div className="text-slate-500 text-[10px]">Ground Truth</div>
                <div className="text-white font-bold capitalize mt-0.5">{prediction.ground_truth_label}</div>
              </div>
              <div className="p-2.5 rounded bg-lab-card border border-lab-border">
                <div className="text-slate-500 text-[10px]">Timestep & Split</div>
                <div className="text-cyan-300 font-bold mt-0.5">t = {prediction.timestep} ({prediction.split})</div>
              </div>
              <div className="p-2.5 rounded bg-lab-card border border-lab-border">
                <div className="text-slate-500 text-[10px]">In / Out Degree</div>
                <div className="text-white font-bold mt-0.5">{prediction.in_degree} in / {prediction.out_degree} out</div>
              </div>
              <div className="p-2.5 rounded bg-lab-card border border-lab-border">
                <div className="text-slate-500 text-[10px]">1-Hop Neighbors</div>
                <div className="text-indigo-300 font-bold mt-0.5">{prediction.neighborhood_size_1hop} transactions</div>
              </div>
            </div>
          </div>

          {/* Right: Explainability / Structural Context */}
          <div className="lg:col-span-7 lab-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <BarChart2 className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">
                    {prediction.is_gnn
                      ? 'Relational Subgraph Context & Neighborhood Structure'
                      : 'Model-Derived Top Feature Importances'}
                  </h3>
                </div>
                <span className="text-[10px] font-mono text-lab-muted">
                  {prediction.is_gnn ? 'GNN Inductive Context' : 'Tree Gini Importance'}
                </span>
              </div>

              {/* Tabular Feature Importance Chart */}
              {prediction.feature_contributions && prediction.feature_contributions.length > 0 && (
                <div className="space-y-4">
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={prediction.feature_contributions}
                        layout="vertical"
                        margin={{ top: 5, right: 20, left: 40, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2940" />
                        <XAxis type="number" stroke="#627294" tick={{ fontSize: 10, fill: '#627294' }} />
                        <YAxis
                          type="category"
                          dataKey="feature_name"
                          stroke="#627294"
                          tick={{ fontSize: 10, fill: '#627294' }}
                          width={80}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f1320', borderColor: '#1f2940', borderRadius: '8px', fontSize: '11px' }}
                          formatter={(val: any) => [Number(val || 0).toFixed(4), 'Relative Importance']}
                        />
                        <Bar dataKey="importance_or_weight" fill="#6366f1" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="space-y-1 text-xs font-mono">
                    <span className="text-slate-400 text-[11px] block">Top Ranked Feature Values:</span>
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      {prediction.feature_contributions.slice(0, 4).map((fc) => (
                        <div key={fc.feature_name} className="p-2 rounded bg-lab-card border border-lab-border flex justify-between">
                          <span className="text-slate-400">{fc.feature_name}:</span>
                          <span className="text-cyan-300 font-bold">{fc.value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* GNN Graph Context */}
              {prediction.graph_context && (
                <div className="space-y-4 font-mono text-xs">
                  <p className="text-slate-300 font-sans text-xs leading-relaxed">
                    Graph Neural Networks compute predictions by aggregating structural information across the transaction neighborhood.
                    Below is the local subnetwork composition for this query node:
                  </p>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <div className="p-3 rounded bg-lab-card border border-lab-border text-center">
                      <div className="text-lg font-bold text-white">
                        {prediction.graph_context.subgraph_nodes_2hop}
                      </div>
                      <div className="text-[10px] text-slate-500">2-Hop Subgraph Nodes</div>
                    </div>
                    <div className="p-3 rounded bg-lab-card border border-lab-border text-center">
                      <div className="text-lg font-bold text-white">
                        {prediction.graph_context.subgraph_edges_2hop}
                      </div>
                      <div className="text-[10px] text-slate-500">2-Hop Subgraph Edges</div>
                    </div>
                    <div className="p-3 rounded bg-lab-card border border-lab-border text-center">
                      <div className="text-lg font-bold text-cyan-400">
                        {prediction.graph_context.homophily_ratio}
                      </div>
                      <div className="text-[10px] text-slate-500">Labeled Homophily</div>
                    </div>
                  </div>

                  <div className="p-3 rounded bg-lab-card border border-lab-border space-y-1.5">
                    <span className="text-slate-400 font-semibold block mb-1">
                      1-Hop Neighbor Class Breakdown:
                    </span>
                    <div className="flex items-center justify-between">
                      <span className="text-rose-400">Illicit Neighbors:</span>
                      <span className="text-white font-bold">{prediction.graph_context.neighbor_class_composition.illicit}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-emerald-400">Licit Neighbors:</span>
                      <span className="text-white font-bold">{prediction.graph_context.neighbor_class_composition.licit}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Unknown Neighbors:</span>
                      <span className="text-slate-200 font-bold">{prediction.graph_context.neighbor_class_composition.unknown}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
