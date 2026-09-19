import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { Database, Calendar, BarChart2, PieChart as PieIcon, Layers, Info } from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { DatasetSummary, TimestepInfo } from '../types/api';
import { api } from '../services/api';

export const DatasetPage: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [timesteps, setTimesteps] = useState<TimestepInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([api.getSummary(), api.getTimesteps()])
      .then(([summaryData, timestepsData]) => {
        setSummary(summaryData);
        setTimesteps(timestepsData);
      })
      .catch((err) => console.error('Error fetching dataset info:', err))
      .finally(() => setLoading(false));
  }, []);

  const classPieData = [
    { name: 'Unknown (Unlabeled)', value: 157205, color: '#64748b' },
    { name: 'Licit (Class 2)', value: 42019, color: '#10b981' },
    { name: 'Illicit (Class 1)', value: 4545, color: '#f43f5e' },
  ];

  const featurePieData = [
    { name: 'Local Features', value: 93, color: '#6366f1' },
    { name: 'Aggregated Features (1-hop)', value: 72, color: '#38bdf8' },
  ];

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
          <Database className="w-3.5 h-3.5" />
          <span>DATASET EXPLORER</span>
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Elliptic Bitcoin Transaction Dataset Profile
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Temporal graph structure, class distribution dynamics, and chronological partitioning boundaries.
        </p>
      </div>

      {/* Dataset Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        <MetricCard label="Total Txs" value="203,769" badge="Nodes" badgeColor="indigo" />
        <MetricCard label="Directed Edges" value="234,355" badge="Edges" badgeColor="indigo" />
        <MetricCard label="Timesteps" value="49" badge="Intervals" badgeColor="cyan" />
        <MetricCard label="ML Features" value="165" badge="Continuous" badgeColor="cyan" />
        <MetricCard label="Illicit Txs" value="4,545" badge="Class 1" badgeColor="rose" />
        <MetricCard label="Licit Txs" value="42,019" badge="Class 2" badgeColor="emerald" />
        <MetricCard label="Unknown Txs" value="157,205" badge="Unlabeled" badgeColor="slate" />
      </div>

      {/* Chronological Temporal Split Banner */}
      <div className="lab-card p-6 bg-gradient-to-r from-lab-surface to-lab-card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Calendar className="w-4 h-4 text-indigo-400" />
            <h2 className="text-sm font-mono uppercase tracking-wider text-slate-200 font-semibold">
              Chronological Partitioning Architecture
            </h2>
          </div>
          <span className="text-xs font-mono text-cyan-400 px-2.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
            Strict Temporal Ordering
          </span>
        </div>

        <p className="text-xs text-slate-300 mb-5 leading-relaxed">
          <strong className="text-white">Crucial Methodological Requirement: </strong>
          The evaluation uses chronological partitioning rather than random splitting. Transactions are grouped by discrete 2-week timesteps.
          Models are trained strictly on past timesteps (1–34), hyperparameter-tuned on validation timesteps (35–39), and evaluated blindly on future unseen test timesteps (40–49).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Train */}
          <div className="p-4 rounded-lg bg-[#0b0e18] border border-indigo-500/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-indigo-400 uppercase">
                TRAINING SPLIT
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300">
                Timesteps 1–34
              </span>
            </div>
            <div className="text-xl font-mono font-bold text-white">
              29,894 <span className="text-xs font-normal text-slate-400">Labeled Txs</span>
            </div>
            <div className="text-xs font-mono text-slate-400 space-y-0.5">
              <div>Illicit: <span className="text-rose-400 font-semibold">3,462 (11.58%)</span></div>
              <div>Licit: <span className="text-emerald-400 font-semibold">26,432 (88.42%)</span></div>
            </div>
          </div>

          {/* Validation */}
          <div className="p-4 rounded-lg bg-[#0b0e18] border border-cyan-500/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
                VALIDATION SPLIT
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">
                Timesteps 35–39
              </span>
            </div>
            <div className="text-xl font-mono font-bold text-white">
              5,486 <span className="text-xs font-normal text-slate-400">Labeled Txs</span>
            </div>
            <div className="text-xs font-mono text-slate-400 space-y-0.5">
              <div>Illicit: <span className="text-rose-400 font-semibold">447 (8.15%)</span></div>
              <div>Licit: <span className="text-emerald-400 font-semibold">5,039 (91.85%)</span></div>
            </div>
          </div>

          {/* Test */}
          <div className="p-4 rounded-lg bg-[#0b0e18] border border-rose-500/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono font-bold text-rose-400 uppercase">
                TEST SPLIT (BLIND)
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">
                Timesteps 40–49
              </span>
            </div>
            <div className="text-xl font-mono font-bold text-white">
              11,184 <span className="text-xs font-normal text-slate-400">Labeled Txs</span>
            </div>
            <div className="text-xs font-mono text-slate-400 space-y-0.5">
              <div>Illicit: <span className="text-rose-400 font-semibold">636 (5.69%)</span></div>
              <div>Licit: <span className="text-emerald-400 font-semibold">10,548 (94.31%)</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Row 1: Volume Per Timestep & Illicit Ratio Dynamics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* A. Transaction Volume Per Timestep */}
        <div className="lab-card p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Transaction Volume by Discrete Timestep
              </h3>
              <p className="text-xs text-lab-muted font-mono">
                Labeled (Licit + Illicit) vs Total Volume across 49 timesteps
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timesteps} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2940" />
                <XAxis dataKey="timestep" stroke="#627294" tick={{ fontSize: 11, fill: '#627294' }} />
                <YAxis stroke="#627294" tick={{ fontSize: 11, fill: '#627294' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f1320', borderColor: '#1f2940', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                <Bar dataKey="total_txs" name="Total Txs" fill="#1f2940" radius={[2, 2, 0, 0]} />
                <Line type="monotone" dataKey="labeled_txs" name="Labeled Txs" stroke="#6366f1" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="illicit_txs" name="Illicit Txs" stroke="#f43f5e" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* B. Illicit Ratio Dynamics Across Timesteps */}
        <div className="lab-card p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Illicit Proportion Dynamics & Regime Shift
              </h3>
              <p className="text-xs text-lab-muted font-mono">
                Illicit ratio per timestep (notice severe drop at t=43–45 during darknet shutdown)
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timesteps} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2940" />
                <XAxis dataKey="timestep" stroke="#627294" tick={{ fontSize: 11, fill: '#627294' }} />
                <YAxis stroke="#627294" tick={{ fontSize: 11, fill: '#627294' }} domain={[0, 0.4]} />
                <Tooltip
                  formatter={(val: number) => [`${(val * 100).toFixed(2)}%`, 'Illicit Ratio']}
                  contentStyle={{ backgroundColor: '#0f1320', borderColor: '#1f2940', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                <Line
                  type="monotone"
                  dataKey="illicit_ratio"
                  name="Illicit Ratio (Illicit / Labeled)"
                  stroke="#f43f5e"
                  strokeWidth={2.5}
                  dot={{ r: 2, fill: '#f43f5e' }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Chart Row 2: Distribution Pies & Feature Space Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Class Breakdown Pie */}
        <div className="lab-card p-6">
          <h3 className="text-sm font-semibold text-white mb-2">
            Class Distribution & Unknown Dominance
          </h3>
          <p className="text-xs text-lab-muted mb-4 font-mono">
            77.15% of all transactions in the Bitcoin graph have no ground-truth label.
          </p>

          <div className="h-60 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, percent }) => `${name.split(' ')[0]} (${(percent * 100).toFixed(1)}%)`}
                >
                  {classPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f1320', borderColor: '#1f2940', borderRadius: '8px', fontSize: '12px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Space Breakdown */}
        <div className="lab-card p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white mb-2">
              Feature Space Composition (165 Total ML Features)
            </h3>
            <p className="text-xs text-lab-muted mb-4 font-mono">
              93 local transaction moments + 72 aggregated 1-hop neighborhood features.
            </p>

            <div className="space-y-4 text-xs font-mono">
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border space-y-1">
                <div className="flex items-center justify-between text-indigo-400 font-bold">
                  <span>93 Local Features (cols 0..92)</span>
                  <span>56.4%</span>
                </div>
                <p className="text-[11px] text-slate-400 font-sans">
                  Statistical moments of the transaction itself: input/output counts, transaction fee, output volume, output scripts, and timestep-level activity metrics.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-lab-card border border-lab-border space-y-1">
                <div className="flex items-center justify-between text-cyan-400 font-bold">
                  <span>72 Aggregated Features (cols 93..164)</span>
                  <span>43.6%</span>
                </div>
                <p className="text-[11px] text-slate-400 font-sans">
                  1-hop neighborhood aggregate statistics: min, max, mean, and standard deviation of neighbor transactions in upstream inputs and downstream outputs.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
