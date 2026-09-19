import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ErrorBar
} from 'recharts';
import { Cpu, BarChart3, Filter, SlidersHorizontal, Info, ShieldCheck } from 'lucide-react';
import { MainResultRecord } from '../types/api';
import { api } from '../services/api';

export const ModelLabPage: React.FC = () => {
  const [mainResults, setMainResults] = useState<MainResultRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedFamily, setSelectedFamily] = useState<string>('all');

  useEffect(() => {
    api.getMainResults()
      .then((data) => setMainResults(data))
      .catch((err) => console.error('Failed to load main results:', err))
      .finally(() => setLoading(false));
  }, []);

  const families = ['all', ...new Set(mainResults.map((m) => m.family))];

  const filteredResults = mainResults.filter((m) =>
    selectedFamily === 'all' ? true : m.family === selectedFamily
  );

  const chartData = filteredResults.map((r) => ({
    model: r.model,
    f1_mean: r.test_f1_mean,
    f1_error: r.test_f1_std,
    prauc_mean: r.test_prauc_mean,
    prauc_error: r.test_prauc_std,
    precision: r.test_precision_mean,
    recall: r.test_recall_mean,
    mcc: r.test_mcc_mean,
  }));

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
            <Cpu className="w-3.5 h-3.5" />
            <span>MODEL COMPARISON LAB</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Observed Model Performance Matrix (7 Architectures)
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Empirical evaluation across conventional tabular models and graph neural networks under chronological test evaluation (N=5 seeds).
          </p>
        </div>

        {/* Filter */}
        <div className="flex items-center space-x-2 bg-lab-card border border-lab-border px-3 py-1.5 rounded-lg text-xs font-mono">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-lab-muted">Family:</span>
          <select
            value={selectedFamily}
            onChange={(e) => setSelectedFamily(e.target.value)}
            className="bg-transparent text-white focus:outline-none cursor-pointer"
          >
            {families.map((f) => (
              <option key={f} value={f} className="bg-lab-surface text-white">
                {f === 'all' ? 'All Model Families' : f}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Charts: F1 Comparison and PR-AUC Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Test F1 Chart */}
        <div className="lab-card p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Observed Test Illicit F1 Performance
              </h3>
              <p className="text-xs text-lab-muted font-mono">
                Mean ± Standard Deviation across 5 random seeds
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 15, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2940" />
                <XAxis
                  dataKey="model"
                  stroke="#627294"
                  tick={{ fontSize: 10, fill: '#627294' }}
                  angle={-25}
                  textAnchor="end"
                />
                <YAxis stroke="#627294" tick={{ fontSize: 11, fill: '#627294' }} domain={[0, 1.0]} />
                <Tooltip
                  formatter={(val: any) => [Number(val || 0).toFixed(4), 'Test Illicit F1']}
                  contentStyle={{ backgroundColor: '#0f1320', borderColor: '#1f2940', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="f1_mean" fill="#6366f1" radius={[4, 4, 0, 0]}>
                  <ErrorBar dataKey="f1_error" width={4} strokeWidth={1.5} stroke="#c7d2fe" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Test PR-AUC Chart */}
        <div className="lab-card p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Observed Test Precision-Recall AUC (PR-AUC)
              </h3>
              <p className="text-xs text-lab-muted font-mono">
                Evaluated under severe 5.69% test class imbalance
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 15, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2940" />
                <XAxis
                  dataKey="model"
                  stroke="#627294"
                  tick={{ fontSize: 10, fill: '#627294' }}
                  angle={-25}
                  textAnchor="end"
                />
                <YAxis stroke="#627294" tick={{ fontSize: 11, fill: '#627294' }} domain={[0, 1.0]} />
                <Tooltip
                  formatter={(val: any) => [Number(val || 0).toFixed(4), 'Test PR-AUC']}
                  contentStyle={{ backgroundColor: '#0f1320', borderColor: '#1f2940', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="prauc_mean" fill="#38bdf8" radius={[4, 4, 0, 0]}>
                  <ErrorBar dataKey="prauc_error" width={4} strokeWidth={1.5} stroke="#bae6fd" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Dynamic Master Results Table */}
      <div className="lab-card overflow-hidden">
        <div className="p-4 border-b border-lab-border flex items-center justify-between bg-lab-card/30">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-indigo-400" />
            <h3 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-semibold">
              Master Experimental Evaluation Table (Dynamic from MASTER_MAIN_RESULTS.csv)
            </h3>
          </div>
          <span className="text-[11px] font-mono text-lab-muted">
            N = 5 Seeds (42, 123, 456, 789, 999)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-lab-card/60 text-slate-400 border-b border-lab-border text-[11px] uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-3">Family</th>
                <th className="py-3 px-3">Val F1</th>
                <th className="py-3 px-3">Val PR-AUC</th>
                <th className="py-3 px-3 text-white font-bold">Test F1 (Mean ± Std)</th>
                <th className="py-3 px-3 text-cyan-300 font-bold">Test PR-AUC</th>
                <th className="py-3 px-3">Test Precision</th>
                <th className="py-3 px-3">Test Recall</th>
                <th className="py-3 px-3">Test MCC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-lab-border/40">
              {filteredResults.map((r, i) => (
                <tr key={r.model} className="hover:bg-lab-card/40 transition-colors">
                  <td className="py-3 px-4 font-semibold text-white flex items-center space-x-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                    <span>{r.model}</span>
                  </td>
                  <td className="py-3 px-3 text-slate-400">{r.family}</td>
                  <td className="py-3 px-3 text-slate-300">
                    {r.val_f1_mean.toFixed(4)} ± {r.val_f1_std.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {r.val_prauc_mean.toFixed(4)} ± {r.val_prauc_std.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-white font-bold">
                    {r.test_f1_mean.toFixed(4)} ± {r.test_f1_std.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-cyan-300 font-bold">
                    {r.test_prauc_mean.toFixed(4)} ± {r.test_prauc_std.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {r.test_precision_mean.toFixed(4)} ± {r.test_precision_std.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {r.test_recall_mean.toFixed(4)} ± {r.test_recall_std.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {r.test_mcc_mean.toFixed(4)} ± {r.test_mcc_std.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
