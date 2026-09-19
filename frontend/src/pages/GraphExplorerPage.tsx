import React, { useState, useEffect } from 'react';
import {
  Search,
  Network,
  Share2,
  Sliders,
  Eye,
  Crosshair,
  ArrowRight,
  ShieldCheck,
  ShieldAlert,
  HelpCircle,
  Hash,
  Clock,
  Layers,
  Sparkles
} from 'lucide-react';
import { GraphCanvas } from '../components/GraphCanvas';
import { GraphNode, SampleTransaction, SubgraphResponse, TransactionDetail } from '../types/api';
import { api } from '../services/api';

export const GraphExplorerPage: React.FC = () => {
  const [searchTxId, setSearchTxId] = useState<string>('232629023');
  const [kHop, setKHop] = useState<number>(1);
  const [subgraph, setSubgraph] = useState<SubgraphResponse | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [nodeDetail, setNodeDetail] = useState<TransactionDetail | null>(null);
  const [samples, setSamples] = useState<SampleTransaction[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load sample transactions on mount
  useEffect(() => {
    api.getSampleTransactions()
      .then((data) => setSamples(data))
      .catch((err) => console.error('Failed to load sample transactions:', err));
  }, []);

  // Fetch neighborhood
  const loadNeighborhood = async (txIdNum: number, hopVal: number) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const data = await api.getNeighborhood(txIdNum, hopVal, 80);
      setSubgraph(data);
      if (data.center_node) {
        setSelectedNode(data.center_node);
        loadNodeDetails(data.center_node.tx_id);
      }
    } catch (err: any) {
      setErrorMsg(err?.response?.data?.detail || `Transaction ID ${txIdNum} not found.`);
      setSubgraph(null);
      setSelectedNode(null);
    } finally {
      setLoading(false);
    }
  };

  const loadNodeDetails = async (txIdNum: number) => {
    try {
      const detail = await api.getTransaction(txIdNum);
      setNodeDetail(detail);
    } catch (err) {
      console.error('Failed to load transaction details:', err);
    }
  };

  // Initial load
  useEffect(() => {
    loadNeighborhood(232629023, kHop);
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const idNum = parseInt(searchTxId.trim(), 10);
    if (isNaN(idNum)) {
      setErrorMsg('Please enter a valid numeric Transaction ID.');
      return;
    }
    loadNeighborhood(idNum, kHop);
  };

  const handleHopChange = (newHop: number) => {
    setKHop(newHop);
    if (subgraph?.center_tx_id) {
      loadNeighborhood(subgraph.center_tx_id, newHop);
    }
  };

  const handleNodeClick = (node: GraphNode) => {
    setSelectedNode(node);
    loadNodeDetails(node.tx_id);
  };

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto flex flex-col h-[calc(100vh-4rem)]">
      {/* Top Header & Search Bar */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 mb-1">
            <Network className="w-3.5 h-3.5" />
            <span>SUBGRAPH NEIGHBORHOOD EXPLORER</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Transaction Graph Local Neighborhood
          </h1>
        </div>

        {/* Search & Hop Toggle */}
        <div className="flex items-center space-x-3">
          {/* Hop Selector */}
          <div className="flex items-center bg-lab-card border border-lab-border p-1 rounded-lg text-xs font-mono">
            <span className="px-2 text-lab-muted">Radius:</span>
            <button
              onClick={() => handleHopChange(1)}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                kHop === 1 ? 'bg-indigo-600 text-white font-bold' : 'text-slate-400 hover:text-white'
              }`}
            >
              1-Hop
            </button>
            <button
              onClick={() => handleHopChange(2)}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                kHop === 2 ? 'bg-indigo-600 text-white font-bold' : 'text-slate-400 hover:text-white'
              }`}
            >
              2-Hop
            </button>
          </div>

          {/* Search Form */}
          <form onSubmit={handleSearch} className="flex items-center space-x-2">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
              <input
                type="text"
                value={searchTxId}
                onChange={(e) => setSearchTxId(e.target.value)}
                placeholder="Enter Tx ID (e.g. 232629023)..."
                className="w-56 pl-9 pr-3 py-1.5 rounded-lg bg-lab-card border border-lab-border text-xs text-white placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
            <button
              type="submit"
              className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium font-mono transition-colors shadow-sm"
            >
              Search
            </button>
          </form>
        </div>
      </div>

      {/* Quick Sample Selector Chips */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1 text-xs font-mono">
        <span className="text-lab-muted shrink-0 text-[11px]">Quick Samples:</span>
        {samples.map((s) => (
          <button
            key={s.tx_id}
            onClick={() => {
              setSearchTxId(s.tx_id.toString());
              loadNeighborhood(s.tx_id, kHop);
            }}
            className={`px-2.5 py-1 rounded-md border text-[11px] shrink-0 transition-all ${
              subgraph?.center_tx_id === s.tx_id
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

      {/* Main Split View: Canvas (Left) + Inspector (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        {/* Left: Interactive Canvas */}
        <div className="lg:col-span-2 h-[560px] relative">
          {errorMsg && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 px-4 py-2 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-mono">
              {errorMsg}
            </div>
          )}
          <GraphCanvas
            subgraph={subgraph}
            selectedNode={selectedNode}
            onSelectNode={handleNodeClick}
            loading={loading}
          />
        </div>

        {/* Right: Node Inspector Panel */}
        <div className="lab-card p-5 flex flex-col justify-between overflow-y-auto h-[560px]">
          {selectedNode ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-lab-border">
                <div className="flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold">
                    Node Inspector
                  </h3>
                </div>
                {selectedNode.is_center && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    Query Center
                  </span>
                )}
              </div>

              {/* Node Basic Info */}
              <div className="p-3 rounded-lg bg-lab-card border border-lab-border space-y-2 font-mono text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Transaction ID:</span>
                  <span className="text-white font-bold">{selectedNode.tx_id}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Ground Truth:</span>
                  <span
                    className={`font-semibold capitalize px-2 py-0.5 rounded ${
                      selectedNode.label === 'illicit'
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                        : selectedNode.label === 'licit'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        : 'bg-slate-500/10 text-slate-400 border border-slate-500/20'
                    }`}
                  >
                    {selectedNode.label}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Timestep & Split:</span>
                  <span className="text-cyan-300">
                    t = {selectedNode.timestep} ({selectedNode.split})
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Node Distance:</span>
                  <span className="text-slate-300">
                    {selectedNode.hop === 0 ? 'Center (0 hops)' : `${selectedNode.hop}-hop neighbor`}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Loaded Radius:</span>
                  <span className="text-indigo-300 font-semibold">
                    {kHop}-Hop
                  </span>
                </div>
              </div>

              {/* Graph Degree Profile */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-mono uppercase text-lab-muted">
                  Degree Connectivity
                </span>
                <div className="grid grid-cols-3 gap-2 font-mono text-center">
                  <div className="p-2 rounded bg-lab-card border border-lab-border">
                    <div className="text-sm font-bold text-white">{selectedNode.total_degree}</div>
                    <div className="text-[10px] text-slate-500">Total Degree</div>
                  </div>
                  <div className="p-2 rounded bg-lab-card border border-lab-border">
                    <div className="text-sm font-bold text-indigo-400">{selectedNode.in_degree}</div>
                    <div className="text-[10px] text-slate-500">In-Degree</div>
                  </div>
                  <div className="p-2 rounded bg-lab-card border border-lab-border">
                    <div className="text-sm font-bold text-cyan-400">{selectedNode.out_degree}</div>
                    <div className="text-[10px] text-slate-500">Out-Degree</div>
                  </div>
                </div>
              </div>

              {/* Subgraph Context Summary */}
              {subgraph && (
                <div className="p-3 rounded-lg bg-lab-card border border-lab-border space-y-1.5 font-mono text-[11px]">
                  <span className="text-slate-400 font-semibold block mb-1">
                    Neighborhood Summary ({kHop}-Hop)
                  </span>
                  <div className="flex justify-between text-slate-400">
                    <span>Total Subgraph Nodes:</span>
                    <span className="text-white font-bold">{subgraph.total_nodes}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Total Subgraph Edges:</span>
                    <span className="text-white font-bold">{subgraph.total_edges}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Direct Neighbors:</span>
                    <span className="text-cyan-300 font-bold">
                      {subgraph.nodes.filter((n) => n.hop === 1).length}
                    </span>
                  </div>
                </div>
              )}

              {/* Feature Preview Sample */}
              {nodeDetail && (
                <div className="space-y-1.5">
                  <span className="text-[11px] font-mono uppercase text-lab-muted">
                    Feature Vector Preview (Sample 4 of 165)
                  </span>
                  <div className="p-2.5 rounded bg-lab-card border border-lab-border space-y-1 font-mono text-[10px]">
                    <div className="flex justify-between text-slate-400">
                      <span>local_0 (Input Vol Moment):</span>
                      <span className="text-cyan-300">
                        {nodeDetail.features_local['local_0']?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>local_1 (Output Fee Moment):</span>
                      <span className="text-cyan-300">
                        {nodeDetail.features_local['local_1']?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>agg_0 (1-Hop Inflow Mean):</span>
                      <span className="text-indigo-300">
                        {nodeDetail.features_agg['agg_0']?.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>agg_1 (1-Hop Inflow Std):</span>
                      <span className="text-indigo-300">
                        {nodeDetail.features_agg['agg_1']?.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500 font-mono text-xs">
              <Crosshair className="w-8 h-8 mb-2 stroke-1" />
              <span>Click on any node in the graph canvas to inspect its relational attributes.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
