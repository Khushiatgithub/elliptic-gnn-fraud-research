"""
Graph Topology and Network Structure Analysis for the Elliptic Dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("graph_analyzer")


class GraphAnalyzer:
    """
    Constructs and audits the Bitcoin transaction network structure,
    evaluating both Directed (financial flow) and Undirected (message-passing)
    graph formulations.
    """

    def __init__(self, df_edges: pd.DataFrame, df_features: pd.DataFrame):
        self.df_edges = df_edges
        self.df_features = df_features
        
        # All known nodes from features
        self.all_nodes = set(df_features["txId"])
        
        # Directed NetworkX graph
        logger.info("Instantiating Directed Graph (DiGraph)...")
        self.G_dir = nx.DiGraph()
        self.G_dir.add_nodes_from(self.all_nodes)
        self.G_dir.add_edges_from(zip(df_edges["txId1"], df_edges["txId2"]))
        
        logger.info(f"DiGraph initialized: {self.G_dir.number_of_nodes():,} nodes, {self.G_dir.number_of_edges():,} edges.")

    def compute_degree_statistics(self) -> Dict[str, Any]:
        """
        Compute in-degree, out-degree, and total degree distribution statistics.
        """
        logger.info("Computing degree distributions (in-degree, out-degree, total degree)...")
        
        in_degrees = np.array([d for _, d in self.G_dir.in_degree()])
        out_degrees = np.array([d for _, d in self.G_dir.out_degree()])
        total_degrees = in_degrees + out_degrees
        
        # Isolated nodes (degree == 0)
        isolated_nodes_count = int(np.sum(total_degrees == 0))
        isolated_nodes_pct = (isolated_nodes_count / len(self.all_nodes)) * 100.0
        
        def _get_dist_stats(arr: np.ndarray, name: str) -> Dict[str, float]:
            return {
                f"{name}_mean": float(np.mean(arr)),
                f"{name}_std": float(np.std(arr)),
                f"{name}_min": int(np.min(arr)),
                f"{name}_25pct": float(np.percentile(arr, 25)),
                f"{name}_median": float(np.median(arr)),
                f"{name}_75pct": float(np.percentile(arr, 75)),
                f"{name}_99pct": float(np.percentile(arr, 99)),
                f"{name}_max": int(np.max(arr)),
            }

        stats = {
            "total_nodes": len(self.all_nodes),
            "total_edges": self.G_dir.number_of_edges(),
            "isolated_nodes_count": isolated_nodes_count,
            "isolated_nodes_pct": isolated_nodes_pct,
            **_get_dist_stats(in_degrees, "in_degree"),
            **_get_dist_stats(out_degrees, "out_degree"),
            **_get_dist_stats(total_degrees, "total_degree"),
        }
        
        return stats

    def compute_connected_components(self) -> Dict[str, Any]:
        """
        Compute Weakly Connected Components (WCC) and Strongly Connected Components (SCC).
        """
        logger.info("Analyzing connected components...")
        
        # Strongly connected components
        scc_sizes = [len(c) for c in nx.strongly_connected_components(self.G_dir)]
        num_scc = len(scc_sizes)
        max_scc = max(scc_sizes) if scc_sizes else 0
        
        # Weakly connected components
        wcc_sizes = [len(c) for c in nx.weakly_connected_components(self.G_dir)]
        num_wcc = len(wcc_sizes)
        max_wcc = max(wcc_sizes) if wcc_sizes else 0
        
        logger.info(
            f"Components: WCC={num_wcc:,} (Largest WCC={max_wcc:,} nodes | {max_wcc/len(self.all_nodes):.2%}) | "
            f"SCC={num_scc:,} (Largest SCC={max_scc:,} nodes)"
        )
        
        return {
            "num_weakly_connected_components": num_wcc,
            "largest_wcc_size": max_wcc,
            "largest_wcc_fraction": float(max_wcc / len(self.all_nodes)),
            "num_strongly_connected_components": num_scc,
            "largest_scc_size": max_scc,
            "is_dag": nx.is_directed_acyclic_graph(self.G_dir)
        }

    def compute_temporal_graph_metrics(self) -> pd.DataFrame:
        """
        Analyze network scale and density across individual time steps.
        """
        logger.info("Computing per-timestep subgraph topology metrics...")
        
        # Merge edge endpoints with their time_steps
        tx_to_time = dict(zip(self.df_features["txId"], self.df_features["time_step"]))
        
        # Map source and target timestamps
        df_edges_time = self.df_edges.copy()
        df_edges_time["t1"] = df_edges_time["txId1"].map(tx_to_time)
        df_edges_time["t2"] = df_edges_time["txId2"].map(tx_to_time)
        
        # Cross-timestep edges check
        cross_time_edges = (df_edges_time["t1"] != df_edges_time["t2"]).sum()
        logger.info(f"Cross-timestep edges count: {cross_time_edges} (Expected: 0, since Elliptic defines disjoint subgraphs per timestep).")
        
        temporal_records = []
        for t in range(1, 50):
            nodes_t = self.df_features[self.df_features["time_step"] == t]["txId"].values
            edges_t = df_edges_time[df_edges_time["t1"] == t]
            
            n_nodes = len(nodes_t)
            n_edges = len(edges_t)
            
            # Density in directed graph: |E| / (|V| * (|V| - 1))
            density = (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
            
            temporal_records.append({
                "time_step": t,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "edge_to_node_ratio": float(n_edges / n_nodes) if n_nodes > 0 else 0.0,
                "graph_density": density
            })
            
        return pd.DataFrame(temporal_records)

    def export_summary_tables(self, tables_dir: Path) -> Dict[str, Path]:
        """Export graph metrics to CSV."""
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Degree & component metrics
        deg_stats = self.compute_degree_statistics()
        comp_stats = self.compute_connected_components()
        
        summary_rows = [
            {"Metric": "Graph Formulation", "Value": "Directed (Bitcoin Fund Flow) & Undirected (Message Passing)", "Research Significance": "Directed edges model Bitcoin input-output lineage"},
            {"Metric": "Total Node Count (|V|)", "Value": f"{deg_stats['total_nodes']:,}", "Research Significance": "Full transaction set across 49 timesteps"},
            {"Metric": "Total Directed Edge Count (|E|)", "Value": f"{deg_stats['total_edges']:,}", "Research Significance": "Observed flow of Bitcoin payments between transactions"},
            {"Metric": "Isolated Nodes (Degree = 0)", "Value": f"{deg_stats['isolated_nodes_count']:,} ({deg_stats['isolated_nodes_pct']:.2f}%)", "Research Significance": "Nodes without 1-hop edges in the temporal slice"},
            {"Metric": "In-Degree (Mean ± Std)", "Value": f"{deg_stats['in_degree_mean']:.3f} ± {deg_stats['in_degree_std']:.3f}", "Research Significance": "Number of input transactions funding a transaction"},
            {"Metric": "In-Degree Max", "Value": str(deg_stats['in_degree_max']), "Research Significance": "Super-hub incoming transaction"},
            {"Metric": "Out-Degree (Mean ± Std)", "Value": f"{deg_stats['out_degree_mean']:.3f} ± {deg_stats['out_degree_std']:.3f}", "Research Significance": "Number of output transactions spawned"},
            {"Metric": "Out-Degree Max", "Value": str(deg_stats['out_degree_max']), "Research Significance": "Super-hub outgoing transaction"},
            {"Metric": "Total Degree (Mean ± Std)", "Value": f"{deg_stats['total_degree_mean']:.3f} ± {deg_stats['total_degree_std']:.3f}", "Research Significance": "Overall neighborhood density"},
            {"Metric": "Weakly Connected Components (WCC)", "Value": f"{comp_stats['num_weakly_connected_components']:,}", "Research Significance": "Partitioned clusters across graph"},
            {"Metric": "Largest WCC Node Fraction", "Value": f"{comp_stats['largest_wcc_fraction']:.2%}", "Research Significance": "Size of primary giant component"},
            {"Metric": "Strongly Connected Components (SCC)", "Value": f"{comp_stats['num_strongly_connected_components']:,}", "Research Significance": "Circularity of transaction chains"},
            {"Metric": "Is Directed Acyclic Graph (DAG)?", "Value": str(comp_stats['is_dag']), "Research Significance": "Absence of cycles reflects temporal causality of Bitcoin UTXO"},
        ]
        
        df_graph_summary = pd.DataFrame(summary_rows)
        graph_csv = tables_dir / "graph_topology_summary.csv"
        df_graph_summary.to_csv(graph_csv, index=False)
        
        # Temporal graph metrics table
        df_temp_graph = self.compute_temporal_graph_metrics()
        temp_graph_csv = tables_dir / "temporal_graph_metrics.csv"
        df_temp_graph.to_csv(temp_graph_csv, index=False)
        
        logger.info(f"Graph topology summary exported to {graph_csv}")
        logger.info(f"Temporal graph metrics exported to {temp_graph_csv}")
        return {"topology": graph_csv, "temporal_graph": temp_graph_csv}

    def generate_plots(self, figures_dir: Path, dpi: int = 300) -> None:
        """
        Generate publication degree distribution plots (in, out, total degrees on log-log scale).
        """
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use("seaborn-v0_8-whitegrid")
        
        in_degrees = np.array([d for _, d in self.G_dir.in_degree()])
        out_degrees = np.array([d for _, d in self.G_dir.out_degree()])
        total_degrees = in_degrees + out_degrees
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=dpi)
        
        # Helper for log-log degree plot
        def _plot_degree_dist(deg_arr: np.ndarray, ax, title: str, color: str):
            vals, counts = np.unique(deg_arr, return_counts=True)
            # Filter non-zero for log plot
            mask = vals > 0
            ax.scatter(vals[mask], counts[mask], color=color, alpha=0.8, s=25, edgecolor="none")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Degree (log scale)", fontsize=10)
            ax.set_ylabel("Frequency / Node Count (log scale)", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            
        _plot_degree_dist(in_degrees, axes[0], "In-Degree Distribution (Log-Log)", "#1f77b4")
        _plot_degree_dist(out_degrees, axes[1], "Out-Degree Distribution (Log-Log)", "#ff7f0e")
        _plot_degree_dist(total_degrees, axes[2], "Total Degree Distribution (Log-Log)", "#2ca02c")
        
        plt.tight_layout()
        deg_plot_path = figures_dir / "degree_distribution.png"
        plt.savefig(deg_plot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {deg_plot_path}")
