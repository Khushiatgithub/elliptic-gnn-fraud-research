"""
Label-Graph Homophily and Class Neighborhood Interaction Analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("homophily_analyzer")


class HomophilyAnalyzer:
    """
    Analyzes class-conditional degree distributions, edge transition matrices,
    and homophily / heterophily dynamics across licit, illicit, and unknown transactions.
    """

    def __init__(self, df_classes: pd.DataFrame, df_edges: pd.DataFrame, df_features: pd.DataFrame):
        self.df_classes = df_classes
        self.df_edges = df_edges
        self.df_features = df_features
        
        # Create node to class mapping
        self.node_to_class = dict(zip(df_classes["txId"], df_classes["class"]))
        
        # Map human-readable class names
        self.class_label_map = {"1": "Illicit", "2": "Licit", "unknown": "Unknown"}
        
        # Annotated edges with source and target classes
        self.df_annotated_edges = self.df_edges.copy()
        self.df_annotated_edges["src_class"] = self.df_annotated_edges["txId1"].map(self.node_to_class)
        self.df_annotated_edges["dst_class"] = self.df_annotated_edges["txId2"].map(self.node_to_class)
        self.df_annotated_edges["src_label"] = self.df_annotated_edges["src_class"].map(self.class_label_map)
        self.df_annotated_edges["dst_label"] = self.df_annotated_edges["dst_class"].map(self.class_label_map)

    def compute_degree_by_class(self, G_dir: nx.DiGraph) -> pd.DataFrame:
        """
        Compute in-degree, out-degree, and total degree grouped by node class
        (Licit, Illicit, Unknown).
        """
        logger.info("Computing degree distributions segmented by transaction class...")
        
        node_degrees = []
        for node in G_dir.nodes():
            cls = self.node_to_class.get(node, "unknown")
            in_d = G_dir.in_degree(node)
            out_d = G_dir.out_degree(node)
            tot_d = in_d + out_d
            
            node_degrees.append({
                "txId": node,
                "class_code": cls,
                "class_label": self.class_label_map.get(cls, "Unknown"),
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": tot_d
            })
            
        df_node_deg = pd.DataFrame(node_degrees)
        
        # Aggregate statistics per class
        agg_stats = df_node_deg.groupby("class_label").agg(
            node_count=("txId", "count"),
            in_degree_mean=("in_degree", "mean"),
            in_degree_median=("in_degree", "median"),
            in_degree_max=("in_degree", "max"),
            out_degree_mean=("out_degree", "mean"),
            out_degree_median=("out_degree", "median"),
            out_degree_max=("out_degree", "max"),
            total_degree_mean=("total_degree", "mean"),
            total_degree_median=("total_degree", "median"),
            total_degree_max=("total_degree", "max"),
            zero_degree_count=("total_degree", lambda s: (s == 0).sum())
        ).reset_index()
        
        agg_stats["zero_degree_pct"] = (agg_stats["zero_degree_count"] / agg_stats["node_count"]) * 100.0
        return agg_stats

    def compute_edge_transition_matrix(self) -> pd.DataFrame:
        """
        Compute full edge transition counts and proportions between all class pairs:
        (Licit -> Licit, Licit -> Illicit, Illicit -> Licit, Illicit -> Illicit, and Unknown interactions).
        """
        logger.info("Computing edge transition matrix across class boundaries...")
        
        # Cross-tabulation of source and destination labels
        crosstab_counts = pd.crosstab(
            self.df_annotated_edges["src_label"],
            self.df_annotated_edges["dst_label"],
            margins=True,
            margins_name="Total"
        )
        
        return crosstab_counts

    def compute_homophily_metrics(self) -> Dict[str, Any]:
        """
        Compute node homophily and edge homophily metrics on the labeled subgraph.
        """
        logger.info("Computing labeled edge homophily ratios...")
        
        # Filter edges where BOTH source and destination are labeled (Licit or Illicit)
        labeled_edges = self.df_annotated_edges[
            (self.df_annotated_edges["src_class"].isin(["1", "2"])) &
            (self.df_annotated_edges["dst_class"].isin(["1", "2"]))
        ]
        
        total_labeled_edges = len(labeled_edges)
        
        if total_labeled_edges > 0:
            same_class_edges = (labeled_edges["src_class"] == labeled_edges["dst_class"]).sum()
            edge_homophily = float(same_class_edges / total_labeled_edges)
            
            licit_to_licit = ((labeled_edges["src_class"] == "2") & (labeled_edges["dst_class"] == "2")).sum()
            illicit_to_illicit = ((labeled_edges["src_class"] == "1") & (labeled_edges["dst_class"] == "1")).sum()
            licit_to_illicit = ((labeled_edges["src_class"] == "2") & (labeled_edges["dst_class"] == "1")).sum()
            illicit_to_licit = ((labeled_edges["src_class"] == "1") & (labeled_edges["dst_class"] == "2")).sum()
        else:
            same_class_edges = 0
            edge_homophily = 0.0
            licit_to_licit = illicit_to_illicit = licit_to_illicit = illicit_to_licit = 0
            
        # Edges involving at least one illicit node
        edges_with_illicit = (
            (self.df_annotated_edges["src_class"] == "1") |
            (self.df_annotated_edges["dst_class"] == "1")
        ).sum()
        total_edges = len(self.df_annotated_edges)
        pct_edges_with_illicit = (edges_with_illicit / total_edges) * 100.0

        # Edges involving unknown nodes
        edges_with_unknown = (
            (self.df_annotated_edges["src_class"] == "unknown") |
            (self.df_annotated_edges["dst_class"] == "unknown")
        ).sum()
        pct_edges_with_unknown = (edges_with_unknown / total_edges) * 100.0

        return {
            "total_network_edges": total_edges,
            "edges_with_illicit_count": int(edges_with_illicit),
            "edges_with_illicit_pct": float(pct_edges_with_illicit),
            "edges_with_unknown_count": int(edges_with_unknown),
            "edges_with_unknown_pct": float(pct_edges_with_unknown),
            "labeled_to_labeled_edges_count": total_labeled_edges,
            "labeled_edge_homophily_ratio": float(edge_homophily),
            "licit_to_licit_count": int(licit_to_licit),
            "illicit_to_illicit_count": int(illicit_to_illicit),
            "licit_to_illicit_count": int(licit_to_illicit),
            "illicit_to_licit_count": int(illicit_to_licit),
        }

    def export_summary_tables(self, tables_dir: Path, G_dir: nx.DiGraph) -> Dict[str, Path]:
        """Export class degree statistics and transition matrix to CSV."""
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Degree by class table
        df_deg_class = self.compute_degree_by_class(G_dir)
        deg_class_path = tables_dir / "degree_distribution_by_class.csv"
        df_deg_class.to_csv(deg_class_path, index=False)
        
        # 2. Edge transition matrix
        df_trans = self.compute_edge_transition_matrix()
        trans_path = tables_dir / "edge_class_transition_matrix.csv"
        df_trans.to_csv(trans_path, index=True)
        
        # 3. Homophily metrics summary
        homophily_stats = self.compute_homophily_metrics()
        df_homophily = pd.DataFrame([
            {"Metric": "Total Network Edges", "Value": f"{homophily_stats['total_network_edges']:,}", "Description": "All directed edges in dataset"},
            {"Metric": "Edges Involving >= 1 Unknown Node", "Value": f"{homophily_stats['edges_with_unknown_count']:,} ({homophily_stats['edges_with_unknown_pct']:.2f}%)", "Description": "Edges connecting with unlabeled context"},
            {"Metric": "Edges Involving >= 1 Illicit Node", "Value": f"{homophily_stats['edges_with_illicit_count']:,} ({homophily_stats['edges_with_illicit_pct']:.2f}%)", "Description": "Fraud exposure across network"},
            {"Metric": "Both-Endpoint Labeled Edges", "Value": f"{homophily_stats['labeled_to_labeled_edges_count']:,}", "Description": "Edges between two labeled nodes"},
            {"Metric": "Labeled Edge Homophily Ratio", "Value": f"{homophily_stats['labeled_edge_homophily_ratio']:.4f}", "Description": "Proportion of same-class connections (Licit-Licit + Illicit-Illicit)"},
            {"Metric": "Licit -> Licit Edges", "Value": f"{homophily_stats['licit_to_licit_count']:,}", "Description": "Clean fund lineage"},
            {"Metric": "Illicit -> Illicit Edges", "Value": f"{homophily_stats['illicit_to_illicit_count']:,}", "Description": "Direct fraud chains/peeling chains"},
            {"Metric": "Licit -> Illicit Edges", "Value": f"{homophily_stats['licit_to_illicit_count']:,}", "Description": "Clean funds entering illicit entities"},
            {"Metric": "Illicit -> Licit Edges", "Value": f"{homophily_stats['illicit_to_licit_count']:,}", "Description": "Laundered funds exiting to licit entities"},
        ])
        homophily_path = tables_dir / "graph_homophily_summary.csv"
        df_homophily.to_csv(homophily_path, index=False)
        
        logger.info(f"Class degree table exported to {deg_class_path}")
        logger.info(f"Edge transition table exported to {trans_path}")
        logger.info(f"Homophily summary exported to {homophily_path}")
        return {"degree_class": deg_class_path, "transition": trans_path, "homophily": homophily_path}

    def generate_plots(self, figures_dir: Path, G_dir: nx.DiGraph, dpi: int = 300) -> None:
        """
        Generate publication-ready visualizations:
        1. Boxplot/violin plot of degree distributions across classes.
        2. Heatmap of edge transition matrix proportions.
        """
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use("seaborn-v0_8-whitegrid")
        palette = {"Licit": "#1f77b4", "Illicit": "#d62728", "Unknown": "#7f7f7f"}
        
        # --- Plot 1: Degree Distribution by Class ---
        node_degrees = []
        for node in G_dir.nodes():
            cls = self.node_to_class.get(node, "unknown")
            label = self.class_label_map.get(cls, "Unknown")
            tot_d = G_dir.in_degree(node) + G_dir.out_degree(node)
            node_degrees.append({"Class": label, "Total Degree": tot_d})
            
        df_deg = pd.DataFrame(node_degrees)
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
        sns.boxplot(
            data=df_deg,
            x="Class",
            y="Total Degree",
            palette=palette,
            showfliers=False,  # Exclude extreme outliers for box clarity
            ax=ax
        )
        ax.set_title("Node Degree Distribution by Class (Interquartile Range, Outliers Hidden)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Degree (In + Out)", fontsize=11)
        ax.set_xlabel("Transaction Class", fontsize=11)
        
        plt.tight_layout()
        plot1_path = figures_dir / "degree_by_class.png"
        plt.savefig(plot1_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot1_path}")

        # --- Plot 2: Edge Transition Probability Matrix ---
        crosstab = pd.crosstab(
            self.df_annotated_edges["src_label"],
            self.df_annotated_edges["dst_label"],
            normalize="index"
        ) * 100.0  # Row normalized percentages
        
        # Ensure standard order: Illicit, Licit, Unknown
        order = [c for c in ["Illicit", "Licit", "Unknown"] if c in crosstab.index]
        crosstab = crosstab.loc[order, order]
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
        sns.heatmap(
            crosstab,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar_kws={"label": "Row-Normalized Transition Probability (%)"},
            ax=ax,
            linewidths=0.5
        )
        ax.set_title("Edge Transition Matrix: Source Class to Destination Class (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Destination Node Class", fontsize=11)
        ax.set_ylabel("Source Node Class", fontsize=11)
        
        plt.tight_layout()
        plot2_path = figures_dir / "edge_homophily_matrix.png"
        plt.savefig(plot2_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot2_path}")
