"""
Label Analysis and Temporal Dynamics for the Elliptic Dataset.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("label_analyzer")


class LabelAnalyzer:
    """
    Analyzes class distributions, class imbalance ratios, and temporal evolution
    across the 49 distinct time steps.
    """

    def __init__(self, df_classes: pd.DataFrame, df_features: pd.DataFrame):
        self.df_classes = df_classes
        self.df_features = df_features
        # Create a merged view for time-step label analysis (txId + time_step + class)
        self.df_merged = self.df_features[["txId", "time_step"]].merge(
            self.df_classes, on="txId", how="inner"
        )

    def compute_class_distribution(self) -> Dict[str, Any]:
        """
        Compute overall class distribution, proportions, and imbalance metrics.
        """
        logger.info("Computing class distribution metrics...")
        
        counts = self.df_classes["class"].value_counts()
        total_nodes = len(self.df_classes)
        
        num_unknown = int(counts.get("unknown", 0))
        num_licit = int(counts.get("2", 0))
        num_illicit = int(counts.get("1", 0))
        num_labeled = num_licit + num_illicit
        
        pct_unknown = (num_unknown / total_nodes) * 100.0
        pct_licit = (num_licit / total_nodes) * 100.0
        pct_illicit = (num_illicit / total_nodes) * 100.0
        
        pct_licit_labeled = (num_licit / num_labeled) * 100.0 if num_labeled > 0 else 0.0
        pct_illicit_labeled = (num_illicit / num_labeled) * 100.0 if num_labeled > 0 else 0.0
        
        # Imbalance ratio: licit to illicit (majority : minority)
        imbalance_ratio = float(num_licit / num_illicit) if num_illicit > 0 else 0.0
        
        stats = {
            "total_transactions": total_nodes,
            "num_unknown": num_unknown,
            "pct_unknown": pct_unknown,
            "num_licit": num_licit,
            "pct_licit_all": pct_licit,
            "pct_licit_labeled": pct_licit_labeled,
            "num_illicit": num_illicit,
            "pct_illicit_all": pct_illicit,
            "pct_illicit_labeled": pct_illicit_labeled,
            "num_labeled": num_labeled,
            "pct_labeled_all": (num_labeled / total_nodes) * 100.0,
            "imbalance_ratio_licit_to_illicit": imbalance_ratio,
        }
        
        logger.info(
            f"Class summary: Total={total_nodes:,} | Unknown={num_unknown:,} ({pct_unknown:.2f}%) | "
            f"Licit={num_licit:,} ({pct_licit_labeled:.2f}% of labeled) | "
            f"Illicit={num_illicit:,} ({pct_illicit_labeled:.2f}% of labeled) | "
            f"Imbalance Ratio (Licit:Illicit)={imbalance_ratio:.2f}:1"
        )
        return stats

    def compute_temporal_distribution(self) -> pd.DataFrame:
        """
        Compute temporal transaction breakdown per time step (1 to 49).
        
        Returns:
            DataFrame indexed by time_step with total, unknown, licit, illicit counts and ratios.
        """
        logger.info("Computing temporal breakdown across time steps...")
        
        time_grouped = self.df_merged.groupby(["time_step", "class"]).size().unstack(fill_value=0)
        
        # Ensure expected columns exist
        for col in ["1", "2", "unknown"]:
            if col not in time_grouped.columns:
                time_grouped[col] = 0
                
        df_temporal = pd.DataFrame()
        df_temporal["time_step"] = time_grouped.index
        df_temporal["num_illicit"] = time_grouped["1"].values
        df_temporal["num_licit"] = time_grouped["2"].values
        df_temporal["num_unknown"] = time_grouped["unknown"].values
        df_temporal["total_txs"] = df_temporal["num_illicit"] + df_temporal["num_licit"] + df_temporal["num_unknown"]
        df_temporal["total_labeled"] = df_temporal["num_illicit"] + df_temporal["num_licit"]
        
        # Proportions
        df_temporal["pct_illicit_labeled"] = np.where(
            df_temporal["total_labeled"] > 0,
            (df_temporal["num_illicit"] / df_temporal["total_labeled"]) * 100.0,
            0.0
        )
        df_temporal["pct_labeled_total"] = (df_temporal["total_labeled"] / df_temporal["total_txs"]) * 100.0
        df_temporal["imbalance_ratio"] = np.where(
            df_temporal["num_illicit"] > 0,
            df_temporal["num_licit"] / df_temporal["num_illicit"],
            np.nan
        )
        
        return df_temporal

    def export_summary_tables(self, tables_dir: Path) -> Tuple[Path, Path]:
        """Export class distribution and temporal analysis tables to CSV."""
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Overall class distribution table
        stats = self.compute_class_distribution()
        df_class_summary = pd.DataFrame([
            {"Category": "Total Transactions", "Count": stats["total_transactions"], "Proportion (%)": 100.0, "Role in Benchmark": "All network nodes"},
            {"Category": "Unknown (Unlabeled)", "Count": stats["num_unknown"], "Proportion (%)": stats["pct_unknown"], "Role in Benchmark": "Contextual graph nodes (unsupervised/GNN message passing)"},
            {"Category": "Total Labeled", "Count": stats["num_labeled"], "Proportion (%)": stats["pct_labeled_all"], "Role in Benchmark": "Supervised training/evaluation set"},
            {"Category": "Licit Transactions (Class 2)", "Count": stats["num_licit"], "Proportion (%)": stats["pct_licit_all"], "Role in Benchmark": "Majority negative class (90.24% of labeled)"},
            {"Category": "Illicit Transactions (Class 1)", "Count": stats["num_illicit"], "Proportion (%)": stats["pct_illicit_all"], "Role in Benchmark": "Minority positive class (9.76% of labeled)"},
        ])
        class_csv = tables_dir / "class_distribution_summary.csv"
        df_class_summary.to_csv(class_csv, index=False)
        
        # 2. Temporal table
        df_temporal = self.compute_temporal_distribution()
        temporal_csv = tables_dir / "temporal_distribution_by_timestep.csv"
        df_temporal.to_csv(temporal_csv, index=False)
        
        logger.info(f"Class distribution table saved to {class_csv}")
        logger.info(f"Temporal distribution table saved to {temporal_csv}")
        return class_csv, temporal_csv

    def generate_plots(self, figures_dir: Path, dpi: int = 300) -> None:
        """
        Generate publication-quality plots:
        1. Class distribution bar / donut chart.
        2. Temporal stack & line plot across 49 timesteps.
        3. Illicit percentage dynamics over time.
        """
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use("seaborn-v0_8-whitegrid")
        palette = {"licit": "#1f77b4", "illicit": "#d62728", "unknown": "#7f7f7f"}
        
        # --- Plot 1: Overall Class Distribution ---
        stats = self.compute_class_distribution()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)
        
        # Left: Complete dataset (All 203,769 nodes)
        cats_all = ["Unknown\n(Unlabeled)", "Licit\n(Class 2)", "Illicit\n(Class 1)"]
        counts_all = [stats["num_unknown"], stats["num_licit"], stats["num_illicit"]]
        colors_all = [palette["unknown"], palette["licit"], palette["illicit"]]
        bars = axes[0].bar(cats_all, counts_all, color=colors_all, edgecolor="black", linewidth=0.8)
        axes[0].set_title("Complete Elliptic Dataset Node Composition\n(N = 203,769)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Number of Transactions", fontsize=11)
        axes[0].set_ylim(0, max(counts_all) * 1.15)
        for bar in bars:
            height = bar.get_height()
            pct = (height / stats["total_transactions"]) * 100.0
            axes[0].annotate(f"{height:,}\n({pct:.1f}%)",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 4), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10, fontweight="bold")
                             
        # Right: Labeled Transactions Only (46,564 nodes)
        cats_lab = ["Licit (Class 2)", "Illicit (Class 1)"]
        counts_lab = [stats["num_licit"], stats["num_illicit"]]
        colors_lab = [palette["licit"], palette["illicit"]]
        bars_lab = axes[1].bar(cats_lab, counts_lab, color=colors_lab, edgecolor="black", linewidth=0.8)
        axes[1].set_title(f"Supervised Evaluation Subset\n(N = {stats['num_labeled']:,} | Imbalance = {stats['imbalance_ratio_licit_to_illicit']:.1f}:1)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Number of Labeled Transactions", fontsize=11)
        axes[1].set_ylim(0, max(counts_lab) * 1.15)
        for bar in bars_lab:
            height = bar.get_height()
            pct = (height / stats["num_labeled"]) * 100.0
            axes[1].annotate(f"{height:,}\n({pct:.1f}%)",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 4), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10, fontweight="bold")
                             
        plt.tight_layout()
        plot1_path = figures_dir / "label_distribution.png"
        plt.savefig(plot1_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot1_path}")

        # --- Plot 2: Temporal Evolution of Transactions across 49 Timesteps ---
        df_temp = self.compute_temporal_distribution()
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        
        t = df_temp["time_step"]
        ax.plot(t, df_temp["total_txs"], label="Total Transactions", color="black", linewidth=2.0, linestyle="--")
        ax.plot(t, df_temp["num_unknown"], label="Unknown (Unlabeled)", color=palette["unknown"], linewidth=1.5)
        ax.plot(t, df_temp["num_licit"], label="Licit (Class 2)", color=palette["licit"], linewidth=1.8)
        ax.plot(t, df_temp["num_illicit"], label="Illicit (Class 1)", color=palette["illicit"], linewidth=1.8)
        
        ax.set_title("Temporal Transaction Distribution across 49 Bitcoin Timesteps (~2-week intervals)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time Step (Discrete Temporal Epochs: 1 to 49)", fontsize=11)
        ax.set_ylabel("Number of Transactions", fontsize=11)
        ax.set_xlim(1, 49)
        ax.set_xticks(range(1, 50, 2))
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        plot2_path = figures_dir / "temporal_evolution.png"
        plt.savefig(plot2_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot2_path}")

        # --- Plot 3: Illicit Rate Dynamics and Shutdown Effect (e.g. Darknet Market Shutdowns) ---
        fig, ax = plt.subplots(figsize=(15, 5), dpi=dpi)
        ax.plot(t, df_temp["pct_illicit_labeled"], color=palette["illicit"], marker="o", markersize=4, linewidth=2.0)
        
        # Highlight mean illicit rate
        mean_rate = df_temp["pct_illicit_labeled"].mean()
        ax.axhline(mean_rate, color="navy", linestyle="--", alpha=0.7, label=f"Mean Illicit Ratio ({mean_rate:.1f}%)")
        
        # Highlight known dark market disruption regime around timestep ~34-35 (PlusToken / AlphaBay / Hydra dynamics)
        ax.axvspan(34, 49, color="orange", alpha=0.15, label="Test Horizon (Timesteps 35-49: Regime Shift)")
        
        ax.set_title("Temporal Fraud Rate Dynamics (% Illicit among Labeled Transactions)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time Step (1 to 49)", fontsize=11)
        ax.set_ylabel("Illicit Transaction Proportion (%)", fontsize=11)
        ax.set_xlim(1, 49)
        ax.set_xticks(range(1, 50, 2))
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        plot3_path = figures_dir / "illicit_ratio_over_time.png"
        plt.savefig(plot3_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot3_path}")
