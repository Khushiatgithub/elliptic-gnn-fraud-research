"""
Publication-Quality Visualization Suite for Phase 3 Graph Neural Network Experiments.
All figures generated at 300 DPI with clear typography, high contrast, and academic formatting.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("gnn_plots")


def generate_all_gnn_figures(
    df_gnn_summary: pd.DataFrame,
    df_baseline_summary: pd.DataFrame,
    df_unknown_ablation: pd.DataFrame,
    df_direction_ablation: pd.DataFrame,
    df_depth_ablation: pd.DataFrame,
    df_benchmark: Optional[pd.DataFrame] = None,
    df_timestep_errors: Optional[pd.DataFrame] = None,
    representative_curve_data: Optional[List[Dict]] = None,
    best_overall_result: Optional[Dict] = None,
    figures_dir: Optional[Path] = None,
    dpi: int = 300
) -> None:
    """
    Generates the 12 required publication-grade 300 DPI figures for Phase 3.
    """
    if figures_dir is None:
        figures_dir = Path("results/figures")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # -------------------------------------------------------------
    # 1. GNN Model Comparison — Illicit F1
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    df_plot1 = df_gnn_summary.copy().sort_values(by="test_illicit_f1_mean", ascending=True)
    df_plot1["label"] = df_plot1["model"] + " [" + df_plot1["feature_space"] + "]"
    y_pos = np.arange(len(df_plot1))
    
    ax.barh(
        y_pos,
        df_plot1["test_illicit_f1_mean"],
        xerr=df_plot1["test_illicit_f1_std"],
        color="#2ca02c",
        edgecolor="#1b611b",
        alpha=0.88,
        capsize=5,
        height=0.65
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot1["label"], fontsize=10, fontweight="medium")
    ax.set_xlabel("Test Illicit F1-Score (Mean ± Std)", fontsize=11, fontweight="bold")
    ax.set_title("GNN Architectures: Test Illicit F1-Score (5 Seeds)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(df_plot1["test_illicit_f1_mean"]):
        ax.text(v + 0.02, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold", color="#1b611b")
    plt.tight_layout()
    p1 = figures_dir / "gnn_model_comparison_f1.png"
    plt.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 1 saved: {p1}")

    # -------------------------------------------------------------
    # 2. GNN Model Comparison — PR-AUC
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    df_plot2 = df_gnn_summary.copy().sort_values(by="test_pr_auc_mean", ascending=True)
    df_plot2["label"] = df_plot2["model"] + " [" + df_plot2["feature_space"] + "]"
    y_pos2 = np.arange(len(df_plot2))
    
    ax.barh(
        y_pos2,
        df_plot2["test_pr_auc_mean"],
        xerr=df_plot2["test_pr_auc_std"],
        color="#9467bd",
        edgecolor="#5c3566",
        alpha=0.88,
        capsize=5,
        height=0.65
    )
    ax.set_yticks(y_pos2)
    ax.set_yticklabels(df_plot2["label"], fontsize=10, fontweight="medium")
    ax.set_xlabel("Test PR-AUC (Mean ± Std)", fontsize=11, fontweight="bold")
    ax.set_title("GNN Architectures: Test Precision-Recall AUC (5 Seeds)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(df_plot2["test_pr_auc_mean"]):
        ax.text(v + 0.02, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold", color="#5c3566")
    plt.tight_layout()
    p2 = figures_dir / "gnn_model_comparison_prauc.png"
    plt.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 2 saved: {p2}")

    # -------------------------------------------------------------
    # 3. Local vs Full Feature Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    sns.barplot(
        data=df_gnn_summary,
        x="model",
        y="test_illicit_f1_mean",
        hue="feature_space",
        palette=["#4a7bb0", "#e26d5c"],
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Feature Space Comparison: Local (93) vs Full (165) Features", fontsize=13, fontweight="bold")
    ax.set_ylabel("Test Illicit F1-Score (Mean)", fontsize=11, fontweight="bold")
    ax.set_xlabel("GNN Architecture", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Feature Configuration", frameon=True, fontsize=10)
    plt.tight_layout()
    p3 = figures_dir / "feature_ablation_local_vs_full.png"
    plt.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 3 saved: {p3}")

    # -------------------------------------------------------------
    # 4. GNN vs Conventional ML Baselines Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    combo_rows = []
    for _, r in df_gnn_summary.iterrows():
        combo_rows.append({
            "Family": "Graph Neural Networks",
            "Model Name": f"{r['model']} ({r['feature_space']})",
            "Test F1": float(r["test_illicit_f1_mean"]),
            "Test F1 Std": float(r["test_illicit_f1_std"]),
        })
    for _, r in df_baseline_summary.iterrows():
        combo_rows.append({
            "Family": "Conventional ML",
            "Model Name": f"{r['model']} ({r['weighting']}, {r['feature_space']})",
            "Test F1": float(r["test_illicit_f1_mean"]),
            "Test F1 Std": float(r["test_illicit_f1_std"]),
        })
    df_combo = pd.DataFrame(combo_rows).sort_values(by="Test F1", ascending=False).head(10)
    
    sns.barplot(
        data=df_combo,
        x="Test F1",
        y="Model Name",
        hue="Family",
        palette={"Graph Neural Networks": "#2ca02c", "Conventional ML": "#1f77b4"},
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Top Detection Models: GNNs vs Conventional ML Baselines", fontsize=13, fontweight="bold")
    ax.set_xlabel("Test Illicit F1-Score (Mean)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model Configuration", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.legend(title="Model Family", frameon=True, fontsize=10)
    plt.tight_layout()
    p4 = figures_dir / "gnn_vs_conventional_baselines.png"
    plt.savefig(p4, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 4 saved: {p4}")

    # -------------------------------------------------------------
    # 5. PR Curves for Representative Models
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 5. PR Curves for Representative Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    if representative_curve_data:
        for item in representative_curve_data:
            pr_data = item.get("pr_curve", {})
            if isinstance(pr_data, str):
                try:
                    pr_data = eval(pr_data)
                except Exception:
                    pr_data = {}
            if isinstance(pr_data, dict) and "recall" in pr_data and "precision" in pr_data:
                ax.plot(
                    pr_data["recall"],
                    pr_data["precision"],
                    lw=2,
                    label=f"{item['name']} (PR-AUC = {float(item['pr_auc']):.4f})"
                )
    ax.set_title("Precision-Recall Curves for GNN Architectures (Seed 42)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Illicit)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Illicit)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p5 = figures_dir / "gnn_precision_recall_curves.png"
    plt.savefig(p5, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 5 saved: {p5}")

    # -------------------------------------------------------------
    # 6. ROC Curves for Representative Models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    if representative_curve_data:
        for item in representative_curve_data:
            roc_data = item.get("roc_curve", {})
            if isinstance(roc_data, str):
                try:
                    roc_data = eval(roc_data)
                except Exception:
                    roc_data = {}
            if isinstance(roc_data, dict) and "fpr" in roc_data and "tpr" in roc_data:
                ax.plot(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    lw=2,
                    label=f"{item['name']} (ROC-AUC = {float(item['roc_auc']):.4f})"
                )
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves (Seed 42)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p6 = figures_dir / "gnn_roc_curves.png"
    plt.savefig(p6, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 6 saved: {p6}")

    # -------------------------------------------------------------
    # 7. Confusion Matrix for Overall Strongest GNN Model
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    if best_overall_result and "test_confusion_matrix" in best_overall_result:
        raw_cm = best_overall_result["test_confusion_matrix"]
        if isinstance(raw_cm, str):
            try:
                raw_cm = eval(raw_cm)
            except Exception:
                raw_cm = [[0, 0], [0, 0]]
        cm = np.array(raw_cm)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Licit", "Illicit"],
            yticklabels=["Licit", "Illicit"],
            annot_kws={"size": 14, "weight": "bold"},
            ax=ax
        )
        opt_t = float(best_overall_result.get("optimal_threshold", 0.5))
        ax.set_title(f"Confusion Matrix: Best GNN Model (Locked Tau* = {opt_t:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p7 = figures_dir / "gnn_confusion_matrix_best.png"
    plt.savefig(p7, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 7 saved: {p7}")

    # -------------------------------------------------------------
    # 8. Temporal Performance by Timestep (Test Timesteps 40–49)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    if df_timestep_errors is not None and len(df_timestep_errors) > 0:
        df_ts_plot = df_timestep_errors.copy()
        df_ts_plot["Illicit F1"] = df_ts_plot["Illicit F1"].astype(float)
        df_ts_plot["Recall"] = df_ts_plot["Recall"].astype(float)
        df_ts_plot["Precision"] = df_ts_plot["Precision"].astype(float)
        
        ax.plot(df_ts_plot["Time Step"], df_ts_plot["Illicit F1"], marker="o", lw=2.5, color="#2ca02c", label="Illicit F1-Score")
        ax.plot(df_ts_plot["Time Step"], df_ts_plot["Recall"], marker="s", lw=2.0, color="#d62728", linestyle="--", label="Recall")
        ax.plot(df_ts_plot["Time Step"], df_ts_plot["Precision"], marker="^", lw=2.0, color="#1f77b4", linestyle=":", label="Precision")
        
        ax.set_title("Temporal Generalization: GNN Detection Performance Across Test Timesteps (40–49)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Test Time Step (Snapshot)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
        ax.set_xticks(df_ts_plot["Time Step"])
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p8 = figures_dir / "gnn_temporal_performance_timesteps.png"
    plt.savefig(p8, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 8 saved: {p8}")

    # -------------------------------------------------------------
    # 9. Unknown-Node Context Ablation (Retained vs Removed)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    sns.barplot(
        data=df_unknown_ablation,
        x="model",
        y="test_illicit_f1_mean",
        hue="unknown_treatment",
        palette=["#2ca02c", "#d62728"],
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Unknown-Node Context Ablation: Impact of 77.15% Unlabeled Graph Nodes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Illicit F1-Score (Mean)", fontsize=11, fontweight="bold")
    ax.set_xlabel("GNN Architecture", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Context Setting", frameon=True, fontsize=10)
    plt.tight_layout()
    p9 = figures_dir / "unknown_node_ablation.png"
    plt.savefig(p9, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 9 saved: {p9}")

    # -------------------------------------------------------------
    # 10. Depth / Over-Smoothing Ablation (1, 2, 3 Layers)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    sns.lineplot(
        data=df_depth_ablation,
        x="num_layers",
        y="test_illicit_f1_mean",
        hue="model",
        marker="o",
        markersize=9,
        lw=2.5,
        ax=ax
    )
    ax.set_title("GNN Depth & Over-Smoothing Analysis across 1, 2, and 3 Layers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Illicit F1-Score (Mean)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of GNN Message Passing Layers", fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 1.0)
    ax.legend(title="GNN Model", frameon=True, fontsize=10)
    plt.tight_layout()
    p10 = figures_dir / "depth_ablation.png"
    plt.savefig(p10, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 10 saved: {p10}")

    # -------------------------------------------------------------
    # 11. Graph Direction Sensitivity Ablation
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    sns.barplot(
        data=df_direction_ablation,
        x="model",
        y="test_illicit_f1_mean",
        hue="direction",
        palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Directional Sensitivity Ablation (Forward vs Backward vs Bidirectional)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Illicit F1-Score (Mean)", fontsize=11, fontweight="bold")
    ax.set_xlabel("GNN Architecture", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Edge Propagation Flow", frameon=True, fontsize=10)
    plt.tight_layout()
    p11 = figures_dir / "direction_sensitivity_ablation.png"
    plt.savefig(p11, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 11 saved: {p11}")

    # -------------------------------------------------------------
    # 12. Computational Benchmark (Accuracy vs Training Time Tradeoff)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    if df_benchmark is not None and len(df_benchmark) > 0:
        df_bench_plot = df_benchmark.copy()
        df_bench_plot["training_time_sec"] = df_bench_plot["training_time_sec"].astype(float)
        df_bench_plot["test_illicit_f1"] = df_bench_plot["test_illicit_f1"].astype(float)
        
        sns.scatterplot(
            data=df_bench_plot,
            x="training_time_sec",
            y="test_illicit_f1",
            hue="model",
            style="feature_space",
            s=120,
            alpha=0.9,
            edgecolor="black",
            ax=ax
        )
        ax.set_title("Computational Cost vs Fraud Detection Accuracy Trade-off", fontsize=13, fontweight="bold")
        ax.set_xlabel("Training Time (Seconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Test Illicit F1-Score", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    p12 = figures_dir / "gnn_compute_benchmark.png"
    plt.savefig(p12, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure 12 saved: {p12}")
