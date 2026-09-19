"""
Publication-Quality Visualization Suite for Baseline Experiments.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("baseline_plots")


def generate_baseline_plots(
    df_summary: pd.DataFrame,
    best_models_data: List[Dict],
    figures_dir: Path,
    dpi: int = 300
) -> None:
    """
    Generate all required publication figures for Phase 2.
    
    Args:
        df_summary: DataFrame containing Mean ± Std metrics for each model configuration.
        best_models_data: List of dictionaries with curve data and confusion matrices.
        figures_dir: Directory to save figure files.
        dpi: Figure resolution (300 DPI).
    """
    figures_dir = Path(figures_dir)
    cm_dir = figures_dir / "confusion_matrices"
    pr_dir = figures_dir / "precision_recall_curves"
    roc_dir = figures_dir / "roc_curves"
    
    for d in [figures_dir, cm_dir, pr_dir, roc_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # -------------------------------------------------------------
    # 1. Main Baseline Comparison: F1 & PR-AUC across all configs
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)
    
    df_plot = df_summary.copy()
    # Clean label name
    df_plot["config_label"] = (
        df_plot["model"] + "\n(" + df_plot["weighting"] + ")\n[" + df_plot["feature_mode"] + "]"
    )
    
    # Sort by test F1 mean
    df_plot = df_plot.sort_values(by="test_illicit_f1_mean", ascending=True)
    y_pos = np.arange(len(df_plot))
    
    # Left: Test Illicit F1-Score
    axes[0].barh(
        y_pos,
        df_plot["test_illicit_f1_mean"],
        xerr=df_plot["test_illicit_f1_std"],
        color="#1f77b4",
        edgecolor="black",
        alpha=0.85,
        capsize=4
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_plot["config_label"], fontsize=8)
    axes[0].set_xlabel("Test Illicit F1-Score (Mean ± Std)", fontsize=11, fontweight="bold")
    axes[0].set_title("Test Illicit F1-Score across 16 Baseline Configurations", fontsize=12, fontweight="bold")
    axes[0].set_xlim(0, 1.0)
    
    # Right: Test PR-AUC
    axes[1].barh(
        y_pos,
        df_plot["test_pr_auc_mean"],
        xerr=df_plot["test_pr_auc_std"],
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.85,
        capsize=4
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([""] * len(df_plot))  # Hide duplicate y-labels
    axes[1].set_xlabel("Test PR-AUC (Mean ± Std)", fontsize=11, fontweight="bold")
    axes[1].set_title("Test PR-AUC across 16 Baseline Configurations", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, 1.0)
    
    plt.tight_layout()
    comp_path = figures_dir / "baseline_model_comparison.png"
    plt.savefig(comp_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {comp_path}")

    # -------------------------------------------------------------
    # 2. Local vs Full Features Impact (F1 & PR-AUC comparison)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    sns.barplot(
        data=df_summary,
        x="model",
        y="test_illicit_f1_mean",
        hue="feature_mode",
        palette=["#4a7bb0", "#e26d5c"],
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Predictive Impact of Neighborhood Aggregated Features on Illicit F1-Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test Illicit F1-Score (Mean)", fontsize=11)
    ax.set_xlabel("Baseline Model", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Feature Space", frameon=True)
    plt.tight_layout()
    f1_path = figures_dir / "f1_comparison.png"
    plt.savefig(f1_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {f1_path}")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    sns.barplot(
        data=df_summary,
        x="model",
        y="test_pr_auc_mean",
        hue="feature_mode",
        palette=["#4a7bb0", "#e26d5c"],
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Predictive Impact of Neighborhood Aggregated Features on PR-AUC", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test PR-AUC (Mean)", fontsize=11)
    ax.set_xlabel("Baseline Model", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Feature Space", frameon=True)
    plt.tight_layout()
    pr_auc_path = figures_dir / "pr_auc_comparison.png"
    plt.savefig(pr_auc_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {pr_auc_path}")

    # -------------------------------------------------------------
    # 3. Confusion Matrices, PR Curves & ROC Curves for Representative Models
    # -------------------------------------------------------------
    for item in best_models_data:
        cfg_name = item["name"]
        cm = np.array(item["confusion_matrix"])
        
        # A. Confusion Matrix Heatmap
        fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Licit (0)", "Illicit (1)"],
            yticklabels=["Licit (0)", "Illicit (1)"],
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {cfg_name}\n(Locked Tau = {item['threshold']:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("Actual Class", fontsize=9)
        plt.tight_layout()
        cm_file = cm_dir / f"cm_{cfg_name.replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(cm_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # B. PR Curve
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        pr_data = item["pr_curve"]
        ax.plot(pr_data["recall"], pr_data["precision"], color="#d62728", lw=2, label=f"PR-AUC = {item['pr_auc']:.4f}")
        ax.set_title(f"Precision-Recall Curve: {cfg_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall (Illicit)", fontsize=10)
        ax.set_ylabel("Precision (Illicit)", fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        pr_file = pr_dir / f"pr_{cfg_name.replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(pr_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # C. ROC Curve
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        roc_data = item["roc_curve"]
        ax.plot(roc_data["fpr"], roc_data["tpr"], color="#1f77b4", lw=2, label=f"ROC-AUC = {item['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_title(f"ROC Curve: {cfg_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        roc_file = roc_dir / f"roc_{cfg_name.replace(' ', '_').replace('+', '_')}.png"
        plt.savefig(roc_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
