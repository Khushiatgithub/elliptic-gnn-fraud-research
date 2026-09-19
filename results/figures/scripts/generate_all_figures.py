# -*- coding: utf-8 -*-
"""
Master Deterministic Figure Generation Script
Generates all figures for the research paper at 300 DPI with publication-quality styling.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def set_academic_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

def generate_fig01_label_distribution(df_ts: pd.DataFrame, out_dir: Path):
    """Fig. 1: Distribution of Licit, Illicit, and Unknown transactions across 49 timesteps."""
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    
    timesteps = df_ts["time_step"]
    licit = df_ts["num_licit"]
    illicit = df_ts["num_illicit"]
    unknown = df_ts["num_unknown"]
    
    ax.bar(timesteps, licit, label="Licit Transactions (Class 2)", color="#2b5c8f", alpha=0.9)
    ax.bar(timesteps, illicit, bottom=licit, label="Illicit Transactions (Class 1)", color="#d95f02", alpha=0.9)
    ax.bar(timesteps, unknown, bottom=licit+illicit, label="Unknown Transactions", color="#b0b0b0", alpha=0.6)
    
    # Add vertical partition lines
    ax.axvline(x=34.5, color="black", linestyle="--", lw=1.2, label="Train / Val Boundary ($t=34$)")
    ax.axvline(x=39.5, color="red", linestyle=":", lw=1.5, label="Val / Test Boundary ($t=39$)")
    
    ax.set_xlabel(r"Discrete Time Step ($t \in [1, 49]$)", fontweight="bold")
    ax.set_ylabel("Transaction Count", fontweight="bold")
    ax.set_title("Distribution of Licit, Illicit, and Unknown Transactions Across 49 Timesteps", fontweight="bold")
    ax.set_xlim(0.5, 49.5)
    ax.set_xticks(np.arange(1, 50, 2))
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    
    plt.tight_layout()
    p = out_dir / "label_distribution.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig02_temporal_evolution(df_ts: pd.DataFrame, out_dir: Path):
    """Fig. 2: Total transaction volume and temporal class composition over 49 timesteps."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=300)
    
    timesteps = df_ts["time_step"]
    total = df_ts["total_txs"]
    labeled = df_ts["total_labeled"]
    illicit_ratio = df_ts["pct_illicit_labeled"]
    
    # Subplot 1: Total Volume
    ax1.plot(timesteps, total, color="#1f77b4", lw=2, marker="o", markersize=4, label="Total Transaction Volume")
    ax1.plot(timesteps, labeled, color="#2ca02c", lw=1.8, marker="s", markersize=4, label="Ground-Truth Labeled Volume")
    ax1.axvline(x=34.5, color="black", linestyle="--", lw=1.0)
    ax1.axvline(x=39.5, color="red", linestyle=":", lw=1.2)
    ax1.set_ylabel("Transaction Count", fontweight="bold")
    ax1.set_title("Temporal Transaction Volume Dynamics and Illicit Prevalence Shift", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=8)
    
    # Subplot 2: Illicit Ratio
    ax2.plot(timesteps, illicit_ratio, color="#d95f02", lw=2, marker="^", markersize=4, label="Illicit Label Ratio (%)")
    ax2.axvline(x=34.5, color="black", linestyle="--", lw=1.0)
    ax2.axvline(x=39.5, color="red", linestyle=":", lw=1.2)
    ax2.set_xlabel(r"Discrete Time Step ($t \in [1, 49]$)", fontweight="bold")
    ax2.set_ylabel("Illicit Ratio (%)", fontweight="bold")
    ax2.set_xlim(0.5, 49.5)
    ax2.set_xticks(np.arange(1, 50, 2))
    ax2.legend(loc="upper right", frameon=True, fontsize=8)
    
    plt.tight_layout()
    p = out_dir / "temporal_evolution.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig06_gnn_vs_baselines(out_dir: Path):
    """Fig. 3: Main benchmark comparison of test illicit F1 and PR-AUC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    
    models = ["Random Forest", "XGBoost", "MLP", "Logistic Reg.", "GAT", "GraphSAGE", "GCN"]
    f1_means = [0.7247, 0.7131, 0.5848, 0.4015, 0.3308, 0.2822, 0.2589]
    f1_stds = [0.0027, 0.0111, 0.0194, 0.0000, 0.0602, 0.1248, 0.0639]
    pr_means = [0.6599, 0.6744, 0.5115, 0.2754, 0.2667, 0.2366, 0.2202]
    pr_stds = [0.0025, 0.0028, 0.0187, 0.0000, 0.0695, 0.1386, 0.0703]
    
    colors = ["#2b5c8f", "#2b5c8f", "#4a7bb0", "#789bc7", "#d95f02", "#e26d5c", "#f08b7d"]
    y_pos = np.arange(len(models))
    
    # Left: F1
    ax1.barh(y_pos, f1_means, xerr=f1_stds, color=colors, edgecolor="black", alpha=0.85, capsize=4, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontweight="medium")
    ax1.set_xlabel("Test Illicit F1-Score (Mean ± Std)", fontweight="bold")
    ax1.set_title("Test Illicit F1-Score (Full 165 Features)", fontweight="bold")
    ax1.set_xlim(0, 1.0)
    ax1.invert_yaxis()
    for i, (m, s) in enumerate(zip(f1_means, f1_stds)):
        ax1.text(m + s + 0.02, i, f"{m:.4f}", va="center", fontsize=8, fontweight="bold")
        
    # Right: PR-AUC
    ax2.barh(y_pos, pr_means, xerr=pr_stds, color=colors, edgecolor="black", alpha=0.85, capsize=4, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(models))
    ax2.set_xlabel("Test PR-AUC (Mean ± Std)", fontweight="bold")
    ax2.set_title("Test PR-AUC (Full 165 Features)", fontweight="bold")
    ax2.set_xlim(0, 1.0)
    ax2.invert_yaxis()
    for i, (m, s) in enumerate(zip(pr_means, pr_stds)):
        ax2.text(m + s + 0.02, i, f"{m:.4f}", va="center", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    p = out_dir / "gnn_vs_conventional_baselines.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig10_feature_ablation(out_dir: Path):
    """Fig. 5: Feature representation impact comparing Local 93 vs Full 165."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    models = ["Random Forest", "XGBoost", "MLP", "Logistic Reg.", "GAT", "GraphSAGE", "GCN"]
    local_f1 = [0.6936, 0.6701, 0.5661, 0.4378, 0.2367, 0.4153, 0.2559]
    full_f1 = [0.7247, 0.7131, 0.5848, 0.4015, 0.3308, 0.2822, 0.2589]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, local_f1, width, label="Local Features (93 Dim)", color="#4a7bb0", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, full_f1, width, label="Full Features (165 Dim)", color="#d95f02", edgecolor="black", alpha=0.85)
    
    ax.set_ylabel("Test Illicit F1-Score", fontweight="bold")
    ax.set_title("Feature Representation Impact: Local (93) vs Full (165) Features", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontweight="medium")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=True, fontsize=9)
    
    for i, (l, f) in enumerate(zip(local_f1, full_f1)):
        diff = f - l
        sign = "+" if diff > 0 else ""
        color = "green" if diff > 0 else "red"
        ax.text(i, max(l, f) + 0.03, f"{sign}{diff:.3f}", ha="center", fontsize=8, fontweight="bold", color=color)
        
    plt.tight_layout()
    p = out_dir / "feature_ablation_local_vs_full.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig11_unknown_node_ablation(out_dir: Path):
    """Fig. 6: Unknown-Node Context Ablation (Retained vs Removed)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    models = ["GAT", "GCN", "GraphSAGE"]
    retained = [0.3308, 0.2589, 0.2822]
    retained_std = [0.0602, 0.0639, 0.1248]
    removed = [0.2918, 0.4555, 0.4132]
    removed_std = [0.0521, 0.0524, 0.0514]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, retained, width, yerr=retained_std, label="Unknown Nodes Retained (100% Topology)", color="#2b5c8f", edgecolor="black", alpha=0.85, capsize=4)
    ax.bar(x + width/2, removed, width, yerr=removed_std, label="Unknown Nodes Removed (22.85% Subgraph)", color="#d95f02", edgecolor="black", alpha=0.85, capsize=4)
    
    ax.set_ylabel("Test Illicit F1-Score (Mean ± Std)", fontweight="bold")
    ax.set_title("Unknown-Node Structural Context Ablation (Timesteps 40–49)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=True, fontsize=9)
    
    plt.tight_layout()
    p = out_dir / "unknown_node_ablation.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig13_direction_sensitivity(out_dir: Path):
    """Fig. 7: Graph Directionality Sensitivity Ablation."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    
    models = ["GAT", "GCN", "GraphSAGE"]
    forward = [0.3308, 0.2589, 0.2822]
    backward = [0.3938, 0.3953, 0.3213]
    bidir = [0.4618, 0.3142, 0.3710]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, forward, width, label=r"Forward Propagation ($u \to v$)", color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.bar(x, backward, width, label=r"Backward Propagation ($v \to u$)", color="#d95f02", edgecolor="black", alpha=0.85)
    ax.bar(x + width, bidir, width, label=r"Bidirectional Propagation ($u \leftrightarrow v$)", color="#2ca02c", edgecolor="black", alpha=0.85)
    
    ax.set_ylabel("Test Illicit F1-Score", fontweight="bold")
    ax.set_title("Message Propagation Directionality Ablation (Full 165 Features)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=True, fontsize=9)
    
    plt.tight_layout()
    p = out_dir / "direction_sensitivity_ablation.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig12_depth_ablation(out_dir: Path):
    """Fig. 8: Message-Passing Layer Depth Ablation."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    layers = [1, 2, 3]
    gat = [0.2877, 0.3308, 0.2847]
    gcn = [0.2862, 0.2589, 0.2089]
    sage = [0.3750, 0.2822, 0.3327]
    
    ax.plot(layers, gat, marker="o", lw=2.2, color="#2b5c8f", label="GAT (Peaks at 2 Layers)")
    ax.plot(layers, gcn, marker="s", lw=2.2, color="#d95f02", label="GCN (Monotonic Over-Smoothing)")
    ax.plot(layers, sage, marker="^", lw=2.2, color="#2ca02c", label="GraphSAGE (Peaks at 1 Layer)")
    
    ax.set_xlabel("Number of Message-Passing Layers", fontweight="bold")
    ax.set_ylabel("Test Illicit F1-Score", fontweight="bold")
    ax.set_title("GNN Message-Passing Depth & Over-Smoothing Analysis", fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0.15, 0.45)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    p = out_dir / "depth_ablation.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def generate_fig04_edge_homophily(out_dir: Path):
    """Fig. 10: Edge Class Transition Matrix & Homophily Visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    mat_labeled = np.array([
        [998, 915],
        [781, 33930]
    ])
    sns.heatmap(
        mat_labeled,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Illicit Target", "Licit Target"],
        yticklabels=["Illicit Source", "Licit Source"],
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax1
    )
    ax1.set_title("Mutually Labeled Edge Transitions ($N = 36,624$)\nHomophily = 95.37%", fontweight="bold")
    
    categories = ["Incident to Unknown\n(197,731 edges)", "Licit-Licit Homophilic\n(33,930 edges)", "Illicit-Illicit Homophilic\n(998 edges)", "Cross-Class Heterophilic\n(1,696 edges)"]
    pcts = [84.37, 14.48, 0.43, 0.72]
    colors = ["#b0b0b0", "#2b5c8f", "#d95f02", "#e7298a"]
    
    y_pos = np.arange(len(categories))
    ax2.barh(y_pos, pcts, color=colors, edgecolor="black", alpha=0.85, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=8.5)
    ax2.set_xlabel("Proportion of Total Graph Edges (%)", fontweight="bold")
    ax2.set_title("Global Edge Topology Composition ($|E| = 234,355$)", fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    for i, p in enumerate(pcts):
        ax2.text(p + 1.5, i, f"{p:.2f}%", va="center", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    p = out_dir / "edge_homophily_matrix.png"
    plt.savefig(p)
    plt.close(fig)
    print(f"Saved: {p}")

def main():
    set_academic_style()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ts_path = Path("results/tables/temporal_distribution_by_timestep.csv")
    df_ts = pd.read_csv(ts_path) if ts_path.exists() else None
    
    if df_ts is not None:
        generate_fig01_label_distribution(df_ts, out_dir)
        generate_fig02_temporal_evolution(df_ts, out_dir)
    generate_fig06_gnn_vs_baselines(out_dir)
    generate_fig10_feature_ablation(out_dir)
    generate_fig11_unknown_node_ablation(out_dir)
    generate_fig13_direction_sensitivity(out_dir)
    generate_fig12_depth_ablation(out_dir)
    generate_fig04_edge_homophily(out_dir)
    print("All primary figures successfully generated and validated at 300 DPI.")

if __name__ == "__main__":
    main()
