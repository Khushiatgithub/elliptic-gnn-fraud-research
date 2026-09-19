"""
Ablation Study Orchestrator for Phase 3 GNN Experiments.
Executes:
1. Feature Ablation (Local 93 vs Full 165)
2. Unknown-Node Context Ablation (Retained vs Removed)
3. Graph Direction Ablation (Forward vs Backward vs Bidirectional)
4. GNN Depth / Over-Smoothing Ablation (1, 2, 3 Layers)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.data.graph_builder import build_pyg_graph
from src.data.loader import load_classes, load_edgelist, load_features
from src.models.gat import GATNet
from src.models.gcn import GCNNet
from src.models.graphsage import GraphSAGENet
from src.training.gnn_trainer import train_and_evaluate_gnn
from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger("gnn_ablations")


def run_all_ablations(
    df_primary_all_runs: pd.DataFrame,
    cfg_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes all required ablation studies across 5 random seeds.
    
    Returns:
        Tuple of (df_feature_ablation, df_unknown_ablation, df_direction_ablation, df_depth_ablation).
    """
    root = get_project_root()
    if cfg_path is None:
        cfg_path = root / "configs" / "gnn_config.yaml"
    else:
        cfg_path = Path(cfg_path)
        
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    seeds = cfg["experiment"].get("seeds", [42, 123, 456, 789, 999])
    val_interval = cfg["experiment"].get("validation_interval", 2)
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    raw_features_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv"
    raw_classes_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv"
    raw_edgelist_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_edgelist.csv"
    
    df_classes = load_classes(raw_classes_file)
    df_features = load_features(raw_features_file, total_features=165, num_local=93)
    df_edgelist = load_edgelist(raw_edgelist_file)
    
    models = ["GCN", "GraphSAGE", "GAT"]
    
    # -------------------------------------------------------------
    # 1. Feature Ablation (Local 93 vs Full 165)
    # -------------------------------------------------------------
    logger.info("Compiling Feature Space Ablation from primary runs...")
    feature_rows = []
    for (m, f_mode, f_space), grp in df_primary_all_runs.groupby(["model", "feature_mode", "feature_space"]):
        feature_rows.append({
            "model": m,
            "feature_mode": f_mode,
            "feature_space": f_space,
            "test_illicit_f1_mean": grp["test_illicit_f1"].astype(float).mean(),
            "test_illicit_f1_std": grp["test_illicit_f1"].astype(float).std(),
            "test_illicit_f1_formatted": f"{grp['test_illicit_f1'].astype(float).mean():.4f} ± {grp['test_illicit_f1'].astype(float).std():.4f}",
            "test_pr_auc_mean": grp["test_pr_auc"].astype(float).mean(),
            "test_pr_auc_std": grp["test_pr_auc"].astype(float).std(),
            "test_pr_auc_formatted": f"{grp['test_pr_auc'].astype(float).mean():.4f} ± {grp['test_pr_auc'].astype(float).std():.4f}",
            "test_precision_mean": grp["test_illicit_precision"].astype(float).mean(),
            "test_recall_mean": grp["test_illicit_recall"].astype(float).mean(),
            "test_mcc_mean": grp["test_mcc"].astype(float).mean() if "test_mcc" in grp.columns else 0.0,
        })
    df_feature_ablation = pd.DataFrame(feature_rows)
    df_feature_ablation.to_csv(tables_dir / "gnn_ablation_feature_space.csv", index=False)
    df_feature_ablation.to_csv(tables_dir / "feature_ablation.csv", index=False)
    
    # -------------------------------------------------------------
    # 2. Unknown-Node Context Ablation (Retained vs Removed)
    # -------------------------------------------------------------
    unk_csv_path = tables_dir / "gnn_ablation_unknown_context.csv"
    if unk_csv_path.exists() and (tables_dir / "unknown_node_ablation.csv").exists():
        logger.info(f"Loading existing Unknown-Node Context Ablation from: {unk_csv_path}")
        df_unknown_ablation = pd.read_csv(unk_csv_path)
    else:
        logger.info("Running Unknown-Node Context Ablation (Removed Graph)...")
        # Build graph with remove_unknown=True
        pyg_removed = build_pyg_graph(
            df_features=df_features,
            df_classes=df_classes,
            df_edgelist=df_edgelist,
            feature_mode="full",
            direction="forward",
            remove_unknown=True,
            scale=True,
            train_timesteps=tuple(cfg["splits"]["train_timesteps"]),
            val_timesteps=tuple(cfg["splits"]["val_timesteps"]),
            test_timesteps=tuple(cfg["splits"]["test_timesteps"])
        )
        in_dim_full = pyg_removed.num_features
        
        unknown_ablation_runs = []
        # Add retained runs from primary experiments
        for _, row in df_primary_all_runs[df_primary_all_runs["feature_mode"] == "full"].iterrows():
            unknown_ablation_runs.append({
                "model": row["model"],
                "unknown_treatment": "Retained (100% Graph)",
                "seed": int(row["seed"]),
                "test_illicit_f1": float(row["test_illicit_f1"]),
                "test_pr_auc": float(row["test_pr_auc"]),
                "test_precision": float(row["test_illicit_precision"]),
                "test_recall": float(row["test_illicit_recall"]),
                "test_mcc": float(row.get("test_mcc", 0.0))
            })
            
        for model_name in models:
            m_cfg = cfg["models"][model_name.lower()]
            for seed in seeds:
                if model_name == "GCN":
                    net = GCNNet(in_dim_full, hidden_channels=128, num_layers=2, dropout=0.3)
                elif model_name == "GraphSAGE":
                    net = GraphSAGENet(in_dim_full, hidden_channels=128, num_layers=2, dropout=0.3, aggr="mean")
                elif model_name == "GAT":
                    net = GATNet(in_dim_full, hidden_channels=128, num_layers=2, heads=8, dropout=0.3)
                    
                res = train_and_evaluate_gnn(
                    net, pyg_removed, seed=seed, epochs=m_cfg.get("epochs", 100),
                    lr=float(m_cfg.get("lr", 1e-3)), weight_decay=float(m_cfg.get("weight_decay", 1e-4)),
                    patience=m_cfg.get("patience", 15), val_interval=val_interval
                )
                unknown_ablation_runs.append({
                    "model": model_name,
                    "unknown_treatment": "Removed (Labeled Only)",
                    "seed": seed,
                    "test_illicit_f1": res["test_illicit_f1"],
                    "test_pr_auc": res["test_pr_auc"],
                    "test_precision": res["test_illicit_precision"],
                    "test_recall": res["test_illicit_recall"],
                    "test_mcc": res["test_mcc"]
                })
                
        df_unknown_runs = pd.DataFrame(unknown_ablation_runs)
        unk_summary_rows = []
        for (m, trt), grp in df_unknown_runs.groupby(["model", "unknown_treatment"]):
            unk_summary_rows.append({
                "model": m,
                "unknown_treatment": trt,
                "test_illicit_f1_mean": grp["test_illicit_f1"].mean(),
                "test_illicit_f1_std": grp["test_illicit_f1"].std(),
                "test_illicit_f1_formatted": f"{grp['test_illicit_f1'].mean():.4f} ± {grp['test_illicit_f1'].std():.4f}",
                "test_pr_auc_mean": grp["test_pr_auc"].mean(),
                "test_pr_auc_std": grp["test_pr_auc"].std(),
                "test_pr_auc_formatted": f"{grp['test_pr_auc'].mean():.4f} ± {grp['test_pr_auc'].std():.4f}",
                "test_precision_formatted": f"{grp['test_precision'].mean():.4f} ± {grp['test_precision'].std():.4f}",
                "test_recall_formatted": f"{grp['test_recall'].mean():.4f} ± {grp['test_recall'].std():.4f}",
                "test_mcc_formatted": f"{grp['test_mcc'].mean():.4f} ± {grp['test_mcc'].std():.4f}",
            })
        df_unknown_ablation = pd.DataFrame(unk_summary_rows)
        df_unknown_ablation.to_csv(unk_csv_path, index=False)
        df_unknown_ablation.to_csv(tables_dir / "unknown_node_ablation.csv", index=False)
        logger.info(f"Unknown node ablation saved to: {unk_csv_path}")

    # -------------------------------------------------------------
    # 3. Graph Direction Ablation (Forward vs Backward vs Bidirectional)
    # -------------------------------------------------------------
    dir_csv_path = tables_dir / "gnn_ablation_direction.csv"
    if dir_csv_path.exists() and (tables_dir / "graph_direction_ablation.csv").exists():
        logger.info(f"Loading existing Graph Direction Ablation from: {dir_csv_path}")
        df_direction_ablation = pd.read_csv(dir_csv_path)
    else:
        logger.info("Running Graph Direction Ablation (Backward & Bidirectional)...")
        direction_runs = []
        
        # Add forward from primary
        for _, row in df_primary_all_runs[df_primary_all_runs["feature_mode"] == "full"].iterrows():
            direction_runs.append({
                "model": row["model"],
                "direction": "Forward (u -> v)",
                "seed": int(row["seed"]),
                "test_illicit_f1": float(row["test_illicit_f1"]),
                "test_pr_auc": float(row["test_pr_auc"]),
                "test_precision": float(row["test_illicit_precision"]),
                "test_recall": float(row["test_illicit_recall"]),
                "test_mcc": float(row.get("test_mcc", 0.0))
            })
            
        for dir_mode, dir_label in [("backward", "Backward (v -> u)"), ("bidirectional", "Bidirectional (u <-> v)")]:
            pyg_dir = build_pyg_graph(
                df_features=df_features,
                df_classes=df_classes,
                df_edgelist=df_edgelist,
                feature_mode="full",
                direction=dir_mode,
                remove_unknown=False,
                scale=True,
                train_timesteps=tuple(cfg["splits"]["train_timesteps"]),
                val_timesteps=tuple(cfg["splits"]["val_timesteps"]),
                test_timesteps=tuple(cfg["splits"]["test_timesteps"])
            )
            in_dim_full = pyg_dir.num_features
            for model_name in models:
                m_cfg = cfg["models"][model_name.lower()]
                for seed in seeds:
                    if model_name == "GCN":
                        net = GCNNet(in_dim_full, hidden_channels=128, num_layers=2, dropout=0.3)
                    elif model_name == "GraphSAGE":
                        net = GraphSAGENet(in_dim_full, hidden_channels=128, num_layers=2, dropout=0.3, aggr="mean")
                    elif model_name == "GAT":
                        net = GATNet(in_dim_full, hidden_channels=128, num_layers=2, heads=8, dropout=0.3)
                        
                    res = train_and_evaluate_gnn(
                        net, pyg_dir, seed=seed, epochs=m_cfg.get("epochs", 100),
                        lr=float(m_cfg.get("lr", 1e-3)), weight_decay=float(m_cfg.get("weight_decay", 1e-4)),
                        patience=m_cfg.get("patience", 15), val_interval=val_interval
                    )
                    direction_runs.append({
                        "model": model_name,
                        "direction": dir_label,
                        "seed": seed,
                        "test_illicit_f1": res["test_illicit_f1"],
                        "test_pr_auc": res["test_pr_auc"],
                        "test_precision": res["test_illicit_precision"],
                        "test_recall": res["test_illicit_recall"],
                        "test_mcc": res["test_mcc"]
                    })
                    
        df_dir_runs = pd.DataFrame(direction_runs)
        dir_summary_rows = []
        for (m, d), grp in df_dir_runs.groupby(["model", "direction"]):
            dir_summary_rows.append({
                "model": m,
                "direction": d,
                "test_illicit_f1_mean": grp["test_illicit_f1"].mean(),
                "test_illicit_f1_std": grp["test_illicit_f1"].std(),
                "test_illicit_f1_formatted": f"{grp['test_illicit_f1'].mean():.4f} ± {grp['test_illicit_f1'].std():.4f}",
                "test_pr_auc_mean": grp["test_pr_auc"].mean(),
                "test_pr_auc_std": grp["test_pr_auc"].std(),
                "test_pr_auc_formatted": f"{grp['test_pr_auc'].mean():.4f} ± {grp['test_pr_auc'].std():.4f}",
                "test_precision_formatted": f"{grp['test_precision'].mean():.4f} ± {grp['test_precision'].std():.4f}",
                "test_recall_formatted": f"{grp['test_recall'].mean():.4f} ± {grp['test_recall'].std():.4f}",
                "test_mcc_formatted": f"{grp['test_mcc'].mean():.4f} ± {grp['test_mcc'].std():.4f}",
            })
        df_direction_ablation = pd.DataFrame(dir_summary_rows)
        df_direction_ablation.to_csv(dir_csv_path, index=False)
        df_direction_ablation.to_csv(tables_dir / "graph_direction_ablation.csv", index=False)
        logger.info(f"Graph direction ablation saved to: {dir_csv_path}")

    # -------------------------------------------------------------
    # 4. GNN Depth / Over-Smoothing Ablation (1, 2, 3 Layers)
    # -------------------------------------------------------------
    depth_csv_path = tables_dir / "gnn_ablation_depth.csv"
    if depth_csv_path.exists() and (tables_dir / "depth_ablation.csv").exists():
        logger.info(f"Loading existing Depth Ablation from: {depth_csv_path}")
        df_depth_ablation = pd.read_csv(depth_csv_path)
    else:
        logger.info("Running GNN Depth / Over-Smoothing Ablation (Layers 1 & 3)...")
        # Forward full graph
        pyg_forward = build_pyg_graph(
            df_features=df_features,
            df_classes=df_classes,
            df_edgelist=df_edgelist,
            feature_mode="full",
            direction="forward",
            remove_unknown=False,
            scale=True,
            train_timesteps=tuple(cfg["splits"]["train_timesteps"]),
            val_timesteps=tuple(cfg["splits"]["val_timesteps"]),
            test_timesteps=tuple(cfg["splits"]["test_timesteps"])
        )
        in_dim_full = pyg_forward.num_features
        
        depth_runs = []
        # Add 2 layers from primary runs
        for _, row in df_primary_all_runs[df_primary_all_runs["feature_mode"] == "full"].iterrows():
            depth_runs.append({
                "model": row["model"],
                "num_layers": 2,
                "seed": int(row["seed"]),
                "test_illicit_f1": float(row["test_illicit_f1"]),
                "test_pr_auc": float(row["test_pr_auc"]),
                "test_precision": float(row["test_illicit_precision"]),
                "test_recall": float(row["test_illicit_recall"]),
                "test_mcc": float(row.get("test_mcc", 0.0))
            })
            
        for layer_k in [1, 3]:
            for model_name in models:
                m_cfg = cfg["models"][model_name.lower()]
                for seed in seeds:
                    if model_name == "GCN":
                        net = GCNNet(in_dim_full, hidden_channels=128, num_layers=layer_k, dropout=0.3)
                    elif model_name == "GraphSAGE":
                        net = GraphSAGENet(in_dim_full, hidden_channels=128, num_layers=layer_k, dropout=0.3, aggr="mean")
                    elif model_name == "GAT":
                        net = GATNet(in_dim_full, hidden_channels=128, num_layers=layer_k, heads=8, dropout=0.3)
                        
                    res = train_and_evaluate_gnn(
                        net, pyg_forward, seed=seed, epochs=m_cfg.get("epochs", 100),
                        lr=float(m_cfg.get("lr", 1e-3)), weight_decay=float(m_cfg.get("weight_decay", 1e-4)),
                        patience=m_cfg.get("patience", 15), val_interval=val_interval
                    )
                    depth_runs.append({
                        "model": model_name,
                        "num_layers": layer_k,
                        "seed": seed,
                        "test_illicit_f1": res["test_illicit_f1"],
                        "test_pr_auc": res["test_pr_auc"],
                        "test_precision": res["test_illicit_precision"],
                        "test_recall": res["test_illicit_recall"],
                        "test_mcc": res["test_mcc"]
                    })
                    
        df_dep_runs = pd.DataFrame(depth_runs)
        dep_summary_rows = []
        for (m, k), grp in df_dep_runs.groupby(["model", "num_layers"]):
            dep_summary_rows.append({
                "model": m,
                "num_layers": int(k),
                "test_illicit_f1_mean": grp["test_illicit_f1"].mean(),
                "test_illicit_f1_std": grp["test_illicit_f1"].std(),
                "test_illicit_f1_formatted": f"{grp['test_illicit_f1'].mean():.4f} ± {grp['test_illicit_f1'].std():.4f}",
                "test_pr_auc_mean": grp["test_pr_auc"].mean(),
                "test_pr_auc_std": grp["test_pr_auc"].std(),
                "test_pr_auc_formatted": f"{grp['test_pr_auc'].mean():.4f} ± {grp['test_pr_auc'].std():.4f}",
                "test_precision_formatted": f"{grp['test_precision'].mean():.4f} ± {grp['test_precision'].std():.4f}",
                "test_recall_formatted": f"{grp['test_recall'].mean():.4f} ± {grp['test_recall'].std():.4f}",
                "test_mcc_formatted": f"{grp['test_mcc'].mean():.4f} ± {grp['test_mcc'].std():.4f}",
            })
        df_depth_ablation = pd.DataFrame(dep_summary_rows)
        df_depth_ablation.to_csv(depth_csv_path, index=False)
        df_depth_ablation.to_csv(tables_dir / "depth_ablation.csv", index=False)
        logger.info(f"Depth ablation saved to: {depth_csv_path}")
        
    return df_feature_ablation, df_unknown_ablation, df_direction_ablation, df_depth_ablation
