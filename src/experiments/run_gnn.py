"""
Primary Graph Neural Network Experiment Orchestrator (30 Runs).
Executes GCN, GraphSAGE, and GAT across Local & Full features and 5 seeds with safe resume.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.graph_builder import build_pyg_graph
from src.data.loader import load_classes, load_edgelist, load_features
from src.models.gat import GATNet
from src.models.gcn import GCNNet
from src.models.graphsage import GraphSAGENet
from src.training.gnn_trainer import evaluate_saved_checkpoint, setup_device_and_log, train_and_evaluate_gnn
from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger("gnn_runner")


def instantiate_gnn(
    model_name: str,
    in_channels: int,
    m_cfg: Dict
) -> torch.nn.Module:
    """Helper to instantiate GNN module from config parameters."""
    hidden_dim = m_cfg.get("hidden_dim", 128)
    num_layers = m_cfg.get("num_layers", 2)
    dropout = m_cfg.get("dropout", 0.3)
    
    if model_name == "GCN":
        return GCNNet(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_name == "GraphSAGE":
        return GraphSAGENet(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggr=m_cfg.get("aggr", "mean")
        )
    elif model_name == "GAT":
        return GATNet(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            heads=m_cfg.get("heads", 8),
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")


def run_primary_gnn_experiments(
    cfg_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict], Dict]:
    """
    Executes the 30 primary GNN experiment runs (3 models x 2 feature sets x 5 seeds).
    Safely resumes from existing completed runs in results/tables/gnn_all_runs.csv.
    
    Returns:
        Tuple of (df_all_runs, df_summary, df_best, representative_curve_data, best_overall_run_dict).
    """
    root = get_project_root()
    if cfg_path is None:
        cfg_path = root / "configs" / "gnn_config.yaml"
    else:
        cfg_path = Path(cfg_path)
        
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    seeds = cfg["experiment"].get("seeds", [42, 123, 456, 789, 999])
    primary_dir = cfg["experiment"].get("primary_direction", "forward")
    val_interval = cfg["experiment"].get("validation_interval", 2)
    
    tables_dir = root / "results" / "tables"
    ckpt_dir = root / "models" / "checkpoints"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Hardware setup
    device_obj = setup_device_and_log()
    
    # 2. Check for existing runs in CSV
    all_runs_csv = tables_dir / "gnn_all_runs.csv"
    summary_csv = tables_dir / "gnn_summary_mean_std.csv"
    best_csv = tables_dir / "gnn_best_validation.csv"
    best_csv_legacy = tables_dir / "gnn_best_validation_config.csv"
    benchmark_csv = tables_dir / "gnn_compute_benchmark.csv"
    
    all_run_records = []
    completed_keys = set()
    
    if all_runs_csv.exists():
        df_existing = pd.read_csv(all_runs_csv)
        # Verify required columns exist
        required_cols = ["model", "feature_mode", "seed", "test_illicit_f1", "test_pr_auc"]
        if all(col in df_existing.columns for col in required_cols):
            # Only count rows that have non-null test metrics
            valid_existing = df_existing.dropna(subset=["test_illicit_f1", "test_pr_auc"]).copy()
            all_run_records = valid_existing.to_dict("records")
            for r in all_run_records:
                completed_keys.add((str(r["model"]), str(r["feature_mode"]), int(r["seed"])))
            logger.info(f"Loaded {len(all_run_records)} valid completed runs from {all_runs_csv}.")
        else:
            logger.warning(f"Existing {all_runs_csv} missing required columns. Rebuilding run table.")
            
    # 3. Load Raw Data ONCE
    raw_features_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv"
    raw_classes_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv"
    raw_edgelist_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_edgelist.csv"
    
    logger.info("Loading raw dataset components for GNN graph builder...")
    df_classes = load_classes(raw_classes_file)
    df_features = load_features(raw_features_file, total_features=165, num_local=93)
    df_edgelist = load_edgelist(raw_edgelist_file)
    
    # 4. Cache Graphs in memory: pyg_local and pyg_full
    logger.info("Building and caching PyG graphs (Local 93 and Full 165)...")
    pyg_graphs = {
        "local": build_pyg_graph(
            df_features=df_features,
            df_classes=df_classes,
            df_edgelist=df_edgelist,
            feature_mode="local",
            direction=primary_dir,
            remove_unknown=False,
            scale=True,
            train_timesteps=tuple(cfg["splits"]["train_timesteps"]),
            val_timesteps=tuple(cfg["splits"]["val_timesteps"]),
            test_timesteps=tuple(cfg["splits"]["test_timesteps"])
        ),
        "full": build_pyg_graph(
            df_features=df_features,
            df_classes=df_classes,
            df_edgelist=df_edgelist,
            feature_mode="full",
            direction=primary_dir,
            remove_unknown=False,
            scale=True,
            train_timesteps=tuple(cfg["splits"]["train_timesteps"]),
            val_timesteps=tuple(cfg["splits"]["val_timesteps"]),
            test_timesteps=tuple(cfg["splits"]["test_timesteps"])
        )
    }
    
    # Primary Execution Sequence matching required order:
    # 1. GCN Local, 2. GraphSAGE Local, 3. GAT Local
    # 4. GCN Full, 5. GraphSAGE Full, 6. GAT Full
    execution_plan = [
        ("GCN", "local", "Local Only (93)"),
        ("GraphSAGE", "local", "Local Only (93)"),
        ("GAT", "local", "Local Only (93)"),
        ("GCN", "full", "Full Features (165)"),
        ("GraphSAGE", "full", "Full Features (165)"),
        ("GAT", "full", "Full Features (165)"),
    ]
    
    total_runs = len(execution_plan) * len(seeds)
    logger.info(f"Total Primary Runs in Matrix: {total_runs} (Already completed: {len(completed_keys)})")
    
    representative_curve_data = []
    best_overall_result = None
    best_overall_val_f1 = -1.0
    
    run_idx = 1
    
    for model_name, feat_mode, feat_label in execution_plan:
        pyg_data = pyg_graphs[feat_mode]
        in_dim = pyg_data.num_features
        m_cfg = cfg["models"][model_name.lower()]
        m_epochs = m_cfg.get("epochs", 100)
        m_patience = m_cfg.get("patience", 15)
        m_lr = float(m_cfg.get("lr", 1e-3))
        m_wd = float(m_cfg.get("weight_decay", 1e-4))
        m_val_interval = m_cfg.get("val_interval", val_interval)
        
        logger.info(f"\n{'='*70}\nProcessing Group: {model_name} | {feat_label}\n{'='*70}")
        
        group_completed = 0
        
        for seed in seeds:
            key = (model_name, feat_mode, seed)
            if key in completed_keys:
                print(f"[SKIP] {model_name} | {feat_label} | seed={seed} | already completed")
                logger.info(f"[{run_idx}/{total_runs}] [SKIP] {model_name} | {feat_label} | Seed {seed} | already completed")
                
                # Check if we have curve data for seed 42
                if seed == 42:
                    # Find matching record in existing runs
                    for r in all_run_records:
                        if r["model"] == model_name and r["feature_mode"] == feat_mode and int(r["seed"]) == 42:
                            if r.get("test_confusion_matrix") is not None:
                                representative_curve_data.append({
                                    "name": f"{model_name} [{feat_label}]",
                                    "model": model_name,
                                    "threshold": r.get("optimal_threshold", 0.5),
                                    "pr_auc": r.get("test_pr_auc", 0.0),
                                    "roc_auc": r.get("test_roc_auc", 0.0),
                                    "confusion_matrix": eval(str(r["test_confusion_matrix"])) if isinstance(r["test_confusion_matrix"], str) else r["test_confusion_matrix"],
                                    "pr_curve": eval(str(r["test_pr_curve"])) if isinstance(r.get("test_pr_curve"), str) else r.get("test_pr_curve", {}),
                                    "roc_curve": eval(str(r["test_roc_curve"])) if isinstance(r.get("test_roc_curve"), str) else r.get("test_roc_curve", {}),
                                })
                            break
                            
                run_idx += 1
                group_completed += 1
                continue
                
            print(f"[RUN] {model_name} | {feat_label} | seed={seed}")
            logger.info(f"[{run_idx}/{total_runs}] [RUN] {model_name} | {feat_label} | Seed {seed}")
            
            # Instantiate fresh network
            net = instantiate_gnn(model_name, in_channels=in_dim, m_cfg=m_cfg)
            
            # Train and evaluate
            eval_res = train_and_evaluate_gnn(
                model=net,
                data=pyg_data,
                seed=seed,
                epochs=m_epochs,
                lr=m_lr,
                weight_decay=m_wd,
                patience=m_patience,
                val_interval=m_val_interval,
                device=str(device_obj)
            )
            
            # Save model checkpoint
            ckpt_file = ckpt_dir / f"{model_name.lower()}_{feat_mode}_seed{seed}.pt"
            torch.save(eval_res["best_state_dict"], ckpt_file)
            
            record = {
                "run_id": len(all_run_records) + 1,
                "model": model_name,
                "feature_mode": feat_mode,
                "feature_space": feat_label,
                "num_features": in_dim,
                "direction": primary_dir,
                "unknown_treatment": "retained",
                "seed": seed,
                "num_params": eval_res["num_params"],
                "best_epoch": eval_res["best_epoch"],
                "epochs_trained": eval_res["epochs_trained"],
                "optimal_threshold": eval_res["optimal_threshold"],
                "training_time_sec": eval_res["training_time_sec"],
                "inference_time_sec": eval_res["inference_time_sec"],
                "peak_ram_mb": eval_res["peak_ram_mb"],
                "device_used": eval_res["device_used"],
                "pos_weight": eval_res["pos_weight"],
                # Validation Metrics
                "val_illicit_f1": eval_res["val_illicit_f1"],
                "val_pr_auc": eval_res["val_pr_auc"],
                "val_illicit_precision": eval_res["val_illicit_precision"],
                "val_illicit_recall": eval_res["val_illicit_recall"],
                "val_roc_auc": eval_res["val_roc_auc"],
                "val_loss": eval_res["val_loss"],
                # Test Metrics (Locked Threshold)
                "test_illicit_f1": eval_res["test_illicit_f1"],
                "test_pr_auc": eval_res["test_pr_auc"],
                "test_illicit_precision": eval_res["test_illicit_precision"],
                "test_illicit_recall": eval_res["test_illicit_recall"],
                "test_macro_f1": eval_res["test_macro_f1"],
                "test_roc_auc": eval_res["test_roc_auc"],
                "test_mcc": eval_res["test_mcc"],
                "test_accuracy": eval_res["test_accuracy"],
                "test_tp": eval_res["test_tp"],
                "test_fp": eval_res["test_fp"],
                "test_tn": eval_res["test_tn"],
                "test_fn": eval_res["test_fn"],
                "test_confusion_matrix": str(eval_res["test_confusion_matrix"]),
                "test_pr_curve": str(eval_res["test_pr_curve"]),
                "test_roc_curve": str(eval_res["test_roc_curve"]),
            }
            
            all_run_records.append(record)
            completed_keys.add((model_name, feat_mode, seed))
            
            # Incremental CSV write
            df_current = pd.DataFrame(all_run_records)
            df_current.to_csv(all_runs_csv, index=False)
            
            # Print completion summary
            print(
                f"[COMPLETED]\n"
                f"{model_name} | {feat_label} | seed={seed}\n"
                f"Best epoch: {eval_res['best_epoch']} / {eval_res['epochs_trained']}\n"
                f"Val PR-AUC: {eval_res['val_pr_auc']:.4f} | Val F1: {eval_res['val_illicit_f1']:.4f}\n"
                f"Test PR-AUC: {eval_res['test_pr_auc']:.4f} | Test F1: {eval_res['test_illicit_f1']:.4f}\n"
                f"Test Precision: {eval_res['test_illicit_precision']:.4f} | Test Recall: {eval_res['test_illicit_recall']:.4f}\n"
                f"Time: {eval_res['training_time_sec']:.2f} sec | Tau*: {eval_res['optimal_threshold']:.2f}\n"
            )
            
            if seed == 42:
                representative_curve_data.append({
                    "name": f"{model_name} [{feat_label}]",
                    "model": model_name,
                    "threshold": eval_res["optimal_threshold"],
                    "pr_auc": eval_res["test_pr_auc"],
                    "roc_auc": eval_res["test_roc_auc"],
                    "confusion_matrix": eval_res["test_confusion_matrix"],
                    "pr_curve": eval_res["test_pr_curve"],
                    "roc_curve": eval_res["test_roc_curve"],
                })
                
            if eval_res["val_illicit_f1"] > best_overall_val_f1:
                best_overall_val_f1 = eval_res["val_illicit_f1"]
                best_overall_result = eval_res
                
            run_idx += 1
            group_completed += 1
            
        print(f"--- Aggregate Group Summary: {model_name} [{feat_label}] Complete ({group_completed}/5 seeds) ---")
        
    # 5. Compile Final All Runs DataFrame
    df_all_runs = pd.DataFrame(all_run_records)
    df_all_runs.to_csv(all_runs_csv, index=False)
    logger.info(f"All {len(df_all_runs)} primary GNN runs verified in: {all_runs_csv}")
    
    # 6. Generate Summary Table (Mean ± Std)
    group_cols = ["model", "feature_mode", "feature_space", "num_features", "direction"]
    numeric_metric_cols = [
        "optimal_threshold", "training_time_sec", "inference_time_sec",
        "val_illicit_f1", "val_pr_auc", "val_illicit_precision", "val_illicit_recall", "val_roc_auc",
        "test_illicit_f1", "test_pr_auc", "test_illicit_precision", "test_illicit_recall",
        "test_macro_f1", "test_roc_auc", "test_mcc", "test_accuracy"
    ]
    
    summary_rows = []
    for keys, group in df_all_runs.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        for col in numeric_metric_cols:
            if col in group.columns:
                mean_val = float(group[col].astype(float).mean())
                std_val = float(group[col].astype(float).std())
                rec[f"{col}_mean"] = mean_val
                rec[f"{col}_std"] = std_val
                rec[f"{col}_formatted"] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_rows.append(rec)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by="test_illicit_f1_mean", ascending=False).reset_index(drop=True)
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Summary table saved to: {summary_csv}")
    
    # 7. Best Configurations Table
    models_tested = ["GCN", "GraphSAGE", "GAT"]
    best_configs = []
    for m in models_tested:
        sub = df_summary[df_summary["model"] == m]
        if len(sub) > 0:
            best_m = sub.loc[sub["val_illicit_f1_mean"].idxmax()].to_dict()
            best_configs.append({
                "Selection Category": f"Best {m} (Val F1)",
                "Model": best_m["model"],
                "Feature Space": best_m["feature_space"],
                "Validation Illicit F1": best_m["val_illicit_f1_formatted"],
                "Validation PR-AUC": best_m["val_pr_auc_formatted"],
                "Test Illicit F1": best_m["test_illicit_f1_formatted"],
                "Test PR-AUC": best_m["test_pr_auc_formatted"],
                "Test Precision": best_m["test_illicit_precision_formatted"],
                "Test Recall": best_m["test_illicit_recall_formatted"],
                "Test ROC-AUC": best_m["test_roc_auc_formatted"],
                "Test MCC": best_m["test_mcc_formatted"],
            })
            
    best_overall_row = df_summary.loc[df_summary["val_illicit_f1_mean"].idxmax()].to_dict()
    best_configs.insert(0, {
        "Selection Category": "★ Overall Best GNN (Val F1)",
        "Model": best_overall_row["model"],
        "Feature Space": best_overall_row["feature_space"],
        "Validation Illicit F1": best_overall_row["val_illicit_f1_formatted"],
        "Validation PR-AUC": best_overall_row["val_pr_auc_formatted"],
        "Test Illicit F1": best_overall_row["test_illicit_f1_formatted"],
        "Test PR-AUC": best_overall_row["test_pr_auc_formatted"],
        "Test Precision": best_overall_row["test_illicit_precision_formatted"],
        "Test Recall": best_overall_row["test_illicit_recall_formatted"],
        "Test ROC-AUC": best_overall_row["test_roc_auc_formatted"],
        "Test MCC": best_overall_row["test_mcc_formatted"],
    })
    
    df_best = pd.DataFrame(best_configs)
    df_best.to_csv(best_csv, index=False)
    df_best.to_csv(best_csv_legacy, index=False)
    logger.info(f"Best GNN validation configs saved to: {best_csv}")
    
    # 8. Computational Benchmark Table
    benchmark_cols = [
        "model", "feature_space", "seed", "num_params", "epochs_trained", "best_epoch",
        "training_time_sec", "inference_time_sec", "peak_ram_mb", "device_used",
        "val_pr_auc", "test_pr_auc", "val_illicit_f1", "test_illicit_f1"
    ]
    avail_bench_cols = [c for c in benchmark_cols if c in df_all_runs.columns]
    df_benchmark = df_all_runs[avail_bench_cols].copy()
    df_benchmark.to_csv(benchmark_csv, index=False)
    logger.info(f"Computational benchmark table saved to: {benchmark_csv}")
    
    # 9. Ensure best_overall_result is populated
    if best_overall_result is None and len(df_all_runs) > 0:
        # Identify top run by validation F1
        best_run_row = df_all_runs.loc[df_all_runs["val_illicit_f1"].astype(float).idxmax()]
        b_model = str(best_run_row["model"])
        b_feat_mode = str(best_run_row["feature_mode"])
        b_seed = int(best_run_row["seed"])
        
        ckpt_file = ckpt_dir / f"{b_model.lower()}_{b_feat_mode}_seed{b_seed}.pt"
        pyg_eval = pyg_graphs[b_feat_mode]
        in_dim_eval = pyg_eval.num_features
        m_cfg_eval = cfg["models"][b_model.lower()]
        net_eval = instantiate_gnn(b_model, in_channels=in_dim_eval, m_cfg=m_cfg_eval)
        
        if ckpt_file.exists():
            logger.info(f"Reconstructing best_overall_result from checkpoint: {ckpt_file}")
            best_overall_result = evaluate_saved_checkpoint(
                model=net_eval,
                data=pyg_eval,
                checkpoint_path=ckpt_file,
                device=str(device_obj)
            )
        else:
            logger.info(f"Checkpoint not found on disk. Re-evaluating best model {b_model} ({b_feat_mode}, seed {b_seed})...")
            best_overall_result = train_and_evaluate_gnn(
                model=net_eval,
                data=pyg_eval,
                seed=b_seed,
                epochs=m_cfg_eval.get("epochs", 100),
                lr=float(m_cfg_eval.get("lr", 1e-3)),
                weight_decay=float(m_cfg_eval.get("weight_decay", 1e-4)),
                patience=m_cfg_eval.get("patience", 15),
                val_interval=m_cfg_eval.get("val_interval", val_interval),
                device=str(device_obj)
            )
            
    return df_all_runs, df_summary, df_best, representative_curve_data, best_overall_result
