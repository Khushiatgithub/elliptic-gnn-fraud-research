"""
Master Baseline Training and Multi-Seed Evaluation Pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.features.preprocessing import TabularPreprocessor
from src.features.schema import verify_and_export_feature_schema
from src.models.logistic_regression import LogisticRegressionBaseline
from src.models.mlp import MLPBaseline
from src.models.random_forest import RandomForestBaseline
from src.models.xgboost_model import XGBoostBaseline
from src.training.temporal_split import create_temporal_splits
from src.training.threshold_selection import find_optimal_threshold
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.plots import generate_baseline_plots
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger("baseline_runner")


def run_baseline_experiments(
    baseline_cfg_path: Optional[Path] = None,
    reuse_existing: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the 16-experiment matrix across 5 random seeds (80 runs total).
    
    Returns:
        Tuple of (df_all_runs, df_summary_mean_std, df_best_configs)
    """
    root = get_project_root()
    if baseline_cfg_path is None:
        baseline_cfg_path = root / "configs" / "baseline_config.yaml"
    else:
        baseline_cfg_path = Path(baseline_cfg_path)
        
    with open(baseline_cfg_path, "r", encoding="utf-8") as f:
        b_cfg = yaml.safe_load(f)
        
    seeds = b_cfg["experiment"].get("seeds", [42, 123, 456, 789, 999])
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    logs_dir = root / "results" / "logs"
    
    for d in [tables_dir, figures_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    all_runs_csv = tables_dir / "baseline_all_runs.csv"
    summary_csv = tables_dir / "baseline_summary_mean_std.csv"
    best_csv = tables_dir / "baseline_best_validation_config.csv"
    
    if reuse_existing and all_runs_csv.exists() and summary_csv.exists() and best_csv.exists():
        logger.info(f"Reusing existing baseline run records from {all_runs_csv}...")
        df_all_runs = pd.read_csv(all_runs_csv)
        df_summary = pd.read_csv(summary_csv)
        df_best = pd.read_csv(best_csv)
        return df_all_runs, df_summary, df_best

    # 1. Feature Schema Verification
    raw_features_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv"
    raw_classes_file = root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv"
    
    verify_and_export_feature_schema(raw_features_file, tables_dir, logs_dir)
    
    # 2. Data Loading & Temporal Splitting
    from src.data.loader import load_classes, load_features
    df_classes = load_classes(raw_classes_file)
    df_features = load_features(raw_features_file, total_features=165, num_local=93)
    
    (df_train, y_train), (df_val, y_val), (df_test, y_test), df_splits = create_temporal_splits(
        df_features=df_features,
        df_classes=df_classes,
        train_range=tuple(b_cfg["splits"]["train_timesteps"]),
        val_range=tuple(b_cfg["splits"]["val_timesteps"]),
        test_range=tuple(b_cfg["splits"]["test_timesteps"]),
        output_dir=tables_dir
    )
    
    # 3. Model Grid Matrix
    model_types = ["Logistic Regression", "Random Forest", "XGBoost", "MLP"]
    feature_modes = ["local", "full"]
    weightings = ["Standard", "Weighted"]
    
    all_run_records = []
    representative_curve_data = []
    
    total_experiments = len(feature_modes) * len(model_types) * len(weightings) * len(seeds)
    logger.info(f"Starting {total_experiments} experiment runs (16 configs x {len(seeds)} seeds)...")
    
    run_idx = 1
    for feat_mode in feature_modes:
        feat_label = "Local Only (93)" if feat_mode == "local" else "Full Features (165)"
        
        # Build Preprocessors (Strictly fit on training split)
        # Scaled preprocessor for LR and MLP
        prep_scaled = TabularPreprocessor(feature_mode=feat_mode, scale=True)
        X_train_scaled = prep_scaled.fit_transform(df_train)
        X_val_scaled = prep_scaled.transform(df_val)
        X_test_scaled = prep_scaled.transform(df_test)
        
        # Unscaled preprocessor for RF and XGBoost
        prep_unscaled = TabularPreprocessor(feature_mode=feat_mode, scale=False)
        X_train_unscaled = prep_unscaled.fit_transform(df_train)
        X_val_unscaled = prep_unscaled.transform(df_val)
        X_test_unscaled = prep_unscaled.transform(df_test)
        
        for model_name in model_types:
            for weight_mode in weightings:
                is_weighted = (weight_mode == "Weighted")
                config_id = f"{model_name}_{weight_mode}_{feat_mode}"
                
                seed_records = []
                for seed in seeds:
                    logger.info(
                        f"[{run_idx}/{total_experiments}] Training {model_name} | {weight_mode} | "
                        f"{feat_label} | Seed: {seed}"
                    )
                    
                    # Instantiate model
                    if model_name == "Logistic Regression":
                        X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
                        m_cfg = b_cfg["models"]["logistic_regression"]
                        clf = LogisticRegressionBaseline(
                            weighted=is_weighted,
                            C=m_cfg.get("C", 1.0),
                            max_iter=m_cfg.get("max_iter", 1000),
                            random_state=seed
                        )
                    elif model_name == "Random Forest":
                        X_tr, X_v, X_te = X_train_unscaled, X_val_unscaled, X_test_unscaled
                        m_cfg = b_cfg["models"]["random_forest"]
                        clf = RandomForestBaseline(
                            weighted=is_weighted,
                            n_estimators=m_cfg.get("n_estimators", 100),
                            max_depth=m_cfg.get("max_depth", 15),
                            min_samples_split=m_cfg.get("min_samples_split", 5),
                            min_samples_leaf=m_cfg.get("min_samples_leaf", 2),
                            random_state=seed,
                            n_jobs=-1
                        )
                    elif model_name == "XGBoost":
                        X_tr, X_v, X_te = X_train_unscaled, X_val_unscaled, X_test_unscaled
                        m_cfg = b_cfg["models"]["xgboost"]
                        clf = XGBoostBaseline(
                            weighted=is_weighted,
                            n_estimators=m_cfg.get("n_estimators", 100),
                            max_depth=m_cfg.get("max_depth", 6),
                            learning_rate=m_cfg.get("learning_rate", 0.1),
                            subsample=m_cfg.get("subsample", 0.8),
                            colsample_bytree=m_cfg.get("colsample_bytree", 0.8),
                            random_state=seed,
                            n_jobs=-1
                        )
                    elif model_name == "MLP":
                        X_tr, X_v, X_te = X_train_scaled, X_val_scaled, X_test_scaled
                        m_cfg = b_cfg["models"]["mlp"]
                        clf = MLPBaseline(
                            weighted=is_weighted,
                            hidden_dims=m_cfg.get("hidden_dims", [128, 64]),
                            dropout=m_cfg.get("dropout", 0.2),
                            learning_rate=m_cfg.get("learning_rate", 0.001),
                            weight_decay=m_cfg.get("weight_decay", 1e-4),
                            batch_size=m_cfg.get("batch_size", 256),
                            epochs=m_cfg.get("epochs", 40),
                            random_state=seed
                        )
                        
                    # 1. Fit model strictly on train split
                    clf.fit(X_tr, y_train)
                    
                    # 2. Predict on Validation Split & Optimize Decision Threshold
                    val_probs = clf.predict_proba(X_v)
                    opt_tau, opt_val_f1, _ = find_optimal_threshold(
                        y_true_val=y_val,
                        y_prob_val=val_probs,
                        metric="f1"
                    )
                    val_metrics = compute_classification_metrics(y_val, val_probs, threshold=opt_tau)
                    
                    # 3. Evaluate on Test Split with LOCKED Threshold
                    test_probs = clf.predict_proba(X_te)
                    test_metrics = compute_classification_metrics(y_test, test_probs, threshold=opt_tau)
                    
                    record = {
                        "run_id": run_idx,
                        "model": model_name,
                        "weighting": weight_mode,
                        "feature_mode": feat_mode,
                        "feature_space": feat_label,
                        "num_features": X_tr.shape[1],
                        "seed": seed,
                        "optimal_threshold": opt_tau,
                        # Validation Metrics
                        "val_illicit_f1": val_metrics["illicit_f1"],
                        "val_pr_auc": val_metrics["pr_auc"],
                        "val_illicit_precision": val_metrics["illicit_precision"],
                        "val_illicit_recall": val_metrics["illicit_recall"],
                        "val_roc_auc": val_metrics["roc_auc"],
                        # Test Metrics (Locked Threshold)
                        "test_illicit_f1": test_metrics["illicit_f1"],
                        "test_pr_auc": test_metrics["pr_auc"],
                        "test_illicit_precision": test_metrics["illicit_precision"],
                        "test_illicit_recall": test_metrics["illicit_recall"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_roc_auc": test_metrics["roc_auc"],
                        "test_mcc": test_metrics["mcc"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_tp": test_metrics["tp"],
                        "test_fp": test_metrics["fp"],
                        "test_tn": test_metrics["tn"],
                        "test_fn": test_metrics["fn"],
                    }
                    all_run_records.append(record)
                    seed_records.append(record)
                    
                    # Store seed 42 curve data for visualization
                    if seed == 42:
                        representative_curve_data.append({
                            "name": f"{model_name} ({weight_mode}) [{feat_mode}]",
                            "model": model_name,
                            "threshold": opt_tau,
                            "pr_auc": test_metrics["pr_auc"],
                            "roc_auc": test_metrics["roc_auc"],
                            "confusion_matrix": test_metrics["confusion_matrix"],
                            "pr_curve": test_metrics["pr_curve"],
                            "roc_curve": test_metrics["roc_curve"],
                        })
                        
                    run_idx += 1

    # 4. Compile DataFrames
    df_all_runs = pd.DataFrame(all_run_records)
    all_runs_csv = tables_dir / "baseline_all_runs.csv"
    df_all_runs.to_csv(all_runs_csv, index=False)
    logger.info(f"All {len(df_all_runs)} runs saved to: {all_runs_csv}")
    
    # 5. Compute Mean ± Std across 5 seeds for all 16 configurations
    group_cols = ["model", "weighting", "feature_mode", "feature_space", "num_features"]
    numeric_metric_cols = [
        "optimal_threshold",
        "val_illicit_f1", "val_pr_auc",
        "test_illicit_f1", "test_pr_auc",
        "test_illicit_precision", "test_illicit_recall",
        "test_macro_f1", "test_roc_auc", "test_mcc", "test_accuracy"
    ]
    
    summary_rows = []
    for keys, group in df_all_runs.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        for col in numeric_metric_cols:
            mean_val = float(group[col].mean())
            std_val = float(group[col].std())
            rec[f"{col}_mean"] = mean_val
            rec[f"{col}_std"] = std_val
            rec[f"{col}_formatted"] = f"{mean_val:.4f} ± {std_val:.4f}"
        summary_rows.append(rec)
        
    df_summary = pd.DataFrame(summary_rows)
    # Sort by test_illicit_f1_mean descending
    df_summary = df_summary.sort_values(by="test_illicit_f1_mean", ascending=False).reset_index(drop=True)
    summary_csv = tables_dir / "baseline_summary_mean_std.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Baseline Mean ± Std summary saved to: {summary_csv}")
    
    # 6. Identify Best Configuration by Validation F1
    best_configs = []
    # Best per model
    for m in model_types:
        sub = df_summary[df_summary["model"] == m]
        best_m = sub.loc[sub["val_illicit_f1_mean"].idxmax()].to_dict()
        best_configs.append({
            "Selection Category": f"Best {m} (Val F1)",
            "Model": best_m["model"],
            "Weighting": best_m["weighting"],
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
        
    # Overall best by Validation F1
    best_overall = df_summary.loc[df_summary["val_illicit_f1_mean"].idxmax()].to_dict()
    best_configs.insert(0, {
        "Selection Category": "★ Overall Best Baseline (Val F1)",
        "Model": best_overall["model"],
        "Weighting": best_overall["weighting"],
        "Feature Space": best_overall["feature_space"],
        "Validation Illicit F1": best_overall["val_illicit_f1_formatted"],
        "Validation PR-AUC": best_overall["val_pr_auc_formatted"],
        "Test Illicit F1": best_overall["test_illicit_f1_formatted"],
        "Test PR-AUC": best_overall["test_pr_auc_formatted"],
        "Test Precision": best_overall["test_illicit_precision_formatted"],
        "Test Recall": best_overall["test_illicit_recall_formatted"],
        "Test ROC-AUC": best_overall["test_roc_auc_formatted"],
        "Test MCC": best_overall["test_mcc_formatted"],
    })
    
    df_best = pd.DataFrame(best_configs)
    best_csv = tables_dir / "baseline_best_validation_config.csv"
    df_best.to_csv(best_csv, index=False)
    logger.info(f"Best validation configurations saved to: {best_csv}")
    
    # 7. Generate Visualizations
    logger.info("Generating Phase 2 publication figures...")
    generate_baseline_plots(df_summary, representative_curve_data, figures_dir, dpi=300)
    
    return df_all_runs, df_summary, df_best
