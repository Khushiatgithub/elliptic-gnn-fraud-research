"""
Comprehensive Error and Failure-Mode Analysis Module for Graph Neural Networks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.logger import get_logger

logger = get_logger("error_analysis")


def perform_gnn_error_analysis(
    data: Data,
    best_result: Dict,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs in-depth error taxonomy for the top-performing GNN model.
    
    Args:
        data: PyG Data object.
        best_result: Result dictionary from train_and_evaluate_gnn containing
                     test_probabilities, test_predictions, test_labels, test_node_indices.
        output_dir: Optional path to save tables.
        
    Returns:
        Tuple of (df_error_summary, df_timestep_error_distribution).
    """
    test_indices = best_result["test_node_indices"]
    y_true = best_result["test_labels"]
    y_pred = best_result["test_predictions"]
    y_prob = best_result["test_probabilities"]
    
    time_steps = data.time_step[test_indices].cpu().numpy()
    tx_ids = data.tx_id[test_indices].cpu().numpy()
    
    # Compute in-degree and out-degree for all nodes
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    out_degree = np.bincount(edge_index[0], minlength=num_nodes)
    in_degree = np.bincount(edge_index[1], minlength=num_nodes)
    total_degree = out_degree + in_degree
    
    test_in_deg = in_degree[test_indices]
    test_out_deg = out_degree[test_indices]
    test_tot_deg = total_degree[test_indices]
    
    # Classify error types
    # 0 = Licit, 1 = Illicit
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    high_conf_fp = fp_mask & (y_prob >= 0.80)
    high_conf_fn = fn_mask & (y_prob <= 0.20)
    
    # 1. Error Summary Table
    summary_rows = [
        {"Category": "Total Test Nodes", "Count": len(y_true), "Percentage": "100.00%", "Mean In-Degree": f"{test_in_deg.mean():.2f}", "Mean Out-Degree": f"{test_out_deg.mean():.2f}"},
        {"Category": "True Positives (TP)", "Count": int(np.sum(tp_mask)), "Percentage": f"{np.sum(tp_mask) / len(y_true) * 100:.2f}%", "Mean In-Degree": f"{test_in_deg[tp_mask].mean() if np.sum(tp_mask) > 0 else 0:.2f}", "Mean Out-Degree": f"{test_out_deg[tp_mask].mean() if np.sum(tp_mask) > 0 else 0:.2f}"},
        {"Category": "True Negatives (TN)", "Count": int(np.sum(tn_mask)), "Percentage": f"{np.sum(tn_mask) / len(y_true) * 100:.2f}%", "Mean In-Degree": f"{test_in_deg[tn_mask].mean():.2f}", "Mean Out-Degree": f"{test_out_deg[tn_mask].mean():.2f}"},
        {"Category": "False Positives (FP)", "Count": int(np.sum(fp_mask)), "Percentage": f"{np.sum(fp_mask) / len(y_true) * 100:.2f}%", "Mean In-Degree": f"{test_in_deg[fp_mask].mean() if np.sum(fp_mask) > 0 else 0:.2f}", "Mean Out-Degree": f"{test_out_deg[fp_mask].mean() if np.sum(fp_mask) > 0 else 0:.2f}"},
        {"Category": "False Negatives (FN)", "Count": int(np.sum(fn_mask)), "Percentage": f"{np.sum(fn_mask) / len(y_true) * 100:.2f}%", "Mean In-Degree": f"{test_in_deg[fn_mask].mean() if np.sum(fn_mask) > 0 else 0:.2f}", "Mean Out-Degree": f"{test_out_deg[fn_mask].mean() if np.sum(fn_mask) > 0 else 0:.2f}"},
        {"Category": "High-Confidence False Positives (p >= 0.80)", "Count": int(np.sum(high_conf_fp)), "Percentage": f"{np.sum(high_conf_fp) / len(y_true) * 100:.2f}%", "Mean In-Degree": f"{test_in_deg[high_conf_fp].mean() if np.sum(high_conf_fp) > 0 else 0:.2f}", "Mean Out-Degree": f"{test_out_deg[high_conf_fp].mean() if np.sum(high_conf_fp) > 0 else 0:.2f}"},
        {"Category": "High-Confidence False Negatives (p <= 0.20)", "Count": int(np.sum(high_conf_fn)), "Percentage": f"{np.sum(high_conf_fn) / len(y_true) * 100:.2f}%", "Mean In-Degree": f"{test_in_deg[high_conf_fn].mean() if np.sum(high_conf_fn) > 0 else 0:.2f}", "Mean Out-Degree": f"{test_out_deg[high_conf_fn].mean() if np.sum(high_conf_fn) > 0 else 0:.2f}"},
    ]
    df_error_summary = pd.DataFrame(summary_rows)
    
    # 2. Timestep Breakdown
    ts_rows = []
    for ts in range(40, 50):
        ts_sel = time_steps == ts
        if not np.any(ts_sel):
            continue
        ts_y_true = y_true[ts_sel]
        ts_y_pred = y_pred[ts_sel]
        ts_fp = np.sum((ts_y_true == 0) & (ts_y_pred == 1))
        ts_fn = np.sum((ts_y_true == 1) & (ts_y_pred == 0))
        ts_tp = np.sum((ts_y_true == 1) & (ts_y_pred == 1))
        ts_tn = np.sum((ts_y_true == 0) & (ts_y_pred == 0))
        ts_illicit = np.sum(ts_y_true == 1)
        ts_recall = float(ts_tp) / ts_illicit if ts_illicit > 0 else 0.0
        ts_prec = float(ts_tp) / (ts_tp + ts_fp) if (ts_tp + ts_fp) > 0 else 0.0
        ts_f1 = 2 * ts_prec * ts_recall / (ts_prec + ts_recall) if (ts_prec + ts_recall) > 0 else 0.0
        
        ts_rows.append({
            "Time Step": ts,
            "Total Samples": int(np.sum(ts_sel)),
            "Illicit Count": int(ts_illicit),
            "True Positives": int(ts_tp),
            "False Positives": int(ts_fp),
            "False Negatives": int(ts_fn),
            "Precision": f"{ts_prec:.4f}",
            "Recall": f"{ts_recall:.4f}",
            "Illicit F1": f"{ts_f1:.4f}"
        })
    df_ts_errors = pd.DataFrame(ts_rows)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_error_summary.to_csv(output_dir / "gnn_error_analysis.csv", index=False)
        df_error_summary.to_csv(output_dir / "error_analysis.csv", index=False)
        df_ts_errors.to_csv(output_dir / "error_analysis_timesteps.csv", index=False)
        logger.info(f"Error analysis tables exported to: {output_dir / 'gnn_error_analysis.csv'}")
        
    return df_error_summary, df_ts_errors
