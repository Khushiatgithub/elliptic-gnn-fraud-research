"""
Paired Statistical Significance Testing Suite (GNN vs Phase 2 Baselines).
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger("statistical_significance")


def compute_paired_significance_tests(
    gnn_runs: List[Dict],
    baseline_all_runs_csv: Path,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Computes paired t-tests, Wilcoxon signed-rank tests, and Cohen's d effect sizes
    comparing the best GNN against Phase 2 baseline models across identical 5 random seeds.
    
    Args:
        gnn_runs: List of dictionary run records for the best GNN across 5 seeds.
        baseline_all_runs_csv: Path to 'results/tables/baseline_all_runs.csv'.
        output_dir: Optional path to save 'gnn_statistical_tests.csv'.
        
    Returns:
        DataFrame containing statistical test metrics.
    """
    baseline_all_runs_csv = Path(baseline_all_runs_csv)
    df_base = pd.read_csv(baseline_all_runs_csv)
    
    df_gnn = pd.DataFrame(gnn_runs).sort_values(by="seed").reset_index(drop=True)
    gnn_seeds = df_gnn["seed"].values
    gnn_f1 = df_gnn["test_illicit_f1"].astype(float).values
    gnn_pr_auc = df_gnn["test_pr_auc"].astype(float).values
    gnn_name = df_gnn["model"].iloc[0] + " (" + df_gnn["feature_space"].iloc[0] + ")"
    
    comparisons = [
        ("Random Forest (Full 165, Standard)", "Random Forest", "full", "Standard"),
        ("XGBoost (Full 165, Weighted)", "XGBoost", "full", "Weighted"),
        ("MLP (Full 165, Standard)", "MLP", "full", "Standard"),
        ("Logistic Regression (Full 165, Weighted)", "Logistic Regression", "full", "Weighted"),
    ]
    
    results = []
    
    for label, b_model, b_feat, b_weight in comparisons:
        sub_base = df_base[
            (df_base["model"] == b_model) &
            (df_base["feature_mode"] == b_feat) &
            (df_base["weighting"] == b_weight)
        ].sort_values(by="seed").reset_index(drop=True)
        
        if len(sub_base) != len(df_gnn):
            logger.warning(f"Seed count mismatch for {label}: base={len(sub_base)}, gnn={len(df_gnn)}")
            continue
            
        base_f1 = sub_base["test_illicit_f1"].astype(float).values
        base_pr_auc = sub_base["test_pr_auc"].astype(float).values
        
        # Test Illicit F1 Comparison
        diff_f1 = gnn_f1 - base_f1
        mean_diff_f1 = float(np.mean(diff_f1))
        std_diff_f1 = float(np.std(diff_f1, ddof=1)) if len(diff_f1) > 1 else 0.0
        cohen_d_f1 = mean_diff_f1 / std_diff_f1 if std_diff_f1 > 0 else 0.0
        
        t_stat_f1, p_val_t_f1 = stats.ttest_rel(gnn_f1, base_f1)
        try:
            w_stat_f1, p_val_w_f1 = stats.wilcoxon(gnn_f1, base_f1)
        except Exception:
            w_stat_f1, p_val_w_f1 = np.nan, np.nan
            
        # Test PR-AUC Comparison
        diff_pr = gnn_pr_auc - base_pr_auc
        mean_diff_pr = float(np.mean(diff_pr))
        std_diff_pr = float(np.std(diff_pr, ddof=1)) if len(diff_pr) > 1 else 0.0
        cohen_d_pr = mean_diff_pr / std_diff_pr if std_diff_pr > 0 else 0.0
        
        t_stat_pr, p_val_t_pr = stats.ttest_rel(gnn_pr_auc, base_pr_auc)
        try:
            w_stat_pr, p_val_w_pr = stats.wilcoxon(gnn_pr_auc, base_pr_auc)
        except Exception:
            w_stat_pr, p_val_w_pr = np.nan, np.nan
            
        results.append({
            "Baseline Model": label,
            "GNN Model": gnn_name,
            "Metric": "Test Illicit F1",
            "GNN Mean ± Std": f"{np.mean(gnn_f1):.4f} ± {np.std(gnn_f1):.4f}",
            "Baseline Mean ± Std": f"{np.mean(base_f1):.4f} ± {np.std(base_f1):.4f}",
            "Mean Difference (Δ)": f"{mean_diff_f1:+.4f}",
            "Paired t-stat": f"{t_stat_f1:.4f}",
            "t-test p-value": f"{p_val_t_f1:.4e}",
            "Wilcoxon p-value": f"{p_val_w_f1:.4e}" if not np.isnan(p_val_w_f1) else "N/A",
            "Cohen's d": f"{cohen_d_f1:.2f}",
            "Statistically Significant (p < 0.05)": "YES" if p_val_t_f1 < 0.05 else "NO"
        })
        
        results.append({
            "Baseline Model": label,
            "GNN Model": gnn_name,
            "Metric": "Test PR-AUC",
            "GNN Mean ± Std": f"{np.mean(gnn_pr_auc):.4f} ± {np.std(gnn_pr_auc):.4f}",
            "Baseline Mean ± Std": f"{np.mean(base_pr_auc):.4f} ± {np.std(base_pr_auc):.4f}",
            "Mean Difference (Δ)": f"{mean_diff_pr:+.4f}",
            "Paired t-stat": f"{t_stat_pr:.4f}",
            "t-test p-value": f"{p_val_t_pr:.4e}",
            "Wilcoxon p-value": f"{p_val_w_pr:.4e}" if not np.isnan(p_val_w_pr) else "N/A",
            "Cohen's d": f"{cohen_d_pr:.2f}",
            "Statistically Significant (p < 0.05)": "YES" if p_val_t_pr < 0.05 else "NO"
        })
        
    df_results = pd.DataFrame(results)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_dir / "gnn_statistical_tests.csv", index=False)
        df_results.to_csv(output_dir / "statistical_significance.csv", index=False)
        logger.info(f"Statistical significance table exported to: {output_dir / 'gnn_statistical_tests.csv'}")
        
    return df_results
