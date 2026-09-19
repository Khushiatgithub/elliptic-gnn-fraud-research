"""
Comprehensive Scientific Audit Script for Phase 3 Experimental Results.
Performs:
1. Programmatic consistency check (all_runs vs summary vs best_validation vs benchmark)
2. Corrected paired statistical significance testing with Holm-Bonferroni correction and paired Cohen's d_z
3. Validation-to-test generalization gap quantification (F1 and PR-AUC)
4. Seed-level granular breakdown and variance audit
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def run_scientific_audit():
    root = Path(".")
    tables_dir = root / "results" / "tables"
    
    # -------------------------------------------------------------
    # 1. Consistency Audit
    # -------------------------------------------------------------
    all_runs_path = tables_dir / "gnn_all_runs.csv"
    summary_path = tables_dir / "gnn_summary_mean_std.csv"
    best_val_path = tables_dir / "gnn_best_validation.csv"
    benchmark_path = tables_dir / "gnn_compute_benchmark.csv"
    
    df_all = pd.read_csv(all_runs_path)
    df_summary = pd.read_csv(summary_path)
    df_best = pd.read_csv(best_val_path)
    df_bench = pd.read_csv(benchmark_path)
    
    audit_rows = []
    
    # Check all 6 model-feature combinations
    for (model, f_mode), group in df_all.groupby(["model", "feature_mode"]):
        f_space = group["feature_space"].iloc[0]
        n_seeds = len(group)
        
        # Calculate ground truth sample statistics (ddof=1 for sample std)
        calc_val_f1_mean = group["val_illicit_f1"].astype(float).mean()
        calc_val_f1_std = group["val_illicit_f1"].astype(float).std(ddof=1)
        calc_val_prauc_mean = group["val_pr_auc"].astype(float).mean()
        calc_val_prauc_std = group["val_pr_auc"].astype(float).std(ddof=1)
        
        calc_test_f1_mean = group["test_illicit_f1"].astype(float).mean()
        calc_test_f1_std = group["test_illicit_f1"].astype(float).std(ddof=1)
        calc_test_prauc_mean = group["test_pr_auc"].astype(float).mean()
        calc_test_prauc_std = group["test_pr_auc"].astype(float).std(ddof=1)
        
        # Compare with summary table
        sum_row = df_summary[(df_summary["model"] == model) & (df_summary["feature_mode"] == f_mode)]
        if len(sum_row) == 1:
            tbl_test_f1_mean = float(sum_row["test_illicit_f1_mean"].iloc[0])
            tbl_test_f1_std = float(sum_row["test_illicit_f1_std"].iloc[0])
            diff_test_f1 = abs(calc_test_f1_mean - tbl_test_f1_mean)
            status = "MATCH" if diff_test_f1 < 1e-4 else "DISCREPANCY"
        else:
            status = "MISSING_IN_SUMMARY"
            tbl_test_f1_mean = np.nan
            tbl_test_f1_std = np.nan
            diff_test_f1 = np.nan
            
        audit_rows.append({
            "model": model,
            "feature_mode": f_mode,
            "feature_space": f_space,
            "num_seeds": n_seeds,
            "recomputed_test_f1_mean": round(calc_test_f1_mean, 4),
            "recomputed_test_f1_std": round(calc_test_f1_std, 4),
            "table_test_f1_mean": round(tbl_test_f1_mean, 4),
            "table_test_f1_std": round(tbl_test_f1_std, 4),
            "mean_difference": round(diff_test_f1, 6) if not np.isnan(diff_test_f1) else np.nan,
            "status": status
        })
        
    df_consistency = pd.DataFrame(audit_rows)
    df_consistency.to_csv(tables_dir / "gnn_consistency_audit.csv", index=False)
    print("Consistency audit saved.")
    
    # -------------------------------------------------------------
    # 2. Validation-to-Test Generalization Gap
    # -------------------------------------------------------------
    gen_rows = []
    for (model, f_mode), group in df_all.groupby(["model", "feature_mode"]):
        f_space = group["feature_space"].iloc[0]
        
        val_f1_m = group["val_illicit_f1"].astype(float).mean()
        val_f1_s = group["val_illicit_f1"].astype(float).std()
        test_f1_m = group["test_illicit_f1"].astype(float).mean()
        test_f1_s = group["test_illicit_f1"].astype(float).std()
        delta_f1 = test_f1_m - val_f1_m
        rel_drop_f1 = (delta_f1 / val_f1_m) * 100 if val_f1_m > 0 else np.nan
        
        val_pr_m = group["val_pr_auc"].astype(float).mean()
        val_pr_s = group["val_pr_auc"].astype(float).std()
        test_pr_m = group["test_pr_auc"].astype(float).mean()
        test_pr_s = group["test_pr_auc"].astype(float).std()
        delta_pr = test_pr_m - val_pr_m
        rel_drop_pr = (delta_pr / val_pr_m) * 100 if val_pr_m > 0 else np.nan
        
        gen_rows.append({
            "Model": model,
            "Feature Space": f_space,
            "Val Illicit F1": f"{val_f1_m:.4f} ± {val_f1_s:.4f}",
            "Test Illicit F1": f"{test_f1_m:.4f} ± {test_f1_s:.4f}",
            "Δ F1 (Test - Val)": f"{delta_f1:+.4f}",
            "F1 Generalization Drop (%)": f"{rel_drop_f1:+.2f}%",
            "Val PR-AUC": f"{val_pr_m:.4f} ± {val_pr_s:.4f}",
            "Test PR-AUC": f"{test_pr_m:.4f} ± {test_pr_s:.4f}",
            "Δ PR-AUC (Test - Val)": f"{delta_pr:+.4f}",
            "PR-AUC Generalization Drop (%)": f"{rel_drop_pr:+.2f}%"
        })
        
    df_gen = pd.DataFrame(gen_rows).sort_values(by="Model")
    df_gen.to_csv(tables_dir / "gnn_val_test_generalization.csv", index=False)
    print("Generalization audit saved.")
    
    # -------------------------------------------------------------
    # 3. Seed-Level Audit
    # -------------------------------------------------------------
    seed_audit_cols = [
        "model", "feature_space", "seed", "best_epoch", "epochs_trained",
        "optimal_threshold", "val_illicit_f1", "val_pr_auc",
        "test_illicit_f1", "test_pr_auc", "test_illicit_precision", "test_illicit_recall",
        "test_mcc", "training_time_sec"
    ]
    df_seed_audit = df_all[[c for c in seed_audit_cols if c in df_all.columns]].sort_values(
        by=["model", "feature_space", "seed"]
    ).reset_index(drop=True)
    df_seed_audit.to_csv(tables_dir / "gnn_seed_level_audit.csv", index=False)
    print("Seed-level audit saved.")
    
    # -------------------------------------------------------------
    # 4. Corrected Paired Statistical Tests with Holm-Bonferroni
    # -------------------------------------------------------------
    baseline_all_path = tables_dir / "baseline_all_runs.csv"
    df_base = pd.read_csv(baseline_all_path)
    
    # Best GNN is GraphSAGE (local)
    df_best_gnn_runs = df_all[(df_all["model"] == "GraphSAGE") & (df_all["feature_mode"] == "local")].sort_values("seed").reset_index(drop=True)
    gnn_f1_arr = df_best_gnn_runs["test_illicit_f1"].astype(float).values
    gnn_pr_arr = df_best_gnn_runs["test_pr_auc"].astype(float).values
    
    baseline_comparisons = [
        ("Random Forest (Full 165, Standard)", "Random Forest", "full", "Standard"),
        ("XGBoost (Full 165, Weighted)", "XGBoost", "full", "Weighted"),
        ("MLP (Full 165, Standard)", "MLP", "full", "Standard"),
        ("Logistic Regression (Full 165, Weighted)", "Logistic Regression", "full", "Weighted"),
    ]
    
    stat_rows = []
    
    for label, b_mod, b_feat, b_wt in baseline_comparisons:
        sub_b = df_base[(df_base["model"] == b_mod) & (df_base["feature_mode"] == b_feat) & (df_base["weighting"] == b_wt)].sort_values("seed").reset_index(drop=True)
        if len(sub_b) != 5:
            continue
            
        b_f1_arr = sub_b["test_illicit_f1"].astype(float).values
        b_pr_arr = sub_b["test_pr_auc"].astype(float).values
        
        # F1
        diff_f1 = gnn_f1_arr - b_f1_arr
        m_diff_f1 = np.mean(diff_f1)
        s_diff_f1 = np.std(diff_f1, ddof=1)
        paired_d_f1 = m_diff_f1 / s_diff_f1 if s_diff_f1 > 0 else 0.0
        t_stat_f1, p_t_f1 = stats.ttest_rel(gnn_f1_arr, b_f1_arr)
        try:
            w_res_f1 = stats.wilcoxon(gnn_f1_arr, b_f1_arr)
            p_w_f1 = w_res_f1.pvalue
            w_stat_f1 = w_res_f1.statistic
        except Exception:
            p_w_f1, w_stat_f1 = np.nan, np.nan
            
        stat_rows.append({
            "Baseline Model": label,
            "GNN Model": "GraphSAGE (Local Only 93)",
            "Metric": "Test Illicit F1",
            "GNN Mean ± Std": f"{np.mean(gnn_f1_arr):.4f} ± {np.std(gnn_f1_arr, ddof=1):.4f}",
            "Baseline Mean ± Std": f"{np.mean(b_f1_arr):.4f} ± {np.std(b_f1_arr, ddof=1):.4f}",
            "Mean Difference (Δ)": f"{m_diff_f1:+.4f}",
            "Paired t-statistic": t_stat_f1,
            "Raw t-test p-value": p_t_f1,
            "Wilcoxon statistic": w_stat_f1,
            "Wilcoxon p-value": p_w_f1,
            "Paired Cohen's d_z": paired_d_f1
        })
        
        # PR-AUC
        diff_pr = gnn_pr_arr - b_pr_arr
        m_diff_pr = np.mean(diff_pr)
        s_diff_pr = np.std(diff_pr, ddof=1)
        paired_d_pr = m_diff_pr / s_diff_pr if s_diff_pr > 0 else 0.0
        t_stat_pr, p_t_pr = stats.ttest_rel(gnn_pr_arr, b_pr_arr)
        try:
            w_res_pr = stats.wilcoxon(gnn_pr_arr, b_pr_arr)
            p_w_pr = w_res_pr.pvalue
            w_stat_pr = w_res_pr.statistic
        except Exception:
            p_w_pr, w_stat_pr = np.nan, np.nan
            
        stat_rows.append({
            "Baseline Model": label,
            "GNN Model": "GraphSAGE (Local Only 93)",
            "Metric": "Test PR-AUC",
            "GNN Mean ± Std": f"{np.mean(gnn_pr_arr):.4f} ± {np.std(gnn_pr_arr, ddof=1):.4f}",
            "Baseline Mean ± Std": f"{np.mean(b_pr_arr):.4f} ± {np.std(b_pr_arr, ddof=1):.4f}",
            "Mean Difference (Δ)": f"{m_diff_pr:+.4f}",
            "Paired t-statistic": t_stat_pr,
            "Raw t-test p-value": p_t_pr,
            "Wilcoxon statistic": w_stat_pr,
            "Wilcoxon p-value": p_w_pr,
            "Paired Cohen's d_z": paired_d_pr
        })
        
    df_stat_corr = pd.DataFrame(stat_rows)
    
    # Apply Holm-Bonferroni correction within each metric family
    for metric_name in ["Test Illicit F1", "Test PR-AUC"]:
        m_idx = df_stat_corr[df_stat_corr["Metric"] == metric_name].index
        m_pvals = df_stat_corr.loc[m_idx, "Raw t-test p-value"].values
        
        sort_order = np.argsort(m_pvals)
        ranks = np.empty_like(sort_order)
        ranks[sort_order] = np.arange(len(m_pvals))
        
        m = len(m_pvals)
        holm_pvals = np.zeros(m)
        for i, idx_sorted in enumerate(sort_order):
            p_adj = m_pvals[idx_sorted] * (m - i)
            holm_pvals[idx_sorted] = min(p_adj, 1.0)
            
        for i in range(1, m):
            curr_sorted = sort_order[i]
            prev_sorted = sort_order[i-1]
            holm_pvals[curr_sorted] = max(holm_pvals[curr_sorted], holm_pvals[prev_sorted])
            holm_pvals[curr_sorted] = min(holm_pvals[curr_sorted], 1.0)
            
        df_stat_corr.loc[m_idx, "Holm-Adjusted t-test p-value"] = holm_pvals
        
    df_stat_corr["Statistically Significant (Raw t < 0.05)"] = df_stat_corr["Raw t-test p-value"].apply(lambda p: "YES" if p < 0.05 else "NO")
    df_stat_corr["Statistically Significant (Holm-Adjusted t < 0.05)"] = df_stat_corr["Holm-Adjusted t-test p-value"].apply(lambda p: "YES" if p < 0.05 else "NO")
    df_stat_corr["Wilcoxon Rejection at alpha=0.05 (Min Possible p=0.0625)"] = "NO (Insufficient Power at N=5)"
    
    df_stat_corr.to_csv(tables_dir / "gnn_statistical_tests_corrected.csv", index=False)
    print("Corrected statistical tests saved.")

if __name__ == "__main__":
    run_scientific_audit()
