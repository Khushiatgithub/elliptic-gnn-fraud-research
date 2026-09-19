"""
Master Execution Pipeline for Phase 3: Graph Neural Network Experiments.
Optimized, Leakage-Safe, Reproducible, and Resumable.
"""

from pathlib import Path
import time
import pandas as pd

from src.analysis.error_analysis import perform_gnn_error_analysis
from src.analysis.homophily import compute_graph_homophily
from src.analysis.statistical_significance import compute_paired_significance_tests
from src.data.graph_builder import build_pyg_graph
from src.data.loader import load_classes, load_edgelist, load_features
from src.evaluation.compare_gnn_baselines import generate_phase3_report
from src.evaluation.gnn_plots import generate_all_gnn_figures
from src.experiments.run_ablation import run_all_ablations
from src.experiments.run_gnn import run_primary_gnn_experiments
from src.utils.config import get_project_root
from src.utils.logger import get_logger


def main():
    start_time = time.time()
    root = get_project_root()
    log_dir = root / "results" / "logs"
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    
    for d in [log_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    logger = get_logger("phase3_master", log_file=log_dir / "phase3_gnn.log")
    logger.info("=" * 80)
    logger.info("STARTING OPTIMIZED PHASE 3: GRAPH NEURAL NETWORK EXPERIMENTS")
    logger.info(f"Project Root: {root}")
    logger.info("=" * 80)
    
    # 1. Primary GNN Experiments (30 Runs with Safe Resumption)
    logger.info("--- STEP 1: Running Primary GNN Experiments (30 Runs Matrix) ---")
    df_all_runs, df_gnn_summary, df_gnn_best, rep_curve_data, best_overall_res = run_primary_gnn_experiments()
    
    # 2. Ablation Studies (Feature, Unknown Nodes, Direction, Depth)
    logger.info("--- STEP 2: Running Controlled Ablation Studies ---")
    df_feat_abl, df_unk_abl, df_dir_abl, df_depth_abl = run_all_ablations(df_all_runs)
    
    # 3. Homophily & Relational Neighborhood Analysis
    logger.info("--- STEP 3: Running Homophily & Relational Structure Analysis ---")
    df_classes = load_classes(root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv")
    df_features = load_features(root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv")
    df_edgelist = load_edgelist(root / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_edgelist.csv")
    
    pyg_primary = build_pyg_graph(
        df_features=df_features,
        df_classes=df_classes,
        df_edgelist=df_edgelist,
        feature_mode="full",
        direction="forward",
        remove_unknown=False,
        scale=True
    )
    homo_dict, df_homophily = compute_graph_homophily(pyg_primary, output_dir=tables_dir)
    
    # 4. Error Analysis on Best GNN Model
    logger.info("--- STEP 4: Running Error Taxonomy & Failure-Mode Analysis ---")
    df_err_summary, df_ts_err = perform_gnn_error_analysis(pyg_primary, best_overall_res, output_dir=tables_dir)
    
    # 5. Statistical Significance Tests against Phase 2 Baselines
    logger.info("--- STEP 5: Running Paired Statistical Significance Testing ---")
    best_model_name = df_gnn_best.loc[0, "Model"]
    best_feat_space = df_gnn_best.loc[0, "Feature Space"]
    best_gnn_runs = df_all_runs[
        (df_all_runs["model"] == best_model_name) &
        (df_all_runs["feature_space"] == best_feat_space)
    ].to_dict("records")
    
    baseline_runs_csv = tables_dir / "baseline_all_runs.csv"
    df_sig = compute_paired_significance_tests(
        gnn_runs=best_gnn_runs,
        baseline_all_runs_csv=baseline_runs_csv,
        output_dir=tables_dir
    )
    
    # 6. Generate 12 Publication-Grade Figures (300 DPI)
    logger.info("--- STEP 6: Generating 12 Publication-Grade Visualizations (300 DPI) ---")
    baseline_summary_csv = tables_dir / "baseline_summary_mean_std.csv"
    df_base_summary = pd.read_csv(baseline_summary_csv) if baseline_summary_csv.exists() else pd.DataFrame()
    benchmark_csv = tables_dir / "gnn_compute_benchmark.csv"
    df_bench = pd.read_csv(benchmark_csv) if benchmark_csv.exists() else None
    
    generate_all_gnn_figures(
        df_gnn_summary=df_gnn_summary,
        df_baseline_summary=df_base_summary,
        df_unknown_ablation=df_unk_abl,
        df_direction_ablation=df_dir_abl,
        df_depth_ablation=df_depth_abl,
        df_benchmark=df_bench,
        df_timestep_errors=df_ts_err,
        representative_curve_data=rep_curve_data,
        best_overall_result=best_overall_res,
        figures_dir=figures_dir,
        dpi=300
    )
    
    # 7. Compile Final Phase 3 Markdown Report (19 Sections)
    logger.info("--- STEP 7: Compiling Final Phase 3 Markdown Report ---")
    baseline_best_csv = tables_dir / "baseline_best_validation_config.csv"
    df_base_best = pd.read_csv(baseline_best_csv) if baseline_best_csv.exists() else pd.DataFrame()
    
    report_paths = [
        log_dir / "phase3_gnn_experiments.md",
        root / "results" / "phase3_gnn_experiments.md"
    ]
    for p in report_paths:
        generate_phase3_report(
            df_gnn_summary=df_gnn_summary,
            df_gnn_best=df_gnn_best,
            df_baseline_best=df_base_best,
            df_feature_ablation=df_feat_abl,
            df_unknown_ablation=df_unk_abl,
            df_direction_ablation=df_dir_abl,
            df_depth_ablation=df_depth_abl,
            df_homophily=df_homophily,
            df_error_summary=df_err_summary,
            df_sig_tests=df_sig,
            df_benchmark=df_bench,
            report_output_path=p
        )
        
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"PHASE 3 GNN EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ({elapsed/60:.2f} MIN).")
    logger.info(f"Report: {log_dir / 'phase3_gnn_experiments.md'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
