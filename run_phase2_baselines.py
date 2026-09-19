"""
Master CLI Runner for Phase 2: Conventional Machine Learning Baselines.
"""

from pathlib import Path
import time
import pandas as pd

from src.evaluation.compare_models import generate_phase2_report
from src.training.train_baselines import run_baseline_experiments
from src.utils.config import get_project_root
from src.utils.logger import get_logger


def main():
    start_time = time.time()
    root = get_project_root()
    log_dir = root / "results" / "logs"
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    
    logger = get_logger("phase2_master", log_file=log_dir / "phase2_baselines.log")
    logger.info("=" * 80)
    logger.info("STARTING PHASE 2: CONVENTIONAL MACHINE LEARNING BASELINES")
    logger.info(f"Project Root: {root}")
    logger.info("=" * 80)

    # 1. Run Baseline Experiments Matrix
    df_all_runs, df_summary, df_best = run_baseline_experiments(reuse_existing=True)
    
    # 2. Read Temporal Splits Table
    splits_csv = tables_dir / "temporal_split_distribution.csv"
    df_splits = pd.read_csv(splits_csv)

    # 3. Generate Markdown Report
    report_paths = [
        log_dir / "phase2_baseline_experiments.md",
        root / "results" / "phase2_baseline_experiments.md"
    ]
    for p in report_paths:
        generate_phase2_report(
            df_splits=df_splits,
            df_summary=df_summary,
            df_best=df_best,
            report_output_path=p
        )

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"PHASE 2 EXPERIMENTS COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS.")
    logger.info(f"All 80 baseline runs recorded in: {tables_dir / 'baseline_all_runs.csv'}")
    logger.info(f"Summary table saved to: {tables_dir / 'baseline_summary_mean_std.csv'}")
    logger.info(f"Best configurations saved to: {tables_dir / 'baseline_best_validation_config.csv'}")
    logger.info(f"Reports saved to: {log_dir / 'phase2_baseline_experiments.md'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
