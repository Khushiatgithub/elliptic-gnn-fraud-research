"""
Chronological Temporal Splitting and Dataset Partitioning for Supervised Learning.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("temporal_split")


def create_temporal_splits(
    df_features: pd.DataFrame,
    df_classes: pd.DataFrame,
    train_range: Tuple[int, int] = (1, 34),
    val_range: Tuple[int, int] = (35, 39),
    test_range: Tuple[int, int] = (40, 49),
    output_dir: Optional[Path] = None
) -> Tuple[
    Tuple[pd.DataFrame, np.ndarray],
    Tuple[pd.DataFrame, np.ndarray],
    Tuple[pd.DataFrame, np.ndarray],
    pd.DataFrame
]:
    """
    Creates chronological train/validation/test splits using only labeled instances.
    
    Mapping:
    - class "1" -> 1 (Illicit / Fraud)
    - class "2" -> 0 (Licit / Clean)
    - class "unknown" -> EXCLUDED from supervised evaluation sets
    
    Args:
        df_features: Full features DataFrame (including txId and time_step).
        df_classes: Full classes DataFrame (txId, class).
        train_range: (min_t, max_t) for training split (default: 1 to 34).
        val_range: (min_t, max_t) for validation split (default: 35 to 39).
        test_range: (min_t, max_t) for test split (default: 40 to 49).
        output_dir: Directory to save temporal split distribution CSV.
        
    Returns:
        ((X_train_df, y_train), (X_val_df, y_val), (X_test_df, y_test), split_summary_df)
    """
    logger.info("Creating chronological temporal splits...")
    
    # 1. Merge features with labels
    df_labeled = df_features.merge(df_classes, on="txId", how="inner")
    
    # 2. Filter strictly labeled transactions
    df_labeled = df_labeled[df_labeled["class"].isin(["1", "2"])].copy()
    
    # 3. Explicit binary mapping: class '1' (illicit) -> 1, class '2' (licit) -> 0
    df_labeled["label"] = (df_labeled["class"] == "1").astype(np.int64)
    
    # 4. Partition by time_step
    train_mask = (df_labeled["time_step"] >= train_range[0]) & (df_labeled["time_step"] <= train_range[1])
    val_mask = (df_labeled["time_step"] >= val_range[0]) & (df_labeled["time_step"] <= val_range[1])
    test_mask = (df_labeled["time_step"] >= test_range[0]) & (df_labeled["time_step"] <= test_range[1])
    
    df_train = df_labeled[train_mask].copy().reset_index(drop=True)
    df_val = df_labeled[val_mask].copy().reset_index(drop=True)
    df_test = df_labeled[test_mask].copy().reset_index(drop=True)
    
    y_train = df_train["label"].values
    y_val = df_val["label"].values
    y_test = df_test["label"].values
    
    # 5. Audit split statistics
    def _get_split_stats(df_split: pd.DataFrame, split_name: str, t_range: Tuple[int, int]) -> Dict:
        n_total = len(df_split)
        n_illicit = int((df_split["label"] == 1).sum())
        n_licit = int((df_split["label"] == 0).sum())
        pct_illicit = (n_illicit / n_total) * 100.0 if n_total > 0 else 0.0
        imbalance_ratio = float(n_licit / n_illicit) if n_illicit > 0 else np.nan
        
        return {
            "Split": split_name,
            "Timestep Range": f"{t_range[0]}–{t_range[1]}",
            "Total Samples": n_total,
            "Licit Count (Class 0)": n_licit,
            "Illicit Count (Class 1)": n_illicit,
            "Illicit Rate (%)": round(pct_illicit, 2),
            "Imbalance Ratio (Licit:Illicit)": f"{imbalance_ratio:.2f}:1" if not np.isnan(imbalance_ratio) else "N/A"
        }

    stats_list = [
        _get_split_stats(df_train, "Training", train_range),
        _get_split_stats(df_val, "Validation", val_range),
        _get_split_stats(df_test, "Test", test_range),
    ]
    df_summary = pd.DataFrame(stats_list)
    
    # Add total row
    total_samples = len(df_train) + len(df_val) + len(df_test)
    total_licit = int((df_labeled["label"] == 0).sum())
    total_illicit = int((df_labeled["label"] == 1).sum())
    df_summary.loc[len(df_summary)] = {
        "Split": "Total Labeled Set",
        "Timestep Range": f"{train_range[0]}–{test_range[1]}",
        "Total Samples": total_samples,
        "Licit Count (Class 0)": total_licit,
        "Illicit Count (Class 1)": total_illicit,
        "Illicit Rate (%)": round((total_illicit / total_samples) * 100.0, 2),
        "Imbalance Ratio (Licit:Illicit)": f"{(total_licit / total_illicit):.2f}:1"
    }
    
    logger.info(
        f"Split breakdown: Train={len(df_train):,} ({sum(y_train==1):,} illicit) | "
        f"Val={len(df_val):,} ({sum(y_val==1):,} illicit) | "
        f"Test={len(df_test):,} ({sum(y_test==1):,} illicit)"
    )
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "temporal_split_distribution.csv"
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Temporal split distribution saved to: {csv_path}")
        
    return (df_train, y_train), (df_val, y_val), (df_test, y_test), df_summary
