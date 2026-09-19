"""
Safe and Typed Data Loading for the Elliptic Bitcoin Dataset.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_loader")


def get_feature_column_names(total_features: int = 165, num_local: int = 93) -> list:
    """
    Generate standard descriptive column names for features.csv.
    
    Structure:
    - col 0: txId
    - col 1: time_step
    - cols 2..94 (93 cols): local_feat_0 .. local_feat_92
    - cols 95..166 (72 cols): agg_feat_0 .. agg_feat_71
    
    Args:
        total_features: Total number of features (165).
        num_local: Number of local features (93).
        
    Returns:
        List of 167 column names.
    """
    num_agg = total_features - num_local
    cols = ["txId", "time_step"]
    cols.extend([f"local_feat_{i}" for i in range(num_local)])
    cols.extend([f"agg_feat_{i}" for i in range(num_agg)])
    return cols


def load_classes(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load elliptic_txs_classes.csv safely.
    
    Args:
        filepath: Path to classes CSV file.
        
    Returns:
        DataFrame with columns ['txId', 'class'], where 'class' is categorical string.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found at: {path}")
        
    logger.info(f"Loading classes from {path}...")
    df_classes = pd.read_csv(
        path,
        dtype={"txId": np.int64, "class": str}
    )
    # Clean whitespace if any
    df_classes["class"] = df_classes["class"].str.strip()
    logger.info(f"Loaded classes: {df_classes.shape[0]} rows, {df_classes.shape[1]} cols.")
    return df_classes


def load_edgelist(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load elliptic_txs_edgelist.csv safely.
    
    Args:
        filepath: Path to edgelist CSV file.
        
    Returns:
        DataFrame with columns ['txId1', 'txId2'].
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Edgelist file not found at: {path}")
        
    logger.info(f"Loading edgelist from {path}...")
    df_edges = pd.read_csv(
        path,
        dtype={"txId1": np.int64, "txId2": np.int64}
    )
    logger.info(f"Loaded edgelist: {df_edges.shape[0]} edges.")
    return df_edges


def load_features(filepath: Union[str, Path], total_features: int = 165, num_local: int = 93) -> pd.DataFrame:
    """
    Load elliptic_txs_features.csv safely.
    
    Args:
        filepath: Path to features CSV file (headerless).
        total_features: Total features (165).
        num_local: Local features count (93).
        
    Returns:
        DataFrame with standardized column names and float32 features.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found at: {path}")
        
    logger.info(f"Loading features from {path} (header=None)...")
    cols = get_feature_column_names(total_features=total_features, num_local=num_local)
    
    dtype_dict = {"txId": np.int64, "time_step": np.int32}
    for col in cols[2:]:
        dtype_dict[col] = np.float32
        
    df_features = pd.read_csv(
        path,
        header=None,
        names=cols,
        dtype=dtype_dict
    )
    logger.info(f"Loaded features: {df_features.shape[0]} rows, {df_features.shape[1]} cols.")
    return df_features


def load_complete_dataset(
    raw_dir: Union[str, Path],
    total_features: int = 165,
    num_local: int = 93
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three dataset files without modifying original files.
    
    Args:
        raw_dir: Directory containing raw CSV files.
        total_features: Total feature count (165).
        num_local: Local feature count (93).
        
    Returns:
        Tuple of (df_classes, df_edges, df_features).
    """
    raw_path = Path(raw_dir)
    classes_path = raw_path / "elliptic_txs_classes.csv"
    edgelist_path = raw_path / "elliptic_txs_edgelist.csv"
    features_path = raw_path / "elliptic_txs_features.csv"
    
    df_classes = load_classes(classes_path)
    df_edges = load_edgelist(edgelist_path)
    df_features = load_features(features_path, total_features=total_features, num_local=num_local)
    
    return df_classes, df_edges, df_features
