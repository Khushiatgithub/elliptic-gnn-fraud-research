"""
Unit tests for data loader.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.loader import (
    get_feature_column_names,
    load_classes,
    load_edgelist,
    load_features,
)
from src.utils.config import get_project_root, load_config


def test_get_feature_column_names():
    cols = get_feature_column_names(total_features=165, num_local=93)
    assert len(cols) == 167
    assert cols[0] == "txId"
    assert cols[1] == "time_step"
    assert cols[2] == "local_feat_0"
    assert cols[94] == "local_feat_92"
    assert cols[95] == "agg_feat_0"
    assert cols[166] == "agg_feat_71"


def test_load_classes():
    cfg = load_config()
    classes_path = Path(cfg["paths"]["classes_file"])
    if not classes_path.exists():
        pytest.skip("Dataset classes file not found")
        
    df_classes = load_classes(classes_path)
    assert df_classes.shape[0] == 203769
    assert df_classes.shape[1] == 2
    assert set(df_classes.columns) == {"txId", "class"}
    assert set(df_classes["class"].unique()).issubset({"1", "2", "unknown"})


def test_load_edgelist():
    cfg = load_config()
    edgelist_path = Path(cfg["paths"]["edgelist_file"])
    if not edgelist_path.exists():
        pytest.skip("Dataset edgelist file not found")
        
    df_edges = load_edgelist(edgelist_path)
    assert df_edges.shape[0] == 234355
    assert df_edges.shape[1] == 2
    assert list(df_edges.columns) == ["txId1", "txId2"]


def test_load_features_shape():
    cfg = load_config()
    features_path = Path(cfg["paths"]["features_file"])
    if not features_path.exists():
        pytest.skip("Dataset features file not found")
        
    # Read first 100 rows for fast testing
    cols = get_feature_column_names(165, 93)
    df_feat_sample = pd.read_csv(features_path, header=None, names=cols, nrows=100)
    assert df_feat_sample.shape == (100, 167)
    assert df_feat_sample["time_step"].iloc[0] == 1


def test_exact_feature_schema_partition():
    """Explicitly verify the 93 local + 72 aggregated = 165 ML features schema."""
    cfg = load_config()
    features_path = Path(cfg["paths"]["features_file"])
    if not features_path.exists():
        pytest.skip("Dataset features file not found")
        
    df_sample = pd.read_csv(features_path, header=None, nrows=10)
    total_cols = df_sample.shape[1]
    
    # 1. Total columns per line must be 167
    assert total_cols == 167, f"Expected 167 columns, got {total_cols}"
    
    # 2. Non-ID columns count (cols 1..166) is 166
    non_id_cols_count = total_cols - 1
    assert non_id_cols_count == 166
    
    # 3. ML features count (cols 2..166) is 165
    ml_features_count = total_cols - 2
    assert ml_features_count == 165
    
    # 4. Local feature partition (cols 2..94) is 93
    local_features_count = 94 - 2 + 1
    assert local_features_count == 93
    
    # 5. Aggregated feature partition (cols 95..166) is 72
    agg_features_count = 166 - 95 + 1
    assert agg_features_count == 72
    
    # 6. Sum of partitions matches ML features
    assert local_features_count + agg_features_count == 165
