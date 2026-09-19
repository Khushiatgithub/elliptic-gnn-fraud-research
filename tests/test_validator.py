"""
Unit tests for dataset validator.
"""

import pandas as pd
import numpy as np
import pytest

from src.data.validator import DatasetValidator


@pytest.fixture
def sample_dataset():
    df_classes = pd.DataFrame({
        "txId": [101, 102, 103, 104],
        "class": ["1", "2", "unknown", "2"]
    })
    
    df_edges = pd.DataFrame({
        "txId1": [101, 102],
        "txId2": [102, 103]
    })
    
    # 4 rows, 2 metadata + 4 features
    df_features = pd.DataFrame({
        "txId": [101, 102, 103, 104],
        "time_step": [1, 1, 1, 1],
        "local_feat_0": [0.1, 0.2, 0.3, 0.4],
        "local_feat_1": [-0.5, 0.0, 0.5, 1.0],
        "agg_feat_0": [1.0, 2.0, 3.0, 4.0],
        "agg_feat_1": [0.0, 0.1, 0.2, 0.3]
    })
    
    return df_classes, df_edges, df_features


def test_validator_clean_data(sample_dataset):
    df_classes, df_edges, df_features = sample_dataset
    validator = DatasetValidator(df_classes, df_edges, df_features)
    results = validator.validate_all()
    
    assert results["summary_status"] == "PASSED"
    assert results["missing_values"]["has_missing_values"] is False
    assert results["duplicates"]["has_duplicates"] is False
    assert results["edge_integrity"]["self_loops_count"] == 0
    assert results["relational_consistency"]["isolated_nodes_in_graph_count"] == 1  # tx 104 is isolated


def test_validator_detects_self_loops(sample_dataset):
    df_classes, df_edges, df_features = sample_dataset
    df_edges_corrupt = pd.concat([
        df_edges,
        pd.DataFrame([{"txId1": 101, "txId2": 101}])
    ], ignore_index=True)
    
    validator = DatasetValidator(df_classes, df_edges_corrupt, df_features)
    results = validator.validate_all()
    assert results["edge_integrity"]["self_loops_count"] == 1


def test_validator_detects_orphan_edges(sample_dataset):
    df_classes, df_edges, df_features = sample_dataset
    df_edges_orphan = pd.concat([
        df_edges,
        pd.DataFrame([{"txId1": 999, "txId2": 102}])
    ], ignore_index=True)
    
    validator = DatasetValidator(df_classes, df_edges_orphan, df_features)
    results = validator.validate_all()
    assert results["relational_consistency"]["edges_not_in_features_count"] == 1
    assert results["summary_status"] == "WARNINGS_FOUND"
