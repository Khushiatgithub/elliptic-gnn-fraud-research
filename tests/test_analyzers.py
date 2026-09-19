"""
Unit tests for label, feature, and graph analyzers.
"""

import pandas as pd
import numpy as np
import pytest

from src.data.label_analyzer import LabelAnalyzer
from src.features.feature_analyzer import FeatureAnalyzer
from src.graph.graph_analyzer import GraphAnalyzer
from src.graph.homophily_analyzer import HomophilyAnalyzer


@pytest.fixture
def synthetic_data():
    df_classes = pd.DataFrame({
        "txId": [1, 2, 3, 4, 5, 6],
        "class": ["1", "2", "2", "unknown", "1", "2"]
    })
    
    df_edges = pd.DataFrame({
        "txId1": [1, 2, 3, 5],
        "txId2": [2, 3, 4, 1]
    })
    
    df_features = pd.DataFrame({
        "txId": [1, 2, 3, 4, 5, 6],
        "time_step": [1, 1, 1, 2, 2, 2],
        "local_feat_0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "local_feat_1": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # constant
        "agg_feat_0": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0], # perfectly correlated with local_feat_0
        "agg_feat_1": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    })
    
    return df_classes, df_edges, df_features


def test_label_analyzer(synthetic_data):
    df_classes, _, df_features = synthetic_data
    analyzer = LabelAnalyzer(df_classes, df_features)
    stats = analyzer.compute_class_distribution()
    
    assert stats["total_transactions"] == 6
    assert stats["num_illicit"] == 2
    assert stats["num_licit"] == 3
    assert stats["num_unknown"] == 1
    assert stats["num_labeled"] == 5
    assert stats["imbalance_ratio_licit_to_illicit"] == 1.5


def test_feature_analyzer_constant_and_corr(synthetic_data):
    _, _, df_features = synthetic_data
    analyzer = FeatureAnalyzer(df_features, num_local=2, num_agg=2)
    
    const_feats, near_zero = analyzer.detect_low_variance_features()
    assert "local_feat_1" in const_feats
    
    high_corr = analyzer.compute_high_correlations(threshold=0.99)
    assert len(high_corr) >= 1
    assert (high_corr["feature_1"].iloc[0] == "local_feat_0" and high_corr["feature_2"].iloc[0] == "agg_feat_0") or \
           (high_corr["feature_1"].iloc[0] == "agg_feat_0" and high_corr["feature_2"].iloc[0] == "local_feat_0")


def test_graph_and_homophily_analyzer(synthetic_data):
    df_classes, df_edges, df_features = synthetic_data
    g_analyzer = GraphAnalyzer(df_edges, df_features)
    deg_stats = g_analyzer.compute_degree_statistics()
    
    assert deg_stats["total_nodes"] == 6
    assert deg_stats["total_edges"] == 4
    assert deg_stats["isolated_nodes_count"] == 1  # node 6
    
    h_analyzer = HomophilyAnalyzer(df_classes, df_edges, df_features)
    h_stats = h_analyzer.compute_homophily_metrics()
    assert h_stats["total_network_edges"] == 4
    assert h_stats["edges_with_illicit_count"] >= 1
