"""
Unit tests for Phase 3: Graph Neural Networks, Masking, Builder, and Analyses.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.homophily import compute_graph_homophily
from src.data.graph_builder import build_pyg_graph
from src.data.masks import create_node_masks
from src.models.gat import GATNet
from src.models.gcn import GCNNet
from src.models.graphsage import GraphSAGENet
from src.training.gnn_trainer import train_and_evaluate_gnn


@pytest.fixture
def dummy_graph_data():
    np.random.seed(42)
    n_nodes = 100
    tx_ids = np.arange(1000, 1000 + n_nodes)
    
    # 93 local + 72 agg = 165 features
    feat_dict = {"txId": tx_ids, "time_step": np.random.choice([1, 2, 35, 36, 40, 41], size=n_nodes)}
    for i in range(93):
        feat_dict[f"local_feat_{i}"] = np.random.randn(n_nodes).astype(np.float32)
    for i in range(72):
        feat_dict[f"agg_feat_{i}"] = np.random.randn(n_nodes).astype(np.float32)
    df_feat = pd.DataFrame(feat_dict)
    
    # Classes: 10 illicit (1), 40 licit (2), 50 unknown
    cls_vals = ["1"] * 10 + ["2"] * 40 + ["unknown"] * 50
    df_cls = pd.DataFrame({"txId": tx_ids, "class": cls_vals})
    
    # Edges: within same time steps
    edges = []
    for ts in np.unique(feat_dict["time_step"]):
        ts_nodes = tx_ids[feat_dict["time_step"] == ts]
        if len(ts_nodes) > 1:
            for _ in range(len(ts_nodes) * 2):
                src, dst = np.random.choice(ts_nodes, size=2, replace=False)
                edges.append((src, dst))
    df_edges = pd.DataFrame(edges, columns=["txId1", "txId2"]).drop_duplicates()
    return df_feat, df_cls, df_edges


def test_masks_integrity(dummy_graph_data):
    df_feat, df_cls, _ = dummy_graph_data
    df_merged = pd.merge(df_feat, df_cls, on="txId")
    masks = create_node_masks(df_merged, (1, 34), (35, 39), (40, 49))
    
    # Unknown nodes must never be in train/val/test masks
    unknown_mask = masks["unknown_mask"]
    assert not torch.any(masks["train_mask"] & unknown_mask)
    assert not torch.any(masks["val_mask"] & unknown_mask)
    assert not torch.any(masks["test_mask"] & unknown_mask)
    assert not torch.any(masks["train_mask"] & masks["val_mask"])
    assert not torch.any(masks["train_mask"] & masks["test_mask"])


def test_graph_builder(dummy_graph_data):
    df_feat, df_cls, df_edges = dummy_graph_data
    
    # Forward full
    data = build_pyg_graph(df_feat, df_cls, df_edges, feature_mode="full", direction="forward", remove_unknown=False)
    assert data.num_nodes == 100
    assert data.num_features == 165
    assert data.edge_index.shape[0] == 2
    
    # Removed unknown
    data_rem = build_pyg_graph(df_feat, df_cls, df_edges, feature_mode="local", direction="bidirectional", remove_unknown=True)
    assert data_rem.num_nodes == 50
    assert data_rem.num_features == 93


def test_gnn_model_forward_passes(dummy_graph_data):
    df_feat, df_cls, df_edges = dummy_graph_data
    data = build_pyg_graph(df_feat, df_cls, df_edges, feature_mode="local", direction="forward")
    
    in_dim = data.num_features
    gcn = GCNNet(in_channels=in_dim, hidden_channels=32, num_layers=2)
    sage = GraphSAGENet(in_channels=in_dim, hidden_channels=32, num_layers=2)
    gat = GATNet(in_channels=in_dim, hidden_channels=32, num_layers=2, heads=4)
    
    out_gcn = gcn(data.x, data.edge_index)
    out_sage = sage(data.x, data.edge_index)
    out_gat = gat(data.x, data.edge_index)
    
    assert out_gcn.shape == (data.num_nodes,)
    assert out_sage.shape == (data.num_nodes,)
    assert out_gat.shape == (data.num_nodes,)


def test_gnn_training_loop(dummy_graph_data):
    df_feat, df_cls, df_edges = dummy_graph_data
    data = build_pyg_graph(df_feat, df_cls, df_edges, feature_mode="local", direction="forward")
    
    gcn = GCNNet(in_channels=data.num_features, hidden_channels=16, num_layers=2)
    res = train_and_evaluate_gnn(gcn, data, seed=42, epochs=5, lr=0.01, patience=3)
    
    assert "val_illicit_f1" in res
    assert "test_illicit_f1" in res
    assert "optimal_threshold" in res
    assert 0.0 <= res["optimal_threshold"] <= 1.0


def test_homophily_computation(dummy_graph_data):
    df_feat, df_cls, df_edges = dummy_graph_data
    data = build_pyg_graph(df_feat, df_cls, df_edges, feature_mode="full", direction="forward")
    homo_dict, df_homo = compute_graph_homophily(data)
    assert "overall_labeled_homophily" in homo_dict
    assert len(df_homo) > 0
