"""
Backend API and Demonstration Engine Test Suite.
"""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Elliptic" in data["dataset"]


def test_dataset_summary():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 203769
    assert data["total_edges"] == 234355
    assert data["total_timesteps"] == 49
    assert data["total_features"] == 165
    assert data["local_features"] == 93
    assert data["aggregated_features"] == 72
    assert data["illicit_count"] == 4545
    assert data["licit_count"] == 42019
    assert data["unknown_count"] == 157205


def test_dataset_timesteps():
    response = client.get("/api/dataset/timesteps")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 49
    # Check timestep 1 is train
    assert data[0]["timestep"] == 1
    assert data[0]["split"] == "train"
    # Check timestep 35 is val
    assert data[34]["timestep"] == 35
    assert data[34]["split"] == "val"
    # Check timestep 40 is test
    assert data[39]["timestep"] == 40
    assert data[39]["split"] == "test"


def test_sample_transactions():
    response = client.get("/api/dataset/sample-transactions")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) > 0
    assert any(s["label"] == "illicit" for s in samples)
    assert any(s["label"] == "licit" for s in samples)


def test_transaction_detail():
    response = client.get("/api/dataset/transaction/232629023")
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == 232629023
    assert data["ground_truth_label"] == "illicit"
    assert len(data["features_local"]) == 93
    assert len(data["features_agg"]) == 72


def test_graph_neighborhood():
    # 1-hop
    response = client.get("/api/transaction/230425980/neighbors?k_hop=1")
    assert response.status_code == 200
    subgraph = response.json()
    assert subgraph["center_tx_id"] == 230425980
    assert subgraph["total_nodes"] > 0
    assert subgraph["center_node"]["tx_id"] == 230425980

    # 2-hop
    response_2hop = client.get("/api/transaction/230425980/neighbors?k_hop=2")
    assert response_2hop.status_code == 200
    subgraph_2hop = response_2hop.json()
    assert subgraph_2hop["total_nodes"] >= subgraph["total_nodes"]


def test_results_endpoints():
    # Main results
    resp_main = client.get("/api/results/main")
    assert resp_main.status_code == 200
    main_data = resp_main.json()
    assert len(main_data) == 7  # 7 models
    model_names = [m["model"] for m in main_data]
    assert "Random Forest" in model_names
    assert "XGBoost" in model_names
    assert "GAT" in model_names
    assert "GraphSAGE" in model_names
    assert "GCN" in model_names

    # Feature comparison
    resp_feat = client.get("/api/results/features")
    assert resp_feat.status_code == 200
    assert len(resp_feat.json()) > 0

    # Weighting comparison
    resp_wt = client.get("/api/results/weighting")
    assert resp_wt.status_code == 200
    assert len(resp_wt.json()) > 0

    # Ablations
    resp_abl = client.get("/api/results/ablations")
    assert resp_abl.status_code == 200
    assert len(resp_abl.json()) > 0

    # Seeds
    resp_seeds = client.get("/api/results/seeds")
    assert resp_seeds.status_code == 200
    assert len(resp_seeds.json()) > 0

    # Statistics
    resp_stat = client.get("/api/results/statistics")
    assert resp_stat.status_code == 200
    assert len(resp_stat.json()) > 0

    # Error analysis
    resp_err = client.get("/api/results/error-analysis")
    assert resp_err.status_code == 200
    assert len(resp_err.json()) > 0


def test_figures_catalog():
    response = client.get("/api/figures/list")
    assert response.status_code == 200
    figures = response.json()
    assert len(figures) > 0
    assert any(f["filename"] == "f1_comparison.png" for f in figures)


def test_models_available():
    response = client.get("/api/models/available")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 7
    available_ids = [m["model_id"] for m in models if m["available"]]
    assert "random_forest" in available_ids
    assert "xgboost" in available_ids
    assert "gat" in available_ids


def test_live_prediction():
    # Test Random Forest Prediction
    payload_rf = {"tx_id": 230425980, "model_id": "random_forest"}
    resp_rf = client.post("/api/predict", json=payload_rf)
    assert resp_rf.status_code == 200
    data_rf = resp_rf.json()
    assert data_rf["tx_id"] == 230425980
    assert 0.0 <= data_rf["predicted_probability"] <= 1.0
    assert data_rf["model_id"] == "random_forest"
    assert data_rf["feature_contributions"] is not None
    assert len(data_rf["feature_contributions"]) > 0

    # Test XGBoost Prediction
    payload_xgb = {"tx_id": 232438397, "model_id": "xgboost"}
    resp_xgb = client.post("/api/predict", json=payload_xgb)
    assert resp_xgb.status_code == 200
    data_xgb = resp_xgb.json()
    assert 0.0 <= data_xgb["predicted_probability"] <= 1.0

    # Test GAT Prediction
    payload_gat = {"tx_id": 230425980, "model_id": "gat"}
    resp_gat = client.post("/api/predict", json=payload_gat)
    assert resp_gat.status_code == 200
    data_gat = resp_gat.json()
    assert 0.0 <= data_gat["predicted_probability"] <= 1.0
    assert data_gat["graph_context"] is not None
