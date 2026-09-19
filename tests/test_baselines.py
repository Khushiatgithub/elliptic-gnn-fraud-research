"""
Unit tests for Phase 2: Conventional ML Baselines, Preprocessing, and Metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import TabularPreprocessor
from src.training.temporal_split import create_temporal_splits
from src.training.threshold_selection import find_optimal_threshold
from src.evaluation.metrics import compute_classification_metrics
from src.models.logistic_regression import LogisticRegressionBaseline
from src.models.random_forest import RandomForestBaseline
from src.models.xgboost_model import XGBoostBaseline
from src.models.mlp import MLPBaseline


@pytest.fixture
def dummy_tabular_data():
    np.random.seed(42)
    n_samples = 200
    
    # Create 93 local + 72 agg = 165 features
    data = {
        "txId": list(range(1000, 1000 + n_samples)),
        "time_step": np.random.choice([1, 2, 35, 36, 40, 41], size=n_samples),
    }
    for i in range(93):
        data[f"local_feat_{i}"] = np.random.randn(n_samples).astype(np.float32)
    for i in range(72):
        data[f"agg_feat_{i}"] = np.random.randn(n_samples).astype(np.float32)
        
    df_feat = pd.DataFrame(data)
    df_cls = pd.DataFrame({
        "txId": list(range(1000, 1000 + n_samples)),
        "class": np.random.choice(["1", "2"], size=n_samples, p=[0.1, 0.9])
    })
    return df_feat, df_cls


def test_temporal_split_integrity(dummy_tabular_data):
    df_feat, df_cls = dummy_tabular_data
    (df_tr, y_tr), (df_v, y_v), (df_te, y_te), df_summary = create_temporal_splits(
        df_features=df_feat,
        df_classes=df_cls,
        train_range=(1, 34),
        val_range=(35, 39),
        test_range=(40, 49)
    )
    
    assert len(df_tr) + len(df_v) + len(df_te) == 200
    assert set(np.unique(y_tr)).issubset({0, 1})
    assert len(df_summary) == 4
    assert df_summary.loc[0, "Split"] == "Training"


def test_tabular_preprocessor(dummy_tabular_data):
    df_feat, _ = dummy_tabular_data
    
    # Local only
    prep_local = TabularPreprocessor(feature_mode="local", scale=True)
    X_loc = prep_local.fit_transform(df_feat)
    assert X_loc.shape == (200, 93)
    assert np.allclose(np.mean(X_loc, axis=0), 0.0, atol=1e-4)
    
    # Full features
    prep_full = TabularPreprocessor(feature_mode="full", scale=False)
    X_full = prep_full.fit_transform(df_feat)
    assert X_full.shape == (200, 165)


def test_threshold_selection_and_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.7, 0.6, 0.3, 0.2, 0.9])
    
    opt_tau, best_f1, _ = find_optimal_threshold(y_true, y_prob)
    assert 0.01 <= opt_tau <= 0.99
    assert best_f1 > 0.8
    
    metrics = compute_classification_metrics(y_true, y_prob, threshold=opt_tau)
    assert "illicit_f1" in metrics
    assert "pr_auc" in metrics
    assert metrics["pr_auc"] > 0.8


def test_all_baseline_models_fit(dummy_tabular_data):
    df_feat, df_cls = dummy_tabular_data
    (df_tr, y_tr), (df_v, y_v), (df_te, y_te), _ = create_temporal_splits(
        df_features=df_feat,
        df_classes=df_cls,
        train_range=(1, 34),
        val_range=(35, 39),
        test_range=(40, 49)
    )
    
    prep = TabularPreprocessor(feature_mode="local", scale=True)
    X_tr = prep.fit_transform(df_tr)
    X_te = prep.transform(df_te)
    
    # 1. Logistic Regression
    lr = LogisticRegressionBaseline(weighted=True, random_state=42)
    lr.fit(X_tr, y_tr)
    probs_lr = lr.predict_proba(X_te)
    assert probs_lr.shape == (len(df_te),)
    assert (probs_lr >= 0.0).all() and (probs_lr <= 1.0).all()
    
    # 2. Random Forest
    rf = RandomForestBaseline(weighted=True, n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_tr, y_tr)
    probs_rf = rf.predict_proba(X_te)
    assert probs_rf.shape == (len(df_te),)
    
    # 3. XGBoost
    xgb = XGBoostBaseline(weighted=True, n_estimators=10, max_depth=3, random_state=42)
    xgb.fit(X_tr, y_tr)
    probs_xgb = xgb.predict_proba(X_te)
    assert probs_xgb.shape == (len(df_te),)
    
    # 4. MLP
    mlp = MLPBaseline(weighted=True, hidden_dims=[32, 16], epochs=3, random_state=42)
    mlp.fit(X_tr, y_tr)
    probs_mlp = mlp.predict_proba(X_te)
    assert probs_mlp.shape == (len(df_te),)
