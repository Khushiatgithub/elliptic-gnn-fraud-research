"""
Serialization utility to train and save baseline tabular models and preprocessor
strictly on the training split (timesteps 1–34) with seed 42 to enable instant,
authentic live multi-model prediction on the dashboard.
"""

from pathlib import Path
import joblib
import torch
import yaml
import numpy as np

from src.data.loader import load_classes, load_features
from src.features.preprocessing import TabularPreprocessor
from src.models.logistic_regression import LogisticRegressionBaseline
from src.models.mlp import MLPBaseline
from src.models.random_forest import RandomForestBaseline
from src.models.xgboost_model import XGBoostBaseline
from src.training.temporal_split import create_temporal_splits
from src.utils.config import get_project_root
from src.utils.logger import get_logger

logger = get_logger("tabular_serializer")


def serialize_tabular_models():
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "elliptic_bitcoin_dataset"
    ckpt_dir = root / "models" / "checkpoints" / "tabular"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = root / "configs" / "baseline_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        b_cfg = yaml.safe_load(f)
        
    logger.info("Loading raw dataset for baseline serialization...")
    df_classes = load_classes(raw_dir / "elliptic_txs_classes.csv")
    df_features = load_features(raw_dir / "elliptic_txs_features.csv", total_features=165, num_local=93)
    
    (df_train, y_train), (df_val, y_val), (df_test, y_test), _ = create_temporal_splits(
        df_features=df_features,
        df_classes=df_classes,
        train_range=tuple(b_cfg["splits"]["train_timesteps"]),
        val_range=tuple(b_cfg["splits"]["val_timesteps"]),
        test_range=tuple(b_cfg["splits"]["test_timesteps"])
    )
    
    # 1. Fit & Save Preprocessors
    logger.info("Fitting and saving TabularPreprocessor for Full (165) and Local (93) features...")
    prep_scaled_full = TabularPreprocessor(feature_mode="full", scale=True).fit(df_train)
    prep_unscaled_full = TabularPreprocessor(feature_mode="full", scale=False).fit(df_train)
    prep_scaled_local = TabularPreprocessor(feature_mode="local", scale=True).fit(df_train)
    prep_unscaled_local = TabularPreprocessor(feature_mode="local", scale=False).fit(df_train)
    
    joblib.dump(prep_scaled_full, ckpt_dir / "prep_scaled_full.joblib")
    joblib.dump(prep_unscaled_full, ckpt_dir / "prep_unscaled_full.joblib")
    joblib.dump(prep_scaled_local, ckpt_dir / "prep_scaled_local.joblib")
    joblib.dump(prep_unscaled_local, ckpt_dir / "prep_unscaled_local.joblib")
    
    X_train_scaled = prep_scaled_full.transform(df_train)
    X_train_unscaled = prep_unscaled_full.transform(df_train)
    
    seed = 42
    
    # 2. Random Forest (Full 165, Standard)
    rf_path = ckpt_dir / "random_forest_full_seed42.joblib"
    if not rf_path.exists():
        logger.info("Fitting Random Forest (Full 165, Standard, Seed 42)...")
        rf_cfg = b_cfg["models"]["random_forest"]
        rf_model = RandomForestBaseline(
            weighted=False,
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 15),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=seed,
            n_jobs=-1
        )
        rf_model.fit(X_train_unscaled, y_train)
        joblib.dump(rf_model, rf_path)
        logger.info(f"Saved Random Forest model to {rf_path}")
    else:
        logger.info(f"Random Forest model already exists at {rf_path}")
        
    # 3. XGBoost (Full 165, Standard)
    xgb_path = ckpt_dir / "xgboost_full_seed42.joblib"
    if not xgb_path.exists():
        logger.info("Fitting XGBoost (Full 165, Standard, Seed 42)...")
        xgb_cfg = b_cfg["models"]["xgboost"]
        xgb_model = XGBoostBaseline(
            weighted=False,
            n_estimators=xgb_cfg.get("n_estimators", 100),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            random_state=seed,
            n_jobs=-1
        )
        xgb_model.fit(X_train_unscaled, y_train)
        joblib.dump(xgb_model, xgb_path)
        logger.info(f"Saved XGBoost model to {xgb_path}")
    else:
        logger.info(f"XGBoost model already exists at {xgb_path}")
        
    # 4. Logistic Regression (Full 165, Standard)
    lr_path = ckpt_dir / "logistic_regression_full_seed42.joblib"
    if not lr_path.exists():
        logger.info("Fitting Logistic Regression (Full 165, Standard, Seed 42)...")
        lr_cfg = b_cfg["models"]["logistic_regression"]
        lr_model = LogisticRegressionBaseline(
            weighted=False,
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            random_state=seed
        )
        lr_model.fit(X_train_scaled, y_train)
        joblib.dump(lr_model, lr_path)
        logger.info(f"Saved Logistic Regression model to {lr_path}")
    else:
        logger.info(f"Logistic Regression model already exists at {lr_path}")
        
    # 5. MLP (Full 165, Standard)
    mlp_path = ckpt_dir / "mlp_full_seed42.pt"
    if not mlp_path.exists():
        logger.info("Fitting MLP Baseline (Full 165, Standard, Seed 42)...")
        mlp_cfg = b_cfg["models"]["mlp"]
        mlp_model = MLPBaseline(
            weighted=False,
            hidden_dims=mlp_cfg.get("hidden_dims", [128, 64]),
            dropout=mlp_cfg.get("dropout", 0.2),
            learning_rate=mlp_cfg.get("learning_rate", 0.001),
            weight_decay=mlp_cfg.get("weight_decay", 1e-4),
            batch_size=mlp_cfg.get("batch_size", 256),
            epochs=mlp_cfg.get("epochs", 40),
            random_state=seed
        )
        mlp_model.fit(X_train_scaled, y_train)
        torch.save(mlp_model.model.state_dict(), mlp_path)
        logger.info(f"Saved MLP state dict to {mlp_path}")
    else:
        logger.info(f"MLP checkpoint already exists at {mlp_path}")
        
    logger.info("All tabular models and preprocessors serialized successfully.")


if __name__ == "__main__":
    serialize_tabular_models()
