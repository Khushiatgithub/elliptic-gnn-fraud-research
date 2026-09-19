"""
Leakage-Free Preprocessing Pipeline for Tabular Features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("tabular_preprocessing")


class TabularPreprocessor:
    """
    Fits feature scalers strictly on the training set and transforms
    validation/test sets without leaking future statistical moments.
    """

    def __init__(self, feature_mode: str = "full", scale: bool = True):
        """
        Args:
            feature_mode: "local" (93 features) or "full" (165 features).
            scale: If True, apply StandardScaler; if False, pass raw features.
        """
        if feature_mode not in ["local", "full"]:
            raise ValueError(f"feature_mode must be 'local' or 'full', got {feature_mode}")
            
        self.feature_mode = feature_mode
        self.scale = scale
        self.scaler: Optional[StandardScaler] = StandardScaler() if scale else None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def get_feature_columns(self, df_features: pd.DataFrame) -> List[str]:
        """Extract appropriate feature column subset based on mode."""
        if self.feature_mode == "local":
            return [c for c in df_features.columns if c.startswith("local_feat_")]
        else:
            return [c for c in df_features.columns if c.startswith("local_feat_") or c.startswith("agg_feat_")]

    def fit(self, X_train: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit preprocessing strictly on training data.
        
        Args:
            X_train: Training feature DataFrame.
        """
        self.feature_names = self.get_feature_columns(X_train)
        X_mat = X_train[self.feature_names].values.astype(np.float32)
        
        if self.scale and self.scaler is not None:
            self.scaler.fit(X_mat)
            logger.info(
                f"Fitted StandardScaler strictly on {len(X_train)} training instances "
                f"across {len(self.feature_names)} features ({self.feature_mode} mode)."
            )
        else:
            logger.info(
                f"Using unscaled features on {len(X_train)} training instances "
                f"across {len(self.feature_names)} features ({self.feature_mode} mode)."
            )
            
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform feature DataFrame using fitted parameters.
        
        Args:
            X: Feature DataFrame to transform.
            
        Returns:
            Numpy array of shape (N, D).
        """
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transforming data.")
            
        X_mat = X[self.feature_names].values.astype(np.float32)
        if self.scale and self.scaler is not None:
            return self.scaler.transform(X_mat)
        return X_mat

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed training matrix."""
        return self.fit(X_train).transform(X_train)

    def get_metadata(self) -> Dict[str, Union[str, int, bool]]:
        """Return preprocessor configuration metadata."""
        return {
            "feature_mode": self.feature_mode,
            "num_features": len(self.feature_names),
            "scaled": self.scale,
            "is_fitted": self.is_fitted,
        }
