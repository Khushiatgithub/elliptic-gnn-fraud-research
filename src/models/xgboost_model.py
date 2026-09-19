"""
XGBoost Baseline Wrapper.
"""

from typing import Any, Dict, Optional
import numpy as np
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger("xgboost_model")


class XGBoostBaseline:
    """
    XGBoost classifier supporting standard and scale_pos_weight imbalance handling.
    """

    def __init__(
        self,
        weighted: bool = False,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.weighted = weighted
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scale_pos_weight: float = 1.0
        self.model: Optional[XGBClassifier] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostBaseline":
        """
        Fit XGBoost model on training data.
        If weighted is True, dynamically calculate scale_pos_weight strictly from y_train.
        """
        if self.weighted:
            n_licit = int((y_train == 0).sum())
            n_illicit = int((y_train == 1).sum())
            self.scale_pos_weight = float(n_licit / n_illicit) if n_illicit > 0 else 1.0
        else:
            self.scale_pos_weight = 1.0
            
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss"
        )
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probability of class 1 (Illicit)."""
        if self.model is None:
            raise RuntimeError("XGBoost model must be fitted before predict_proba.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using specified threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "model_type": "XGBoost",
            "weighted": self.weighted,
            "scale_pos_weight": float(self.scale_pos_weight),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state
        }
