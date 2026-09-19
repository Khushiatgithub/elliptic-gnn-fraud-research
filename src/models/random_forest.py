"""
Random Forest Baseline Wrapper.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import get_logger

logger = get_logger("random_forest")


class RandomForestBaseline:
    """
    Random Forest classifier supporting standard and balanced class weighting.
    """

    def __init__(
        self,
        weighted: bool = False,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        self.weighted = weighted
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        class_weight = "balanced" if weighted else None
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseline":
        """Fit model on training data."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probability of class 1 (Illicit)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using specified threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "model_type": "RandomForest",
            "weighted": self.weighted,
            "class_weight": "balanced" if self.weighted else "None",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state
        }
