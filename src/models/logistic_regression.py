"""
Logistic Regression Baseline Wrapper.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

logger = get_logger("logistic_regression")


class LogisticRegressionBaseline:
    """
    Logistic Regression classifier supporting standard and class-weighted formulations.
    """

    def __init__(
        self,
        weighted: bool = False,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        self.weighted = weighted
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        class_weight = "balanced" if weighted else None
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
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
            "model_type": "LogisticRegression",
            "weighted": self.weighted,
            "class_weight": "balanced" if self.weighted else "None",
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state
        }
