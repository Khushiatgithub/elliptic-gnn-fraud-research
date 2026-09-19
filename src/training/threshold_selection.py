"""
Validation-Locked Classification Threshold Optimization.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score

from src.utils.logger import get_logger

logger = get_logger("threshold_selection")


def find_optimal_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_steps: int = 99
) -> Tuple[float, float, Dict[float, float]]:
    """
    Fast vectorized threshold sweep on validation set.
    """
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)
    y_true_bool = (y_true_val == 1)
    
    # Broadcast: [num_steps, N]
    preds = y_prob_val[None, :] >= thresholds[:, None]
    tp = np.sum(preds & y_true_bool[None, :], axis=1)
    fp = np.sum(preds & (~y_true_bool)[None, :], axis=1)
    fn = np.sum((~preds) & y_true_bool[None, :], axis=1)
    
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    denom = prec + rec
    f1_scores = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(round(thresholds[best_idx], 4))
    best_score = float(f1_scores[best_idx])
    
    threshold_scores = {float(round(t, 4)): float(s) for t, s in zip(thresholds, f1_scores)}
    return best_threshold, best_score, threshold_scores
