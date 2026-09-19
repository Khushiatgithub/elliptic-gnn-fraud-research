"""
Comprehensive Evaluation Metrics for Imbalanced Tabular Fraud Classification.
"""

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics given ground-truth labels and positive class probabilities.
    
    Args:
        y_true: Binary ground-truth array (0 = Licit, 1 = Illicit).
        y_prob: Predicted probabilities for class 1 (Illicit).
        threshold: Decision threshold for binary classification.
        
    Returns:
        Dictionary of scalar performance metrics and curve arrays.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Primary Metrics
    illicit_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    # 2. Secondary Metrics
    illicit_precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    illicit_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5
        
    mcc = float(matthews_corrcoef(y_true, y_pred))
    accuracy = float(accuracy_score(y_true, y_pred))
    
    # 3. Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 4. Curves
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_prob)
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true, y_prob)
    
    return {
        # Primary
        "illicit_f1": illicit_f1,
        "pr_auc": pr_auc,
        # Secondary
        "illicit_precision": illicit_precision,
        "illicit_recall": illicit_recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "accuracy": accuracy,
        "threshold": float(threshold),
        # Confusion Matrix
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
        # Curve Data
        "pr_curve": {
            "precision": precision_arr.tolist(),
            "recall": recall_arr.tolist(),
        },
        "roc_curve": {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
        }
    }
