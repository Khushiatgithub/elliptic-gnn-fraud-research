"""
Full-Batch GNN Training, Validation-Locked Threshold Selection, and Evaluation Engine.
Optimized for computational efficiency, robust device selection, and leakage-safe evaluation.
"""

import copy
import os
from pathlib import Path
import platform
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import psutil
from sklearn.metrics import precision_recall_curve, auc
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.evaluation.metrics import compute_classification_metrics
from src.training.threshold_selection import find_optimal_threshold
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("gnn_trainer")

_HARDWARE_LOGGED = False


def setup_device_and_log() -> torch.device:
    """
    Detects hardware, configures optimal multi-threading, and logs environment details.
    """
    global _HARDWARE_LOGGED
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if not _HARDWARE_LOGGED:
                logger.info("=" * 70)
                logger.info("HARDWARE ENVIRONMENT: CUDA GPU DETECTED")
                logger.info(f"Device: CUDA (cuda:0)")
                logger.info(f"GPU Name: {gpu_name}")
                logger.info(f"VRAM: {vram_gb:.2f} GB")
                logger.info(f"CPU: {platform.processor()}")
                logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
                logger.info("=" * 70)
                _HARDWARE_LOGGED = True
            return device
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
            
    # CPU fallback
    device = torch.device("cpu")
    num_cpus = os.cpu_count() or 4
    # Set PyTorch threads to optimal core count (up to 10)
    optimal_threads = min(10, num_cpus)
    torch.set_num_threads(optimal_threads)
    
    if not _HARDWARE_LOGGED:
        logger.info("=" * 70)
        logger.info("HARDWARE ENVIRONMENT: CPU ACCELERATION")
        logger.info(f"Device: CPU")
        logger.info(f"GPU: None (Reason: CUDA not available or CPU-only PyTorch build)")
        logger.info(f"CPU: {platform.processor()} ({num_cpus} logical cores)")
        logger.info(f"Torch Threads: {torch.get_num_threads()}")
        logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        logger.info("=" * 70)
        _HARDWARE_LOGGED = True
        
    return device


def train_and_evaluate_gnn(
    model: nn.Module,
    data: Data,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 15,
    val_interval: int = 2,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Trains a GNN model with full-batch message passing, anti-leakage class weighting,
    validation PR-AUC guided early stopping, and validation-locked test evaluation.
    
    Args:
        model: PyTorch GNN Module.
        data: PyG Data object (features, edge_index, masks, labels).
        seed: Random seed for deterministic reproducibility.
        epochs: Maximum training epochs.
        lr: Adam learning rate.
        weight_decay: L2 regularization penalty.
        patience: Early stopping patience epochs.
        val_interval: Interval (in epochs) between validation forward passes.
        device: 'cpu', 'cuda', or None for automatic device setup.
        
    Returns:
        Dictionary containing validation and test metrics, optimal threshold, timing, and curve data.
    """
    set_seed(seed)
    
    if device is None:
        device_obj = setup_device_and_log()
    else:
        device_obj = torch.device(device)
        
    model = model.to(device_obj)
    data = data.to(device_obj)
    
    # Calculate parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 1. Compute class imbalance weight ONLY from training split (Strict Anti-Leakage)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    y_train = data.y[train_mask]
    n_train_licit = (y_train == 0.0).sum().item()
    n_train_illicit = (y_train == 1.0).sum().item()
    
    if n_train_illicit == 0:
        raise ValueError("Zero illicit transactions found in training mask!")
        
    pos_weight_val = float(n_train_licit) / float(n_train_illicit)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_pr_auc = -1.0
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0
    last_val_loss = 0.0
    
    y_val_np = data.y[val_mask].cpu().numpy().astype(int)
    y_test_np = data.y[test_mask].cpu().numpy().astype(int)
    
    start_train_time = time.time()
    
    # 2. Optimized Training Loop with Validation-Interval Early Stopping
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation evaluation at interval or final epoch
        if (epoch % val_interval == 0) or (epoch == epochs):
            model.eval()
            with torch.no_grad():
                eval_logits = model(data.x, data.edge_index)
                last_val_loss = criterion(eval_logits[val_mask], data.y[val_mask]).item()
                val_probs = torch.sigmoid(eval_logits[val_mask]).cpu().numpy()
                
                # Fast PR-AUC computation
                prec_arr, rec_arr, _ = precision_recall_curve(y_val_np, val_probs)
                val_pr_auc = float(auc(rec_arr, prec_arr))
                
            if val_pr_auc > best_val_pr_auc + 1e-4:
                best_val_pr_auc = val_pr_auc
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += val_interval
                if epochs_no_improve >= patience:
                    logger.debug(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(Best Epoch: {best_epoch}, Best Val PR-AUC: {best_val_pr_auc:.4f})"
                    )
                    break
                    
    train_time = time.time() - start_train_time
    
    # 3. Restore Best Checkpoint for Threshold Optimization & Blind Test Evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        best_model_state = copy.deepcopy(model.state_dict())
        
    model.eval()
    start_inf_time = time.time()
    with torch.no_grad():
        final_logits = model(data.x, data.edge_index)
        val_probs_best = torch.sigmoid(final_logits[val_mask]).cpu().numpy()
        test_probs_best = torch.sigmoid(final_logits[test_mask]).cpu().numpy()
    inf_time = time.time() - start_inf_time
    
    # 4. Threshold Optimization (Locked strictly from Validation split)
    opt_tau, val_f1_opt, _ = find_optimal_threshold(
        y_true_val=y_val_np,
        y_prob_val=val_probs_best,
        metric="f1"
    )
    
    val_final_metrics = compute_classification_metrics(y_val_np, val_probs_best, threshold=opt_tau)
    test_final_metrics = compute_classification_metrics(y_test_np, test_probs_best, threshold=opt_tau)
    
    # Track peak memory
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 ** 2)
    
    result = {
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "num_params": num_params,
        "training_time_sec": train_time,
        "inference_time_sec": inf_time,
        "peak_ram_mb": peak_ram_mb,
        "device_used": str(device_obj),
        "pos_weight": pos_weight_val,
        "optimal_threshold": opt_tau,
        # Validation Metrics
        "val_illicit_f1": val_final_metrics["illicit_f1"],
        "val_pr_auc": val_final_metrics["pr_auc"],
        "val_illicit_precision": val_final_metrics["illicit_precision"],
        "val_illicit_recall": val_final_metrics["illicit_recall"],
        "val_roc_auc": val_final_metrics["roc_auc"],
        "val_loss": last_val_loss,
        # Test Metrics (Locked Threshold)
        "test_illicit_f1": test_final_metrics["illicit_f1"],
        "test_pr_auc": test_final_metrics["pr_auc"],
        "test_illicit_precision": test_final_metrics["illicit_precision"],
        "test_illicit_recall": test_final_metrics["illicit_recall"],
        "test_macro_f1": test_final_metrics["macro_f1"],
        "test_roc_auc": test_final_metrics["roc_auc"],
        "test_mcc": test_final_metrics["mcc"],
        "test_accuracy": test_final_metrics["accuracy"],
        "test_tp": test_final_metrics["tp"],
        "test_fp": test_final_metrics["fp"],
        "test_tn": test_final_metrics["tn"],
        "test_fn": test_final_metrics["fn"],
        "test_confusion_matrix": test_final_metrics["confusion_matrix"],
        "test_pr_curve": test_final_metrics["pr_curve"],
        "test_roc_curve": test_final_metrics["roc_curve"],
        "test_probabilities": test_probs_best,
        "test_predictions": (test_probs_best >= opt_tau).astype(int),
        "test_labels": y_test_np,
        "test_node_indices": np.where(test_mask.cpu().numpy())[0],
        "best_state_dict": best_model_state,
    }
    return result


def evaluate_saved_checkpoint(
    model: torch.nn.Module,
    data: Data,
    checkpoint_path: Path,
    device: Optional[str] = None
) -> Dict:
    """
    Loads a saved checkpoint and runs inference on validation and test masks
    with threshold optimization locked on the validation set.
    """
    if device is None:
        device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)
        
    model = model.to(device_obj)
    data = data.to(device_obj)
    
    state_dict = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.eval()
    
    val_mask = data.val_mask & data.labeled_mask
    test_mask = data.test_mask & data.labeled_mask
    y_val_np = data.y[val_mask].cpu().numpy()
    y_test_np = data.y[test_mask].cpu().numpy()
    
    start_inf = time.time()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
        test_probs = torch.sigmoid(logits[test_mask]).cpu().numpy()
    inf_time = time.time() - start_inf
    
    opt_tau, val_f1_opt, _ = find_optimal_threshold(
        y_true_val=y_val_np,
        y_prob_val=val_probs,
        metric="f1"
    )
    val_metrics = compute_classification_metrics(y_val_np, val_probs, threshold=opt_tau)
    test_metrics = compute_classification_metrics(y_test_np, test_probs, threshold=opt_tau)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    process = psutil.Process(os.getpid())
    peak_ram = process.memory_info().rss / (1024 ** 2)
    
    return {
        "num_params": num_params,
        "inference_time_sec": inf_time,
        "peak_ram_mb": peak_ram,
        "device_used": str(device_obj),
        "optimal_threshold": opt_tau,
        "val_illicit_f1": val_metrics["illicit_f1"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_illicit_precision": val_metrics["illicit_precision"],
        "val_illicit_recall": val_metrics["illicit_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_illicit_f1": test_metrics["illicit_f1"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_illicit_precision": test_metrics["illicit_precision"],
        "test_illicit_recall": test_metrics["illicit_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_mcc": test_metrics["mcc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_tp": test_metrics["tp"],
        "test_fp": test_metrics["fp"],
        "test_tn": test_metrics["tn"],
        "test_fn": test_metrics["fn"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_pr_curve": test_metrics["pr_curve"],
        "test_roc_curve": test_metrics["roc_curve"],
        "test_probabilities": test_probs,
        "test_predictions": (test_probs >= opt_tau).astype(int),
        "test_labels": y_test_np,
        "test_node_indices": np.where(test_mask.cpu().numpy())[0],
        "best_state_dict": state_dict,
    }
