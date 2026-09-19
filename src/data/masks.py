"""
Temporal and Label Mask Generator for Graph Neural Network Experiments.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch


def create_node_masks(
    df_merged: pd.DataFrame,
    train_timesteps: Tuple[int, int] = (1, 34),
    val_timesteps: Tuple[int, int] = (35, 39),
    test_timesteps: Tuple[int, int] = (40, 49)
) -> Dict[str, torch.Tensor]:
    """
    Creates boolean tensor masks for training, validation, and testing.
    
    CRITICAL RULE:
    Unknown nodes (class == 'unknown') MUST ALWAYS have:
      train_mask = False
      val_mask = False
      test_mask = False
      loss_mask = False
      eval_mask = False
    
    Args:
        df_merged: DataFrame containing 'time_step' and 'class' columns.
        train_timesteps: (min_t, max_t) for training split.
        val_timesteps: (min_t, max_t) for validation split.
        test_timesteps: (min_t, max_t) for test split.
        
    Returns:
        Dict of PyTorch boolean tensors:
          - 'train_mask'
          - 'val_mask'
          - 'test_mask'
          - 'labeled_mask'
          - 'unknown_mask'
    """
    time_steps = df_merged["time_step"].values
    classes = df_merged["class"].astype(str).values
    
    is_labeled = (classes == "1") | (classes == "2")
    is_unknown = classes == "unknown"
    
    train_bool = (time_steps >= train_timesteps[0]) & (time_steps <= train_timesteps[1]) & is_labeled
    val_bool = (time_steps >= val_timesteps[0]) & (time_steps <= val_timesteps[1]) & is_labeled
    test_bool = (time_steps >= test_timesteps[0]) & (time_steps <= test_timesteps[1]) & is_labeled
    
    # Assert zero overlap
    assert not np.any(train_bool & val_bool), "Overlap detected between Train and Val masks!"
    assert not np.any(train_bool & test_bool), "Overlap detected between Train and Test masks!"
    assert not np.any(val_bool & test_bool), "Overlap detected between Val and Test masks!"
    assert not np.any(train_bool & is_unknown), "Unknown node included in Train mask!"
    assert not np.any(val_bool & is_unknown), "Unknown node included in Val mask!"
    assert not np.any(test_bool & is_unknown), "Unknown node included in Test mask!"

    return {
        "train_mask": torch.tensor(train_bool, dtype=torch.bool),
        "val_mask": torch.tensor(val_bool, dtype=torch.bool),
        "test_mask": torch.tensor(test_bool, dtype=torch.bool),
        "labeled_mask": torch.tensor(is_labeled, dtype=torch.bool),
        "unknown_mask": torch.tensor(is_unknown, dtype=torch.bool),
    }
