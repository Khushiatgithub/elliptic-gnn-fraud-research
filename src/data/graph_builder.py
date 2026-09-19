"""
PyTorch Geometric Graph Builder for Elliptic Bitcoin Dataset.
Supports Directional, Bidirectional, and Unknown-Node Ablation Graphs.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from src.data.masks import create_node_masks
from src.utils.logger import get_logger

logger = get_logger("graph_builder")


def build_pyg_graph(
    df_features: pd.DataFrame,
    df_classes: pd.DataFrame,
    df_edgelist: pd.DataFrame,
    feature_mode: str = "full",
    direction: str = "forward",
    remove_unknown: bool = False,
    scale: bool = True,
    train_timesteps: Tuple[int, int] = (1, 34),
    val_timesteps: Tuple[int, int] = (35, 39),
    test_timesteps: Tuple[int, int] = (40, 49),
) -> Data:
    """
    Constructs a PyTorch Geometric Data object with strict anti-leakage scaling,
    explicit directional edges, and temporal masks.
    
    Args:
        df_features: DataFrame containing 'txId', 'time_step', and 165 feature columns.
        df_classes: DataFrame containing 'txId' and 'class'.
        df_edgelist: DataFrame containing 'txId1' and 'txId2'.
        feature_mode: 'local' (93 features) or 'full' (165 features).
        direction: 'forward' (src->dst), 'backward' (dst->src), or 'bidirectional' (both).
        remove_unknown: If True, unknown nodes and their incident edges are removed (Ablation).
        scale: If True, features are standardized with StandardScaler fit ONLY on train split.
        train_timesteps: (min_t, max_t) for training split.
        val_timesteps: (min_t, max_t) for validation split.
        test_timesteps: (min_t, max_t) for test split.
        
    Returns:
        PyG Data object.
    """
    logger.info(
        f"Building PyG Graph: feature_mode='{feature_mode}', direction='{direction}', "
        f"remove_unknown={remove_unknown}, scale={scale}"
    )
    
    # 1. Merge Features and Classes
    df_merged = pd.merge(df_features, df_classes, on="txId", how="inner")
    if len(df_merged) != len(df_features):
        raise ValueError(f"Feature and Class merge mismatch: {len(df_merged)} vs {len(df_features)}")
        
    # 2. Filter Unknown Nodes if requested (Unknown Node Ablation)
    if remove_unknown:
        initial_nodes = len(df_merged)
        df_merged = df_merged[df_merged["class"].isin(["1", "2"])].copy().reset_index(drop=True)
        logger.info(f"Removed unknown nodes: {initial_nodes} -> {len(df_merged)} labeled nodes.")
        
    # 3. Create Node ID to Contiguous Index Mapping
    node_tx_ids = df_merged["txId"].values
    tx_to_idx: Dict[int, int] = {tx: idx for idx, tx in enumerate(node_tx_ids)}
    num_nodes = len(df_merged)
    
    # 4. Generate Node Masks
    masks = create_node_masks(
        df_merged,
        train_timesteps=train_timesteps,
        val_timesteps=val_timesteps,
        test_timesteps=test_timesteps
    )
    train_mask_np = masks["train_mask"].numpy()
    
    # 5. Extract and Scale Features (Zero Data Leakage: Scaler fit ONLY on train_mask)
    if feature_mode == "local":
        feature_cols = [c for c in df_merged.columns if c.startswith("local_feat_")]
    else:
        feature_cols = [c for c in df_merged.columns if c.startswith("local_feat_") or c.startswith("agg_feat_")]
        
    raw_x = df_merged[feature_cols].values.astype(np.float32)
    
    if scale:
        scaler = StandardScaler()
        # Fit scaler ONLY on training nodes
        train_features = raw_x[train_mask_np]
        if len(train_features) == 0:
            raise ValueError("No training samples found to fit StandardScaler!")
        scaler.fit(train_features)
        scaled_x = scaler.transform(raw_x)
        x_tensor = torch.tensor(scaled_x, dtype=torch.float32)
    else:
        x_tensor = torch.tensor(raw_x, dtype=torch.float32)
        
    # 6. Map Labels
    # '1' (illicit) -> 1.0, '2' (licit) -> 0.0, 'unknown' -> -1.0
    class_str = df_merged["class"].astype(str).values
    y_arr = np.full(num_nodes, -1.0, dtype=np.float32)
    y_arr[class_str == "1"] = 1.0
    y_arr[class_str == "2"] = 0.0
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)
    
    # 7. Build Edge Tensor
    # Filter edgelist to valid nodes present in df_merged
    valid_tx_set = set(tx_to_idx.keys())
    edge_df = df_edgelist[
        df_edgelist["txId1"].isin(valid_tx_set) & df_edgelist["txId2"].isin(valid_tx_set)
    ]
    
    src_nodes = [tx_to_idx[tx] for tx in edge_df["txId1"]]
    dst_nodes = [tx_to_idx[tx] for tx in edge_df["txId2"]]
    
    if direction == "forward":
        src_t = torch.tensor(src_nodes, dtype=torch.long)
        dst_t = torch.tensor(dst_nodes, dtype=torch.long)
        edge_index = torch.stack([src_t, dst_t], dim=0)
    elif direction == "backward":
        src_t = torch.tensor(src_nodes, dtype=torch.long)
        dst_t = torch.tensor(dst_nodes, dtype=torch.long)
        edge_index = torch.stack([dst_t, src_t], dim=0)
    elif direction == "bidirectional":
        src_t = torch.tensor(src_nodes, dtype=torch.long)
        dst_t = torch.tensor(dst_nodes, dtype=torch.long)
        forward_edges = torch.stack([src_t, dst_t], dim=0)
        backward_edges = torch.stack([dst_t, src_t], dim=0)
        raw_edges = torch.cat([forward_edges, backward_edges], dim=1)
        edge_index = coalesce(raw_edges, num_nodes=num_nodes)
    else:
        raise ValueError(f"Invalid graph direction: '{direction}'. Expected 'forward', 'backward', or 'bidirectional'.")
        
    time_steps_tensor = torch.tensor(df_merged["time_step"].values, dtype=torch.long)
    tx_ids_tensor = torch.tensor(node_tx_ids, dtype=torch.long)
    
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor,
        train_mask=masks["train_mask"],
        val_mask=masks["val_mask"],
        test_mask=masks["test_mask"],
        labeled_mask=masks["labeled_mask"],
        unknown_mask=masks["unknown_mask"],
        time_step=time_steps_tensor,
        tx_id=tx_ids_tensor,
        num_nodes=num_nodes
    )
    
    logger.info(
        f"Graph Constructed Successfully: Nodes={data.num_nodes}, Edges={data.num_edges}, "
        f"Features={data.num_features}, TrainMask={data.train_mask.sum().item()}, "
        f"ValMask={data.val_mask.sum().item()}, TestMask={data.test_mask.sum().item()}"
    )
    return data
