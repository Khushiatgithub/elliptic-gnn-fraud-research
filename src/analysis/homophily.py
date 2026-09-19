"""
Homophily and Relational Neighborhood Consistency Analyzer for Elliptic Graph.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils.logger import get_logger

logger = get_logger("homophily_analysis")


def compute_graph_homophily(
    data: Data,
    output_dir: Optional[Path] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Computes labeled edge homophily, class transition distribution, and unknown edge connectivity.
    
    Args:
        data: PyG Data object containing edge_index, y, and labeled_mask.
        output_dir: Optional directory to save homophily tables.
        
    Returns:
        Tuple of (homophily_dict, homophily_dataframe).
    """
    edge_index = data.edge_index.cpu().numpy()
    y = data.y.cpu().numpy()
    labeled_mask = data.labeled_mask.cpu().numpy()
    
    src = edge_index[0]
    dst = edge_index[1]
    total_edges = len(src)
    
    src_labeled = labeled_mask[src]
    dst_labeled = labeled_mask[dst]
    both_labeled = src_labeled & dst_labeled
    
    labeled_src = src[both_labeled]
    labeled_dst = dst[both_labeled]
    num_labeled_edges = len(labeled_src)
    
    # Extract labels for both-labeled edges (0.0 = Licit, 1.0 = Illicit)
    src_y = y[labeled_src]
    dst_y = y[labeled_dst]
    
    ll_edges = np.sum((src_y == 0.0) & (dst_y == 0.0))
    ii_edges = np.sum((src_y == 1.0) & (dst_y == 1.0))
    li_edges = np.sum((src_y == 0.0) & (dst_y == 1.0))
    il_edges = np.sum((src_y == 1.0) & (dst_y == 0.0))
    same_class_edges = ll_edges + ii_edges
    cross_class_edges = li_edges + il_edges
    
    edge_homophily = float(same_class_edges) / float(num_labeled_edges) if num_labeled_edges > 0 else 0.0
    unknown_incident_edges = total_edges - num_labeled_edges
    unknown_edge_ratio = float(unknown_incident_edges) / float(total_edges) if total_edges > 0 else 0.0
    
    homophily_dict = {
        "total_graph_edges": total_edges,
        "labeled_edges_count": num_labeled_edges,
        "unknown_incident_edges": unknown_incident_edges,
        "unknown_incident_edge_pct": unknown_edge_ratio * 100.0,
        "same_class_edges": same_class_edges,
        "cross_class_edges": cross_class_edges,
        "overall_labeled_homophily": edge_homophily,
        "licit_to_licit_edges": int(ll_edges),
        "licit_to_licit_pct": float(ll_edges) / num_labeled_edges * 100.0 if num_labeled_edges > 0 else 0.0,
        "illicit_to_illicit_edges": int(ii_edges),
        "illicit_to_illicit_pct": float(ii_edges) / num_labeled_edges * 100.0 if num_labeled_edges > 0 else 0.0,
        "licit_to_illicit_edges": int(li_edges),
        "licit_to_illicit_pct": float(li_edges) / num_labeled_edges * 100.0 if num_labeled_edges > 0 else 0.0,
        "illicit_to_licit_edges": int(il_edges),
        "illicit_to_licit_pct": float(il_edges) / num_labeled_edges * 100.0 if num_labeled_edges > 0 else 0.0,
    }
    
    df_homophily = pd.DataFrame([
        {"Metric": "Total Directed Edges (|E|)", "Value": f"{total_edges:,}"},
        {"Metric": "Edges with Both Endpoints Labeled", "Value": f"{num_labeled_edges:,} ({num_labeled_edges / total_edges * 100:.2f}%)"},
        {"Metric": "Edges Incident to Unknown Nodes", "Value": f"{unknown_incident_edges:,} ({unknown_edge_ratio * 100:.2f}%)"},
        {"Metric": "Overall Labeled Edge Homophily", "Value": f"{edge_homophily * 100:.2f}%"},
        {"Metric": "Licit → Licit Edges", "Value": f"{ll_edges:,} ({float(ll_edges) / num_labeled_edges * 100:.2f}%)"},
        {"Metric": "Illicit → Illicit Edges", "Value": f"{ii_edges:,} ({float(ii_edges) / num_labeled_edges * 100:.2f}%)"},
        {"Metric": "Licit → Illicit (Cross-Class)", "Value": f"{li_edges:,} ({float(li_edges) / num_labeled_edges * 100:.2f}%)"},
        {"Metric": "Illicit → Licit (Cross-Class)", "Value": f"{il_edges:,} ({float(il_edges) / num_labeled_edges * 100:.2f}%)"},
    ])
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_homophily.to_csv(output_dir / "gnn_homophily_analysis.csv", index=False)
        df_homophily.to_csv(output_dir / "homophily_analysis.csv", index=False)
        logger.info(f"Homophily table exported to: {output_dir / 'gnn_homophily_analysis.csv'}")
        
    return homophily_dict, df_homophily
