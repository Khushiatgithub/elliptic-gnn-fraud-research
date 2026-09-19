"""
Graph and Neighborhood Pydantic Schemas.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    tx_id: int
    label: str  # "licit", "illicit", "unknown"
    timestep: int
    split: str  # "train", "val", "test"
    in_degree: int
    out_degree: int
    total_degree: int
    is_center: bool = False
    hop: int = 0
    x: Optional[float] = None
    y: Optional[float] = None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    source_tx_id: int
    target_tx_id: int


class SubgraphResponse(BaseModel):
    center_tx_id: int
    k_hop: int
    total_nodes: int
    total_edges: int
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    center_node: Optional[GraphNode] = None
