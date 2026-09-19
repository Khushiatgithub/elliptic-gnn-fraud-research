"""
Graph Service for Fast In-Memory Adjacency and K-Hop Subgraph Extraction.
"""

from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from backend.app.config import settings
from backend.app.schemas.graph import GraphEdge, GraphNode, SubgraphResponse
from backend.app.services.dataset_service import DatasetService, get_dataset_service
from src.data.loader import load_edgelist


class GraphService:
    _instance: Optional["GraphService"] = None

    def __init__(self, dataset_service: Optional[DatasetService] = None):
        self.edgelist_path = settings.DATA_RAW_DIR / "elliptic_txs_edgelist.csv"
        self.dataset_service = dataset_service or get_dataset_service()
        
        self.adj_out: Dict[int, List[int]] = {}  # txId1 -> list of txId2
        self.adj_in: Dict[int, List[int]] = {}   # txId2 -> list of txId1
        self._is_loaded = False

    def load_graph(self):
        if self._is_loaded:
            return

        print("[GraphService] Loading edgelist...")
        df_edges = load_edgelist(self.edgelist_path)
        
        for row in df_edges.itertuples(index=False):
            u, v = int(row.txId1), int(row.txId2)
            
            if u not in self.adj_out:
                self.adj_out[u] = []
            self.adj_out[u].append(v)
            
            if v not in self.adj_in:
                self.adj_in[v] = []
            self.adj_in[v].append(u)
            
        self._is_loaded = True
        print(f"[GraphService] Loaded {len(df_edges)} edges across {len(set(self.adj_out.keys()) | set(self.adj_in.keys()))} connected nodes.")

    def get_in_degree(self, tx_id: int) -> int:
        self.load_graph()
        return len(self.adj_in.get(tx_id, []))

    def get_out_degree(self, tx_id: int) -> int:
        self.load_graph()
        return len(self.adj_out.get(tx_id, []))

    def get_total_degree(self, tx_id: int) -> int:
        return self.get_in_degree(tx_id) + self.get_out_degree(tx_id)

    def extract_k_hop_subgraph(
        self,
        center_tx_id: int,
        k_hop: int = 1,
        max_nodes: int = 80
    ) -> SubgraphResponse:
        self.load_graph()
        self.dataset_service.load_data()
        
        if center_tx_id not in self.dataset_service.tx_to_row:
            return SubgraphResponse(
                center_tx_id=center_tx_id,
                k_hop=k_hop,
                total_nodes=0,
                total_edges=0,
                nodes=[],
                edges=[],
                center_node=None
            )

        visited_hops: Dict[int, int] = {center_tx_id: 0}
        queue = deque([(center_tx_id, 0)])
        
        while queue and len(visited_hops) < max_nodes:
            curr_tx, curr_hop = queue.popleft()
            if curr_hop >= k_hop:
                continue
                
            neighbors = set(self.adj_out.get(curr_tx, [])) | set(self.adj_in.get(curr_tx, []))
            for nbr in neighbors:
                if nbr not in visited_hops:
                    visited_hops[nbr] = curr_hop + 1
                    queue.append((nbr, curr_hop + 1))
                    if len(visited_hops) >= max_nodes:
                        break

        # Collect node details
        nodes: List[GraphNode] = []
        center_node: Optional[GraphNode] = None
        
        for tx, hop in visited_hops.items():
            label = self.dataset_service.class_map.get(tx, "unknown")
            ts = self.dataset_service.timestep_map.get(tx, 1)
            
            if 1 <= ts <= 34:
                split = "train"
            elif 35 <= ts <= 39:
                split = "val"
            else:
                split = "test"
                
            in_deg = self.get_in_degree(tx)
            out_deg = self.get_out_degree(tx)
            is_center = (tx == center_tx_id)
            
            node_obj = GraphNode(
                id=str(tx),
                tx_id=tx,
                label=label,
                timestep=ts,
                split=split,
                in_degree=in_deg,
                out_degree=out_deg,
                total_degree=in_deg + out_deg,
                is_center=is_center,
                hop=hop
            )
            nodes.append(node_obj)
            if is_center:
                center_node = node_obj

        # Collect induced edges between visited nodes
        node_set = set(visited_hops.keys())
        edges: List[GraphEdge] = []
        edge_id = 0
        
        for u in node_set:
            for v in self.adj_out.get(u, []):
                if v in node_set:
                    edges.append(GraphEdge(
                        id=f"e_{u}_{v}_{edge_id}",
                        source=str(u),
                        target=str(v),
                        source_tx_id=u,
                        target_tx_id=v
                    ))
                    edge_id += 1

        return SubgraphResponse(
            center_tx_id=center_tx_id,
            k_hop=k_hop,
            total_nodes=len(nodes),
            total_edges=len(edges),
            nodes=nodes,
            edges=edges,
            center_node=center_node
        )


@lru_cache()
def get_graph_service() -> GraphService:
    service = GraphService()
    service.load_graph()
    return service
