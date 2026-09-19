"""
Graph API Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.schemas.graph import SubgraphResponse
from backend.app.services.graph_service import GraphService, get_graph_service

router = APIRouter(tags=["Graph"])


@router.get("/graph/neighborhood", response_model=SubgraphResponse)
@router.get("/transaction/{tx_id}/neighbors", response_model=SubgraphResponse)
def get_neighborhood(
    tx_id: int,
    k_hop: int = Query(1, ge=1, le=2, description="Hop radius (1 or 2)"),
    max_nodes: int = Query(80, ge=1, le=150, description="Max node cutoff"),
    service: GraphService = Depends(get_graph_service)
):
    """
    Extract localized 1-hop or 2-hop transaction graph neighborhood around tx_id.
    Safely limits rendered subgraph nodes to avoid DOM explosion.
    """
    subgraph = service.extract_k_hop_subgraph(center_tx_id=tx_id, k_hop=k_hop, max_nodes=max_nodes)
    if subgraph.total_nodes == 0:
        raise HTTPException(status_code=404, detail=f"Transaction ID {tx_id} not found in Elliptic dataset.")
    return subgraph
