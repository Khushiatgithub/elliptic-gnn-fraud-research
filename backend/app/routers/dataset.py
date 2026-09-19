"""
Dataset API Endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.schemas.dataset import DatasetSummary, SampleTransaction, TimestepInfo, TransactionDetail
from backend.app.services.dataset_service import DatasetService, get_dataset_service
from backend.app.services.graph_service import GraphService, get_graph_service

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/summary", response_model=DatasetSummary)
def get_summary(service: DatasetService = Depends(get_dataset_service)):
    """Return dataset overview metrics and temporal partition summaries."""
    return service.get_summary()


@router.get("/timesteps", response_model=List[TimestepInfo])
def get_timesteps(service: DatasetService = Depends(get_dataset_service)):
    """Return per-timestep class distributions and split allocations."""
    return service.get_timesteps()


@router.get("/sample-transactions", response_model=List[SampleTransaction])
def get_sample_transactions(service: DatasetService = Depends(get_dataset_service)):
    """Return curated sample transaction IDs representing various classes and temporal regimes."""
    return service.get_sample_transactions()


@router.get("/transaction/{tx_id}", response_model=TransactionDetail)
def get_transaction(
    tx_id: int,
    dataset_service: DatasetService = Depends(get_dataset_service),
    graph_service: GraphService = Depends(get_graph_service)
):
    """Return full transaction details, split, label, degree, and 165 feature values."""
    in_deg = graph_service.get_in_degree(tx_id)
    out_deg = graph_service.get_out_degree(tx_id)
    detail = dataset_service.get_transaction_detail(tx_id, in_degree=in_deg, out_degree=out_deg)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Transaction ID {tx_id} not found in Elliptic dataset.")
    return detail
