"""
Dataset and Transaction Pydantic Schemas.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    total_transactions: int = Field(203769, description="Total node count")
    total_edges: int = Field(234355, description="Total directed edge count")
    total_timesteps: int = Field(49, description="Number of distinct discrete timesteps")
    total_features: int = Field(165, description="Total ML features")
    local_features: int = Field(93, description="Local timestep-level features")
    aggregated_features: int = Field(72, description="Aggregated 1-hop neighborhood features")
    illicit_count: int = Field(4545, description="Class 1: Illicit transactions")
    licit_count: int = Field(42019, description="Class 2: Licit transactions")
    unknown_count: int = Field(157205, description="Unlabeled/Unknown transactions")
    labeled_count: int = Field(46564, description="Licit + Illicit labeled transactions")
    illicit_labeled_ratio: float = Field(..., description="Illicit / Labeled proportion")
    train_timesteps: str = Field("1–34", description="Train chronological partition")
    val_timesteps: str = Field("35–39", description="Validation chronological partition")
    test_timesteps: str = Field("40–49", description="Test chronological partition")
    train_count: int = Field(29894, description="Labeled transactions in train split")
    train_illicit_count: int = Field(3462, description="Illicit transactions in train split")
    val_count: int = Field(5486, description="Labeled transactions in validation split")
    val_illicit_count: int = Field(447, description="Illicit transactions in validation split")
    test_count: int = Field(11184, description="Labeled transactions in test split")
    test_illicit_count: int = Field(636, description="Illicit transactions in test split")


class TimestepInfo(BaseModel):
    timestep: int
    total_txs: int
    licit_txs: int
    illicit_txs: int
    unknown_txs: int
    labeled_txs: int
    illicit_ratio: float
    split: str  # "train", "val", "test"


class TransactionDetail(BaseModel):
    tx_id: int
    timestep: int
    split: str
    ground_truth_label: str  # "illicit", "licit", "unknown"
    in_degree: int
    out_degree: int
    total_degree: int
    features_local: Dict[str, float]
    features_agg: Dict[str, float]


class SampleTransaction(BaseModel):
    tx_id: int
    timestep: int
    split: str
    label: str
    description: str
