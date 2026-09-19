"""
Prediction and Model Inference Pydantic Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model_id: str
    name: str
    family: str
    is_gnn: bool
    feature_mode: str
    optimal_threshold: float
    description: str
    available: bool


class PredictionRequest(BaseModel):
    tx_id: int = Field(..., description="Transaction ID to analyze")
    model_id: str = Field("random_forest", description="Model identifier")


class FeatureContribution(BaseModel):
    feature_name: str
    value: float
    importance_or_weight: float
    description: str


class PredictionResponse(BaseModel):
    tx_id: int
    timestep: int
    split: str
    ground_truth_label: str
    model_id: str
    model_name: str
    model_family: str
    is_gnn: bool
    predicted_probability: float
    classification_threshold: float
    predicted_class: int  # 1 for illicit, 0 for licit
    predicted_label: str  # "Model prediction: Illicit" or "Model prediction: Licit"
    confidence_score: float
    in_degree: int
    out_degree: int
    neighborhood_size_1hop: int
    feature_contributions: Optional[List[FeatureContribution]] = None
    graph_context: Optional[Dict[str, Any]] = None
