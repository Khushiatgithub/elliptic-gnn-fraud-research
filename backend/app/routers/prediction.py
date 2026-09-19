"""
Prediction and Inference API Endpoints.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.schemas.prediction import ModelInfo, PredictionRequest, PredictionResponse
from backend.app.services.inference_service import InferenceService, get_inference_service

router = APIRouter(tags=["Prediction"])


@router.get("/models", response_model=List[ModelInfo])
@router.get("/models/available", response_model=List[ModelInfo])
def get_available_models(service: InferenceService = Depends(get_inference_service)):
    """Return catalog of available trained models and their hyperparameters."""
    return service.get_available_models()


@router.post("/predict", response_model=PredictionResponse)
def predict_transaction(
    request: PredictionRequest,
    service: InferenceService = Depends(get_inference_service)
):
    """
    Execute live, authentic model prediction on transaction ID using selected model.
    Applies exact training preprocessing and returns probability, decision, feature importances, and subgraph context.
    """
    try:
        response = service.predict(tx_id=request.tx_id, model_id=request.model_id)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
