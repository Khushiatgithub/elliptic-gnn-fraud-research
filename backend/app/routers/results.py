"""
Results and Master Evaluation API Endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from backend.app.schemas.results import (
    AblationRecord,
    ErrorAnalysisRecord,
    ExperimentInventoryRecord,
    FeatureComparisonRecord,
    MainResultRecord,
    SeedStabilityRecord,
    StatisticalTestRecord,
    WeightingComparisonRecord,
)
from backend.app.services.results_service import ResultsService, get_results_service

router = APIRouter(prefix="/results", tags=["Results"])


@router.get("/main", response_model=List[MainResultRecord])
def get_main_results(service: ResultsService = Depends(get_results_service)):
    """Return primary model performance metrics dynamically from MASTER_MAIN_RESULTS.csv."""
    return service.get_main_results()


@router.get("/features", response_model=List[FeatureComparisonRecord])
def get_feature_comparison(service: ResultsService = Depends(get_results_service)):
    """Return Local (93) vs Full (165) feature comparison metrics."""
    return service.get_feature_comparison()


@router.get("/weighting", response_model=List[WeightingComparisonRecord])
def get_weighting_comparison(service: ResultsService = Depends(get_results_service)):
    """Return Standard vs Weighted loss ablation metrics."""
    return service.get_weighting_comparison()


@router.get("/ablations", response_model=List[AblationRecord])
def get_ablations(
    ablation_type: Optional[str] = Query(None, description="Filter by ablation domain (Unknown, Direction, Depth)"),
    service: ResultsService = Depends(get_results_service)
):
    """Return GNN ablations for Unknown Nodes, Edge Directionality, and Network Depth."""
    return service.get_ablations(ablation_type=ablation_type)


@router.get("/seeds", response_model=List[SeedStabilityRecord])
def get_seed_stability(
    model: Optional[str] = Query(None, description="Filter by model name"),
    service: ResultsService = Depends(get_results_service)
):
    """Return multi-seed stability evaluations across seeds 42, 123, 456, 789, 999."""
    return service.get_seed_stability(model_filter=model)


@router.get("/statistics", response_model=List[StatisticalTestRecord])
def get_statistical_tests(service: ResultsService = Depends(get_results_service)):
    """Return paired t-test, Wilcoxon signed-rank, Holm-adjusted, and Cohen's dz statistics."""
    return service.get_statistical_tests()


@router.get("/error-analysis", response_model=List[ErrorAnalysisRecord])
def get_error_analysis(service: ResultsService = Depends(get_results_service)):
    """Return test set error taxonomy, degree breakdowns, and timestep regime shift analysis."""
    return service.get_error_analysis()


@router.get("/inventory", response_model=List[ExperimentInventoryRecord])
def get_inventory(service: ResultsService = Depends(get_results_service)):
    """Return complete master inventory of all 112 evaluated experiments."""
    return service.get_inventory()
