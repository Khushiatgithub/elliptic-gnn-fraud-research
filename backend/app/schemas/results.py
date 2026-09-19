"""
Research Results Pydantic Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MainResultRecord(BaseModel):
    model: str
    family: str
    features: str
    weighting: str
    val_f1_mean: float
    val_f1_std: float
    val_prauc_mean: float
    val_prauc_std: float
    test_f1_mean: float
    test_f1_std: float
    test_prauc_mean: float
    test_prauc_std: float
    test_precision_mean: float
    test_precision_std: float
    test_recall_mean: float
    test_recall_std: float
    test_mcc_mean: float
    test_mcc_std: float
    seeds: str


class FeatureComparisonRecord(BaseModel):
    model: str
    feature_space: str
    feature_count: int
    val_f1: str
    val_prauc: str
    test_f1: str
    test_prauc: str
    test_mcc: str
    delta_f1: str
    delta_prauc: str
    delta_mcc: str


class WeightingComparisonRecord(BaseModel):
    model: str
    feature_space: str
    weighting: str
    val_f1: str
    val_prauc: str
    test_f1: str
    test_prauc: str
    test_precision: str
    test_recall: str
    test_mcc: str
    delta_f1: str
    delta_prauc: str
    delta_precision: str
    delta_recall: str
    delta_mcc: str


class AblationRecord(BaseModel):
    ablation: str
    model: str
    setting: str
    test_f1_formatted: str
    test_prauc_formatted: str
    test_precision_formatted: str
    test_recall_formatted: str
    test_mcc_formatted: str
    test_f1_mean: float
    test_f1_std: float
    test_prauc_mean: float
    test_prauc_std: float
    interpretation: str


class SeedStabilityRecord(BaseModel):
    model: str
    phase: str
    seed: int
    val_f1: float
    val_prauc: float
    test_f1: float
    test_prauc: float
    test_mcc: float
    test_f1_mean: float
    test_f1_std: float
    test_f1_min: float
    test_f1_max: float
    test_f1_range: float
    test_prauc_mean: float
    test_prauc_std: float
    test_prauc_range: float


class StatisticalTestRecord(BaseModel):
    model_a: str
    model_b: str
    metric: str
    gnn_mean_std: str
    baseline_mean_std: str
    mean_difference: float
    paired_t_statistic: float
    paired_t_test_p_raw: float
    holm_adjusted_p: float
    wilcoxon_w_statistic: float
    wilcoxon_p_value: float
    paired_cohen_dz: float
    sample_size_n: int
    notes: str


class ErrorAnalysisRecord(BaseModel):
    domain: str
    category_or_timestep: str
    sample_count: str
    proportion_or_rate: str
    mean_in_degree: str
    mean_out_degree: str
    precision: str
    recall: str
    illicit_f1: str
    key_observation: str


class ExperimentInventoryRecord(BaseModel):
    experiment_id: str
    phase: str
    model_family: str
    model: str
    feature_space: str
    feature_count: int
    weighting: str
    graph_context: str
    graph_direction: str
    depth: str
    seed: int
    train_timesteps: str
    val_timesteps: str
    test_timesteps: str
    val_f1: float
    val_prauc: float
    test_f1: float
    test_prauc: float
    test_precision: float
    test_recall: float
    test_mcc: float
    status: str
    section: str
    notes: str
