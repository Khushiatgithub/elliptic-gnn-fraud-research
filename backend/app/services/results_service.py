"""
Results Service to dynamically read, parse, and serve master experimental tables.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.app.config import settings
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


class ResultsService:
    def __init__(self):
        self.tables_dir = settings.TABLES_DIR

    def get_main_results(self) -> List[MainResultRecord]:
        path = self.tables_dir / "MASTER_MAIN_RESULTS.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        records = []
        for _, r in df.iterrows():
            records.append(MainResultRecord(
                model=str(r["Model"]),
                family=str(r["Family"]),
                features=str(r["Features"]),
                weighting=str(r["Weighting"]),
                val_f1_mean=float(r["Validation F1 Mean"]),
                val_f1_std=float(r["Validation F1 Std"]),
                val_prauc_mean=float(r["Validation PR-AUC Mean"]),
                val_prauc_std=float(r["Validation PR-AUC Std"]),
                test_f1_mean=float(r["Test F1 Mean"]),
                test_f1_std=float(r["Test F1 Std"]),
                test_prauc_mean=float(r["Test PR-AUC Mean"]),
                test_prauc_std=float(r["Test PR-AUC Std"]),
                test_precision_mean=float(r["Test Precision Mean"]),
                test_precision_std=float(r["Test Precision Std"]),
                test_recall_mean=float(r["Test Recall Mean"]),
                test_recall_std=float(r["Test Recall Std"]),
                test_mcc_mean=float(r["Test MCC Mean"]),
                test_mcc_std=float(r["Test MCC Std"]),
                seeds=str(r["Seeds"])
            ))
        return records

    def get_feature_comparison(self) -> List[FeatureComparisonRecord]:
        path = self.tables_dir / "MASTER_FEATURE_COMPARISON.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        records = []
        for _, r in df.iterrows():
            records.append(FeatureComparisonRecord(
                model=str(r["Model"]),
                feature_space=str(r["Feature Space"]),
                feature_count=int(r["Feature Count"]),
                val_f1=str(r["Validation F1"]),
                val_prauc=str(r["Validation PR-AUC"]),
                test_f1=str(r["Test F1"]),
                test_prauc=str(r["Test PR-AUC"]),
                test_mcc=str(r["Test MCC"]),
                delta_f1=str(r["Δ F1 (Full - Local)"]),
                delta_prauc=str(r["Δ PR-AUC (Full - Local)"]),
                delta_mcc=str(r["Δ MCC (Full - Local)"])
            ))
        return records

    def get_weighting_comparison(self) -> List[WeightingComparisonRecord]:
        path = self.tables_dir / "MASTER_WEIGHTING_COMPARISON.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        records = []
        for _, r in df.iterrows():
            records.append(WeightingComparisonRecord(
                model=str(r["Model"]),
                feature_space=str(r["Feature Space"]),
                weighting=str(r["Weighting"]),
                val_f1=str(r["Validation F1"]),
                val_prauc=str(r["Validation PR-AUC"]),
                test_f1=str(r["Test F1"]),
                test_prauc=str(r["Test PR-AUC"]),
                test_precision=str(r["Test Precision"]),
                test_recall=str(r["Test Recall"]),
                test_mcc=str(r["Test MCC"]),
                delta_f1=str(r["Δ (Weighted - Standard) F1"]),
                delta_prauc=str(r["Δ (Weighted - Standard) PR-AUC"]),
                delta_precision=str(r["Δ (Weighted - Standard) Precision"]),
                delta_recall=str(r["Δ (Weighted - Standard) Recall"]),
                delta_mcc=str(r["Δ (Weighted - Standard) MCC"])
            ))
        return records

    def get_ablations(self, ablation_type: Optional[str] = None) -> List[AblationRecord]:
        path = self.tables_dir / "MASTER_GNN_ABLATIONS.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        if ablation_type:
            df = df[df["Ablation"].str.contains(ablation_type, case=False, na=False)]
            
        records = []
        for _, r in df.iterrows():
            records.append(AblationRecord(
                ablation=str(r["Ablation"]),
                model=str(r["Model"]),
                setting=str(r["Setting"]),
                test_f1_formatted=str(r["Test F1 Formatted"]),
                test_prauc_formatted=str(r["Test PR-AUC Formatted"]),
                test_precision_formatted=str(r["Test Precision Formatted"]),
                test_recall_formatted=str(r["Test Recall Formatted"]),
                test_mcc_formatted=str(r["Test MCC Formatted"]),
                test_f1_mean=float(r["Test F1 Mean"]),
                test_f1_std=float(r["Test F1 Std"]),
                test_prauc_mean=float(r["Test PR-AUC Mean"]),
                test_prauc_std=float(r["Test PR-AUC Std"]),
                interpretation=str(r["Interpretation"])
            ))
        return records

    def get_seed_stability(self, model_filter: Optional[str] = None) -> List[SeedStabilityRecord]:
        path = self.tables_dir / "MASTER_SEED_STABILITY.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        if model_filter:
            df = df[df["Model"].str.contains(model_filter, case=False, na=False)]
            
        records = []
        for _, r in df.iterrows():
            records.append(SeedStabilityRecord(
                model=str(r["Model"]),
                phase=str(r["Phase"]),
                seed=int(r["Seed"]),
                val_f1=float(r["Validation F1"]),
                val_prauc=float(r["Validation PR-AUC"]),
                test_f1=float(r["Test F1"]),
                test_prauc=float(r["Test PR-AUC"]),
                test_mcc=float(r["Test MCC"]),
                test_f1_mean=float(r["Test F1 Mean"]),
                test_f1_std=float(r["Test F1 Std"]),
                test_f1_min=float(r["Test F1 Min"]),
                test_f1_max=float(r["Test F1 Max"]),
                test_f1_range=float(r["Test F1 Range"]),
                test_prauc_mean=float(r["Test PR-AUC Mean"]),
                test_prauc_std=float(r["Test PR-AUC Std"]),
                test_prauc_range=float(r["Test PR-AUC Range"])
            ))
        return records

    def get_statistical_tests(self) -> List[StatisticalTestRecord]:
        path = self.tables_dir / "MASTER_STATISTICAL_TESTS.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        records = []
        for _, r in df.iterrows():
            records.append(StatisticalTestRecord(
                model_a=str(r["Model A (GNN)"]),
                model_b=str(r["Model B (Baseline)"]),
                metric=str(r["Metric"]),
                gnn_mean_std=str(r["GNN Mean ± Std"]),
                baseline_mean_std=str(r["Baseline Mean ± Std"]),
                mean_difference=float(r["Mean Difference (Δ)"]),
                paired_t_statistic=float(r["Paired t-statistic"]),
                paired_t_test_p_raw=float(r["Paired t-test p (Raw)"]),
                holm_adjusted_p=float(r["Holm-Adjusted t-test p"]),
                wilcoxon_w_statistic=float(r["Wilcoxon W-statistic"]),
                wilcoxon_p_value=float(r["Wilcoxon p-value"]),
                paired_cohen_dz=float(r["Paired Cohen dz"]),
                sample_size_n=int(r["Sample Size N"]),
                notes=str(r["Notes"])
            ))
        return records

    def get_error_analysis(self) -> List[ErrorAnalysisRecord]:
        path = self.tables_dir / "MASTER_ERROR_ANALYSIS.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        records = []
        for _, r in df.iterrows():
            records.append(ErrorAnalysisRecord(
                domain=str(r["Analysis_Domain"]),
                category_or_timestep=str(r["Category_or_Timestep"]),
                sample_count=str(r["Sample_Count"]),
                proportion_or_rate=str(r["Proportion_or_Rate"]),
                mean_in_degree=str(r["Mean_In_Degree"]),
                mean_out_degree=str(r["Mean_Out_Degree"]),
                precision=str(r["Precision"]),
                recall=str(r["Recall"]),
                illicit_f1=str(r["Illicit_F1"]),
                key_observation=str(r["Key_Observation"])
            ))
        return records

    def get_inventory(self) -> List[ExperimentInventoryRecord]:
        path = self.tables_dir / "MASTER_EXPERIMENT_INVENTORY.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        
        records = []
        for _, r in df.iterrows():
            records.append(ExperimentInventoryRecord(
                experiment_id=str(r["Experiment_ID"]),
                phase=str(r["Phase"]),
                model_family=str(r["Model_Family"]),
                model=str(r["Model"]),
                feature_space=str(r["Feature_Space"]),
                feature_count=int(r["Feature_Count"]),
                weighting=str(r["Weighting"]),
                graph_context=str(r["Graph_Context"]),
                graph_direction=str(r["Graph_Direction"]),
                depth=str(r["Depth"]),
                seed=int(r["Seed"]),
                train_timesteps=str(r["Train_Timesteps"]),
                val_timesteps=str(r["Val_Timesteps"]),
                test_timesteps=str(r["Test_Timesteps"]),
                val_f1=float(r["Validation_F1"]),
                val_prauc=float(r["Validation_PR_AUC"]),
                test_f1=float(r["Test_F1"]),
                test_prauc=float(r["Test_PR_AUC"]),
                test_precision=float(r["Test_Precision"]),
                test_recall=float(r["Test_Recall"]),
                test_mcc=float(r["Test_MCC"]),
                status=str(r["Status"]),
                section=str(r["Recommended_Paper_Section"]),
                notes=str(r["Notes"])
            ))
        return records


@lru_cache()
def get_results_service() -> ResultsService:
    return ResultsService()
