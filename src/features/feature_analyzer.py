"""
Statistical Profiling and Quality Analysis for Elliptic Features.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("feature_analyzer")


class FeatureAnalyzer:
    """
    Performs rigorous descriptive statistics, distribution profiling,
    collinearity analysis, variance screening, and outlier detection across
    the 165 elliptic feature dimensions (93 local + 72 aggregated).
    """

    def __init__(self, df_features: pd.DataFrame, num_local: int = 93, num_agg: int = 72):
        self.df_features = df_features
        self.num_local = num_local
        self.num_agg = num_agg
        
        self.local_cols = [c for c in df_features.columns if c.startswith("local_feat_")]
        self.agg_cols = [c for c in df_features.columns if c.startswith("agg_feat_")]
        self.all_feat_cols = self.local_cols + self.agg_cols

    def compute_summary_statistics(self) -> pd.DataFrame:
        """
        Compute mean, std, median, min, max, IQR, skewness, and variance
        for each feature.
        """
        logger.info(f"Computing summary statistics across {len(self.all_feat_cols)} features...")
        feat_df = self.df_features[self.all_feat_cols]
        
        desc = feat_df.describe().T
        desc["variance"] = feat_df.var(axis=0)
        desc["median"] = feat_df.median(axis=0)
        desc["iqr"] = desc["75%"] - desc["25%"]
        desc["skewness"] = feat_df.skew(axis=0)
        desc["feature_type"] = [
            "Local" if col.startswith("local_feat_") else "Aggregated"
            for col in desc.index
        ]
        
        # Reorder columns cleanly
        cols_order = [
            "feature_type", "count", "mean", "std", "variance",
            "min", "25%", "median", "75%", "max", "iqr", "skewness"
        ]
        return desc[cols_order]

    def detect_low_variance_features(self, threshold: float = 1e-5) -> Tuple[List[str], List[str]]:
        """
        Detect constant (var == 0) and near-zero variance features (var < threshold).
        """
        feat_df = self.df_features[self.all_feat_cols]
        variances = feat_df.var(axis=0)
        
        constant_features = list(variances[variances == 0.0].index)
        near_zero_variance = list(variances[(variances > 0.0) & (variances < threshold)].index)
        
        logger.info(
            f"Variance check: Constant features={len(constant_features)}, "
            f"Near-zero variance (<{threshold})={len(near_zero_variance)}"
        )
        return constant_features, near_zero_variance

    def detect_duplicate_features(self) -> List[Tuple[str, str]]:
        """
        Detect identical/duplicate feature columns via SHA-256 byte hashing.
        """
        import hashlib
        feat_df = self.df_features[self.all_feat_cols]
        hash_to_cols: Dict[str, List[str]] = {}
        for col in self.all_feat_cols:
            col_bytes = np.ascontiguousarray(feat_df[col].values).tobytes()
            col_hash = hashlib.sha256(col_bytes).hexdigest()
            hash_to_cols.setdefault(col_hash, []).append(col)
            
        duplicate_pairs = []
        for cols in hash_to_cols.values():
            if len(cols) > 1:
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        duplicate_pairs.append((cols[i], cols[j]))
                        
        logger.info(f"Duplicate feature pairs detected: {len(duplicate_pairs)}")
        return duplicate_pairs

    def compute_high_correlations(self, threshold: float = 0.95, sample_size: int = 50000) -> pd.DataFrame:
        """
        Compute highly correlated feature pairs (|r| >= threshold).
        Uses a representative stratified/random sample if full dataset is memory-intensive.
        """
        logger.info(f"Computing correlation matrix (threshold >= {threshold})...")
        if len(self.df_features) > sample_size:
            feat_sample = self.df_features[self.all_feat_cols].sample(n=sample_size, random_state=42)
        else:
            feat_sample = self.df_features[self.all_feat_cols]
            
        corr_matrix = feat_sample.corr(method="pearson").abs()
        
        # Extract upper triangle without diagonal
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corr_pairs = []
        
        feature_names = self.all_feat_cols
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                val = corr_matrix.iloc[i, j]
                if val >= threshold:
                    corr_pairs.append({
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "feature_1_type": "Local" if feature_names[i].startswith("local_feat_") else "Aggregated",
                        "feature_2_type": "Local" if feature_names[j].startswith("local_feat_") else "Aggregated",
                        "abs_pearson_corr": float(val)
                    })
                    
        df_high_corr = pd.DataFrame(corr_pairs)
        if not df_high_corr.empty:
            df_high_corr = df_high_corr.sort_values(by="abs_pearson_corr", ascending=False).reset_index(drop=True)
            
        logger.info(f"Identified {len(df_high_corr)} highly correlated feature pairs (|r| >= {threshold}).")
        return df_high_corr

    def analyze_outliers(self, iqr_multiplier: float = 1.5, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Quantify extreme values and outliers per feature dimension using IQR and Z-score methods.
        """
        logger.info("Quantifying feature outliers (IQR & Z-score)...")
        feat_df = self.df_features[self.all_feat_cols]
        
        outlier_records = []
        n_samples = len(feat_df)
        
        for col in self.all_feat_cols:
            vals = feat_df[col].values
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            
            # IQR bounds
            lower_bound = q25 - iqr_multiplier * iqr
            upper_bound = q75 + iqr_multiplier * iqr
            iqr_outliers = np.sum((vals < lower_bound) | (vals > upper_bound))
            
            # Z-score bounds
            mean = np.mean(vals)
            std = np.std(vals)
            z_outliers = np.sum(np.abs((vals - mean) / (std + 1e-9)) > z_threshold) if std > 0 else 0
            
            outlier_records.append({
                "feature": col,
                "feature_type": "Local" if col.startswith("local_feat_") else "Aggregated",
                "iqr_outlier_count": int(iqr_outliers),
                "iqr_outlier_pct": float((iqr_outliers / n_samples) * 100.0),
                "zscore_outlier_count": int(z_outliers),
                "zscore_outlier_pct": float((z_outliers / n_samples) * 100.0),
            })
            
        return pd.DataFrame(outlier_records)

    def export_summary_tables(self, tables_dir: Path) -> Dict[str, Path]:
        """Export all feature statistical summaries to CSV."""
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature descriptive statistics
        df_desc = self.compute_summary_statistics()
        desc_path = tables_dir / "feature_summary_statistics.csv"
        df_desc.to_csv(desc_path, index=True)
        
        # 2. High correlation pairs
        df_corr = self.compute_high_correlations(threshold=0.95)
        corr_path = tables_dir / "feature_high_correlations.csv"
        df_corr.to_csv(corr_path, index=False)
        
        # 3. Outlier statistics
        df_outliers = self.analyze_outliers()
        outlier_path = tables_dir / "feature_outlier_statistics.csv"
        df_outliers.to_csv(outlier_path, index=False)
        
        # 4. Low variance and constant features summary
        const_feats, near_zero_feats = self.detect_low_variance_features()
        dup_feats = self.detect_duplicate_features()
        df_quality = pd.DataFrame([
            {"Metric": "Total Features", "Count": len(self.all_feat_cols), "Description": "93 local + 72 aggregated features"},
            {"Metric": "Local Features", "Count": len(self.local_cols), "Description": "Transaction-specific local attributes"},
            {"Metric": "Aggregated Features", "Count": len(self.agg_cols), "Description": "1-hop neighborhood aggregated statistics"},
            {"Metric": "Constant Features (Var = 0)", "Count": len(const_feats), "Description": "Features with zero variance across all nodes"},
            {"Metric": "Near-Zero Variance (Var < 1e-5)", "Count": len(near_zero_feats), "Description": "Quasi-constant features"},
            {"Metric": "Exact Duplicate Feature Pairs", "Count": len(dup_feats), "Description": "Identical column pairs"},
            {"Metric": "Highly Collinear Pairs (|r| >= 0.95)", "Count": len(df_corr), "Description": "Pairs exhibiting extreme linear dependence"},
        ])
        quality_path = tables_dir / "feature_quality_summary.csv"
        df_quality.to_csv(quality_path, index=False)
        
        return {
            "statistics": desc_path,
            "correlations": corr_path,
            "outliers": outlier_path,
            "quality": quality_path
        }

    def generate_plots(self, figures_dir: Path, dpi: int = 300) -> None:
        """
        Generate publication-ready visualizations:
        1. Feature variance distribution (Local vs Aggregated).
        2. Correlation heatmap of representative feature subset.
        """
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # --- Plot 1: Feature Variance Spectrum (Log Scale) ---
        df_desc = self.compute_summary_statistics()
        fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
        
        local_vars = df_desc[df_desc["feature_type"] == "Local"]["variance"]
        agg_vars = df_desc[df_desc["feature_type"] == "Aggregated"]["variance"]
        
        ax.scatter(range(len(local_vars)), local_vars, color="#1f77b4", label="Local Features (93)", alpha=0.7, s=30)
        ax.scatter(range(len(local_vars), len(local_vars) + len(agg_vars)), agg_vars, color="#ff7f0e", label="Aggregated Features (72)", alpha=0.7, s=30)
        
        ax.set_title("Variance Spectrum across 165 Feature Dimensions (Log Scale)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Feature Index (0 to 164)", fontsize=11)
        ax.set_ylabel("Variance (log scale)", fontsize=11)
        ax.set_yscale("log")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Standard Variance (1.0)")
        ax.legend(loc="upper right", frameon=True)
        
        plt.tight_layout()
        plot1_path = figures_dir / "feature_variance_distribution.png"
        plt.savefig(plot1_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot1_path}")

        # --- Plot 2: Correlation Heatmap of Selected Salient Local & Aggregated Features ---
        sample_local = self.local_cols[:15]
        sample_agg = self.agg_cols[:15]
        selected_cols = sample_local + sample_agg
        
        corr_sub = self.df_features[selected_cols].sample(n=30000, random_state=42).corr()
        
        fig, ax = plt.subplots(figsize=(13, 11), dpi=dpi)
        sns.heatmap(
            corr_sub,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
            ax=ax,
            xticklabels=[f"L_{i}" if i < 15 else f"A_{i-15}" for i in range(30)],
            yticklabels=[f"L_{i}" if i < 15 else f"A_{i-15}" for i in range(30)],
        )
        ax.set_title("Pairwise Feature Correlation Matrix (Subset: 15 Local + 15 Aggregated)", fontsize=13, fontweight="bold")
        
        plt.tight_layout()
        plot2_path = figures_dir / "feature_correlation_heatmap.png"
        plt.savefig(plot2_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {plot2_path}")
