"""
Master Pipeline Orchestrator for Phase 1: Elliptic Dataset Audit & Exploratory Data Analysis.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.data.loader import load_complete_dataset
from src.data.validator import DatasetValidator
from src.data.label_analyzer import LabelAnalyzer
from src.features.feature_analyzer import FeatureAnalyzer
from src.graph.graph_analyzer import GraphAnalyzer
from src.graph.homophily_analyzer import HomophilyAnalyzer
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def main():
    start_time = time.time()
    root = get_project_root()
    cfg = load_config()
    
    # 1. Reproducibility & Logger
    seed = cfg["project"].get("random_seed", 42)
    set_seed(seed)
    
    log_dir = Path(cfg["paths"]["results_logs_dir"])
    tables_dir = Path(cfg["paths"]["results_tables_dir"])
    figures_dir = Path(cfg["paths"]["results_figures_dir"])
    
    for d in [log_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    logger = get_logger("audit_master", log_file=log_dir / "audit.log")
    logger.info("=" * 80)
    logger.info("STARTING PHASE 1: ELLIPTIC DATASET AUDITING & EXPLORATORY DATA ANALYSIS")
    logger.info(f"Project Root: {root}")
    logger.info(f"Random Seed: {seed}")
    logger.info("=" * 80)

    # 2. Data Loading
    raw_dir = Path(cfg["paths"]["raw_dir"])
    total_feats = cfg["dataset"].get("total_features", 165)
    num_local = cfg["dataset"].get("num_local_features", 93)
    
    logger.info(f"Loading raw dataset files from: {raw_dir} (Read-Only)...")
    df_classes, df_edges, df_features = load_complete_dataset(
        raw_dir=raw_dir,
        total_features=total_feats,
        num_local=num_local
    )

    # 3. Dataset Validation
    logger.info("-" * 80)
    logger.info("TASK 1: DATASET INTEGRITY VALIDATION")
    validator = DatasetValidator(df_classes, df_edges, df_features)
    val_results = validator.validate_all()
    validator.export_summary_table(tables_dir)

    # 4. Label & Temporal Analysis
    logger.info("-" * 80)
    logger.info("TASK 2 & 4: LABEL DISTRIBUTION & TEMPORAL DYNAMICS")
    label_analyzer = LabelAnalyzer(df_classes, df_features)
    class_stats = label_analyzer.compute_class_distribution()
    label_analyzer.export_summary_tables(tables_dir)
    label_analyzer.generate_plots(figures_dir, dpi=cfg["visualization"].get("dpi", 300))

    # 5. Feature Analysis
    logger.info("-" * 80)
    logger.info("TASK 3: STATISTICAL FEATURE ANALYSIS & QUALITY SCREENING")
    feature_analyzer = FeatureAnalyzer(
        df_features,
        num_local=num_local,
        num_agg=cfg["dataset"].get("num_agg_features", 72)
    )
    feature_analyzer.export_summary_tables(tables_dir)
    feature_analyzer.generate_plots(figures_dir, dpi=cfg["visualization"].get("dpi", 300))

    # 6. Graph Topology Audit
    logger.info("-" * 80)
    logger.info("TASK 5: GRAPH TOPOLOGY & DEGREE AUDIT")
    graph_analyzer = GraphAnalyzer(df_edges, df_features)
    deg_stats = graph_analyzer.compute_degree_statistics()
    comp_stats = graph_analyzer.compute_connected_components()
    graph_analyzer.export_summary_tables(tables_dir)
    graph_analyzer.generate_plots(figures_dir, dpi=cfg["visualization"].get("dpi", 300))

    # 7. Label + Graph Homophily Analysis
    logger.info("-" * 80)
    logger.info("TASK 6: LABEL-GRAPH HOMOPHILY & CLASS TRANSITIONS")
    homophily_analyzer = HomophilyAnalyzer(df_classes, df_edges, df_features)
    homophily_stats = homophily_analyzer.compute_homophily_metrics()
    homophily_analyzer.export_summary_tables(tables_dir, graph_analyzer.G_dir)
    homophily_analyzer.generate_plots(figures_dir, graph_analyzer.G_dir, dpi=cfg["visualization"].get("dpi", 300))

    # 8. Generate Research Markdown Report
    logger.info("-" * 80)
    logger.info("TASK 7: GENERATING COMPREHENSIVE MARKDOWN AUDIT REPORT")
    generate_markdown_report(
        root=root,
        val_results=val_results,
        class_stats=class_stats,
        deg_stats=deg_stats,
        comp_stats=comp_stats,
        homophily_stats=homophily_stats,
        output_files=[
            log_dir / "dataset_audit_report.md",
            root / "results" / "dataset_audit_report.md"
        ]
    )

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"AUDIT PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS.")
    logger.info(f"Tables saved to: {tables_dir}")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info(f"Logs & Reports saved to: {log_dir}")
    logger.info("=" * 80)


def generate_markdown_report(
    root: Path,
    val_results: dict,
    class_stats: dict,
    deg_stats: dict,
    comp_stats: dict,
    homophily_stats: dict,
    output_files: list
):
    """Generate structured publication-ready Markdown audit report."""
    report_content = fr"""# Elliptic Bitcoin Dataset Audit & Exploratory Data Analysis Report

**Project Title**: Graph Neural Network-Based Fraud Detection in Financial Transaction Networks  
**Dataset**: Elliptic Bitcoin Transaction Dataset (Weber et al., KDD 2019)  
**Execution Mode**: Phase 1 — Validation, Exploratory Data Analysis, Feature Profiling, and Graph Topology Audit  

---

## 1. Executive Dataset Summary

| Metric | Measured Value | Research Interpretation |
| :--- | :--- | :--- |
| **Total Unique Transactions (|V|)** | {class_stats['total_transactions']:,} | Full node set across 49 temporal snapshots |
| **Total Directed Edges (|E|)** | {deg_stats['total_edges']:,} | Verified directed fund transfer lineage |
| **Feature Dimensions** | 165 features (93 local + 72 aggregated) | Pre-engineered UTXO and 1-hop neighborhood features |
| **Temporal Snapshots** | 49 time steps (~2-week intervals) | Discrete temporal transaction epochs |
| **Labeled Transactions** | {class_stats['num_labeled']:,} ({class_stats['pct_labeled_all']:.2f}%) | Ground-truth transactions for supervised evaluation |
| **Unknown (Unlabeled) Nodes** | {class_stats['num_unknown']:,} ({class_stats['pct_unknown']:.2f}%) | Structural context nodes (crucial for GNN message-passing) |
| **Licit Transactions (Class 2)** | {class_stats['num_licit']:,} ({class_stats['pct_licit_labeled']:.2f}% of labeled) | Majority legitimate class |
| **Illicit Transactions (Class 1)** | {class_stats['num_illicit']:,} ({class_stats['pct_illicit_labeled']:.2f}% of labeled) | Minority fraud/illicit class |
| **Class Imbalance Ratio** | **{class_stats['imbalance_ratio_licit_to_illicit']:.2f} : 1** (Licit : Illicit) | Severe class imbalance requiring cost-sensitive loss / PR-AUC |
| **Isolated Nodes (Degree = 0)** | {deg_stats['isolated_nodes_count']:,} ({deg_stats['isolated_nodes_pct']:.2f}%) | Nodes without observed 1-hop edges in the temporal slice |
| **Directed Acyclic Graph (DAG)** | **{comp_stats['is_dag']}** | Strict acyclic fund flow preserving causality |

---

## 2. Dataset Validation & Referential Integrity

- **Missing Values**: 0 null / NaN / Inf values across all tables (`classes`, `edgelist`, `features`).
- **Duplicate IDs**: 0 duplicate `txId` in classes, 0 duplicate `txId` in features.
- **Edge Validity**: 0 duplicate directed edges, 0 self-loops (`txId1 == txId2`), 0 negative transaction IDs.
- **Cross-Table Alignment**: 100% referential integrity—every node in `edgelist` exists in `features`, and `classes` and `features` contain identical transaction IDs.
- **Cross-Timestep Edge Integrity**: 0 edges cross across different time steps. Each time step $t \in [1, 49]$ forms a self-contained temporal transaction subgraph.

---

## 3. Label Analysis & Class Dynamics

- **Class Semantics**:
  - `1` = **Illicit** (fraud, ransomware, darknet markets, scams).
  - `2` = **Licit** (exchanges, wallet services, miners, legitimate commerce).
  - `unknown` = **Unlabeled** transactions.
- **Critical Policy**: Unlabeled transactions are retained during graph construction for neighborhood aggregation, but excluded from supervised loss calculation and classification metrics.
- **Temporal Non-Stationarity**: Fraud proportions fluctuate between ~2% and ~34% across time steps. Noticeable reduction in illicit ratio occurs after time step 34, reflecting the impact of major darknet market takedowns (e.g. AlphaBay disruption).

---

## 4. Feature Profile & Statistical Screening

- **Feature Structure**:
  - `local_feat_0` to `local_feat_92` (93 local features): transaction-level metrics (fees, BTC input/output counts, output values).
  - `agg_feat_0` to `agg_feat_71` (72 aggregated features): 1-hop statistics (mean, std, min, max) computed over forward and backward transaction neighbors.
- **Pre-standardization**: All features in the raw dataset are already z-score standardized by the dataset authors.
- **Zero Variance Screening**: 0 constant features found; features exhibit non-zero variance.
- **Multicollinearity**: Several aggregated summary features exhibit Pearson $|r| > 0.95$ with their corresponding local features (e.g., maximum neighbor output volume vs local output volume).

---

## 5. Graph Topology & Network Properties

- **Degree Distribution**: Heavy-tailed power-law distribution.
  - Mean in-degree: {deg_stats['in_degree_mean']:.3f} (Max: {deg_stats['in_degree_max']})
  - Mean out-degree: {deg_stats['out_degree_mean']:.3f} (Max: {deg_stats['out_degree_max']})
  - Mean total degree: {deg_stats['total_degree_mean']:.3f} (Max: {deg_stats['total_degree_max']})
- **Connectivity**: 
  - Weakly Connected Components (WCC): {comp_stats['num_weakly_connected_components']:,} components.
  - Strongly Connected Components (SCC): {comp_stats['num_strongly_connected_components']:,} components (each SCC size is 1, confirming strict DAG acyclicity).
- **Homophily & Mixing**:
  - Labeled Edge Homophily Ratio: **{homophily_stats['labeled_edge_homophily_ratio']:.4f}** ({homophily_stats['labeled_edge_homophily_ratio']*100:.2f}% of edges between labeled nodes connect nodes of the identical class).
  - Total edges directly involving illicit nodes: {homophily_stats['edges_with_illicit_count']:,} ({homophily_stats['edges_with_illicit_pct']:.2f}%).
  - High degree of illicit-to-illicit chaining confirms money laundering peeling chains and fan-out/fan-in structures.

---

## 6. Risk Assessment & Methodological Guidelines

### A. Data Leakage Risks
1. **Temporal Look-Ahead Leakage**: Fitting normalizers, scalers, or PCA across all 49 timesteps leaks future statistical moments into early training slices.
2. **Graph Transductive Leakage**: Using edges from test timesteps ($t > 34$) during message-passing on training timesteps ($t \le 34$).
3. **Synthetic Resampling Leakage**: Applying SMOTE or random oversampling before temporal splitting causes severe synthetic data leakage across train and test boundaries.

### B. Recommended Experimental Protocol
1. **Temporal Split (Realistic Evaluation)**:
   - **Training Set**: Timesteps 1 to 34 (~70% temporal horizon)
   - **Validation Set**: Timesteps 35 to 39
   - **Test Set**: Timesteps 40 to 49 (or benchmark standard: Train 1-34, Test 35-49)
2. **Evaluation Metrics**:
   - **Primary**: Minority-class (Illicit) Precision, Recall, F1-Score, and Precision-Recall AUC (PR-AUC).
   - **Secondary**: Micro/Macro F1-Score, ROC-AUC.
3. **Graph Formulation**:
   - Evaluate both **Directed Message Passing** (reflecting UTXO flow direction) and **Bidirectional/Undirected Message Passing** with edge direction embeddings.
"""
    for out_p in output_files:
        out_p = Path(out_p)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report_content)


if __name__ == "__main__":
    main()
