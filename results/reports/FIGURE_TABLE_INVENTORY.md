# Final Publication Figure and Table Master Inventory

**Document ID:** `results/reports/FIGURE_TABLE_INVENTORY.md`  
**Target Manuscript:** `results/IEEE_MANUSCRIPT.md` and `results/latex/main.tex`  
**Date:** September 18, 2026  
**Auditor:** Autonomous Scientific Verification System  
**Audit Standard:** IEEE Transactions / Conference Formatting Guidelines  
**Status:** **PASS** (100% Verified, Zero Mismatches, Zero Broken References)

---

## 1. Complete Figure Master Inventory

All figures are stored in `results/figures/` and backed by reproducible deterministic Python generation scripts in `results/figures/scripts/`.

```
+---------------------------------------------------------------------------------------------------------------------------------------------+
| FIGURE MASTER INVENTORY TABLE                                                                                                               |
+---------------+---------------------------------+-----------------------------------+-------------------+---------------+---------+---------+
| Figure Number | Filename                        | Source Data / Script              | Primary Metric    | Target Width  | DPI     | Status  |
+---------------+---------------------------------+-----------------------------------+-------------------+---------------+---------+---------+
| **Fig. 1**    | label_distribution.png          | temporal_distribution_by_ts.csv   | Transaction Count | Double-Column | 300 DPI | PASSED  |
| **Fig. 2**    | temporal_evolution.png          | temporal_distribution_by_ts.csv   | Volume & Ratio (%)| Double-Column | 300 DPI | PASSED  |
| **Fig. 3**    | gnn_vs_conventional_baseline.png| MASTER_MAIN_RESULTS.csv           | Illicit F1 & PR   | Double-Column | 300 DPI | PASSED  |
| **Fig. 4**    | gnn_precision_recall_curves.png | Model evaluation logs             | Precision / Recall| Single-Column | 300 DPI | PASSED  |
| **Fig. 5**    | feature_ablation_local_vs_full  | MASTER_FEATURE_COMPARISON.csv     | Illicit F1 & ΔF1  | Double-Column | 300 DPI | PASSED  |
| **Fig. 6**    | unknown_node_ablation.png       | MASTER_GNN_ABLATIONS.csv          | Illicit F1 (Mean) | Single-Column | 300 DPI | PASSED  |
| **Fig. 7**    | direction_sensitivity_ablation  | MASTER_GNN_ABLATIONS.csv          | Illicit F1 (Mean) | Single-Column | 300 DPI | PASSED  |
| **Fig. 8**    | depth_ablation.png              | MASTER_GNN_ABLATIONS.csv          | Illicit F1 (Mean) | Single-Column | 300 DPI | PASSED  |
| **Fig. 9**    | gnn_model_comparison_f1.png     | MASTER_SEED_STABILITY.csv         | Cross-Seed F1     | Single-Column | 300 DPI | PASSED  |
| **Fig. 10**   | edge_homophily_matrix.png       | edge_class_transition_matrix.csv  | Homophily (95.4%) | Double-Column | 300 DPI | PASSED  |
| **Fig. 11**   | gnn_temporal_performance_tim... | MASTER_ERROR_ANALYSIS.csv         | Timestep F1 / Prec| Double-Column | 300 DPI | PASSED  |
| **Fig. S1**   | baseline_model_comparison.png   | baseline_summary_mean_std.csv     | Baseline F1 / PR  | Double-Column | 300 DPI | PASSED  |
| **Fig. S2**   | gnn_roc_curves.png              | Model evaluation logs             | ROC-AUC / FPR-TPR | Single-Column | 300 DPI | PASSED  |
| **Fig. S3**   | gnn_compute_benchmark.png       | gnn_compute_benchmark.csv         | Latency vs F1     | Single-Column | 300 DPI | PASSED  |
| **Fig. S4**   | illicit_ratio_over_time.png     | temporal_distribution_by_ts.csv   | Illicit Ratio (%) | Double-Column | 300 DPI | PASSED  |
+---------------+---------------------------------+-----------------------------------+-------------------+---------------+---------+---------+
```

---

## 2. Complete Table Master Inventory

All 13 tables are integrated into the manuscript with standardized precision and self-contained headers.

```
+---------------------------------------------------------------------------------------------------------------------------------------------+
| TABLE MASTER INVENTORY TABLE                                                                                                                |
+---------------+-------------------------------------------------------+-----------------------------------+---------------+---------+---------+
| Table Number  | Table Title / Scientific Purpose                      | Authoritative Source Table        | Target Width  | Format  | Status  |
+---------------+-------------------------------------------------------+-----------------------------------+---------------+---------+---------+
| **TABLE I**   | Elliptic Dataset Structure & Temporal Split Summary   | dataset_validation_audit.csv      | Double (t*)   | Booktabs| PASSED  |
| **TABLE II**  | Model Architecture & Hyperparameter Specification     | Methodological Specification      | Single (t)    | Booktabs| PASSED  |
| **TABLE III** | Main Benchmark Results on Full Features (165 Dim)     | MASTER_MAIN_RESULTS.csv           | Double (t*)   | Booktabs| PASSED  |
| **TABLE IV**  | Feature Representation Comparison (Local vs. Full)    | MASTER_FEATURE_COMPARISON.csv     | Single (t)    | Booktabs| PASSED  |
| **TABLE V**   | Class Loss Weighting Comparison (Full 165 Features)   | MASTER_WEIGHTING_COMPARISON.csv   | Single (t)    | Booktabs| PASSED  |
| **TABLE VI**  | Multi-Seed Stability and Parameter Initialization     | MASTER_SEED_STABILITY.csv         | Single (t)    | Booktabs| PASSED  |
| **TABLE VII** | Unknown-Node Context Ablation (165 Full Features)     | MASTER_GNN_ABLATIONS.csv          | Single (t)    | Booktabs| PASSED  |
| **TABLE VIII**| Graph Directionality Ablation (165 Full Features)     | MASTER_GNN_ABLATIONS.csv          | Single (t)    | Booktabs| PASSED  |
| **TABLE IX**  | GNN Depth Ablation (1, 2, 3 Layers, 165 Features)     | MASTER_GNN_ABLATIONS.csv          | Single (t)    | Booktabs| PASSED  |
| **TABLE X**   | Paired Statistical Significance Tests (N=5 Seeds)     | MASTER_STATISTICAL_TESTS.csv      | Double (t*)   | Booktabs| PASSED  |
| **TABLE XI**  | Error and Graph-Structural Taxonomy (Test Set, N=11k) | MASTER_ERROR_ANALYSIS.csv         | Single (t)    | Booktabs| PASSED  |
| **TABLE XII** | Research Question Answer Matrix                       | RQ Synthesis (PAPER_STEP_4F.md)   | Double (t*)   | Booktabs| PASSED  |
| **TABLE XIII**| Final Claim-Boundary Audit Checklist                  | Epistemic Audit (FINAL_MANUSCRIPT)| Double (t*)   | Booktabs| PASSED  |
+---------------+-------------------------------------------------------+-----------------------------------+---------------+---------+---------+
```

---

## 3. Inventory Sign-Off

- Total Figures in Main Manuscript: **11**
- Total Supplementary Figures: **4**
- Total Master Tables: **13**
- Unreferenced Figures: **0**
- Unreferenced Tables: **0**
- Overall Master Inventory Status: **APPROVED**
