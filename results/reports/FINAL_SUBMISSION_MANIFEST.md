# FINAL SUBMISSION MANIFEST

This manifest provides the definitive, comprehensive file-by-file accounting of all submission, source, data, figure, and verification artifacts for the research paper:

*“Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning”*

---

## 1. Master Publication Files

| File Path | Exists | Purpose / Description | Submission Required? | Notes |
|---|:---:|---|:---:|---|
| `results/latex/main.pdf` | **YES** | Camera-ready 10-page compiled PDF (2.13 MB) | **YES (Primary)** | Compiled with Tectonic 0.17.0 (XeTeX + Type 1 fonts) |
| `results/latex/main.tex` | **YES** | Master LaTeX manuscript source (47,474 bytes) | **YES (Primary)** | Complete two-column IEEE format |
| `results/latex/references.bib` | **YES** | Master BibTeX bibliography (34 entries, 13,338 bytes) | **YES (Primary)** | 100% cited & resolved |
| `results/IEEE_MANUSCRIPT.md` | **YES** | Authoritative IEEE-formatted Markdown manuscript | Supporting | Synchronized with LaTeX source |
| `results/FINAL_MANUSCRIPT.md` | **YES** | Authoritative assembled scientific manuscript (APA style) | Supporting | Master reference manuscript |
| `results/references.bib` | **YES** | Root BibTeX database (byte-identical to latex/references.bib) | Supporting | 13,338 bytes |

---

## 2. Primary & Supplementary Figure Assets (All 300 DPI)

| Figure ID | File Path | Exists | Resolution | Target Width | Submission Required? |
|---|---|:---:|:---:|:---:|:---:|
| **Fig. 1** | `results/figures/label_distribution.png` | **YES** | 3570x1320 (300 DPI) | Double-Column | **YES** |
| **Fig. 2** | `results/figures/temporal_evolution.png` | **YES** | 3570x1920 (300 DPI) | Double-Column | **YES** |
| **Fig. 3** | `results/figures/gnn_vs_conventional_baselines.png` | **YES** | 3270x1470 (300 DPI) | Double-Column | **YES** |
| **Fig. 4** | `results/figures/gnn_precision_recall_curves.png` | **YES** | 2370x1770 (300 DPI) | Single-Column | **YES** |
| **Fig. 5** | `results/figures/feature_ablation_local_vs_full.png` | **YES** | 2970x1470 (300 DPI) | Double-Column | **YES** |
| **Fig. 6** | `results/figures/gnn_model_comparison_f1.png` | **YES** | 2670x1470 (300 DPI) | Single-Column | **YES** |
| **Fig. 7** | `results/figures/unknown_node_ablation.png` | **YES** | 2370x1320 (300 DPI) | Single-Column | **YES** |
| **Fig. 8** | `results/figures/direction_sensitivity_ablation.png` | **YES** | 2520x1320 (300 DPI) | Single-Column | **YES** |
| **Fig. 9** | `results/figures/depth_ablation.png` | **YES** | 2370x1320 (300 DPI) | Single-Column | **YES** |
| **Fig. 10** | `results/figures/edge_homophily_matrix.png` | **YES** | 3271x1320 (300 DPI) | Double-Column | **YES** |
| **Fig. 11** | `results/figures/gnn_temporal_performance_timesteps.png` | **YES** | 2970x1470 (300 DPI) | Double-Column | **YES** |
| **Fig. S1** | `results/figures/baseline_model_comparison.png` | **YES** | 300 DPI | Double-Column | Optional / Archive |
| **Fig. S2** | `results/figures/gnn_roc_curves.png` | **YES** | 300 DPI | Single-Column | Optional / Archive |
| **Fig. S3** | `results/figures/gnn_compute_benchmark.png` | **YES** | 300 DPI | Single-Column | Optional / Archive |
| **Fig. S4** | `results/figures/illicit_ratio_over_time.png` | **YES** | 300 DPI | Double-Column | Optional / Archive |

---

## 3. Master Experimental Evidence Data Tables

| Table File Path | Exists | Description / Contents |
|---|:---:|---|
| `results/tables/MASTER_MAIN_RESULTS.csv` | **YES** | Authoritative 7-model test performance (F1, PR-AUC, Precision, Recall, MCC) |
| `results/tables/MASTER_FEATURE_COMPARISON.csv` | **YES** | Local 93 vs. Full 165 dimensional feature ablation |
| `results/tables/MASTER_WEIGHTING_COMPARISON.csv` | **YES** | Standard cross-entropy vs. cost-sensitive loss weighting comparison |
| `results/tables/MASTER_GNN_ABLATIONS.csv` | **YES** | Unknown-node context, edge directionality, and layer depth ablations |
| `results/tables/MASTER_SEED_STABILITY.csv` | **YES** | 5-seed initialization sensitivity, min/max/range F1 metrics |
| `results/tables/MASTER_STATISTICAL_TESTS.csv` | **YES** | Paired t-tests, Wilcoxon signed-rank tests, Holm-adjusted p-values, Cohen dz |
| `results/tables/MASTER_ERROR_ANALYSIS.csv` | **YES** | Error taxonomy, in/out degree distribution by class and prediction status |
| `results/tables/MASTER_EXPERIMENT_INVENTORY.csv` | **YES** | Comprehensive run inventory across all models and configurations |

---

## 4. Quality Control & Audit Reports Trail

| Report Path | Exists | Purpose / Scope | Status |
|---|:---:|---|:---:|
| `results/reports/STEP_5H_RESEARCH_INTEGRITY_AUDIT.md` | **YES** | Step 5H Final Research Integrity & Submission Audit | **READY** |
| `results/reports/STEP_5G_FINAL_COMPILATION_AUDIT.md` | **YES** | Step 5G Compilation & PDF Verification Audit | **PASS** |
| `results/reports/STEP_5F_REFERENCE_AUDIT.md` | **YES** | Step 5F Reference, Citation & Bibliography Audit | **PASS** |
| `results/reports/REFERENCE_INVENTORY.md` | **YES** | Step 5F 34-Row Master Reference Inventory | **PASS** |
| `results/reports/CITATION_COVERAGE_REPORT.md` | **YES** | Step 5F 28-Category Scholarly Claim Coverage | **PASS** |
| `results/reports/STEP_5E_VISUAL_AUDIT.md` | **YES** | Step 5E Visual Asset & Table Formatting Audit | **PASS** |
| `results/reports/FIGURE_TABLE_INVENTORY.md` | **YES** | Step 5E Master Figure/Table Inventory | **PASS** |
| `results/reports/STEP_5D_FORMATTING_AUDIT.md` | **YES** | Step 5D IEEE Formatting Audit | **PASS** |
| `results/reports/STEP_5C_ASSEMBLY_AUDIT.md` | **YES** | Step 5C Manuscript Assembly Audit | **PASS** |
| `results/reports/PAPER_FACT_SHEET.md` | **YES** | Master Scientific Fact Sheet & Evidence Pack | **PASS** |
