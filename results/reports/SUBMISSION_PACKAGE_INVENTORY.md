# SUBMISSION PACKAGE INVENTORY

This inventory catalogs all authoritative manuscript, source code, graphic, data, and audit artifacts prepared for submission of the research paper:

*“Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning”*

---

## Master Artifact Manifest

| # | File Path | Purpose / Description | Status | Required for Submission? |
|---|---|---|---|:---:|
| 1 | `results/latex/main.tex` | Master LaTeX publication source | Verified & Compiled | **YES** |
| 2 | `results/latex/references.bib` | Master BibTeX reference database (34 entries) | Verified (100% sync) | **YES** |
| 3 | `results/latex/main.pdf` | Final compiled 10-page camera-ready publication PDF | Compiled (10 pages, 2.13 MB) | **YES** |
| 4 | `results/IEEE_MANUSCRIPT.md` | Authoritative IEEE-formatted Markdown manuscript | Fully Synchronized | Supporting |
| 5 | `results/FINAL_MANUSCRIPT.md` | Authoritative assembled scientific manuscript (APA style) | Fully Synchronized | Supporting |
| 6 | `results/references.bib` | Root reference database (byte-identical to latex/references.bib) | Verified (13,338 bytes) | Supporting |
| 7 | `results/figures/label_distribution.png` | Fig. 1: Class distribution across 49 timesteps (300 DPI) | Verified | **YES** |
| 8 | `results/figures/temporal_evolution.png` | Fig. 2: Transaction volume dynamics & class ratio (300 DPI) | Verified | **YES** |
| 9 | `results/figures/gnn_vs_conventional_baselines.png` | Fig. 3: Main benchmark test F1 & PR-AUC comparison (300 DPI) | Verified | **YES** |
| 10 | `results/figures/gnn_precision_recall_curves.png` | Fig. 4: Precision-Recall curves across GNN architectures (300 DPI) | Verified | **YES** |
| 11 | `results/figures/feature_ablation_local_vs_full.png` | Fig. 5: Local 93 vs. Full 165 feature ablation (300 DPI) | Verified | **YES** |
| 12 | `results/figures/gnn_model_comparison_f1.png` | Fig. 6: Cross-seed test F1 score distributions (300 DPI) | Verified | **YES** |
| 13 | `results/figures/unknown_node_ablation.png` | Fig. 7: Unknown-node context ablation (300 DPI) | Verified | **YES** |
| 14 | `results/figures/direction_sensitivity_ablation.png` | Fig. 8: Message-passing directionality ablation (300 DPI) | Verified | **YES** |
| 15 | `results/figures/depth_ablation.png` | Fig. 9: Message-passing layer depth ablation (300 DPI) | Verified | **YES** |
| 16 | `results/figures/edge_homophily_matrix.png` | Fig. 10: Relational edge transition & homophily matrix (300 DPI) | Verified | **YES** |
| 17 | `results/figures/gnn_temporal_performance_timesteps.png` | Fig. 11: Timestep-by-timestep test performance breakdown (300 DPI) | Verified | **YES** |
| 18 | `results/figures/baseline_model_comparison.png` | Fig. S1: Supplementary baseline model performance breakdown | Verified | Optional / Archive |
| 19 | `results/figures/gnn_roc_curves.png` | Fig. S2: Supplementary ROC curves across GNN models | Verified | Optional / Archive |
| 20 | `results/figures/gnn_compute_benchmark.png` | Fig. S3: Supplementary compute efficiency & training latency | Verified | Optional / Archive |
| 21 | `results/figures/illicit_ratio_over_time.png` | Fig. S4: Supplementary illicit transaction prevalence curve | Verified | Optional / Archive |
| 22 | `results/tables/MASTER_MAIN_RESULTS.csv` | Authoritative master benchmark numerical results | Verified | Data Archive |
| 23 | `results/tables/MASTER_FEATURE_COMPARISON.csv` | Authoritative feature ablation numerical results | Verified | Data Archive |
| 24 | `results/tables/MASTER_WEIGHTING_COMPARISON.csv` | Authoritative loss weighting numerical results | Verified | Data Archive |
| 25 | `results/tables/MASTER_GNN_ABLATIONS.csv` | Authoritative GNN structural ablation numerical results | Verified | Data Archive |
| 26 | `results/tables/MASTER_SEED_STABILITY.csv` | Authoritative multi-seed stability numerical results | Verified | Data Archive |
| 27 | `results/tables/MASTER_STATISTICAL_TESTS.csv` | Authoritative hypothesis testing & effect size results | Verified | Data Archive |
| 28 | `results/tables/MASTER_ERROR_ANALYSIS.csv` | Authoritative error taxonomy & degree distribution results | Verified | Data Archive |
| 29 | `results/reports/STEP_5G_FINAL_COMPILATION_AUDIT.md` | Step 5G final compilation & layout audit report | Verified (PASS) | Audit Trail |
| 30 | `results/reports/STEP_5F_REFERENCE_AUDIT.md` | Step 5F reference & bibliography validation report | Verified (PASS) | Audit Trail |
| 31 | `results/reports/REFERENCE_INVENTORY.md` | Complete 34-row scholarly reference inventory | Verified (PASS) | Audit Trail |
| 32 | `results/reports/CITATION_COVERAGE_REPORT.md` | Complete 28-category scholarly claim coverage report | Verified (PASS) | Audit Trail |
| 33 | `results/reports/FIGURE_TABLE_INVENTORY.md` | Complete publication figure and table master inventory | Verified (PASS) | Audit Trail |

---

## Verification Sign-Off

- **Manuscript LaTeX Compilation**: **PASS** (`main.pdf`, 10 pages, Type 1 fonts embedded).
- **Bibliography Integration**: **PASS** (34/34 references resolved, 0 `??`).
- **Figure Assets**: **PASS** (11 primary + 4 supplementary at 300 DPI).
- **Numerical Alignment**: **PASS** (Zero discrepancies across all tables and text).
- **Submission Readiness**: **APPROVED**.
