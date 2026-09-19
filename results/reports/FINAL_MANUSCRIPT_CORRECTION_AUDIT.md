# Final Scientific Correction Audit: Research Manuscript

**Project:** *Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning*  
**Authoritative LaTeX Source:** [`results/latex/main.tex`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main.tex)  
**Primary PDF Target:** [`results/latex/main.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main.pdf) (Fresh build also at [`results/latex/main_corrected.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main_corrected.pdf))  
**Audit Timestamp:** 2026-09-19T17:59:00+05:30  
**Audit Status:** **PASS (100% Verified against Code & Master CSVs)**

---

## A. Corrections Made

| # | Item Category | Prior Description | Corrected Description | Authoritative Source |
|---|---|---|---|---|
| 1 | **Random Forest Hyperparameters** | Unlimited depth, `min_samples_split=2` | `n_estimators=100`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2` | `configs/baseline_config.yaml` |
| 2 | **MLP Hyperparameters** | 2 hidden layers of 128 units, dropout 0.3, weight decay $10^{-5}$ | `hidden_dims=[128, 64]`, `dropout=0.2`, Adam `lr=0.001`, `weight_decay=1e-4`, `batch_size=256`, `max_epochs=40`, `early_stopping_patience=7` | `configs/baseline_config.yaml` |
| 3 | **GAT Architecture** | 4 heads (32 channels/head) | `hidden_dim=128`, `heads=8`, 16 channels per head in multi-head layer ($8 \times 16 = 128$), 2 message-passing layers, `dropout=0.3`, Adam `lr=0.001`, `weight_decay=1e-4` | `src/models/gat.py`, `configs/gnn_config.yaml` |
| 4 | **GNN Model Selection & Early Stopping** | Monitored validation illicit F1 | Early stopping monitors validation **PR-AUC**; restores checkpoint with highest validation PR-AUC; tunes decision threshold on validation **illicit F1** ($\tau^* = \arg\max \text{F1}_{\text{val}}$); freezes threshold for blind test evaluation | `src/training/gnn_trainer.py` |
| 5 | **Primary GNN Depth Wording** | "All GNN models employ 2 message-passing layers" | "The primary GNN benchmark configuration uses two message-passing layers for all three architectures; a separate depth ablation evaluates one-, two-, and three-layer variants." | `results/latex/main.tex` |
| 6 | **Timestep 43 Non-Stationarity** | Asserted causal darknet market shutdowns/takedowns | Replaced with scientifically safe non-stationarity wording: *"The sharp change in illicit prevalence around timestep 43 coincides with a substantial reduction in per-timestep illicit prevalence ($11.10\% \to 1.75\%$) and a corresponding degradation in model F1 (dropping from $0.5234$ to $0.0718$). Because the dataset provides anonymized discrete timesteps rather than verified event annotations, the cause of this regime change cannot be established from the benchmark alone."* | Manuscript methodology and limitations |
| 7 | **Error Analysis Model Identification** | Generic test set taxonomy label | Explicitly labeled: *"Error taxonomy for the validation-selected GraphSAGE model using the Local 93-feature configuration."* | `src/analysis/error_analysis.py`, `results/tables/error_analysis.csv` |
| 8 | **Table VII Delta Sign Convention** | Mixed sign representations | Explicitly defined $\Delta\text{F1} = \text{Retained} - \text{Removed}$:<br>• GAT: $+0.0390$ (Retained Higher)<br>• GCN: $-0.1966$ (Removed Higher)<br>• GraphSAGE: $-0.1310$ (Removed Higher) | `results/tables/gnn_ablation_unknown_context.csv` |
| 9 | **Terminology Standardization** | "Fraudulent entities" / "fraud detection" | Standardized to "illicit transaction detection", "illicit transaction", "licit transaction", "unknown/unlabeled transaction" | Task specification |
| 10 | **Production Claims Qualification** | Broad claims of production readiness | Qualified: *"In this evaluated benchmark, well-tuned tree ensembles combined with 1-hop pre-aggregated features demonstrated strong detection capability... establishing production readiness requires additional evaluation of latency, calibration, drift, and operational constraints."* | Scientific boundary guidelines |
| 11 | **Author Block** | Preserved exact metadata | `Author: Khushi`, `Department of Artificial Intelligence and Data Science`, `Indira Gandhi Delhi Technical University for Women`, `khushi086btcseai23@igdtuw.ac.in` | Authoritative manuscript metadata |

---

## B. Numerical Verification against Master CSVs

All values reported in the manuscript match the authoritative result CSVs:

| Metric / Table | Model / Configuration | CSV Value | LaTeX Text Value | Verification Status |
|---|---|---|---|---|
| **Table III (Main Results)** | Random Forest (165, Std) | Val F1: $0.9489 \pm 0.0023$, Test F1: $0.7247 \pm 0.0027$, PR-AUC: $0.6599 \pm 0.0025$ | Val F1: $0.9489$, Test F1: $0.7247$, PR-AUC: $0.6599$ | **MATCH (PASS)** |
| **Table III (Main Results)** | XGBoost (165, Std) | Val F1: $0.9399 \pm 0.0053$, Test F1: $0.7131 \pm 0.0111$, PR-AUC: $0.6744 \pm 0.0028$ | Val F1: $0.9399$, Test F1: $0.7131$, PR-AUC: $0.6744$ | **MATCH (PASS)** |
| **Table III (Main Results)** | MLP (165, Std) | Val F1: $0.8212 \pm 0.0344$, Test F1: $0.5848 \pm 0.0194$, PR-AUC: $0.5115 \pm 0.0187$ | Val F1: $0.8212$, Test F1: $0.5848$, PR-AUC: $0.5115$ | **MATCH (PASS)** |
| **Table III (Main Results)** | GAT (165, Weighted) | Val F1: $0.6006 \pm 0.0982$, Test F1: $0.3308 \pm 0.0602$, PR-AUC: $0.2667 \pm 0.0695$ | Val F1: $0.6006$, Test F1: $0.3308$, PR-AUC: $0.2667$ | **MATCH (PASS)** |
| **Table III (Main Results)** | GraphSAGE (165, Weighted) | Val F1: $0.4473 \pm 0.1590$, Test F1: $0.2822 \pm 0.1248$, PR-AUC: $0.2366 \pm 0.1386$ | Val F1: $0.4473$, Test F1: $0.2822$, PR-AUC: $0.2366$ | **MATCH (PASS)** |
| **Table III (Main Results)** | GCN (165, Weighted) | Val F1: $0.4293 \pm 0.0923$, Test F1: $0.2589 \pm 0.0639$, PR-AUC: $0.2202 \pm 0.0703$ | Val F1: $0.4293$, Test F1: $0.2589$, PR-AUC: $0.2202$ | **MATCH (PASS)** |
| **Table IV (Feature Comp.)** | GraphSAGE (Local 93) | Test F1: $0.4153 \pm 0.1914$, PR-AUC: $0.3659 \pm 0.1859$ | Test F1: $0.4153$, PR-AUC: $0.3659$ | **MATCH (PASS)** |
| **Table VII (Unknown Ablation)** | GAT (Retained vs Removed) | Retained: $0.3308$, Removed: $0.2918$, $\Delta\text{F1} = +0.0390$ | Retained: $0.3308$, Removed: $0.2918$, $\Delta\text{F1} = +0.0390$ | **MATCH (PASS)** |
| **Table VII (Unknown Ablation)** | GCN (Retained vs Removed) | Retained: $0.2589$, Removed: $0.4555$, $\Delta\text{F1} = -0.1966$ | Retained: $0.2589$, Removed: $0.4555$, $\Delta\text{F1} = -0.1966$ | **MATCH (PASS)** |
| **Table VII (Unknown Ablation)** | GraphSAGE (Retained vs Removed)| Retained: $0.2822$, Removed: $0.4132$, $\Delta\text{F1} = -0.1310$ | Retained: $0.2822$, Removed: $0.4132$, $\Delta\text{F1} = -0.1310$ | **MATCH (PASS)** |
| **Table X (Statistical Tests)** | GraphSAGE vs. RF | Raw $t$ $p = 0.0229$, Holm $p = 0.0917$, Wilcoxon $p = 0.0625$ | Raw $t$ $p = 0.0229$, Holm $p = 0.0917$, Wilcoxon $p = 0.0625$ | **MATCH (PASS)** |
| **Table XI (Error Taxonomy)** | GraphSAGE Local 93 (TP/TN/FP/FN)| TP: $402$, TN: $8,165$, FP: $2,383$, FN: $234$ | TP: $402$, TN: $8,165$, FP: $2,383$, FN: $234$ | **MATCH (PASS)** |

---

## C. Hyperparameter Verification

- **Random Forest:** `n_estimators=100`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2` (Config file: [`configs/baseline_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/baseline_config.yaml#L29-L34)).
- **MLP:** `hidden_dims=[128, 64]`, `dropout=0.2`, `learning_rate=0.001`, `weight_decay=1e-4`, `batch_size=256`, `epochs=40`, `early_stopping_patience=7` (Config file: [`configs/baseline_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/baseline_config.yaml#L45-L53)).
- **GAT:** `hidden_dim=128`, `heads=8`, `head_dim=16`, `num_layers=2`, `dropout=0.3`, `lr=0.001`, `weight_decay=1e-4` (Implementation: [`src/models/gat.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/models/gat.py#L12-L59)).
- **GCN & GraphSAGE:** `hidden_dim=128`, `num_layers=2`, `dropout=0.3`, `lr=0.001`, `weight_decay=1e-4` (Config file: [`configs/gnn_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/gnn_config.yaml#L14-L44)).

---

## D. Claim-Boundary Verification

- **Timestep 43:** All external causal assertions regarding law enforcement actions or specific market shutdowns have been removed. The text strictly documents the empirical reduction in prevalence ($11.10\% \to 1.75\%$) and model degradation ($0.5234 \to 0.0718$) while explicitly noting that the anonymized dataset cannot establish external causality.
- **Production Claims:** Claims of immediate production deployment readiness have been replaced with benchmark-scoped performance descriptions accompanied by qualifications regarding latency, calibration, temporal drift, and data streaming requirements.

---

## E. Table Consistency Verification

- **Table I (Splits):** Train $t \in [1, 34]$ ($29,894$ labeled), Val $t \in [35, 39]$ ($5,486$ labeled), Test $t \in [40, 49]$ ($11,184$ labeled).
- **Table II (Model Specs):** Accurate hyperparameters for all 7 models.
- **Table III (Main Results):** Correct Mean $\pm$ Std across 5 seeds for all 7 models.
- **Table IV (Feature Ablation):** Local 93 vs. Full 165 $\Delta\text{F1}$ and $\Delta\text{PR-AUC}$.
- **Table V (Loss Weighting):** Standard vs. Cost-sensitive weighted cross-entropy.
- **Table VI (Seed Stability):** Minimum, Maximum, Range, and Mean $\pm$ Std across $N=5$ seeds.
- **Table VII (Unknown Context):** $\Delta\text{F1} = \text{Retained} - \text{Removed}$ (+0.0390 for GAT, -0.1966 for GCN, -0.1310 for GraphSAGE).
- **Table VIII (Directionality):** Forward, Backward, Bidirectional comparisons.
- **Table IX (Depth):** 1, 2, 3 layer evaluations.
- **Table X (Statistical Tests):** Raw paired $t$, Holm-Bonferroni, Wilcoxon signed-rank, Cohen's $d_z$.
- **Table XI (Error Taxonomy):** Explicitly attributed to validation-selected GraphSAGE (Local 93).
- **Table XII (RQ Matrix):** Comprehensive summary across RQ1–RQ4.

---

## F. Reference Verification

- **Bibliography File:** [`results/latex/references.bib`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/references.bib) contains 24 complete, verified entries.
- All citations referenced in `main.tex` resolve with zero missing citations in the generated `.bbl`.

---

## G. PDF Compilation Result

- **Compilation Tool:** Tectonic TeX Engine (`scratch/bin/tectonic.exe`)
- **Compilation Status:** **Exit Code 0 (Success)**
- **Output PDF:** [`scratch/pdf_out/main.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/scratch/pdf_out/main.pdf) (2,238,062 bytes, 11 pages)
- **Direct Copy:** [`results/latex/main_corrected.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main_corrected.pdf)
- *Note:* If [`results/latex/main.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main.pdf) was locked by an active external viewer during compilation, close the viewer and copy from `scratch/pdf_out/main.pdf` or access `results/latex/main_corrected.pdf`.

---

## H. Remaining Manual Items

1. **Author Name & Affiliation Confirmation:**
   - Manuscript author block: `Khushi`, `Department of Artificial Intelligence and Data Science`, `Indira Gandhi Delhi Technical University for Women`, `khushi086btcseai23@igdtuw.ac.in`.
   - Verified present and unmodified.
2. **Camera-Ready Visual Check:**
   - Open [`results/latex/main_corrected.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main_corrected.pdf) to visually inspect page boundaries, equation formatting, and figure placement.
