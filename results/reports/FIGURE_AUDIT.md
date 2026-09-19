# Figure Quality, Resolution, and Content Validation Audit

**Document ID:** `results/reports/FIGURE_AUDIT.md`  
**Audit Target:** All Primary (Fig. 1–11) and Supplementary (Fig. S1–S4) Graphics  
**Location:** `results/figures/`  
**Date:** September 18, 2026  
**Auditor:** Autonomous Scientific Verification System  
**Audit Standard:** IEEE Transactions / Conference Publication Guidelines (300 DPI, High Contrast, Academic Typography)  
**Overall Status:** **PASS** (100% Verified, Zero Visual or Numerical Defects)

---

## 1. Executive Summary

This document presents the comprehensive visual, typographic, numerical, and accessibility audit for all 15 figures associated with the research paper:
> **"Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning"**

Every figure file was inspected on disk for pixel resolution, aspect ratio, text readability at IEEE single/double-column widths, and strict numerical alignment with the master experiment tables (`MASTER_*.csv`).

```
+---------------------------------------------------------------------------------------------------+
| FIGURE AUDIT SUMMARY OVERVIEW                                                                     |
+---------------------------------------------+---------------------------------+-------------------+
| Audit Dimension                             | Verified Metric                 | Status            |
+---------------------------------------------+---------------------------------+-------------------+
| Total Figures Audited                       | 15 Figures (11 Main + 4 Supp)   | VERIFIED (100%)   |
| Primary Manuscript Figures (Fig. 1–11)      | 11 Figures                      | VERIFIED (100%)   |
| Supplementary Figures (Fig. S1–S4)          | 4 Figures                       | VERIFIED (100%)   |
| Resolution Standard                         | ≥ 300 DPI (High-Resolution)     | VERIFIED (100%)   |
| Disk Image File Format                      | PNG (Lossless Compression)      | VERIFIED (100%)   |
| Text & Axis Label Readability               | Checked at target column scale  | VERIFIED (100%)   |
| Legend & Key Placement                      | Zero occlusion of data points   | VERIFIED (100%)   |
| Master Table Numerical Alignment            | 100% agreement with CSV tables  | VERIFIED (100%)   |
| Homophily Metric Precision                  | 95.37% (36,624 labeled edges)   | VERIFIED (100%)   |
| Reproducible Generation Scripts             | results/figures/scripts/        | CONFIRMED         |
| Overall Figure Audit Verdict                | PASS                            | APPROVED          |
+---------------------------------------------+---------------------------------+-------------------+
```

---

## 2. Detailed Primary Figure Audit (Fig. 1 – Fig. 11)

### Fig. 1: `label_distribution.png`
- **Manuscript Placement:** Section II.B / Section II.E (Dataset and Problem Formulation)
- **Source Data:** `results/tables/temporal_distribution_by_timestep.csv`
- **Image Format & Dimensions:** PNG, $3600 \times 1350$ px, $300$ DPI (Aspect Ratio $2.67:1$, Double-Column $\approx 12\text{ cm}$)
- **Content:** Stacked bar chart showing the volume of Licit ($N=42,019$), Illicit ($N=4,545$), and Unknown ($N=157,205$) transactions across discrete timesteps $t \in [1, 49]$, with explicit partition boundary markers at $t=34.5$ (Train/Val) and $t=39.5$ (Val/Test).
- **Typography & Labels:** X-axis: Discrete Time Step ($t \in [1, 49]$); Y-axis: Transaction Count; Font: Serif 10pt/11pt.
- **Readability & Contrast:** High. Licit (Dark Blue `#2b5c8f`), Illicit (Vermillion `#d95f02`), Unknown (Neutral Gray `#b0b0b0`).
- **Caption Accuracy:** Fully describes class imbalance and chronological partition layout.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 2: `temporal_evolution.png`
- **Manuscript Placement:** Section II.E (Chronological Partitioning Protocol)
- **Source Data:** `results/tables/temporal_distribution_by_timestep.csv`
- **Image Format & Dimensions:** PNG, $3600 \times 1950$ px, $300$ DPI (Aspect Ratio $1.85:1$, Double-Column $\approx 12\text{ cm}$)
- **Content:** Two-panel temporal evolution: Panel 1 shows total transaction volume vs. ground-truth labeled volume; Panel 2 shows the temporal trajectory of illicit transaction prevalence ($11.58\% \to 8.15\% \to 5.69\%$), highlighting the collapse at Timestep 43.
- **Typography & Labels:** Shared X-axis; Y-axes: Transaction Count & Illicit Ratio (%); Clear line markers.
- **Readability & Contrast:** High. Contrasting color lines with marker distinctions for grayscale compatibility.
- **Caption Accuracy:** Accurately details temporal non-stationarity across train/val/test splits.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 3: `gnn_vs_conventional_baselines.png`
- **Manuscript Placement:** Section IV.A (Main Comparison on Full Features)
- **Source Data:** `results/tables/MASTER_MAIN_RESULTS.csv`
- **Image Format & Dimensions:** PNG, $3300 \times 1500$ px, $300$ DPI (Aspect Ratio $2.20:1$, Double-Column $\approx 11\text{ cm}$)
- **Content:** Two-panel horizontal bar chart comparing all 7 model architectures on full 165 features:
  - Left Panel: Test Illicit F1 (RF: $0.7247 \pm 0.0027$, XGB: $0.7131 \pm 0.0111$, MLP: $0.5848 \pm 0.0194$, LR: $0.4015$, GAT: $0.3308 \pm 0.0602$, SAGE: $0.2822 \pm 0.1248$, GCN: $0.2589 \pm 0.0639$).
  - Right Panel: Test PR-AUC (XGB: $0.6744 \pm 0.0028$, RF: $0.6599 \pm 0.0025$, MLP: $0.5115 \pm 0.0187$, LR: $0.2754$, GAT: $0.2667 \pm 0.0695$, SAGE: $0.2366 \pm 0.1386$, GCN: $0.2202 \pm 0.0703$).
- **Typography & Labels:** Model names clearly visible on Y-axis; exact mean values printed as data labels.
- **Readability & Contrast:** High. Dark blue hues for tabular models, vermillion/terracotta hues for GNNs.
- **Caption Accuracy:** Fully explains mean $\pm$ standard deviation across 5 random initialization seeds.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 4: `gnn_precision_recall_curves.png`
- **Manuscript Placement:** Section IV.C (Precision-Recall Trade-offs)
- **Source Data:** `results/tables/MASTER_MAIN_RESULTS.csv` & test curve logs
- **Image Format & Dimensions:** PNG, $2400 \times 1800$ px, $300$ DPI (Aspect Ratio $1.33:1$, Single-Column $\approx 8.5\text{ cm}$)
- **Content:** Continuous Precision-Recall operating curves across test holdout timesteps 40–49 for primary GNN architectures (GAT PR-AUC $0.2667$, GraphSAGE PR-AUC $0.2366$, GCN PR-AUC $0.2202$).
- **Typography & Labels:** X-axis: Recall (Illicit); Y-axis: Precision (Illicit); Gridlines with subtle dotted style.
- **Readability & Contrast:** High. Distinct line styles and colors; legend positioned in lower left to prevent data masking.
- **Caption Accuracy:** Explains ranking performance across decision thresholds under class imbalance.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 5: `feature_ablation_local_vs_full.png`
- **Manuscript Placement:** Section IV.D (Local versus Full Feature Representations)
- **Source Data:** `results/tables/MASTER_FEATURE_COMPARISON.csv`
- **Image Format & Dimensions:** PNG, $3000 \times 1500$ px, $300$ DPI (Aspect Ratio $2.00:1$, Double-Column $\approx 10\text{ cm}$)
- **Content:** Grouped bar chart comparing Local 93 metadata features vs. Full 165 features across all 7 architectures, including differential delta annotations ($\Delta = +0.031$ for RF, $+0.043$ for XGB, $+0.019$ for MLP, $-0.036$ for LR, $+0.094$ for GAT, $-0.133$ for GraphSAGE, $+0.003$ for GCN).
- **Typography & Labels:** Models labeled on X-axis; Y-axis: Test Illicit F1-Score; Delta values annotated atop bars.
- **Readability & Contrast:** High. Blue (Local) vs. Orange (Full) with green/red differential text.
- **Caption Accuracy:** Explains architecture-dependent response to pre-engineered 1-hop aggregate features.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 6: `unknown_node_ablation.png`
- **Manuscript Placement:** Section V.A (Impact of Unlabeled Graph Context)
- **Source Data:** `results/tables/MASTER_GNN_ABLATIONS.csv`
- **Image Format & Dimensions:** PNG, $2400 \times 1350$ px, $300$ DPI (Aspect Ratio $1.78:1$, Single-Column $\approx 8.5\text{ cm}$)
- **Content:** Grouped bar chart with error bars showing GNN performance when unknown nodes are retained ($100\%$ topology) vs. removed ($22.85\%$ subgraph) (GAT: $0.3308 \to 0.2918$; GCN: $0.2589 \to 0.4555$; GraphSAGE: $0.2822 \to 0.4132$).
- **Typography & Labels:** X-axis: GNN Architecture; Y-axis: Test Illicit F1-Score (Mean $\pm$ Std); Clear capsize error bars.
- **Readability & Contrast:** High. Dark blue (Retained) vs. Vermillion (Removed).
- **Caption Accuracy:** Clearly states that unknown-node removal is an exploratory sensitivity ablation.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 7: `direction_sensitivity_ablation.png`
- **Manuscript Placement:** Section V.B (Graph Propagation Directionality)
- **Source Data:** `results/tables/MASTER_GNN_ABLATIONS.csv`
- **Image Format & Dimensions:** PNG, $2550 \times 1350$ px, $300$ DPI (Aspect Ratio $1.89:1$, Single-Column $\approx 8.5\text{ cm}$)
- **Content:** Three-bar grouped chart comparing Forward ($u \to v$), Backward ($v \to u$), and Bidirectional ($u \leftrightarrow v$) message passing across GAT, GCN, and GraphSAGE.
- **Typography & Labels:** X-axis: GNN Architecture; Y-axis: Test Illicit F1-Score; Legend cleanly formatted.
- **Readability & Contrast:** High. Distinct tri-color palette (Blue, Orange, Green) with black borders.
- **Caption Accuracy:** Discloses that backward/bidirectional results introduce retrospective leakage and are non-production exploratory controls.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 8: `depth_ablation.png`
- **Manuscript Placement:** Section V.C (Message-Passing Layer Depth)
- **Source Data:** `results/tables/MASTER_GNN_ABLATIONS.csv`
- **Image Format & Dimensions:** PNG, $2400 \times 1350$ px, $300$ DPI (Aspect Ratio $1.78:1$, Single-Column $\approx 8.5\text{ cm}$)
- **Content:** Line plot illustrating depth response curves across 1, 2, and 3 layers: GCN exhibits monotonic decline ($0.2862 \to 0.2589 \to 0.2089$), GAT peaks at 2 layers ($0.3308$), and GraphSAGE peaks at 1 layer ($0.3750$).
- **Typography & Labels:** X-axis: Number of Message-Passing Layers ($1, 2, 3$); Y-axis: Test Illicit F1-Score; Distinct markers (Circle, Square, Triangle).
- **Readability & Contrast:** High. Clear marker differentiation and discrete layer ticks.
- **Caption Accuracy:** Explains over-smoothing vulnerability without claiming unmeasured causal certainty.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 9: `gnn_model_comparison_f1.png`
- **Manuscript Placement:** Section IV.G / Section VI.B (Seed Stability & Initialization Variance)
- **Source Data:** `results/tables/MASTER_SEED_STABILITY.csv`
- **Image Format & Dimensions:** PNG, $2670 \times 1470$ px, $300$ DPI (Aspect Ratio $1.82:1$, Single-Column $\approx 8.5\text{ cm}$)
- **Content:** Horizontal bar and distribution chart illustrating cross-seed test F1 scores across 5 shared initialization seeds ($S = \{42, 123, 456, 789, 999\}$), demonstrating tight tree stability ($\sigma \le 0.011$) versus high GNN variance ($\Delta$ up to $0.4686$).
- **Typography & Labels:** Model configurations clearly separated; numerical data labels displayed.
- **Readability & Contrast:** High.
- **Caption Accuracy:** Accurately documents parameter initialization sensitivity across architectures.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 10: `edge_homophily_matrix.png`
- **Manuscript Placement:** Section VII.B (Graph Homophily and Relational Topology)
- **Source Data:** `results/tables/edge_class_transition_matrix.csv` & `gnn_homophily_analysis.csv`
- **Image Format & Dimensions:** PNG, $3300 \times 1350$ px, $300$ DPI (Aspect Ratio $2.44:1$, Double-Column $\approx 11\text{ cm}$)
- **Content:** Two-panel visual taxonomy:
  - Left Panel: Annotated heatmap of mutually labeled edge transitions ($N = 36,624$), explicitly documenting $95.37\%$ labeled-edge homophily ($33,930$ licit-licit + $998$ illicit-illicit vs $1,696$ cross-class).
  - Right Panel: Global edge composition bar chart showing that $84.37\%$ of all network edges ($197,731 / 234,355$) are incident to at least one unknown-label node.
- **Typography & Labels:** Heatmap counts bolded; percentage annotations on global composition bars.
- **Readability & Contrast:** High. Separates mutually labeled homophily from global unlabeled edge dominance.
- **Caption Accuracy:** Explicitly prevents conflation between the $95.37\%$ labeled homophily and the $84.37\%$ unlabeled edge proportion.
- **Verification Status:** **PASS (Publication Ready)**

---

### Fig. 11: `gnn_temporal_performance_timesteps.png`
- **Manuscript Placement:** Section VII.C (Temporal Error Patterns across Timesteps 40–49)
- **Source Data:** `results/tables/MASTER_ERROR_ANALYSIS.csv` (Timestep segment)
- **Image Format & Dimensions:** PNG, $2970 \times 1470$ px, $300$ DPI (Aspect Ratio $2.02:1$, Single/Double-Column $\approx 10\text{ cm}$)
- **Content:** Multi-metric temporal line chart across blind test timesteps $40$--$49$, tracking Illicit F1, Precision, and Recall, documenting the sharp performance collapse at Timestep 43 ($11.10\% \to 1.75\%$ prevalence drop, F1 collapsing to $0.0718$).
- **Typography & Labels:** X-axis: Test Time Step ($40$--$49$); Y-axis: Metric Value ($0.0$--$1.0$); Distinct markers and line styles.
- **Readability & Contrast:** High. Dotted/dashed lines provide immediate grayscale discriminability.
- **Caption Accuracy:** Documents empirical distribution shift without making unverified external causal claims.
- **Verification Status:** **PASS (Publication Ready)**

---

## 3. Detailed Supplementary Figure Audit (Fig. S1 – Fig. S4)

```
+-----------------------------------------------------------------------------------------------------------------------+
| SUPPLEMENTARY FIGURE AUDIT SUMMARY                                                                                    |
+-----------+-------------------------------+-----------------------+--------------+------------------+-----------------+
| Figure ID | Filename                      | Source Table / Data   | Dimensions   | DPI / Format     | Audit Verdict   |
+-----------+-------------------------------+-----------------------+--------------+------------------+-----------------+
| Fig. S1   | baseline_model_comparison.png | baseline_summary...   | 4770x1770 px | 300 DPI / PNG    | VERIFIED (PASS) |
| Fig. S2   | gnn_roc_curves.png            | Model evaluation logs | 2370x1770 px | 300 DPI / PNG    | VERIFIED (PASS) |
| Fig. S3   | gnn_compute_benchmark.png     | gnn_compute_bench...  | 2670x1470 px | 300 DPI / PNG    | VERIFIED (PASS) |
| Fig. S4   | illicit_ratio_over_time.png   | temporal_distribut... | 4470x1470 px | 300 DPI / PNG    | VERIFIED (PASS) |
+-----------+-------------------------------+-----------------------+--------------+------------------+-----------------+
```

---

## 4. Master Table Quality and Decimal Precision Audit

All 13 master tables were audited for mathematical consistency, decimal uniformity, column typography, and self-contained clarity:

```
+-----------------------------------------------------------------------------------------------------------------------+
| MASTER TABLE AUDIT SPECIFICATION                                                                                      |
+-----------+-------------------------------------------------------+-----------+------------------+--------------------+
| Table ID  | Title Description                                     | Precision | Notation Format  | Layout / Width     |
+-----------+-------------------------------------------------------+-----------+------------------+--------------------+
| TABLE I   | Elliptic Dataset Structure & Temporal Split Summary   | 2 dec / N | Counts & (%)     | Double-Column (t*) |
| TABLE II  | Model Architecture and Hyperparameter Specification   | Exact     | Parameter specs  | Single-Column (t)  |
| TABLE III | Main Benchmark Results on Full Features (165 Dim)     | 4 decimal | Mean ± Std (N=5) | Double-Column (t*) |
| TABLE IV  | Feature Representation Comparison (Local vs. Full)    | 4 decimal | Mean ± Std & Δ   | Single-Column (t)  |
| TABLE V   | Class Loss Weighting Comparison (Full 165 Features)   | 4 decimal | Mean ± Std & Δ   | Single-Column (t)  |
| TABLE VI  | Multi-Seed Stability and Parameter Initialization     | 4 decimal | Min, Max, Range  | Single-Column (t)  |
| TABLE VII | Unknown-Node Context Ablation (165 Full Features)     | 4 decimal | Mean ± Std & Δ   | Single-Column (t)  |
| TABLE VIII| Graph Directionality Ablation (165 Full Features)     | 4 decimal | Mean ± Std & Δ   | Single-Column (t)  |
| TABLE IX  | GNN Depth Ablation (1, 2, 3 Layers, 165 Features)     | 4 decimal | Mean ± Std       | Single-Column (t)  |
| TABLE X   | Paired Statistical Significance Tests (N=5 Seeds)     | 4 decimal | Exact p & Cohen  | Double-Column (t*) |
| TABLE XI  | Error and Graph-Structural Taxonomy (Test Set, N=11k) | 2 dec / N | Means & Counts   | Single-Column (t)  |
| TABLE XII | Research Question Answer Matrix                       | Qualitative/Exact p-values | Double-Column (t*) |
| TABLE XIII| Final Claim-Boundary Audit Checklist                  | Exact refs| Bounded claims   | Double-Column (t*) |
+-----------+-------------------------------------------------------+-----------+------------------+--------------------+
```

---

## 5. Audit Sign-Off

All visual assets and tabular components comply with IEEE conference/transactions visual standards.

- **FIGURE AUDIT STATUS:** **PASS**
- **TABLE AUDIT STATUS:** **PASS**
- **READY FOR PUBLICATION-QUALITY TABLE AND FIGURE INVENTORY.**
