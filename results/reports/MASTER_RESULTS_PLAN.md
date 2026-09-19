# Master Experiment Results Plan & Publication Layout

**Project:** Graph Neural Network-Based Fraud Detection in Financial Transaction Networks  
**Dataset:** Elliptic Bitcoin Transaction Dataset (203,769 transactions across 49 timesteps)  
**Document Purpose:** Serve as the authoritative, canonical blueprint mapping every verified experimental result, ablation study, and statistical test to its designated section in the research paper.

---

## SECTION 4 — MAIN RESULTS: BENCHMARK COMPARISON

### 1. Context & Scope
This section presents the primary empirical comparison between conventional tabular machine learning baselines and inductive Graph Neural Network architectures on the full 165-feature representation under the strict chronological split (**Training:** Timesteps 1–34, **Validation:** Timesteps 35–39, **Blind Testing:** Timesteps 40–49).

### 2. Designated Tables & Figures
- **Primary Table:** [`results/tables/MASTER_MAIN_RESULTS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_MAIN_RESULTS.csv)
- **Primary Figures:**
  - `FIG_06`: [`results/figures/gnn_vs_conventional_baselines.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_vs_conventional_baselines.png) (Direct comparison of Test Illicit F1 and PR-AUC between tree models and GNNs)
  - `FIG_07`: [`results/figures/gnn_model_comparison_f1.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_model_comparison_f1.png) (GNN architecture comparison on illicit F1)
  - `FIG_08`: [`results/figures/gnn_precision_recall_curves.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_precision_recall_curves.png) (Precision-Recall curves on the blind test split)

### 3. Empirical Summary (Full 165 Features)

| Model Family | Model Architecture | Loss Formulation | Validation F1 | Validation PR-AUC | Test Illicit F1 | Test PR-AUC | Test Precision | Test Recall | Test MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tree Ensemble** | **Random Forest** | Standard | $0.9489 \pm 0.0023$ | $0.9553 \pm 0.0008$ | **$0.7247 \pm 0.0027$** | **$0.6599 \pm 0.0025$** | $0.9586 \pm 0.0198$ | $0.5827 \pm 0.0042$ | $0.7367 \pm 0.0057$ |
| **Gradient Boost** | **XGBoost** | Standard | $0.9399 \pm 0.0053$ | $0.9595 \pm 0.0015$ | **$0.7131 \pm 0.0111$** | **$0.6744 \pm 0.0028$** | $0.9127 \pm 0.0366$ | $0.5855 \pm 0.0062$ | $0.7189 \pm 0.0154$ |
| **Neural Net** | **MLP** | Standard | $0.8212 \pm 0.0344$ | $0.8446 \pm 0.0223$ | **$0.5848 \pm 0.0194$** | **$0.5115 \pm 0.0187$** | $0.7630 \pm 0.0430$ | $0.4761 \pm 0.0335$ | $0.5840 \pm 0.0155$ |
| **Linear Model** | **Logistic Regression** | Standard | $0.5921 \pm 0.0000$ | $0.4826 \pm 0.0000$ | **$0.4015 \pm 0.0000$** | **$0.2754 \pm 0.0000$** | $0.3821 \pm 0.0000$ | $0.4230 \pm 0.0000$ | $0.3640 \pm 0.0000$ |
| **GNN** | **GAT** | Weighted | $0.6006 \pm 0.0982$ | $0.5939 \pm 0.1191$ | **$0.3308 \pm 0.0602$** | **$0.2667 \pm 0.0695$** | $0.2680 \pm 0.0632$ | $0.4412 \pm 0.0281$ | $0.2899 \pm 0.0640$ |
| **GNN** | **GraphSAGE** | Weighted | $0.4473 \pm 0.1590$ | $0.4106 \pm 0.1738$ | **$0.2822 \pm 0.1248$** | **$0.2366 \pm 0.1386$** | $0.2695 \pm 0.1858$ | $0.3588 \pm 0.0756$ | $0.2391 \pm 0.1394$ |
| **GNN** | **GCN** | Weighted | $0.4293 \pm 0.0923$ | $0.4056 \pm 0.1386$ | **$0.2589 \pm 0.0639$** | **$0.2202 \pm 0.0703$** | $0.2788 \pm 0.1255$ | $0.2884 \pm 0.0839$ | $0.2213 \pm 0.0766$ |

### 4. Paper-Safe Interpretation Rules
- Explicitly report that tree ensembles (Random Forest and XGBoost) achieved higher illicit F1 and PR-AUC scores on the test set than the tested end-to-end GNN configurations.
- Attribute performance differences to the presence of precomputed 1-hop aggregate statistics in the 165-feature matrix combined with tree partition robustness under temporal non-stationarity.
- Do not claim universal inferiority or superiority of GNNs; scope all claims to the evaluated inductive setup and static graph snapshots.

---

## SECTION 5 — FEATURE ANALYSIS: LOCAL (93) vs. FULL (165)

### 1. Context & Scope
Analyzes the impact of adding 72 handcrafted 1-hop relational aggregate features (`agg_feat_1` to `agg_feat_72`) to the 93 local features (`local_feat_1` to `local_feat_93`).

### 2. Designated Tables & Figures
- **Primary Table:** [`results/tables/MASTER_FEATURE_COMPARISON.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_FEATURE_COMPARISON.csv)
- **Primary Figure:** `FIG_10`: [`results/figures/feature_ablation_local_vs_full.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/feature_ablation_local_vs_full.png)

### 3. Empirical Summary & Differences

| Model Architecture | Local (93) Test F1 | Full (165) Test F1 | $\Delta\text{F1}$ (Full - Local) | Local (93) Test PR-AUC | Full (165) Test PR-AUC | $\Delta\text{PR-AUC}$ (Full - Local) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | $0.6936 \pm 0.0033$ | $0.7247 \pm 0.0027$ | **$+0.0311$** | $0.6495 \pm 0.0003$ | $0.6599 \pm 0.0025$ | **$+0.0104$** |
| **XGBoost** | $0.6701 \pm 0.0109$ | $0.7131 \pm 0.0111$ | **$+0.0430$** | $0.6554 \pm 0.0031$ | $0.6744 \pm 0.0028$ | **$+0.0190$** |
| **MLP** | $0.5661 \pm 0.0184$ | $0.5848 \pm 0.0194$ | **$+0.0187$** | $0.4082 \pm 0.0584$ | $0.5115 \pm 0.0187$ | **$+0.1032$** |
| **Logistic Regression** | $0.4378 \pm 0.0000$ | $0.4015 \pm 0.0000$ | **$-0.0363$** | $0.2140 \pm 0.0000$ | $0.2754 \pm 0.0000$ | **$+0.0615$** |
| **GAT** | $0.2367 \pm 0.0586$ | $0.3308 \pm 0.0602$ | **$+0.0942$** | $0.1653 \pm 0.0747$ | $0.2667 \pm 0.0695$ | **$+0.1014$** |
| **GraphSAGE** | $0.4153 \pm 0.1914$ | $0.2822 \pm 0.1248$ | **$-0.1331$** | $0.3659 \pm 0.1859$ | $0.2366 \pm 0.1386$ | **$-0.1293$** |
| **GCN** | $0.2559 \pm 0.0674$ | $0.2589 \pm 0.0639$ | **$+0.0030$** | $0.2132 \pm 0.0905$ | $0.2202 \pm 0.0703$ | **$+0.0070$** |

### 4. Paper-Safe Interpretation Rules
- For tree-based baselines, MLP, and GAT, adding aggregated features improves test F1 and PR-AUC.
- For GraphSAGE, full features lead to lower test F1 ($0.4153 \to 0.2822$) due to redundant relational aggregation over features that already encapsulate 1-hop statistics.
- Do not describe either feature space as universally superior; describe the trade-off quantitatively.

---

## SECTION 6 — GNN ABLATION STUDIES

### 1. Context & Scope
Examines three controlled topological ablations across GCN, GraphSAGE, and GAT:
1. **Unknown-Node Context:** Retained (100% full graph) vs. Removed (labeled transactions only).
2. **Graph Direction:** Forward ($u \to v$) vs. Backward ($v \to u$) vs. Bidirectional ($u \leftrightarrow v$).
3. **GNN Depth:** 1 Layer vs. 2 Layers vs. 3 Layers.

### 2. Designated Tables & Figures
- **Primary Table:** [`results/tables/MASTER_GNN_ABLATIONS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_GNN_ABLATIONS.csv)
- **Primary Figures:**
  - `FIG_11`: [`results/figures/unknown_node_ablation.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/unknown_node_ablation.png) (Unknown node ablation)
  - `FIG_12`: [`results/figures/depth_ablation.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/depth_ablation.png) (Depth & over-smoothing ablation)
  - `FIG_13`: [`results/figures/direction_sensitivity_ablation.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/direction_sensitivity_ablation.png) (Directional sensitivity)

### 3. Empirical Summary

#### A. Unknown-Node Context Ablation
- **GAT:** Retained = **$0.3308 \pm 0.0602$** vs. Removed = **$0.2918 \pm 0.0521$** ($\Delta = +0.0390$)
- **GCN:** Retained = **$0.2589 \pm 0.0639$** vs. Removed = **$0.4555 \pm 0.0524$** ($\Delta = -0.1966$)
- **GraphSAGE:** Retained = **$0.2822 \pm 0.1248$** vs. Removed = **$0.4132 \pm 0.0514$** ($\Delta = -0.1310$)
- *Interpretation:* Unlabeled-node context is architecture-dependent. Retaining unknown nodes benefits GAT due to attention-weight filtering, while removing unknown nodes improves GCN and GraphSAGE by avoiding feature dilution from ~77% unlabeled neighbors.

#### B. Graph Direction Sensitivity Ablation
- **GAT:** Forward = **$0.3308 \pm 0.0602$** | Backward = **$0.3938 \pm 0.0706$** | Bidirectional = **$0.4618 \pm 0.0234$**
- **GCN:** Forward = **$0.2589 \pm 0.0639$** | Backward = **$0.3953 \pm 0.0462$** | Bidirectional = **$0.3142 \pm 0.0571$**
- **GraphSAGE:** Forward = **$0.2822 \pm 0.1248$** | Backward = **$0.3213 \pm 0.0558$** | Bidirectional = **$0.3710 \pm 0.1228$**
- *Interpretation:* Backward propagation aggregates upstream payment originators, yielding higher retrospective F1 in static snapshots; forward propagation remains the strict causal temporal standard for streaming deployment.

#### C. GNN Depth / Over-Smoothing Ablation
- **GAT:** Layer 1 = **$0.2877 \pm 0.0319$** | Layer 2 = **$0.3308 \pm 0.0602$** | Layer 3 = **$0.2847 \pm 0.0052$**
- **GCN:** Layer 1 = **$0.2862 \pm 0.0250$** | Layer 2 = **$0.2589 \pm 0.0639$** | Layer 3 = **$0.2089 \pm 0.0892$**
- **GraphSAGE:** Layer 1 = **$0.3750 \pm 0.1215$** | Layer 2 = **$0.2822 \pm 0.1248$** | Layer 3 = **$0.3327 \pm 0.1250$**
- *Interpretation:* Deeper message passing does not monotonically improve performance. GCN suffers continuous degradation with depth due to isotropic over-smoothing.

---

## SECTION 7 — ROBUSTNESS & STATISTICAL SIGNIFICANCE

### 1. Context & Scope
Evaluates hypothesis testing, effect sizes, and cross-seed stability across the 5 shared random seeds (`[42, 123, 456, 789, 999]`).

### 2. Designated Tables
- **Statistical Tests Table:** [`results/tables/MASTER_STATISTICAL_TESTS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_STATISTICAL_TESTS.csv)
- **Seed Stability Table:** [`results/tables/MASTER_SEED_STABILITY.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_SEED_STABILITY.csv)

### 3. Empirical Statistical Comparison (GraphSAGE Local 93 vs. Baselines)

| Comparison Baseline | Metric | Mean Diff $\Delta$ | Paired $t$-Stat | Raw $t$ $p$-val | Holm-Adj $t$ $p$-val | Wilcoxon $W$ | Wilcoxon $p$-val | Paired Cohen $d_z$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Full 165)** | Test Illicit F1 | **$-0.3094$** | $-3.5917$ | $0.0229$ | $0.0917$ | $0.0$ | $0.0625$ | $-1.6062$ |
| **Random Forest (Full 165)** | Test PR-AUC | **$-0.2940$** | $-3.5141$ | $0.0246$ | $0.0816$ | $0.0$ | $0.0625$ | $-1.5716$ |
| **XGBoost (Full 165)** | Test Illicit F1 | **$-0.3044$** | $-3.5092$ | $0.0247$ | $0.0917$ | $0.0$ | $0.0625$ | $-1.5694$ |
| **XGBoost (Full 165)** | Test PR-AUC | **$-0.3064$** | $-3.7246$ | $0.0204$ | $0.0816$ | $0.0$ | $0.0625$ | $-1.6657$ |
| **MLP (Full 165)** | Test Illicit F1 | **$-0.1695$** | $-1.8872$ | $0.1322$ | $0.2644$ | $3.0$ | $0.3125$ | $-0.8440$ |
| **Logistic Regression (Full 165)** | Test Illicit F1 | **$+0.0673$** | $+0.7856$ | $0.4760$ | $0.4760$ | $6.0$ | $0.8125$ | $+0.3513$ |

### 4. Paper-Safe Statistical Disclosures
- Explicitly disclose that with $N=5$ seeds, the non-parametric Wilcoxon signed-rank test has a minimum achievable two-sided $p$-value of $0.0625$, which cannot reach $\alpha=0.05$.
- Parametric paired $t$-tests yield $p_{\text{adj}} = 0.0917$ after Holm-Bonferroni step-down correction.
- Effect sizes are explicitly reported as paired within-subject $d_z = \bar{D}/s_D$.

---

## SECTION 8 — ERROR TAXONOMY, HOMOPHILY & TEMPORAL DRIFT

### 1. Context & Scope
Analyzes failure modes, graph relational structure, and the impact of the Darknet market shutdown at Timestep 43.

### 2. Designated Tables & Figures
- **Primary Table:** [`results/tables/MASTER_ERROR_ANALYSIS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_ERROR_ANALYSIS.csv)
- **Primary Figures:**
  - `FIG_03`: [`results/figures/illicit_ratio_over_time.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/illicit_ratio_over_time.png) (Temporal non-stationarity)
  - `FIG_04`: [`results/figures/edge_homophily_matrix.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/edge_homophily_matrix.png) (Edge homophily structure)
  - `FIG_14`: [`results/figures/gnn_temporal_performance_timesteps.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_temporal_performance_timesteps.png) (Timestep-by-timestep performance)

### 3. Empirical Summary
1. **Edge Homophily:** Among labeled-labeled pairs ($15.63\%$ of edges), homophily is high ($95.37\%$). However, $84.37\%$ of all edges connect to unknown nodes, creating substantial structural uncertainty.
2. **Degree Disparity:** True Negatives have mean in-degree $2.38$ vs. False Positives ($1.23$) and True Positives ($0.87$).
3. **Temporal Breakdown:** At Timesteps 43–45, the illicit base rate drops from $11.1\%$ to $<1.8\%$ (e.g., Timestep 45 contains only 5 illicit transactions out of 1,221), causing sharp test precision and F1 degradation.

---

## SUPPLEMENTARY MATERIAL

### 1. Supplementary Tables
- Complete Experiment Inventory: [`results/tables/MASTER_EXPERIMENT_INVENTORY.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_EXPERIMENT_INVENTORY.csv) (110 total runs)
- Baseline Weighting Comparison: [`results/tables/MASTER_WEIGHTING_COMPARISON.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_WEIGHTING_COMPARISON.csv)
- Computational Benchmark: [`results/tables/gnn_compute_benchmark.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/gnn_compute_benchmark.csv)
- Feature Quality & Verification: [`results/tables/feature_schema_verification.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/feature_schema_verification.csv)

### 2. Supplementary Figures
- `FIG_01`: [`results/figures/label_distribution.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/label_distribution.png)
- `FIG_02`: [`results/figures/temporal_evolution.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/temporal_evolution.png)
- `FIG_05`: [`results/figures/baseline_model_comparison.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/baseline_model_comparison.png)
- `FIG_09`: [`results/figures/gnn_roc_curves.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_roc_curves.png)
- `FIG_15`: [`results/figures/gnn_compute_benchmark.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_compute_benchmark.png)

---

## FINAL SCIENTIFIC AUDIT SUMMARY

| Metric / Dimension | Verified Count | Verification Source |
| :--- | :---: | :--- |
| **Total Experiment Runs** | **110** | 80 Phase 2 Baseline runs + 30 Phase 3 Primary GNN runs ([`MASTER_EXPERIMENT_INVENTORY.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_EXPERIMENT_INVENTORY.csv)) |
| **Primary Model Configurations** | **7** | 4 Baselines (RF, XGB, MLP, LR) + 3 GNNs (GCN, GraphSAGE, GAT) |
| **Controlled Ablation Dimensions** | **4** | Feature Space (2), Unknown Context (2), Graph Direction (3), Depth (3) |
| **Random Seeds per Experiment** | **5** | `[42, 123, 456, 789, 999]` |
| **Model Families Evaluated** | **5** | Tree Ensemble, Gradient Boosting, Multi-Layer Perceptron, Linear Model, GNN |
| **Feature Spaces Evaluated** | **2** | Local 93 features vs. Full 165 features |
| **Statistical Comparisons Conducted** | **8** | 4 Baselines $\times$ 2 Metrics (Illicit F1 and PR-AUC) with Dual Testing & Holm Correction |
| **Publication-Grade Figures (300 DPI)** | **15** | [`MASTER_FIGURE_INVENTORY.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_FIGURE_INVENTORY.csv) |
| **Missing Metrics** | **0** | All required classification, threshold, and curve metrics fully populated |
| **Data Leakage Inconsistencies** | **0** | Strict anti-leakage verified across all 110 runs |
