# Full Manuscript Consistency, Evidence, and Citation Audit (Step 4G)

**Document Reference:** `results/reports/PAPER_STEP_4G.md`  
**Paper Title:** *Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning*  
**Audited Manuscript Components:**
1. [`results/reports/PAPER_STEP_4B.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_STEP_4B.md) (Title, Abstract, Research Questions, Section 1 Introduction)
2. [`results/reports/PAPER_STEP_4C.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_STEP_4C.md) (Section 2 Dataset & Problem Formulation, Section 3 Experimental Methodology, Tables 1–2)
3. [`results/reports/PAPER_STEP_4D.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_STEP_4D.md) (Section 4 Experimental Results, Tables 3–6, Figures 3–5)
4. [`results/reports/PAPER_STEP_4E.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_STEP_4E.md) (Section 5 GNN Ablations, Section 6 Statistical Analysis, Section 7 Error Analysis, Tables 7–11, Figures 6–11)
5. [`results/reports/PAPER_STEP_4F.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_STEP_4F.md) (Section 8 Discussion, Section 9 Limitations, Section 10 Conclusion & Future Work, Tables 12–13)
6. [`results/reports/PAPER_FACT_SHEET.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_FACT_SHEET.md) (Authoritative Evidence Pack)

---

# 1. Executive Audit Summary

A comprehensive, multi-pass scientific audit was conducted across all drafted sections of the research paper to ensure total consistency, reproducibility, adherence to evidence, and rigorous claim boundaries. 

### Key Findings of the Audit:
1. **Total Numerical Integrity:** Every numerical metric (F1-scores, PR-AUC, Precision, Recall, MCC, Degree statistics, Sample counts, $p$-values, effect sizes, and percentages) across all narrative sections and tables perfectly matches the authoritative CSV artifacts ([`MASTER_MAIN_RESULTS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_MAIN_RESULTS.csv), [`MASTER_FEATURE_COMPARISON.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_FEATURE_COMPARISON.csv), [`MASTER_WEIGHTING_COMPARISON.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_WEIGHTING_COMPARISON.csv), [`MASTER_GNN_ABLATIONS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_GNN_ABLATIONS.csv), [`MASTER_SEED_STABILITY.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_SEED_STABILITY.csv), [`MASTER_STATISTICAL_TESTS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_STATISTICAL_TESTS.csv), and [`MASTER_ERROR_ANALYSIS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_ERROR_ANALYSIS.csv)).
2. **Strict Temporal Partitioning:** All sections uniformly adhere to the verified chronological partition: **Train = Timesteps 1–34** ($N=29,894$), **Validation = Timesteps 35–39** ($N=5,486$), and **Blind Test = Timesteps 40–49** ($N=11,184$). Zero occurrences of legacy uncalibrated split bounds (e.g., 1–28 / 29–34 / 35–49) exist in the manuscript.
3. **Exact Feature Accounting:** The manuscript consistently reports 165 machine learning features (93 local transaction features + 72 1-hop relational aggregate features), correctly distinguishing the 165 feature columns from the 167 raw CSV columns (which include `txId` and `time_step`).
4. **Leakage-Free Methodology:** Scaling (via `StandardScaler`) and decision thresholds ($\tau^*$) are confirmed to be strictly locked on training/validation horizons prior to test evaluation. No SMOTE, ADASYN, PCA, or transductive test label leakage occurred.
5. **Cautious Scientific Tone:** All claims of GNN vs. tabular performance are appropriately conditioned on the evaluated protocol, avoiding hyperbolic language (e.g., "superior", "state-of-the-art", "proves"). Theoretical mechanisms (over-smoothing, feature dilution, darknet shutdown events) are carefully distinguished as testable hypotheses rather than measured causal facts.
6. **Statistical Power Transparency:** The mathematical lower bound on the two-sided Wilcoxon signed-rank test at $N=5$ ($p_{\min} = 0.0625$) and the Holm-Bonferroni adjusted $p$-values are explicitly disclosed.

---

# 2. Global Numerical Consistency Audit

Table 1 details the verification of every major empirical claim and metric across the manuscript against the authoritative master CSV tables.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 1: Global Numerical Consistency Audit Matrix                                                                    |
+------------------------------------+------------------+-------------------+-------------------+-----------------------+
| Empirical Claim / Parameter        | Location         | Manuscript Value  | Master CSV Value  | Audit Status          |
+------------------------------------+------------------+-------------------+-------------------+-----------------------+
| Total Transactions (|V|)           | Sec 2, Sec 4B, 4C| 203,769           | 203,769           | VERIFIED (Exact Match)|
| Total Directed Edges (|E|)         | Sec 2, Sec 4B, 4C| 234,355           | 234,355           | VERIFIED (Exact Match)|
| Total Discrete Timesteps (T)       | Sec 2, Sec 4B, 4C| 49                | 49                | VERIFIED (Exact Match)|
| Labeled Transactions (|V_L|)       | Sec 2, Tab 1     | 46,564 (22.85%)   | 46,564 (22.85%)   | VERIFIED (Exact Match)|
| Unlabeled Transactions (|V_U|)     | Sec 2, Tab 1     | 157,205 (77.15%)  | 157,205 (77.15%)  | VERIFIED (Exact Match)|
| Licit Transactions (Class 2 / 0)   | Sec 2, Tab 1     | 42,019 (90.24%)   | 42,019 (90.24%)   | VERIFIED (Exact Match)|
| Illicit Transactions (Class 1 / 1) | Sec 2, Tab 1     | 4,545 (9.76%)     | 4,545 (9.76%)     | VERIFIED (Exact Match)|
| Licit-to-Illicit Imbalance Ratio   | Sec 2, Sec 4C    | 9.245 : 1         | 9.245 : 1         | VERIFIED (Exact Match)|
| Total ML Features (d)              | Sec 2, Sec 4C    | 165 (93 loc+72 ag)| 165 (93 loc+72 ag)| VERIFIED (Exact Match)|
| Raw Dataset Columns                | Sec 2, Sec 4C    | 167 (txId+t+165)  | 167 (txId+t+165)  | VERIFIED (Exact Match)|
| Training Nodes (t in [1, 34])      | Sec 2, Tab 1     | 29,894 (11.58% il)| 29,894 (11.58% il)| VERIFIED (Exact Match)|
| Training Positive Loss Weight      | Sec 3.3, Tab 2   | 7.6349 (balanced) | 7.6349 (balanced) | VERIFIED (Exact Match)|
| Validation Nodes (t in [35, 39])   | Sec 2, Tab 1     | 5,486 (8.15% ill) | 5,486 (8.15% ill) | VERIFIED (Exact Match)|
| Test Nodes (t in [40, 49])         | Sec 2, Tab 1     | 11,184 (5.69% ill)| 11,184 (5.69% ill)| VERIFIED (Exact Match)|
| Evaluation Seeds (S)               | Sec 3, Sec 4.1   | 42,123,456,789,999| 42,123,456,789,999| VERIFIED (Exact Match)|
+------------------------------------+------------------+-------------------+-------------------+-----------------------+
| MAIN BENCHMARK RESULTS (Full 165)  |                  |                   |                   |                       |
| Random Forest (Full, Standard) F1  | Sec 4.1, Tab 3   | 0.7247 ± 0.0027   | 0.7247 ± 0.0027   | VERIFIED (Exact Match)|
| Random Forest (Full, Standard) PR  | Sec 4.1, Tab 3   | 0.6599 ± 0.0025   | 0.6599 ± 0.0025   | VERIFIED (Exact Match)|
| XGBoost (Full, Standard) F1        | Sec 4.1, Tab 3   | 0.7131 ± 0.0111   | 0.7131 ± 0.0111   | VERIFIED (Exact Match)|
| XGBoost (Full, Standard) PR-AUC    | Sec 4.1, Tab 3   | 0.6744 ± 0.0028   | 0.6744 ± 0.0028   | VERIFIED (Exact Match)|
| MLP (Full, Standard) F1            | Sec 4.1, Tab 3   | 0.5848 ± 0.0194   | 0.5848 ± 0.0194   | VERIFIED (Exact Match)|
| MLP (Full, Standard) PR-AUC        | Sec 4.1, Tab 3   | 0.5115 ± 0.0187   | 0.5115 ± 0.0187   | VERIFIED (Exact Match)|
| Logistic Regression (Full, Std) F1 | Sec 4.1, Tab 3   | 0.4015 ± 0.0000   | 0.4015 ± 0.0000   | VERIFIED (Exact Match)|
| Logistic Regression (Full, Std) PR | Sec 4.1, Tab 3   | 0.2754 ± 0.0000   | 0.2754 ± 0.0000   | VERIFIED (Exact Match)|
| GAT (Full, Weighted) F1            | Sec 4.1, Tab 3   | 0.3308 ± 0.0602   | 0.3308 ± 0.0602   | VERIFIED (Exact Match)|
| GAT (Full, Weighted) PR-AUC        | Sec 4.1, Tab 3   | 0.2667 ± 0.0695   | 0.2667 ± 0.0695   | VERIFIED (Exact Match)|
| GraphSAGE (Full, Weighted) F1      | Sec 4.1, Tab 3   | 0.2822 ± 0.1248   | 0.2822 ± 0.1248   | VERIFIED (Exact Match)|
| GraphSAGE (Full, Weighted) PR-AUC  | Sec 4.1, Tab 3   | 0.2366 ± 0.1386   | 0.2366 ± 0.1386   | VERIFIED (Exact Match)|
| GCN (Full, Weighted) F1            | Sec 4.1, Tab 3   | 0.2589 ± 0.0639   | 0.2589 ± 0.0639   | VERIFIED (Exact Match)|
| GCN (Full, Weighted) PR-AUC        | Sec 4.1, Tab 3   | 0.2202 ± 0.0703   | 0.2202 ± 0.0703   | VERIFIED (Exact Match)|
| GraphSAGE (Local 93, Val-Selected) | Sec 4.4, Tab 4   | 0.4153 ± 0.1914   | 0.4153 ± 0.1914   | VERIFIED (Exact Match)|
+------------------------------------+------------------+-------------------+-------------------+-----------------------+
| GNN ABLATIONS & STRUCTURAL TESTS   |                  |                   |                   |                       |
| Unknown Context: GAT Retained/Rem  | Sec 5.1, Tab 7   | 0.3308 / 0.2918   | 0.3308 / 0.2918   | VERIFIED (Exact Match)|
| Unknown Context: GCN Retained/Rem  | Sec 5.1, Tab 7   | 0.2589 / 0.4555   | 0.2589 / 0.4555   | VERIFIED (Exact Match)|
| Unknown Context: SAGE Retained/Rem | Sec 5.1, Tab 7   | 0.2822 / 0.4132   | 0.2822 / 0.4132   | VERIFIED (Exact Match)|
| Direction: GAT (Fwd / Bwd / Bidir) | Sec 5.2, Tab 8   | 0.3308/0.3938/.4618 0.3308/0.3938/.4618| VERIFIED (Exact Match)|
| Direction: GCN (Fwd / Bwd / Bidir) | Sec 5.2, Tab 8   | 0.2589/0.3953/.3142 0.2589/0.3953/.3142| VERIFIED (Exact Match)|
| Direction: SAGE (Fwd/ Bwd / Bidir) | Sec 5.2, Tab 8   | 0.2822/0.3213/.3710 0.2822/0.3213/.3710| VERIFIED (Exact Match)|
| Depth: GCN (1L / 2L / 3L) F1       | Sec 5.3, Tab 9   | 0.2862/0.2589/.2089 0.2862/0.2589/.2089| VERIFIED (Exact Match)|
| Depth: GAT (1L / 2L / 3L) F1       | Sec 5.3, Tab 9   | 0.2877/0.3308/.2847 0.2877/0.3308/.2847| VERIFIED (Exact Match)|
| Depth: GraphSAGE (1L / 2L / 3L) F1 | Sec 5.3, Tab 9   | 0.3750/0.2822/.3327 0.3750/0.2822/.3327| VERIFIED (Exact Match)|
+------------------------------------+------------------+-------------------+-------------------+-----------------------+
| ERROR TAXONOMY & DEGREE METRICS    |                  |                   |                   |                       |
| True Positives (TP) Count / In-Deg | Sec 7.1, Tab 11  | 402 / 0.87        | 402 / 0.87        | VERIFIED (Exact Match)|
| True Negatives (TN) Count / In-Deg | Sec 7.1, Tab 11  | 8,165 / 2.38      | 8,165 / 2.38      | VERIFIED (Exact Match)|
| False Positives (FP) Count / In-Deg| Sec 7.1, Tab 11  | 2,383 / 1.23      | 2,383 / 1.23      | VERIFIED (Exact Match)|
| False Negatives (FN) Count / In-Deg| Sec 7.1, Tab 11  | 234 / 1.61        | 234 / 1.61        | VERIFIED (Exact Match)|
| High-Confidence Errors (FP/FN)     | Sec 7.1, Tab 11  | 0 (0.00%) / 0     | 0 (0.00%) / 0     | VERIFIED (Exact Match)|
| Timestep 43 Illicit Rate / F1      | Sec 7.3, Tab 11  | 1.75% / 0.0718    | 1.75% / 0.0718    | VERIFIED (Exact Match)|
+------------------------------------+------------------+-------------------+-------------------+-----------------------+
```

---

# 3. Split Consistency Audit

A strict audit of all temporal partitioning references across all five draft files was conducted:
- **Verified Partition Standard:**
  - **Training Split:** Timesteps 1–34 ($N = 29,894$ labeled nodes, $11.58\%$ illicit prevalence)
  - **Validation Split:** Timesteps 35–39 ($N = 5,486$ labeled nodes, $8.15\%$ illicit prevalence)
  - **Blind Test Split:** Timesteps 40–49 ($N = 11,184$ labeled nodes, $5.69\%$ illicit prevalence)
- **Search Results:** A regex search across all manuscript documents for legacy uncalibrated split bounds (`1-28`, `1–28`, `29-34`, `35-49`) returned **zero occurrences**.
- **Audit Verdict:** **PASS (100% Consistent across all sections).**

---

# 4. Feature Schema and Dimension Audit

- **Feature Dimensions:** The manuscript uniformly defines the feature schema as **$d = 165$ total machine learning features**, structured into:
  - **93 Local Features (Features 1–93):** Intrinsic transaction metadata (e.g., transacted Bitcoin value, transaction fee, number of inputs/outputs, script version, lock time).
  - **72 Aggregate Features (Features 94–165):** 1-hop neighborhood statistics computed within the discrete timestep (e.g., minimum, maximum, mean, and standard deviation of neighbor transaction values, fees, and degrees).
- **Raw CSV Distinction:** Section 2 and Section 3 correctly distinguish the 165 feature dimensions from the 167 raw CSV columns by noting that `txId` (Transaction ID) and `time_step` (Temporal index) are non-feature identifiers.
- **Audit Verdict:** **PASS (Zero occurrences of erroneous 166-feature descriptions).**

---

# 5. Unknown-Label Handling Audit

- **Entity Accounting:** The dataset contains $157,205$ unlabeled nodes ($77.15\%$) out of $203,769$ total transactions.
- **GNN Primary Graph Formulation:** In all primary GNN experiments, unknown nodes are explicitly retained in the graph topology to preserve relational connectivity, but their classification loss is masked out:
$$\mathcal{L}_{\text{masked}} = -\frac{1}{|\mathcal{V}_{\text{train}}^{\text{labeled}}|} \sum_{v \in \mathcal{V}_{\text{train}}^{\text{labeled}}} \left[ w_{\text{pos}} y_v \log \hat{y}_v + (1 - y_v) \log (1 - \hat{y}_v) \right]$$
- **Tabular Model Formulation:** Unknown-label transactions are strictly excluded from supervised tabular training and evaluation.
- **Ablation Clarity:** Section 5.1 and Table 7 unambiguously describe the ablation as comparing the full $100\%$ topology (retained) against the $22.85\%$ labeled-only induced subgraph (removed).
- **Audit Verdict:** **PASS.**

---

# 6. Graph Propagation Directionality Audit

- **Primary Benchmark:** Forward propagation ($u \to v$, input transactions to output transactions) is confirmed as the primary inductive configuration in Sections 3, 4, 5, and 8, preserving causal fund flows.
- **Sensitivity Analyses:** Backward ($v \to u$, upstream funding provenance) and Bidirectional ($u \leftrightarrow v$) message passing are explicitly bounded as retrospective sensitivity analyses.
- **Production Guardrail Check:** Section 5.2 and Section 8.8 contain explicit statements warning against deploying backward propagation in live, streaming transaction screening systems because downstream outputs do not exist at the moment of live transaction broadcast.
- **Audit Verdict:** **PASS.**

---

# 7. Model Selection and Checkpoint Selection Audit

- **Validation-Locked Checkpointing:** Section 3.4 and Section 6.1 explicitly establish that:
  - GNN training checkpoints were selected based strictly on peak validation PR-AUC ($\arg\max \text{PR-AUC}_{\text{val}}$);
  - Decision thresholds ($\tau^*$) were locked on the validation set ($\arg\max \text{F1}_{\text{val}}$) prior to test set evaluation;
  - No holdout test metrics were utilized for model selection, hyperparameter tuning, or threshold optimization.
- **Representative GNN in Statistical Tests:** Section 6.1 and Section 8.10 properly selected **GraphSAGE on Local 93 Features** based on its top validation F1 score ($0.6060 \pm 0.2270$), strictly avoiding test-score post-hoc selection bias.
- **Audit Verdict:** **PASS.**

---

# 8. Scaling and Preprocessing Audit

- **Standardization Integrity:** `StandardScaler` parameters ($\mu, \sigma$) were fitted strictly on training data (Timesteps 1–34) and applied to validation (35–39) and test (40–49) splits without re-fitting.
- **Zero Global Leakage:** Confirmed that no global dataset-level scaling, PCA transformations, SMOTE/ADASYN oversampling, or transductive test neighbor fitting occurred.
- **Audit Verdict:** **PASS.**

---

# 9. Statistical Hypothesis Testing Audit

All hypothesis tests in Section 6.1 and Table 10 were verified against [`MASTER_STATISTICAL_TESTS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_STATISTICAL_TESTS.csv):
- **GraphSAGE Local vs. Random Forest (Full, Std):**
  - $\bar{\Delta} = -0.3094$, $t = -3.5917$, $p_{\text{raw}} = 0.0229$, $p_{\text{Holm}} = 0.0917$, $W = 0.0$, $p_{\text{Wilcoxon}} = 0.0625$, $d_z = -1.6062$.
- **GraphSAGE Local vs. XGBoost (Full, Weighted):**
  - $\bar{\Delta} = -0.3044$, $t = -3.5092$, $p_{\text{raw}} = 0.0247$, $p_{\text{Holm}} = 0.0917$, $W = 0.0$, $p_{\text{Wilcoxon}} = 0.0625$, $d_z = -1.5694$.
- **Significance Reporting Audit:** The text correctly states that while unadjusted $p$-values were $< 0.05$, **no baseline comparison achieves statistical significance at $\alpha = 0.05$ after Holm-Bonferroni correction ($p_{\text{Holm}} \ge 0.0917$) or under Wilcoxon testing ($p = 0.0625$)**.
- **Wilcoxon Bound Disclosure:** Section 6.2 and Section 8.10 correctly prove that for $N=5$, the minimum achievable two-sided Wilcoxon $p$-value is $2/32 = 0.0625$, providing necessary transparency regarding small-sample inferential power.
- **Audit Verdict:** **PASS.**

---

# 10. Homophily Disambiguation Audit

A critical check was conducted on relational homophily reporting across Sections 2, 7, and 8:
- **Global Labeled Edge Homophily ($95.37\%$):** Formally defined as the proportion of same-class edges ($33,930$ licit-licit + $998$ illicit-illicit = $34,928$) over the $36,624$ edges ($15.63\%$) connecting mutually labeled pairs:
$$h_{\text{labeled}} = \frac{34,928}{36,624} = 95.37\%$$
- **Unlabeled Edge Reality ($84.37\%$):** Section 7.2 and Section 8.11 explicitly emphasize that $197,731$ edges ($84.37\%$) touch unannotated nodes, explaining why high labeled homophily does not shield spatial GNNs from unannotated neighborhood noise.
- **Audit Verdict:** **PASS (Clear conceptual separation maintained; no conflation of global edge homophily with local node-level metrics).**

---

# 11. Timestep 43 Distribution Shift Audit

- **Empirical Statistics:** Section 7.3 and Section 8.12 accurately document that at Timestep 43:
  - Illicit prevalence dropped from $11.10\%$ ($t=42$) to $1.75\%$ ($t=43$);
  - Illicit F1 collapsed from $0.5234$ to $0.0718$;
  - Precision dropped to $0.0409$ (over $95\%$ false positive rate among flagged transactions).
- **Causal Attribution Guardrail:** Section 7.3, Section 8.12, and Table 13 explicitly treat the AlphaBay/Hansa marketplace shutdown as an external hypothesis requiring independent verification (`[CITATION REQUIRED — only if external historical attribution is retained]`), presenting the event within the paper strictly as an observed statistical distribution shift.
- **Audit Verdict:** **PASS.**

---

# 12. Temporal Drop Recalculation Audit

Table 2 verifies the mathematical accuracy of all validation-to-test performance drops reported in Section 4.6 and Section 8.5:
$$\text{Relative Drop} = \frac{\text{Val F1} - \text{Test F1}}{\text{Val F1}} \times 100$$

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 2: Validation-to-Test Generalization Drop Verification                                                          |
+------------------------------+---------------+---------------+---------------+--------------------+-------------------+
| Model Architecture / Config  | Validation F1 | Test F1       | Absolute Δ    | Recalculated Drop  | Manuscript Text   |
+------------------------------+---------------+---------------+---------------+--------------------+-------------------+
| Random Forest (Full 165)     | 0.9489        | 0.7247        | -0.2242       | -23.627% (-23.6%)  | -23.6% (VERIFIED) |
| XGBoost (Full 165)           | 0.9399        | 0.7131        | -0.2268       | -24.130% (-24.1%)  | -24.1% (VERIFIED) |
| MLP (Full 165)               | 0.8212        | 0.5848        | -0.2364       | -28.787% (-28.8%)  | -28.8% (VERIFIED) |
| GAT (Full 165, Weighted)     | 0.6006        | 0.3308        | -0.2698       | -44.922% (-44.9%)  | -44.9% (VERIFIED) |
| GraphSAGE (Local 93, Val-Sel)| 0.6060        | 0.4153        | -0.1907       | -31.469% (-31.5%)  | -31.5% (VERIFIED) |
| GCN (Full 165, Weighted)     | 0.4293        | 0.2589        | -0.1704       | -39.693% (-39.7%)  | -39.7% (VERIFIED) |
+------------------------------+---------------+---------------+---------------+--------------------+-------------------+
```
- **Audit Verdict:** **PASS (All relative percentage drops verified to 1 decimal place).**

---

# 13. Complete Table and Figure Inventory Audit

Table 3 inventories all 13 manuscript tables and 15 figures, verifying their source files, section placement, and formatting status.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 3: Complete Table and Figure Inventory                                                                          |
+-----------+-------------------------------------------------------+---------------------+-----------------------------+
| Item ID   | Title / Description                                   | Section Placement   | Source File / Status        |
+-----------+-------------------------------------------------------+---------------------+-----------------------------+
| Table 1   | Elliptic Dataset Partitioning & Class Composition     | Sec 2 (Dataset)     | PAPER_STEP_4C.md (Verified) |
| Table 2   | Model Architecture & Hyperparameter Specification     | Sec 3 (Methodology) | PAPER_STEP_4C.md (Verified) |
| Table 3   | Main Benchmark Results on Full Features (165 Dim)     | Sec 4.1 (Results)   | MASTER_MAIN_RESULTS.csv     |
| Table 4   | Feature Representation Comparison (Local vs. Full)    | Sec 4.4 (Results)   | MASTER_FEATURE_COMPARISON   |
| Table 5   | Class Weighting Comparison (Standard vs. Weighted)    | Sec 4.5 (Results)   | MASTER_WEIGHTING_COMPARISON |
| Table 6   | Multi-Seed Stability & Parameter Initialization (S=5) | Sec 4.7 (Results)   | MASTER_SEED_STABILITY.csv   |
| Table 7   | Unknown-Node Context Ablation (Retained vs Removed)   | Sec 5.1 (Ablations) | MASTER_GNN_ABLATIONS.csv    |
| Table 8   | Graph Propagation Directionality Ablation             | Sec 5.2 (Ablations) | MASTER_GNN_ABLATIONS.csv    |
| Table 9   | Message-Passing Depth Ablation (1, 2, 3 Layers)       | Sec 5.3 (Ablations) | MASTER_GNN_ABLATIONS.csv    |
| Table 10  | Statistical Significance Tests & Effect Sizes (N=5)   | Sec 6.1 (Stats)     | MASTER_STATISTICAL_TESTS    |
| Table 11  | Error Taxonomy & Degree Disparity Analysis            | Sec 7.1 (Errors)    | MASTER_ERROR_ANALYSIS.csv   |
| Table 12  | Research Question Answer Matrix                       | Sec 11 (Discussion) | PAPER_STEP_4F.md (Verified) |
| Table 13  | Final Claim-Boundary Audit Checklist                  | Sec 12 (Audit)      | PAPER_STEP_4F.md (Verified) |
+-----------+-------------------------------------------------------+---------------------+-----------------------------+
| Figure 1  | Class Distribution across 49 Timesteps (300 DPI)      | Sec 2 / Sec 3       | label_distribution.png      |
| Figure 2  | Temporal Transaction Volume Dynamics (300 DPI)        | Sec 2 / Sec 3       | temporal_evolution.png      |
| Figure 3  | Main Benchmark Comparison (F1 & PR-AUC) (300 DPI)     | Sec 4.1             | gnn_vs_conventional_baseline|
| Figure 4  | Precision-Recall Curves across Primary GNNs (300 DPI) | Sec 4.2             | gnn_precision_recall_curves |
| Figure 5  | Feature Space Impact (Local 93 vs Full 165) (300 DPI) | Sec 4.4             | feature_ablation_local_vs_fu|
| Figure 6  | Unknown-Node Context Ablation (300 DPI)               | Sec 5.1             | unknown_node_ablation.png   |
| Figure 7  | Directionality Sensitivity Ablation (300 DPI)         | Sec 5.2             | direction_sensitivity_ablat |
| Figure 8  | Message-Passing Depth Ablation (300 DPI)              | Sec 5.3             | depth_ablation.png          |
| Figure 9  | Multi-Seed Stability & Variance Boxplots (300 DPI)    | Sec 6.1             | gnn_model_comparison_f1.png |
| Figure 10 | Edge Class Transition & Homophily Matrix (300 DPI)    | Sec 7.2             | edge_homophily_matrix.png   |
| Figure 11 | Timestep-by-Timestep Performance Breakdown (300 DPI)  | Sec 7.3             | gnn_temporal_performance_tim|
| Figure S1 | Conventional ML Baseline Validation/Test Summary      | Supplementary       | baseline_model_comparison.pn|
| Figure S2 | Receiver Operating Characteristic (ROC) Curves        | Supplementary       | gnn_roc_curves.png          |
| Figure S3 | Computational Latency & Throughput Benchmark          | Supplementary       | gnn_compute_benchmark.png   |
| Figure S4 | Illicit Transaction Prevalence Collapse Dynamics      | Supplementary       | illicit_ratio_over_time.png |
+-----------+-------------------------------------------------------+---------------------+-----------------------------+
```

---

# 8. Terminology Standardization Audit

Table 4 catalogs terminology usage across all manuscript files to enforce uniform academic naming conventions.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 4: Terminology Standardization Matrix                                                                           |
+--------------------------+-------------------------------------+-----------------------------------+------------------+
| Domain / Entity          | Standardized Preferred Term         | Disallowed / Discouraged Variants | Audit Status     |
+--------------------------+-------------------------------------+-----------------------------------+------------------+
| Primary Task             | illicit transaction detection       | fraud detection, crime detection  | STANDARDIZED     |
| Target Class             | illicit transactions (Class 1)      | fraudulent nodes, criminal txs    | STANDARDIZED     |
| Majority Class           | licit transactions (Class 2 / 0)    | normal nodes, legal transactions  | STANDARDIZED     |
| Unlabeled Transactions   | unknown (unlabeled) transactions    | ungrounded nodes, missing labels  | STANDARDIZED     |
| Non-Graph Classifiers    | conventional tabular baselines      | traditional ML, classical models  | STANDARDIZED     |
| Tree Models              | tree-based ensembles                | random decision forests, trees    | STANDARDIZED     |
| Feature Representation   | Local (93) vs. Full (165) features  | basic vs advanced, raw vs engin.  | STANDARDIZED     |
| Propagation Direction    | Forward (causal) vs. Backward (prov)| downstream vs upstream flow       | STANDARDIZED     |
| Evaluation Setting       | chronological temporal holdout      | dynamic split, sequential split   | STANDARDIZED     |
+--------------------------+-------------------------------------+-----------------------------------+------------------+
```

---

# 9. Claim and Scientific Tone Audit

A full lexical scan was performed across all manuscript text for absolute, causal, or ranking terms:
- **Scan Targets:** `proves`, `guarantees`, `always`, `never`, `superior`, `inferior`, `best`, `worst`, `state-of-the-art`, `SOTA`, `breakthrough`, `causes`, `oversmoothing`.
- **Findings & Actions:**
  - **"Superior" / "Inferior" / "Best" / "Worst":** Strictly avoided in all model descriptions. Replaced with descriptive comparisons (e.g., *"achieved higher holdout test illicit F1-scores under the evaluated protocol"*).
  - **"Proves" / "Guarantees":** Absent from all analytical interpretations. Replaced with *"the empirical findings indicate"* and *"is consistent with"*.
  - **"Causes" / "Causal":** Bounded strictly to edge flow semantics (*"causal forward payment flow"*). No empirical correlations (degree, homophily) are described as causal drivers of prediction failure.
  - **"Over-smoothing":** Evaluated in Section 5.3 and Section 8.9 as a theoretical explanatory hypothesis consistent with isotropic GCN depth degradation, with explicit disclosure that layer-wise representation distance was not directly measured.
- **Audit Verdict:** **PASS (100% Compliant with Scientific Claim Boundaries).**

---

# 10. Citation Audit and Consolidated Literature Mapping

Table 5 groups all 11 explicit `[CITATION REQUIRED]` placeholders identified across the manuscript into thematic literature categories.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 5: Consolidated Citation Mapping Matrix                                                                         |
+--------------------+---------------------+---------------------------------------------+------------------------------+
| Citation Category  | Manuscript Sections | Required Academic Source                    | Reference Examples           |
+--------------------+---------------------+---------------------------------------------+------------------------------+
| Dataset Benchmark  | Sec 1.2, 2.1, 9.1   | Original Elliptic Bitcoin dataset release   | Weber et al. (KDD 2019)      |
| Bitcoin Forensic   | Sec 1.1, 1.2, 8.2   | Blockchain AML and transaction forensics    | Meiklejohn et al., Moseley   |
| Graph Convolutions | Sec 1.3, 3.2        | Graph Convolutional Networks (GCN)          | Kipf & Welling (ICLR 2017)   |
| Inductive Sampling | Sec 1.3, 3.2        | Graph Sample and Aggregate (GraphSAGE)      | Hamilton et al. (NeurIPS 2017|
| Attention Networks | Sec 1.3, 3.2        | Graph Attention Networks (GAT)              | Veličković et al. (ICLR 2018)|
| Tabular Tree Ensem.| Sec 1.1, 3.1, 8.2   | Random Forest / XGBoost algorithms          | Breiman (2001), Chen & Guestr|
| Class Imbalance    | Sec 1.2, 3.3, 4.2   | Precision-Recall AUC under severe skew      | Davis & Goadrich (ICML 2006) |
| Multiple Hyp. Test.| Sec 3.5, 6.1, 8.10  | Holm-Bonferroni FWER step-down correction   | Holm (1979)                  |
| Non-Parametric Test| Sec 3.5, 6.2, 8.10  | Wilcoxon signed-rank paired test            | Wilcoxon (1945)              |
| Over-Smoothing     | Sec 5.3, 8.9        | Theoretical GNN depth limits & oversmoothing| Li et al. (AAAI 2018), Oono  |
| Dynamic Graph NNs  | Sec 9.4, 10.2       | Continuous-time temporal GNNs (TGAT, TGN)   | Xu et al. (ICLR), Rossi et al|
+--------------------+---------------------+---------------------------------------------+------------------------------+
```

---

# 11. Abstract ↔ Results Alignment Audit

The Abstract ([`PAPER_STEP_4B.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_STEP_4B.md)) was audited line-by-line against the empirical results in Section 4 and Section 8:
- Abstract states: Random Forest $\text{F1} = 0.7247 \pm 0.0027$, $\text{PR-AUC} = 0.6599 \pm 0.0025 \implies$ **Matches Table 3 exactly**.
- Abstract states: XGBoost $\text{F1} = 0.7131 \pm 0.0111$, $\text{PR-AUC} = 0.6744 \pm 0.0028 \implies$ **Matches Table 3 exactly**.
- Abstract states: GAT $\text{F1} = 0.3308 \pm 0.0602$, GraphSAGE $\text{F1} = 0.2822 \pm 0.1248$, GCN $\text{F1} = 0.2589 \pm 0.0639 \implies$ **Matches Table 3 exactly**.
- Abstract states: $203,769$ transactions, $234,355$ edges, $49$ timesteps, $77\%$ unknown nodes, splits 1–34 / 35–39 / 40–49, 5 seeds $\implies$ **Matches Section 2 and 3 exactly**.
- Abstract conclusion correctly qualifies findings without claiming universal GNN inferiority.
- **Audit Verdict:** **PASS.**

---

# 12. Research Contributions Audit

Each of the four major contributions declared in Section 1.4 was checked against the experimental execution:
1. **Systematic Chronological Benchmark:** Supported by Section 4.1, Table 3, Table 6, and Figures 3–4.
2. **Feature Representation & Manual Aggregation Dissection:** Supported by Section 4.4, Table 4, and Figure 5.
3. **Graph Topological Sensitivity & Design Space Ablation:** Supported by Section 5.1–5.3, Tables 7–9, and Figures 6–8.
4. **Temporal Non-Stationarity & Error Characterization:** Supported by Section 6.4, Section 7.1–7.3, Table 10, Table 11, and Figures 10–11.
- **Audit Verdict:** **PASS (All contributions map directly to executed experiments).**

---

# 13. Research Question Closure Audit

- **RQ1 (Benchmark Comparison):** Fully answered in Section 4.1, Section 8.1, and Table 12.
- **RQ2 (Feature Space Representation):** Fully answered in Section 4.4, Section 8.3, and Table 12.
- **RQ3 (Graph Topological Sensitivities):** Fully answered in Section 5.1–5.3, Section 8.7–8.9, and Table 12.
- **RQ4 (Temporal Generalization & Seed Stability):** Fully answered in Section 4.6–4.7, Section 6.1–6.4, Section 8.5–8.6, and Table 12.
- **Audit Verdict:** **PASS (All four RQs rigorously resolved).**

---

# 14. Reproducibility Protocol Audit

The manuscript provides complete specifications for exact experimental replication:
- **Data Loading & Split:** Discrete timestep masks ($t \le 34$ train, $35 \le t \le 39$ val, $40 \le t \le 49$ test) explicitly defined.
- **Preprocessing:** `StandardScaler` fitting rules, feature sub-indexing (Local: cols $0-92$, Full: cols $0-164$), and loss masking equations given in Section 3.
- **Hyperparameters:** Exact learning rates ($0.001$), weight decay ($10^{-5}$), hidden dimensions ($128$), layer counts ($2$), dropout ($0.3$), attention heads ($4$), batch size (full graph per timestep), epochs ($100$), and early stopping patience ($15$) specified in Table 2.
- **Evaluation:** Threshold scanning grid ($0.01$ to $0.99$, step $0.01$), locking on validation $\text{F1}$, and multi-seed loop ($S = \{42, 123, 456, 789, 999\}$) completely documented.
- **Audit Verdict:** **PASS.**

---

# 15. Final Integrity Scorecard

Table 6 provides the multi-dimensional integrity scorecard for the research manuscript.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 6: Manuscript Scientific Integrity Scorecard                                                                    |
+-----------------------------------+-----------------------------------+-----------------------------------------------+
| Audit Dimension                   | Integrity Status                  | Audit Observations & Verification Notes       |
+-----------------------------------+-----------------------------------+-----------------------------------------------+
| 1. Numerical Consistency          | PASS                              | 100% exact match across all master CSV tables |
| 2. Temporal Partitioning          | PASS                              | Strict 1–34 / 35–39 / 40–49 adherence         |
| 3. Feature Schema Accounting      | PASS                              | Exactly 165 ML features (93 loc + 72 agg)     |
| 4. Unknown-Label Handling         | PASS                              | 77.15% masked in loss, retained in GNN graph  |
| 5. Graph Directionality Bounds    | PASS                              | Forward causal benchmark strictly maintained  |
| 6. Preprocessing & Leakage Audit  | PASS                              | Scalers and thresholds locked on train/val    |
| 7. Model Selection Protocol       | PASS                              | Validation PR-AUC and F1 locked prior to test |
| 8. Statistical Hypothesis Tests   | PASS                              | Holm-adjusted p-values and N=5 bounds reported|
| 9. Homophily Disambiguation       | PASS                              | 95.37% labeled homophily vs 84.37% unknown    |
| 10. Timestep 43 Attribution       | PASS                              | External darknet event marked for verification|
| 11. Generalization Drop Math      | PASS                              | All relative percentage declines verified     |
| 12. Table / Figure Numbering      | PASS                              | Tables 1–13 and Figures 1–11 aligned          |
| 13. Terminology Standardization   | PASS                              | Academic nomenclature unified across sections |
| 14. Scientific Claim Boundaries   | PASS                              | Zero unsupported causal or ranking assertions |
| 15. Citation Mapping              | PASS                              | 11 targeted [CITATION REQUIRED] markers mapped|
| 16. Abstract Alignment            | PASS                              | Abstract metrics mirror Table 3 exactly       |
| 17. Contribution Grounding        | PASS                              | All 4 contributions supported by experiments  |
| 18. Research Question Closure     | PASS                              | RQ1–RQ4 resolved with empirical evidence      |
| 19. Reproducibility Completeness  | PASS                              | Hyperparameters and protocols fully documented|
+-----------------------------------+-----------------------------------+-----------------------------------------------+
```

---

# 16. Required Corrections and Alignment Notes

During the Step 4G audit, one minor prompt-level discrepancy was identified and verified:
- **Directionality Prompt Examples vs. Master CSV:** The introductory prompt listed preliminary retrospective values for backward/bidirectional propagation. As audited in Section 5.2 and Step 4E, [`MASTER_GNN_ABLATIONS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_GNN_ABLATIONS.csv) is the authoritative empirical source. Step 4E correctly incorporated the authoritative master CSV values:
  - GAT: Forward $= 0.3308$, Backward $= 0.3938$, Bidirectional $= 0.4618$;
  - GCN: Forward $= 0.2589$, Backward $= 0.3953$, Bidirectional $= 0.3142$;
  - GraphSAGE: Forward $= 0.2822$, Backward $= 0.3213$, Bidirectional $= 0.3710$.
- **Action Taken:** Authoritative CSV values are confirmed across all tables and discussion sections. No text modifications required.

---

# 17. Final Recommendation

**The entire manuscript across Steps 4B, 4C, 4D, 4E, and 4F has passed all 19 verification categories of the scientific integrity audit with zero blocking defects.**

The manuscript is ready for compilation into the unified publication paper in **Step 5 / Final Manuscript Assembly**.
