# Scientific Comparability Audit: Phase 2 Baselines vs. Phase 3 Graph Neural Networks

**Project:** Graph Neural Network-Based Fraud Detection in Financial Transaction Networks  
**Dataset:** Elliptic Bitcoin Transaction Dataset (203,769 transactions, 49 discrete timesteps)  
**Audit Purpose:** Verify whether Phase 2 conventional machine learning baselines and Phase 3 GNN experimental results are scientifically comparable for joint inclusion in a research paper.

---

## 1. Temporal Split Verification

| Property | Phase 2 Baselines | Phase 3 GNN Models | Match Status | Source Code / File Verification |
| :--- | :---: | :---: | :---: | :--- |
| **Training Split** | Timesteps **1–34** | Timesteps **1–34** | **EXACT MATCH** | [`configs/baseline_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/baseline_config.yaml#L9)<br>[`configs/gnn_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/gnn_config.yaml#L10) |
| **Validation Split** | Timesteps **35–39** | Timesteps **35–39** | **EXACT MATCH** | [`configs/baseline_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/baseline_config.yaml#L10)<br>[`configs/gnn_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/gnn_config.yaml#L11) |
| **Test Split** | Timesteps **40–49** | Timesteps **40–49** | **EXACT MATCH** | [`configs/baseline_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/baseline_config.yaml#L11)<br>[`configs/gnn_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/gnn_config.yaml#L12) |
| **Train Labeled Nodes** | 29,894 | 29,894 | **EXACT MATCH** | [`results/tables/temporal_split_distribution.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/temporal_split_distribution.csv#L2) |
| **Val Labeled Nodes** | 5,486 | 5,486 | **EXACT MATCH** | [`results/tables/temporal_split_distribution.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/temporal_split_distribution.csv#L3) |
| **Test Labeled Nodes** | 11,184 | 11,184 | **EXACT MATCH** | [`results/tables/temporal_split_distribution.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/temporal_split_distribution.csv#L4) |
| **Train Licit / Illicit** | 26,432 / 3,462 (11.58%) | 26,432 / 3,462 (11.58%) | **EXACT MATCH** | [`src/training/temporal_split.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/temporal_split.py#L72-L88)<br>[`src/data/masks.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/data/masks.py#L48-L66) |
| **Val Licit / Illicit** | 5,039 / 447 (8.15%) | 5,039 / 447 (8.15%) | **EXACT MATCH** | Same |
| **Test Licit / Illicit** | 10,548 / 636 (5.69%) | 10,548 / 636 (5.69%) | **EXACT MATCH** | Same |

---

## 2. Label Handling & Unknown Node Semantics

1. **Label Encoding:**
   - **Phase 2:** Mapped in [`src/training/temporal_split.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/temporal_split.py#L53-L56) as `(df["class"] == "1").astype(np.int64)` $\implies$ Class 1 (Illicit) = `1`, Class 2 (Licit) = `0`.
   - **Phase 3:** Mapped in [`src/data/graph_builder.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/data/graph_builder.py#L101-L108) as `y_arr[class == "1"] = 1.0`, `y_arr[class == "2"] = 0.0`, `y_arr[class == "unknown"] = -1.0`.
2. **Supervised Loss Exclusion:**
   - **Phase 2:** Unknown nodes are filtered out prior to tabular training; supervised loss is computed solely on labeled instances ($N_{\text{train}} = 29,894$).
   - **Phase 3:** Unknown nodes are masked out via boolean tensors (`train_mask`, `val_mask`, `test_mask` all evaluate to `False` for unknown nodes). PyTorch Geometric loss is computed exclusively on `logits[train_mask], data.y[train_mask]` in [`src/training/gnn_trainer.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/gnn_trainer.py#L152).
3. **Graph Message-Passing Role:**
   - **Phase 2:** Conventional models evaluate individual node feature rows independently without graph topology.
   - **Phase 3:** Unknown nodes (157,205 transactions, ~77.1% of graph) remain in the PyTorch Geometric graph structure to preserve connectivity and multi-hop structural paths during message passing, but never contribute to loss gradients or performance evaluation.
4. **Validation and Test Sets:**
   - Both phases evaluate classification performance strictly on labeled validation ($N = 5,486$) and labeled test ($N = 11,184$) instances.

---

## 3. Feature Space Alignment

| Feature Mode | Phase 2 Column Selection | Phase 3 Column Selection | Dimensions | Metadata Exclusions |
| :--- | :--- | :--- | :---: | :--- |
| **Local Only** | `[c for c in cols if c.startswith("local_feat_")]` ([`src/features/preprocessing.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/features/preprocessing.py#L40)) | `[c for c in cols if c.startswith("local_feat_")]` ([`src/data/graph_builder.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/data/graph_builder.py#L83)) | **93** | `txId` excluded<br>`time_step` excluded |
| **Full Features** | `[c for c in cols if c.startswith("local_feat_") or c.startswith("agg_feat_")]` ([`src/features/preprocessing.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/features/preprocessing.py#L42)) | `[c for c in cols if c.startswith("local_feat_") or c.startswith("agg_feat_")]` ([`src/data/graph_builder.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/data/graph_builder.py#L85)) | **165** | `txId` excluded<br>`time_step` excluded |

- **Verification:** The feature subsets, column names, column order, and dimensionality are identical across Phase 2 and Phase 3.

---

## 4. Feature Scaling & Preprocessing

- **Phase 2 Preprocessing:**
  - *Logistic Regression & MLP:* Scaled using `StandardScaler` strictly fit on `df_train` ([`src/training/train_baselines.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/train_baselines.py#L101-L106)). Validation and test sets are transformed using the training-fitted scaler parameters.
  - *Random Forest & XGBoost:* Ingest unscaled raw tabular features ([`src/training/train_baselines.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/train_baselines.py#L108-L111)), as decision tree partition algorithms are scale-invariant.
- **Phase 3 Preprocessing:**
  - *GCN, GraphSAGE, GAT:* Scaled using `StandardScaler` strictly fit on training nodes (`train_features = raw_x[train_mask_np]`) in [`src/data/graph_builder.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/data/graph_builder.py#L89-L97). The full node matrix is transformed with the training moments ($\mu_{\text{train}}, \sigma_{\text{train}}$).
- **Assessment:** Preprocessing follows standard domain best practices. Neural and linear architectures receive standardized inputs to prevent gradient pathology, while tree models utilize raw features. Zero data leakage is verified in both pipelines.

---

## 5. Class Imbalance Handling

- **Class Distribution:** Training split contains 26,432 licit vs. 3,462 illicit transactions (Imbalance Ratio = $7.6349 : 1$, 11.58% illicit).
- **Phase 2 Baseline Protocols:**
  - *Standard (Unweighted):* Models optimized with default empirical loss (`class_weight=None`, `scale_pos_weight=1.0`, standard BCE). Imbalance is addressed via post-hoc threshold tuning on the validation split.
  - *Weighted (Cost-Sensitive):* Models trained with cost-sensitive loss:
    - Random Forest: `class_weight="balanced"`
    - XGBoost: `scale_pos_weight = 7.6349`
    - MLP: `BCEWithLogitsLoss(pos_weight=7.6349)`
    - Logistic Regression: `class_weight="balanced"`
- **Phase 3 GNN Protocol:**
  - GNN models are trained with `BCEWithLogitsLoss(pos_weight=pos_weight)` where `pos_weight = 26432.0 / 3462.0 = 7.6349` computed strictly on `train_mask` ([`src/training/gnn_trainer.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/gnn_trainer.py#L130-L132)).
- **Comparability Rule:** Phase 3 GNNs share the exact positive-weighting formulation as Phase 2 `MLP (Weighted)` and `XGBoost (Weighted)`. In scientific reporting, results must be explicitly categorized by their loss formulation.

---

## 6. Threshold Selection & Optimization Pipeline

Both phases strictly adhere to the same 4-stage thresholding protocol:

$$\text{1. Train Model} \longrightarrow \text{2. Predict on Validation Split} \longrightarrow \text{3. Select } \tau^* = \arg\max_{\tau \in [0.01, 0.99]} \text{F1}_{\text{illicit}}(\tau) \longrightarrow \text{4. Lock } \tau^* \text{ and Evaluate Blind Test}$$

- **Phase 2:** [`src/training/train_baselines.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/train_baselines.py#L178-L189) calls `find_optimal_threshold(y_true_val=y_val, y_prob_val=val_probs, metric="f1")`.
- **Phase 3:** [`src/training/gnn_trainer.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/gnn_trainer.py#L198-L206) calls `find_optimal_threshold(y_true_val=y_val_np, y_prob_val=val_probs_best, metric="f1")`.
- **Zero Test Contamination:** In both phases, the test set is evaluated only after $\tau^*$ has been computed and locked.

---

## 7. Model Selection & Checkpoint Strategy

- **Phase 2:** Conventional models fit for fixed iterations/epochs on the training split. MLP uses validation loss for early stopping (patience = 7).
- **Phase 3:** GNN models train for up to 100 epochs with validation PR-AUC tracking evaluated every 2 epochs (`val_interval = 2`). The checkpoint with the highest validation PR-AUC (`best_model_state`) is restored before validation threshold optimization and test evaluation ([`src/training/gnn_trainer.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/training/gnn_trainer.py#L157-L190)).
- **Assessment:** Both pipelines select models/checkpoints strictly using validation metrics.

---

## 8. Random Seeds & Statistical Aggregation

- **Phase 2 Seeds:** `[42, 123, 456, 789, 999]` (5 seeds) configured in [`configs/baseline_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/baseline_config.yaml#L5).
- **Phase 3 Seeds:** `[42, 123, 456, 789, 999]` (5 seeds) configured in [`configs/gnn_config.yaml`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/configs/gnn_config.yaml#L5).
- **Assessment:** Both phases share the exact same 5 random seeds. Computing and reporting sample mean $\pm$ sample standard deviation across these 5 seeds is fully consistent.

---

## 9. Evaluation Metrics Implementation

Both Phase 2 and Phase 3 utilize the identical evaluation function `compute_classification_metrics()` in [`src/evaluation/metrics.py`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/src/evaluation/metrics.py#L21-L91):
- **Illicit F1:** `f1_score(y_true, y_pred, pos_label=1, zero_division=0)`
- **PR-AUC:** `average_precision_score(y_true, y_prob)` (positive class = 1)
- **Illicit Precision:** `precision_score(y_true, y_pred, pos_label=1, zero_division=0)`
- **Illicit Recall:** `recall_score(y_true, y_pred, pos_label=1, zero_division=0)`
- **Macro F1:** `f1_score(y_true, y_pred, average="macro", zero_division=0)`
- **ROC-AUC:** `roc_auc_score(y_true, y_prob)`
- **MCC:** `matthews_corrcoef(y_true, y_pred)`
- **Accuracy:** `accuracy_score(y_true, y_pred)`

---

## 10. Data Leakage Audit

| Leakage Category | Phase 2 Status | Phase 3 Status | Code Verification Location |
| :--- | :---: | :---: | :--- |
| **1. Fitting Scalers on Val/Test** | **PASS** | **PASS** | `src/features/preprocessing.py:55` (fit on `df_train`)<br>`src/data/graph_builder.py:95` (fit on `train_mask_np`) |
| **2. Feature Selection on Val/Test** | **PASS** | **PASS** | Predefined fixed feature sets (93 / 165); no empirical filtering on val/test |
| **3. PCA / Dimensionality Reduction** | **PASS** | **PASS** | No PCA or unsupervised dimensionality reduction applied |
| **4. Resampling before Temporal Split** | **PASS** | **PASS** | Temporal splitting performed chronologically on raw index; zero resampling |
| **5. Test-Based Threshold Tuning** | **PASS** | **PASS** | `src/training/train_baselines.py:179`<br>`src/training/gnn_trainer.py:199` (tuned on validation only) |
| **6. Test-Based Checkpoint Selection** | **PASS** | **PASS** | `src/training/gnn_trainer.py:168` (tracked strictly on `val_pr_auc`) |
| **7. Test-Based Hyperparameters** | **PASS** | **PASS** | A priori fixed model configurations in `configs/` |
| **8. Inclusion of Target Label in Features** | **PASS** | **PASS** | Label column `'class'` explicitly separated from input feature matrix |
| **9. Inclusion of Transaction ID (txId)** | **PASS** | **PASS** | Excluded via column prefix filtering (`local_feat_`, `agg_feat_`) |
| **10. Inclusion of Time Step (time_step)** | **PASS** | **PASS** | Excluded from feature matrices; used strictly for partitioning masks |
| **11. Graph Temporal Test-Edge Leakage** | **N/A** | **PASS** | Forward edge direction preserves transaction flow; loss strictly masked |

---

## 11. Direct Comparability Matrix

| Property | Phase 2 Baselines | Phase 3 GNNs | Comparable? | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Temporal Split** | Train: 1–34, Val: 35–39, Test: 40–49 | Train: 1–34, Val: 35–39, Test: 40–49 | **YES** | Identical time boundaries and sample counts |
| **Feature Space** | Local 93 & Full 165 | Local 93 & Full 165 | **YES** | Identical columns, definitions, and ordering |
| **Label Encoding** | Class 1 = 1, Class 2 = 0 | Class 1 = 1.0, Class 2 = 0.0, Unknown = -1.0 | **YES** | Supervised loss restricted to labeled nodes |
| **Unknown Nodes** | Excluded from tabular dataset | Retained in graph for message passing | **YES** | Methodologically appropriate for graph models |
| **Feature Scaling** | StandardScaler (fit on train) / Raw (trees) | StandardScaler (fit on train nodes) | **YES** | Model-appropriate and leakage-free |
| **Class Weighting** | Standard (1.0) and Weighted ($7.63$) | Weighted ($pos\_weight = 7.63$) | **YES** | Must compare matching weighting regimes |
| **Threshold Selection** | Val F1 optimization $\to$ locked test | Val F1 optimization $\to$ locked test | **YES** | Identical optimization protocol |
| **Model Selection** | Train convergence / Val loss (MLP) | Val PR-AUC early stopping & checkpointing | **YES** | Identical anti-leakage principle |
| **Random Seeds** | 5 seeds: `[42, 123, 456, 789, 999]` | 5 seeds: `[42, 123, 456, 789, 999]` | **YES** | Identical seed set |
| **Evaluation Metrics** | Illicit F1, PR-AUC, Prec, Rec, MCC | Illicit F1, PR-AUC, Prec, Rec, MCC | **YES** | Exact same metric implementation |
| **Leakage Controls** | 10/10 Passed | 11/11 Passed | **YES** | Zero data contamination |

---

## 12. Empirical Results Comparison Table

The table below presents the verified empirical results across all directly comparable configurations from [`results/tables/baseline_summary_mean_std.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/baseline_summary_mean_std.csv) and [`results/tables/gnn_summary_mean_std.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/gnn_summary_mean_std.csv).

### A. Full Feature Space (165 Features)

| Model Family | Model Configuration | Weighting | Validation F1 | Validation PR-AUC | Test Illicit F1 | Test PR-AUC | Test Precision | Test Recall | Test MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tree Ensemble** | **Random Forest** | Standard | $0.9489 \pm 0.0023$ | $0.9553 \pm 0.0008$ | **$0.7247 \pm 0.0027$** | **$0.6599 \pm 0.0025$** | $0.9586 \pm 0.0198$ | $0.5827 \pm 0.0042$ | $0.7367 \pm 0.0057$ |
| **Tree Ensemble** | **Random Forest** | Weighted | $0.9386 \pm 0.0023$ | $0.9527 \pm 0.0018$ | **$0.7024 \pm 0.0051$** | **$0.6617 \pm 0.0021$** | $0.8830 \pm 0.0221$ | $0.5833 \pm 0.0029$ | $0.7047 \pm 0.0080$ |
| **Gradient Boost** | **XGBoost** | Standard | $0.9399 \pm 0.0053$ | $0.9595 \pm 0.0015$ | **$0.7131 \pm 0.0111$** | **$0.6744 \pm 0.0028$** | $0.9127 \pm 0.0366$ | $0.5855 \pm 0.0062$ | $0.7189 \pm 0.0154$ |
| **Gradient Boost** | **XGBoost** | Weighted | $0.9416 \pm 0.0008$ | $0.9591 \pm 0.0017$ | **$0.7197 \pm 0.0074$** | **$0.6723 \pm 0.0039$** | $0.9238 \pm 0.0250$ | $0.5896 \pm 0.0035$ | $0.7264 \pm 0.0104$ |
| **Neural Net** | **MLP** | Standard | $0.8212 \pm 0.0344$ | $0.8446 \pm 0.0223$ | **$0.5848 \pm 0.0194$** | **$0.5115 \pm 0.0187$** | $0.7630 \pm 0.0430$ | $0.4761 \pm 0.0335$ | $0.5840 \pm 0.0155$ |
| **Neural Net** | **MLP** | Weighted | $0.8254 \pm 0.0135$ | $0.8432 \pm 0.0127$ | **$0.5513 \pm 0.0362$** | **$0.4840 \pm 0.0425$** | $0.7232 \pm 0.0524$ | $0.4456 \pm 0.0287$ | $0.5482 \pm 0.0397$ |
| **Linear** | **Logistic Regression** | Standard | $0.5921 \pm 0.0000$ | $0.4826 \pm 0.0000$ | **$0.4015 \pm 0.0000$** | **$0.2754 \pm 0.0000$** | $0.3821 \pm 0.0000$ | $0.4230 \pm 0.0000$ | $0.3640 \pm 0.0000$ |
| **Linear** | **Logistic Regression** | Weighted | $0.5527 \pm 0.0000$ | $0.4366 \pm 0.0000$ | **$0.3480 \pm 0.0000$** | **$0.2197 \pm 0.0000$** | $0.2678 \pm 0.0000$ | $0.4969 \pm 0.0000$ | $0.3128 \pm 0.0000$ |
| **GNN** | **GAT** | Weighted | $0.6006 \pm 0.0982$ | $0.5939 \pm 0.1191$ | **$0.3308 \pm 0.0602$** | **$0.2667 \pm 0.0695$** | $0.2680 \pm 0.0632$ | $0.4412 \pm 0.0281$ | $0.2899 \pm 0.0640$ |
| **GNN** | **GraphSAGE** | Weighted | $0.4473 \pm 0.1590$ | $0.4106 \pm 0.1738$ | **$0.2822 \pm 0.1248$** | **$0.2366 \pm 0.1386$** | $0.2695 \pm 0.1858$ | $0.3588 \pm 0.0756$ | $0.2391 \pm 0.1394$ |
| **GNN** | **GCN** | Weighted | $0.4293 \pm 0.0923$ | $0.4056 \pm 0.1386$ | **$0.2589 \pm 0.0639$** | **$0.2202 \pm 0.0703$** | $0.2788 \pm 0.1255$ | $0.2884 \pm 0.0839$ | $0.2213 \pm 0.0766$ |

---

### B. Local Feature Space (93 Features)

| Model Family | Model Configuration | Weighting | Validation F1 | Validation PR-AUC | Test Illicit F1 | Test PR-AUC | Test Precision | Test Recall | Test MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tree Ensemble** | **Random Forest** | Standard | $0.9296 \pm 0.0011$ | $0.9491 \pm 0.0011$ | **$0.6936 \pm 0.0033$** | **$0.6495 \pm 0.0003$** | $0.9022 \pm 0.0140$ | $0.5635 \pm 0.0087$ | $0.7004 \pm 0.0024$ |
| **Tree Ensemble** | **Random Forest** | Weighted | $0.9287 \pm 0.0015$ | $0.9494 \pm 0.0003$ | **$0.6723 \pm 0.0035$** | **$0.6499 \pm 0.0017$** | $0.8988 \pm 0.0120$ | $0.5371 \pm 0.0084$ | $0.6818 \pm 0.0018$ |
| **Gradient Boost** | **XGBoost** | Standard | $0.9185 \pm 0.0028$ | $0.9524 \pm 0.0019$ | **$0.6701 \pm 0.0109$** | **$0.6554 \pm 0.0031$** | $0.8746 \pm 0.0322$ | $0.5440 \pm 0.0214$ | $0.6758 \pm 0.0093$ |
| **Gradient Boost** | **XGBoost** | Weighted | $0.9213 \pm 0.0024$ | $0.9546 \pm 0.0013$ | **$0.6756 \pm 0.0108$** | **$0.6570 \pm 0.0040$** | $0.8565 \pm 0.0646$ | $0.5594 \pm 0.0126$ | $0.6774 \pm 0.0202$ |
| **Neural Net** | **MLP** | Standard | $0.8912 \pm 0.0091$ | $0.8914 \pm 0.0115$ | **$0.5661 \pm 0.0184$** | **$0.4082 \pm 0.0584$** | $0.5816 \pm 0.0180$ | $0.5516 \pm 0.0221$ | $0.5409 \pm 0.0193$ |
| **Neural Net** | **MLP** | Weighted | $0.8791 \pm 0.0176$ | $0.8600 \pm 0.0279$ | **$0.5224 \pm 0.0459$** | **$0.3166 \pm 0.0542$** | $0.5164 \pm 0.1014$ | $0.5387 \pm 0.0211$ | $0.4952 \pm 0.0518$ |
| **Linear** | **Logistic Regression** | Standard | $0.6789 \pm 0.0000$ | $0.3935 \pm 0.0000$ | **$0.4378 \pm 0.0000$** | **$0.2140 \pm 0.0000$** | $0.3564 \pm 0.0000$ | $0.5676 \pm 0.0000$ | $0.4081 \pm 0.0000$ |
| **Linear** | **Logistic Regression** | Weighted | $0.6325 \pm 0.0000$ | $0.3540 \pm 0.0000$ | **$0.4068 \pm 0.0000$** | **$0.1982 \pm 0.0000$** | $0.3118 \pm 0.0000$ | $0.5849 \pm 0.0000$ | $0.3804 \pm 0.0000$ |
| **GNN** | **GraphSAGE** | Weighted | $0.6060 \pm 0.2270$ | $0.5892 \pm 0.2208$ | **$0.4153 \pm 0.1914$** | **$0.3659 \pm 0.1859$** | $0.4430 \pm 0.3021$ | $0.4569 \pm 0.1382$ | $0.3905 \pm 0.2095$ |
| **GNN** | **GCN** | Weighted | $0.4276 \pm 0.0898$ | $0.3925 \pm 0.0960$ | **$0.2559 \pm 0.0674$** | **$0.2132 \pm 0.0905$** | $0.2010 \pm 0.0863$ | $0.4299 \pm 0.1278$ | $0.2130 \pm 0.0665$ |
| **GNN** | **GAT** | Weighted | $0.3797 \pm 0.0457$ | $0.2741 \pm 0.1011$ | **$0.2367 \pm 0.0586$** | **$0.1653 \pm 0.0747$** | $0.1964 \pm 0.1177$ | $0.3943 \pm 0.1221$ | $0.1959 \pm 0.0721$ |

---

## 13. Non-Comparable Configurations & Methodological Distinctions

The following ablation configurations must **NOT** be directly compared against standard forward inductive baselines without explicit caveats:

1. **Ablation Study: Backward and Bidirectional Graph Propagation** ([`results/tables/gnn_ablation_direction.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/gnn_ablation_direction.csv))
   - *Status:* **NOT DIRECTLY COMPARABLE** to standard causal temporal baselines.
   - *Reason:* Backward propagation ($tx_2 \to tx_1$) and bidirectional propagation aggregate features against the physical direction of financial transaction flows. In a streaming/real-time deployment, backward edges represent future recipient transactions that may not have occurred yet at the time of incoming transaction evaluation. They serve strictly as retrospective graph-structure ablations.
2. **Ablation Study: Unknown-Node Removal** ([`results/tables/gnn_ablation_unknown_context.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/gnn_ablation_unknown_context.csv))
   - *Status:* **NOT DIRECTLY COMPARABLE** to the 100% full graph benchmark.
   - *Reason:* Pruning unknown nodes alters the graph topology by deleting ~77% of all transaction nodes and disconnecting multi-hop transaction chains.
3. **Cross-Weighting Direct Comparisons (Standard vs. Weighted):**
   - *Status:* **QUALIFIED COMPARISON ONLY**.
   - *Reason:* Comparing an unweighted loss model directly to a cost-sensitive weighted model conflates inductive bias with the optimization objective. Comparisons should explicitly pair Standard baselines with standard models and Weighted baselines with cost-sensitive GNNs.

---

## 14. Paper-Safe Conclusion

### A. Verified Facts
1. **Identical Temporal Split:** Both Phase 2 and Phase 3 strictly execute on Training (Timesteps 1–34), Validation (35–39), and Blind Testing (40–49) across identical sample counts (29,894 train, 5,486 val, 11,184 test labeled instances).
2. **Identical Feature Definitions:** Both phases use identical 93 local and 165 full features with `txId` and `time_step` strictly excluded from model feature matrices.
3. **Zero Data Leakage:** All preprocessors (StandardScaler), loss weights, checkpoint selections, and decision thresholds are derived strictly from training/validation sets without test contamination.
4. **Shared Random Seeds:** All multi-seed experiments in both phases were executed using the identical 5 seeds (`[42, 123, 456, 789, 999]`).
5. **Exact Metric Equivalence:** All reported metrics (Illicit F1, PR-AUC, Precision, Recall, MCC) are computed via the identical metric functions.

### B. Comparability Issues & Methodological Caveats
1. **Graph Topological Context vs. Tabular Independence:** GNNs operate over the interconnected transaction graph (including 157,205 unlabeled intermediate nodes), whereas conventional baselines operate strictly on isolated node feature vectors.
2. **Domain Non-Stationarity & Seed Sensitivity:** On the temporal test set (Timesteps 40–49), GNN models exhibit higher cross-seed variance ($\text{std} \approx 0.06 - 0.19$) compared to tree-based baselines ($\text{std} \le 0.01$). This indicates higher sensitivity of graph message passing to temporal distribution shift.
3. **Statistical Power Limitation:** With $N = 5$ random seeds, the non-parametric Wilcoxon signed-rank test has a minimum possible two-sided $p$-value of $0.0625$, which cannot reach $\alpha = 0.05$. Parametric paired $t$-tests yield $p_{\text{adj}} = 0.0917$ after Holm-Bonferroni correction.

### C. Paper-Safe Interpretation
- In the evaluated static snapshot setting of the Elliptic Bitcoin transaction dataset under strict temporal partitioning (Train: 1–34, Val: 35–39, Test: 40–49), conventional tree-based ensembles (Random Forest Full 165: Test F1 = $0.7247 \pm 0.0027$, Test PR-AUC = $0.6599 \pm 0.0025$; XGBoost Full 165: Test F1 = $0.7131 \pm 0.0111$, Test PR-AUC = $0.6744 \pm 0.0028$) demonstrated higher test-set illicit F1 and PR-AUC scores than the evaluated end-to-end GNN architectures (GAT Full 165: Test F1 = $0.3308 \pm 0.0602$; GraphSAGE Local 93: Test F1 = $0.4153 \pm 0.1914$; GCN Full 165: Test F1 = $0.2589 \pm 0.0639$).
- Handcrafted 1-hop aggregate features (the 72 background features in the 165-feature set) provide strong tabular representations that tree ensembles exploit effectively. End-to-end GNN message passing over transaction graphs containing ~77% unlabeled nodes exhibits vulnerability to neighbor feature dilution in isotropic models (GCN, GraphSAGE) and higher performance degradation under temporal domain shift.
