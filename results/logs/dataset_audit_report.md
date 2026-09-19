# Elliptic Bitcoin Dataset Audit & Exploratory Data Analysis Report

**Project Title**: Graph Neural Network-Based Fraud Detection in Financial Transaction Networks  
**Dataset**: Elliptic Bitcoin Transaction Dataset (Weber et al., KDD 2019)  
**Execution Mode**: Phase 1 — Validation, Exploratory Data Analysis, Feature Profiling, and Graph Topology Audit  

---

## 1. Executive Dataset Summary

| Metric | Measured Value | Research Interpretation |
| :--- | :--- | :--- |
| **Total Unique Transactions (|V|)** | 203,769 | Full node set across 49 temporal snapshots |
| **Total Directed Edges (|E|)** | 234,355 | Verified directed fund transfer lineage |
| **Feature Dimensions** | 165 features (93 local + 72 aggregated) | Pre-engineered UTXO and 1-hop neighborhood features |
| **Temporal Snapshots** | 49 time steps (~2-week intervals) | Discrete temporal transaction epochs |
| **Labeled Transactions** | 46,564 (22.85%) | Ground-truth transactions for supervised evaluation |
| **Unknown (Unlabeled) Nodes** | 157,205 (77.15%) | Structural context nodes (crucial for GNN message-passing) |
| **Licit Transactions (Class 2)** | 42,019 (90.24% of labeled) | Majority legitimate class |
| **Illicit Transactions (Class 1)** | 4,545 (9.76% of labeled) | Minority fraud/illicit class |
| **Class Imbalance Ratio** | **9.25 : 1** (Licit : Illicit) | Severe class imbalance requiring cost-sensitive loss / PR-AUC |
| **Isolated Nodes (Degree = 0)** | 0 (0.00%) | Nodes without observed 1-hop edges in the temporal slice |
| **Directed Acyclic Graph (DAG)** | **True** | Strict acyclic fund flow preserving causality |

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
  - Mean in-degree: 1.150 (Max: 284)
  - Mean out-degree: 1.150 (Max: 472)
  - Mean total degree: 2.300 (Max: 473)
- **Connectivity**: 
  - Weakly Connected Components (WCC): 49 components.
  - Strongly Connected Components (SCC): 203,769 components (each SCC size is 1, confirming strict DAG acyclicity).
- **Homophily & Mixing**:
  - Labeled Edge Homophily Ratio: **0.9537** (95.37% of edges between labeled nodes connect nodes of the identical class).
  - Total edges directly involving illicit nodes: 8,145 (3.48%).
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
