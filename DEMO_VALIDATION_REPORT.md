# DEMO_VALIDATION_REPORT: Interactive Research Demonstration Dashboard

**Project Title:** *Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning*  
**Date of Validation:** September 19, 2026  
**Status:** **100% PASSED (Production Ready)**

---

## 1. Summary of Completed Features

| Module / Page | Status | Description & Verification |
|---|---|---|
| **Overview Page** | **PASS** | Displays verified dataset facts (203,769 transactions, 234,355 edges, 49 timesteps, 165 features, 4,545 illicit, 42,019 licit, 157,205 unknown), 5-stage research pipeline, 7 model family cards, and central research question. |
| **Dataset Explorer** | **PASS** | Dynamic timestep volume charts, illicit ratio dynamics (regime shift at $t=43\text{--}45$), class distribution pie chart (77.15% unknown dominance), chronological split visualization (Train 1–34, Val 35–39, Test 40–49), and feature breakdown (93 local vs 72 aggregated). |
| **Graph Explorer** | **PASS** | Interactive $k$-hop subnetwork visualizer ($k \in \{1, 2\}$) using actual edge list (`elliptic_txs_edgelist.csv`). Features pan/zoom, node selection, degree connectivity, directed arrow payment flows, distinct states for Licit (Emerald), Illicit (Rose), Unknown (Slate), and Selected Center Node (Indigo). |
| **Model Lab** | **PASS** | Dynamic evaluation table and bar charts reading directly from `MASTER_MAIN_RESULTS.csv` (7 models $\times$ 5 random seeds). Displays Illicit F1, PR-AUC, Precision, Recall, MCC (Mean $\pm$ Std), and family filtering with neutral "Observed performance" language. |
| **Experiment Lab** | **PASS** | 6 interactive tabs: Feature Space (Local vs Full), Unknown Nodes Context (Retained vs Removed), Edge Direction (Forward vs Backward vs Bidirectional), Layer Depth (1 vs 2 vs 3 Layers), Seed Stability (5 seeds min/max/range), and Statistical Tests (paired $t$, Wilcoxon $p_{\min}=0.0625$, Holm correction, Cohen's $d_z$). |
| **Prediction Page** | **PASS** | Live, authentic multi-model inference (Random Forest, XGBoost, MLP, Logistic Regression, GAT, GraphSAGE, GCN) using exact train-split preprocessing. Displays predicted probability, optimal validation-locked threshold $\tau$, model decision, feature importance chart (for tree models), and local graph structure. |
| **Research Findings** | **PASS** | 4-panel executive narrative synthesis + 15 publication-grade figures gallery (300 DPI) with click-to-enlarge high-resolution modal viewer. |
| **Presentation Mode** | **PASS** | 10-slide full-screen presentation deck with keyboard Left/Right arrow navigation, slide indicator dots, large metrics, and speaker takeaway bullets. |

---

## 2. API Endpoint Verification Matrix

All backend endpoints were tested with automated tests (`pytest backend/tests/test_api.py`) and verified with 100% pass rate:

```
backend/tests/test_api.py::test_health_endpoint PASSED                   [ 10%]
backend/tests/test_api.py::test_dataset_summary PASSED                   [ 20%]
backend/tests/test_api.py::test_dataset_timesteps PASSED                 [ 30%]
backend/tests/test_api.py::test_sample_transactions PASSED               [ 40%]
backend/tests/test_api.py::test_transaction_detail PASSED                [ 50%]
backend/tests/test_api.py::test_graph_neighborhood PASSED                [ 60%]
backend/tests/test_api.py::test_results_endpoints PASSED                 [ 70%]
backend/tests/test_api.py::test_figures_catalog PASSED                   [ 80%]
backend/tests/test_api.py::test_models_available PASSED                  [ 90%]
backend/tests/test_api.py::test_live_prediction PASSED                   [100%]
============================== 10 passed in 110.64s ==============================
```

---

## 3. Verified Models and Serialization Artifacts

| Model Family | Model Name | Checkpoint File | Preprocessing Mode | Verified Live Inference |
|---|---|---|---|---|
| **Tree Ensemble** | Random Forest | `models/checkpoints/tabular/random_forest_full_seed42.joblib` | Unscaled (Full 165) | **PASS** (Feature Importances generated) |
| **Gradient Boosting**| XGBoost | `models/checkpoints/tabular/xgboost_full_seed42.joblib` | Unscaled (Full 165) | **PASS** (Feature Importances generated) |
| **Linear Model** | Logistic Regression | `models/checkpoints/tabular/logistic_regression_full_seed42.joblib` | Standard Scaled (Full 165) | **PASS** (Coefficients generated) |
| **Neural Network** | MLP Baseline | `models/checkpoints/tabular/mlp_full_seed42.pt` | Standard Scaled (Full 165) | **PASS** (Sigmoid Logits) |
| **Graph Neural Net**| GAT | `models/checkpoints/gat_full_seed42.pt` | Standard Scaled (Full 165) | **PASS** (2-Hop Subgraph Message Passing) |
| **Graph Neural Net**| GraphSAGE | `models/checkpoints/graphsage_full_seed42.pt` | Standard Scaled (Full 165) | **PASS** (2-Hop Subgraph Message Passing) |
| **Graph Neural Net**| GCN | `models/checkpoints/gcn_full_seed42.pt` | Standard Scaled (Full 165) | **PASS** (2-Hop Subgraph Message Passing) |

---

## 4. Transaction IDs Tested in Live Demo

- `232629023`: Known Illicit (Class 1) in Train split (Timestep 1)
- `230389796`: Known Illicit (Class 1) in Train split (Timestep 1)
- `232438397`: Known Licit (Class 2) in Train split (Timestep 1)
- `232029206`: Known Licit (Class 2) in Train split (Timestep 1)
- `230425980`: Structural Unknown node in Train split (Timestep 1)

---

## 5. Exact Commands to Run the Application

### Start the FastAPI Backend:
```bash
python backend/run_backend.py
```
*Backend runs on:* `http://127.0.0.1:8000` (Interactive API docs at `http://127.0.0.1:8000/docs`)

### Start the React + Vite Frontend:
```bash
cd frontend
npm run dev
```
*Frontend runs on:* `http://localhost:5173`

---

## 6. Known Limitations & Research Transparency

1. **Graph Rendering Scope:** In order to maintain 60 FPS in the browser, graph exploration is scoped to 1-hop and 2-hop local transaction subnetworks (up to 80–150 nodes) centered on the queried transaction ID. Rendering all 203,769 nodes simultaneously in the browser DOM is intentionally prevented.
2. **Missing Model Artifacts:** None. All 7 model architectures have active serialized weights and preprocessors in `models/checkpoints/` and `models/checkpoints/tabular/`.
3. **Statistical Significance Note:** As documented in the manuscript and confirmed in the dashboard, paired differences between GNNs and tree baselines after Holm-Bonferroni multi-testing correction yield adjusted $p$-values between $0.0816$ and $0.4760$, which are conservatively reported as not statistically significant at $\alpha = 0.05$.
