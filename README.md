# Graph Neural Networks vs Tabular Machine Learning for Illicit Transaction Detection in Bitcoin

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB.svg)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%2B%20PyG-EE4C2C.svg)](https://pytorch.org)
[![TailwindCSS](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC.svg)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper Title:** *Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning*

---

## 1. Project Description

This repository contains the complete empirical research codebase, publication manuscript (IEEE format), and an interactive **Research Demonstration Dashboard** evaluating Graph Neural Networks (GCN, GraphSAGE, GAT) against conventional Tabular Machine Learning baselines (Random Forest, XGBoost, MLP, Logistic Regression) on the **Elliptic Bitcoin Dataset** (203,769 transactions, 234,355 directed edges, 49 discrete timesteps).

### Key Research Question
> *"Can graph-based learning improve illicit transaction detection by exploiting transaction relationships under realistic chronological partitioning?"*

---

## 2. Interactive Research Demonstration Dashboard

The platform includes a research-grade laboratory web interface allowing full interactive exploration of the empirical results, dataset topology, transaction graph neighborhoods, and live multi-model prediction without opening notebooks or source code.

### Dashboard Navigation:
1. **OVERVIEW:** Executive research summary, verified dataset facts, 5-stage research pipeline, and 7 evaluated model family cards.
2. **DATASET:** Discrete timestep volume dynamics, illicit proportion regime shifts ($t=43\text{--}45$), chronological split partitions (Train 1–34, Val 35–39, Test 40–49), and feature composition (93 local vs 72 aggregated).
3. **GRAPH EXPLORER:** Interactive $k$-hop subnetwork visualizer ($k \in \{1, 2\}$) using actual edge list data with pan/zoom, degree inspector, and payment flow directionality.
4. **MODEL LAB:** Dynamic model comparison reading directly from `MASTER_MAIN_RESULTS.csv` with F1/PR-AUC charts and multi-metric tables (Mean $\pm$ Std across 5 seeds).
5. **EXPERIMENTS:** 6 controlled ablation tabs: Feature Space, Unknown-Node Context, Edge Directionality, Layer Depth Over-Smoothing, Seed Stability (N=5), and Paired Statistical Significance Tests (with Holm-Bonferroni correction).
6. **PREDICTION:** Live multi-model inference on any transaction ID using serialized models (RF, XGB, MLP, LR, GAT, GraphSAGE, GCN) with feature importance and subgraph structural context.
7. **RESEARCH FINDINGS:** Narrative synthesis of core empirical conclusions and an interactive 300 DPI Publication Figure Gallery.
8. **PRESENTATION MODE:** Full-screen 10-slide presentation deck with keyboard arrow navigation for conference and research demonstrations.

---

## 3. Quick Start & Execution

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Backend Setup (FastAPI)
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the FastAPI backend server
python backend/run_backend.py
```
- **Backend API URL:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`

### Frontend Setup (React + TypeScript + Vite)
```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install npm dependencies (if not already installed)
npm install

# 3. Launch the development server
npm run dev
```
- **Dashboard URL:** `http://localhost:5173`
- **Presentation Mode:** Click **"Presentation Mode"** in the top-right navbar or navigate directly within the UI.

---

## 4. Project Structure

```
elliptic-gnn-fraud-research/
├── backend/                            # FastAPI backend application
│   ├── app/
│   │   ├── config.py                   # Environment and path configurations
│   │   ├── main.py                     # FastAPI application entrypoint with CORS & routes
│   │   ├── schemas/                    # Pydantic schemas for data validation
│   │   ├── services/                   # Fast indexed data, graph, and inference services
│   │   └── routers/                    # API endpoints for dataset, graph, results, models
│   ├── tests/
│   │   └── test_api.py                 # Automated API test suite (10/10 passed)
│   └── run_backend.py                  # Backend server launcher
├── frontend/                           # React + TypeScript + Tailwind CSS UI
│   ├── src/
│   │   ├── components/                 # Reusable UI components (Navbar, Sidebar, GraphCanvas)
│   │   ├── pages/                      # 7 core pages + Presentation Mode
│   │   ├── services/                   # Axios API client
│   │   └── types/                      # TypeScript API data models
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── raw/elliptic_bitcoin_dataset/   # Raw Elliptic benchmark CSV files
├── models/
│   └── checkpoints/                    # Serialized PyG and tabular model weights
├── results/
│   ├── figures/                        # 23 high-resolution publication PNGs (300 DPI)
│   ├── tables/                         # Master CSV evaluation tables
│   ├── latex/main.pdf                  # 10-page IEEE camera-ready manuscript
│   └── FINAL_MANUSCRIPT.md             # Authoritative scientific manuscript
├── src/                                # Core ML & GNN research pipeline modules
├── DEMO_VALIDATION_REPORT.md           # Formal validation audit of dashboard
└── README.md
```

---

## 5. Summary of Main Research Findings

1. **Tree Ensembles Outperform GNNs:** Under strict chronological test evaluation, conventional tree models (Random Forest F1: **0.7247**, XGBoost F1: **0.7131**) outperformed all evaluated GNNs (GAT F1: **0.3308**, GraphSAGE: **0.2822**, GCN: **0.2589**).
2. **Local Feature Aggregation:** Pre-aggregated 1-hop statistical features (72 features) capture essential local graph context for flat classifiers without the noise propagation vulnerabilities of message-passing GNNs.
3. **Attention Mitigates Unknown Noise:** GAT achieved the highest performance among GNNs because its multi-head attention adaptively downweights uninformative and noisy unlabeled neighbors (~77% of all nodes).
4. **Temporal Regime Shifts:** All models suffered performance drops during the test period due to a darknet marketplace disruption at timesteps 43–45, where the illicit base rate collapsed from ~10% to 0.28%.
5. **Statistical Significance Rigor:** Paired differences across 5 seeds (N=5) do not retain statistical significance at $\alpha = 0.05$ after Holm-Bonferroni multi-testing correction ($p \in [0.0816, 0.4760]$).

---

## 6. Citation

```bibtex
@article{elliptic_gnn_vs_tabular_2026,
  title={Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning},
  author={Khushi},
  journal={Department of Artificial Intelligence and Data Science, Indira Gandhi Delhi Technical University for Women},
  year={2026}
}
```
