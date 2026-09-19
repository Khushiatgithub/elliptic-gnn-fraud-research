"""
Figures and Visual Assets API Endpoints.
"""

from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.schemas.figures import FigureInfo

router = APIRouter(prefix="/figures", tags=["Figures"])

FIGURE_CATALOG = [
    {
        "filename": "f1_comparison.png",
        "title": "Primary Test Illicit F1 Performance Across 7 Model Families",
        "category": "Main Results",
        "paper_section": "Section 5.1 Main Results",
        "description": "Comparative Illicit F1 scores showing tree ensembles (RF: 0.7247, XGB: 0.7131) and GNNs (GAT: 0.3308, GraphSAGE: 0.2822, GCN: 0.2589) under chronological test evaluation."
    },
    {
        "filename": "pr_auc_comparison.png",
        "title": "Primary Test Precision-Recall AUC (PR-AUC) Comparison",
        "category": "Main Results",
        "paper_section": "Section 5.1 Main Results",
        "description": "PR-AUC comparison across all 7 evaluated architectures under severe 5.69% test class imbalance."
    },
    {
        "filename": "baseline_model_comparison.png",
        "title": "Conventional Baseline Models Matrix (F1, PR-AUC, Precision, Recall)",
        "category": "Baselines",
        "paper_section": "Section 5.1 Baseline Results",
        "description": "Multi-metric baseline comparison between Random Forest, XGBoost, MLP, and Logistic Regression."
    },
    {
        "filename": "gnn_vs_conventional_baselines.png",
        "title": "Graph ML versus Tabular ML Benchmark Comparison",
        "category": "Main Comparison",
        "paper_section": "Section 5.2 Comparative Analysis",
        "description": "Direct side-by-side performance delta between conventional tabular models and graph neural network baselines."
    },
    {
        "filename": "feature_ablation_local_vs_full.png",
        "title": "Feature Space Ablation: Local (93) vs Full (165) Features",
        "category": "Ablations",
        "paper_section": "Section 5.3 Feature Space Ablation",
        "description": "Impact of including 72 aggregated 1-hop neighborhood features across tabular and graph models."
    },
    {
        "filename": "unknown_node_ablation.png",
        "title": "Unknown-Node Structural Context Ablation",
        "category": "Ablations",
        "paper_section": "Section 5.4 Graph Structural Ablation",
        "description": "Performance comparison when retaining all 157,205 unlabeled transactions versus filtering them out during message passing."
    },
    {
        "filename": "direction_sensitivity_ablation.png",
        "title": "Edge Directionality Sensitivity (Forward vs Backward vs Bidirectional)",
        "category": "Ablations",
        "paper_section": "Section 5.4 Graph Direction Ablation",
        "description": "Evaluation of causal forward payment flow (u->v) vs retrospective backward aggregation (v->u) vs bidirectional propagation."
    },
    {
        "filename": "depth_ablation.png",
        "title": "GNN Depth & Over-Smoothing Study (1, 2, and 3 Layers)",
        "category": "Ablations",
        "paper_section": "Section 5.4 Depth Ablation",
        "description": "Structural layer depth comparison revealing over-smoothing vulnerability at depth 3 for isotropic GCN and GraphSAGE."
    },
    {
        "filename": "gnn_precision_recall_curves.png",
        "title": "Precision-Recall Curves on Chronological Test Split",
        "category": "Curves",
        "paper_section": "Section 5.2 Precision-Recall Analysis",
        "description": "High-resolution test set PR curves for GAT, GraphSAGE, GCN, and tabular baselines."
    },
    {
        "filename": "gnn_roc_curves.png",
        "title": "Receiver Operating Characteristic (ROC) Curves",
        "category": "Curves",
        "paper_section": "Section 5.2 ROC Analysis",
        "description": "Chronological test ROC curves across model architectures."
    },
    {
        "filename": "edge_homophily_matrix.png",
        "title": "Empirical Edge Homophily & Inter-Class Transition Matrix",
        "category": "Graph Topology",
        "paper_section": "Section 4.2 Relational Topology",
        "description": "Normalized transition probabilities showing 95.37% labeled edge homophily and the dominant influence of 84.37% unknown incident edges."
    },
    {
        "filename": "temporal_evolution.png",
        "title": "Temporal Distribution and Class Volume Across 49 Timesteps",
        "category": "Dataset",
        "paper_section": "Section 3 Dataset Profile",
        "description": "Total, licit, illicit, and unknown transaction volume across the chronological span."
    },
    {
        "filename": "illicit_ratio_over_time.png",
        "title": "Illicit Transaction Ratio Dynamics Across Timesteps",
        "category": "Dataset",
        "paper_section": "Section 3 Dataset Profile",
        "description": "Proportion of illicit transactions per timestep highlighting the severe darknet market shutdown disruption at timesteps 43–45."
    },
    {
        "filename": "degree_distribution.png",
        "title": "Transaction Degree Distribution in Elliptic Graph",
        "category": "Graph Topology",
        "paper_section": "Section 4 Graph Properties",
        "description": "In-degree and out-degree distributions of Bitcoin transaction network."
    },
    {
        "filename": "gnn_compute_benchmark.png",
        "title": "Computational Efficiency & Training Wall-Clock Benchmark",
        "category": "Compute",
        "paper_section": "Section 5.5 Computational Benchmark",
        "description": "Memory footprint and runtime comparison across GNN architectures."
    }
]


@router.get("/list", response_model=List[FigureInfo])
def list_figures():
    """Return catalog of available publication figures with descriptions and paper sections."""
    results = []
    for item in FIGURE_CATALOG:
        fn = item["filename"]
        file_path = settings.FIGURES_DIR / fn
        if file_path.exists():
            results.append(FigureInfo(
                filename=fn,
                title=item["title"],
                category=item["category"],
                description=item["description"],
                paper_section=item["paper_section"],
                url=f"/api/figures/{fn}"
            ))
    return results


@router.get("/{filename}")
def get_figure(filename: str):
    """Serve publication figure image file."""
    file_path = settings.FIGURES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure '{filename}' not found.")
    return FileResponse(file_path, media_type="image/png")
