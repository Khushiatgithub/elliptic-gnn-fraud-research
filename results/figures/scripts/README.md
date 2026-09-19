# Figure Generation Scripts Suite

This directory contains deterministic, reproducible Python scripts to generate all publication-quality figures (300 DPI) for the research paper:
**"Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning"**

## Master Data Sources

All figures are strictly derived from the verified master experiment tables in `results/tables/`:
- `MASTER_MAIN_RESULTS.csv`
- `MASTER_FEATURE_COMPARISON.csv`
- `MASTER_WEIGHTING_COMPARISON.csv`
- `MASTER_GNN_ABLATIONS.csv`
- `MASTER_SEED_STABILITY.csv`
- `MASTER_STATISTICAL_TESTS.csv`
- `MASTER_ERROR_ANALYSIS.csv`
- `temporal_distribution_by_timestep.csv`
- `edge_class_transition_matrix.csv`
- `gnn_compute_benchmark.csv`

## Figure Inventory & Mapping

| Manuscript Figure | Script Name | Target Image File | Source Data Table | Purpose / Description |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1** | `generate_fig01_label_distribution.py` | `label_distribution.png` | `temporal_distribution_by_timestep.csv` | Class distribution across 49 discrete timesteps |
| **Fig. 2** | `generate_fig02_temporal_evolution.py` | `temporal_evolution.png` | `temporal_distribution_by_timestep.csv` | Total transaction volume and class composition over time |
| **Fig. 3** | `generate_fig06_gnn_vs_baselines.py` | `gnn_vs_conventional_baselines.png` | `MASTER_MAIN_RESULTS.csv` | Primary benchmark comparison: Test illicit F1 & PR-AUC |
| **Fig. 4** | `generate_fig08_gnn_pr_curves.py` | `gnn_precision_recall_curves.png` | `MASTER_MAIN_RESULTS.csv` | Precision-Recall trade-off curves for primary GNNs |
| **Fig. 5** | `generate_fig10_feature_ablation.py` | `feature_ablation_local_vs_full.png` | `MASTER_FEATURE_COMPARISON.csv` | Local (93) vs Full (165) feature space comparison |
| **Fig. 6** | `generate_fig11_unknown_node_ablation.py` | `unknown_node_ablation.png` | `MASTER_GNN_ABLATIONS.csv` | Impact of unlabeled graph context (Retained vs Removed) |
| **Fig. 7** | `generate_fig13_direction_sensitivity.py` | `direction_sensitivity_ablation.png` | `MASTER_GNN_ABLATIONS.csv` | Edge directionality (Forward, Backward, Bidirectional) |
| **Fig. 8** | `generate_fig12_depth_ablation.py` | `depth_ablation.png` | `MASTER_GNN_ABLATIONS.csv` | Message-passing depth response across 1, 2, 3 layers |
| **Fig. 9** | `generate_fig07_gnn_model_comparison_f1.py` | `gnn_model_comparison_f1.png` | `MASTER_SEED_STABILITY.csv` | Cross-seed test F1 score distributions across 5 seeds |
| **Fig. 10** | `generate_fig04_edge_homophily.py` | `edge_homophily_matrix.png` | `edge_class_transition_matrix.csv` | Edge transition matrix and labeled homophily (95.37%) |
| **Fig. 11** | `generate_fig14_temporal_performance.py` | `gnn_temporal_performance_timesteps.png` | `MASTER_ERROR_ANALYSIS.csv` | Timestep-by-timestep test performance and collapse |
| **Fig. S1** | `generate_fig05_baseline_comparison.py` | `baseline_model_comparison.png` | `baseline_summary_mean_std.csv` | Conventional baseline benchmark across all configurations |
| **Fig. S2** | `generate_fig09_gnn_roc_curves.py` | `gnn_roc_curves.png` | Model evaluations | ROC-AUC comparison curves |
| **Fig. S3** | `generate_fig15_compute_benchmark.py` | `gnn_compute_benchmark.png` | `gnn_compute_benchmark.csv` | Training and inference computational latency tradeoff |
| **Fig. S4** | `generate_fig03_illicit_ratio.py` | `illicit_ratio_over_time.png` | `temporal_distribution_by_timestep.csv` | Illicit transaction prevalence collapse dynamics |

## Reproduction Instructions

To regenerate all figures deterministically at 300 DPI:
```bash
python results/figures/scripts/generate_all_figures.py
```
Or run any individual figure script:
```bash
python results/figures/scripts/generate_fig06_gnn_vs_baselines.py
```
