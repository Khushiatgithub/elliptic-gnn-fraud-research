# -*- coding: utf-8 -*-
"""
Verify all figure assets in results/figures/.
"""

import os
from PIL import Image
import sys

sys.stdout.reconfigure(encoding='utf-8')

fig_dir = 'results/figures'
figs = [
    'label_distribution.png',
    'temporal_evolution.png',
    'gnn_vs_conventional_baselines.png',
    'gnn_precision_recall_curves.png',
    'feature_ablation_local_vs_full.png',
    'gnn_model_comparison_f1.png',
    'unknown_node_ablation.png',
    'direction_sensitivity_ablation.png',
    'depth_ablation.png',
    'edge_homophily_matrix.png',
    'gnn_temporal_performance_timesteps.png',
    'supplementary_confusion_matrices.png',
    'supplementary_pr_all_models.png',
    'supplementary_loss_curves.png',
    'supplementary_error_distributions.png'
]

print("=== AUDITING FIGURE ASSETS (11 PRIMARY + 4 SUPPLEMENTARY) ===")
for i, f in enumerate(figs, 1):
    path = os.path.join(fig_dir, f)
    if os.path.exists(path):
        im = Image.open(path)
        dpi = im.info.get('dpi', (72, 72))
        size_kb = os.path.getsize(path) / 1024
        print(f"[{i:2d}] {'SUPP' if 'supp' in f else 'PRIM'} | {f:42s} | Size: {im.size[0]}x{im.size[1]} | DPI: {dpi[0]:.0f}x{dpi[1]:.0f} | {size_kb:.1f} KB | PASS")
    else:
        print(f"[{i:2d}] MISSING: {path}")
