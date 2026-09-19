# -*- coding: utf-8 -*-
"""
Deep inspection of statistical test tables and main results.
"""

import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Read tables
main_df = pd.read_csv('results/tables/MASTER_MAIN_RESULTS.csv')
stats_df = pd.read_csv('results/tables/MASTER_STATISTICAL_TESTS.csv')
feat_df = pd.read_csv('results/tables/MASTER_FEATURE_COMPARISON.csv')
ablation_df = pd.read_csv('results/tables/MASTER_GNN_ABLATIONS.csv')

print("=" * 80)
print("1. MASTER MAIN RESULTS (TABLE I)")
print("=" * 80)
for i, r in main_df.iterrows():
    print(f"{r['Model']:20s} | Fam: {r['Family']:20s} | Test F1: {r['Test F1 Mean']:.4f} +/- {r['Test F1 Std']:.4f} | Test PR-AUC: {r['Test PR-AUC Mean']:.4f} +/- {r['Test PR-AUC Std']:.4f} | Prec: {r['Test Precision Mean']:.4f} | Rec: {r['Test Recall Mean']:.4f}")

print("\n" + "=" * 80)
print("2. MASTER STATISTICAL TESTS (TABLE IX / SECTION VI)")
print("=" * 80)
for i, r in stats_df.iterrows():
    print(f"{r['Model A (GNN)']} vs {r['Model B (Baseline)']}")
    print(f"   Metric: {r['Metric']} | Mean Diff: {r['Mean Difference (Δ)']} | t-stat: {r['Paired t-statistic']} | raw p: {r['Paired t-test p (Raw)']} | Holm p: {r['Holm-Adjusted t-test p']}")
    print(f"   Wilcoxon W: {r['Wilcoxon W-statistic']} | Wilcoxon p: {r['Wilcoxon p-value']} | Cohen dz: {r['Paired Cohen dz']}")

print("\n" + "=" * 80)
print("3. FEATURE COMPARISON (TABLE IV)")
print("=" * 80)
for i, r in feat_df.iterrows():
    print(f"{r['Model']:20s} | Local F1: {r['Local 93 F1 Mean']:.4f} | Full F1: {r['Full 165 F1 Mean']:.4f} | Delta: {r['Delta F1']:.4f} | Local PR-AUC: {r['Local 93 PR-AUC Mean']:.4f} | Full PR-AUC: {r['Full 165 PR-AUC Mean']:.4f}")

print("\n" + "=" * 80)
print("4. GNN ABLATIONS (TABLES VI, VII, VIII)")
print("=" * 80)
for i, r in ablation_df.iterrows():
    print(f"{r['Ablation Category']:25s} | Config: {r['Configuration']:35s} | Test F1: {r['Test F1 Mean']:.4f} +/- {r['Test F1 Std']:.4f}")
