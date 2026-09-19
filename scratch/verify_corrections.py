"""
Verification and Audit Script for Final Scientific Correction Pass.
"""

import re
import pandas as pd
from pathlib import Path

root = Path(".").resolve()
tex_path = root / "results" / "latex" / "main.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    tex = f.read()

checks = []

# 1. RF Hyperparameters
rf_ok = ("max\\_depth=15" in tex and "min\\_samples\\_split=5" in tex and "min\\_samples\\_leaf=2" in tex and "depth 15, split 5, leaf 2" in tex)
rf_bad = ("max\\_depth=None" in tex or "unlimited depth" in tex)
checks.append(("1. Random Forest Hyperparameters", rf_ok and not rf_bad, "max_depth=15, min_samples_split=5, min_samples_leaf=2"))

# 2. MLP Hyperparameters
mlp_ok = ("hidden dimensions $[128, 64]$" in tex and "Dropout ($p=0.2$)" in tex and "weight decay $10^{-4}$" in tex and "batch size 256" in tex and "patience 7" in tex)
mlp_bad = ("2 hidden layers (128 units each)" in tex or ("Dropout ($p=0.3$)" in tex and "MLP" in tex.split("Multi-Layer Perceptron (MLP):")[1][:100]))
checks.append(("2. MLP Hyperparameters", mlp_ok and not mlp_bad, "dims=[128,64], dropout=0.2, lr=0.001, wd=1e-4, bs=256, epochs=40, patience=7"))

# 3. GAT Hyperparameters
gat_ok = ("8 attention heads" in tex and "16 channels per head" in tex and "8 heads (16/head)" in tex)
gat_bad = ("4 attention heads" in tex or "4 heads (32/head)" in tex)
checks.append(("3. GAT Hyperparameters", gat_ok and not gat_bad, "hidden=128, heads=8, 16 ch/head (8x16=128), 2 layers, dropout=0.3, lr=1e-3, wd=1e-4"))

# 4. GNN Model Selection & Early Stopping
gnn_es_ok = ("monitoring validation PR-AUC" in tex and "highest validation PR-AUC is restored" in tex and "decision threshold optimization is conducted on the restored model" in tex)
gnn_es_bad = ("monitoring validation illicit F1" in tex)
checks.append(("4. GNN Model Selection / Early Stopping", gnn_es_ok and not gnn_es_bad, "Early stopping monitors val PR-AUC, best val PR-AUC checkpoint restored, threshold tuned on val F1, frozen for test"))

# 5. Primary GNN Depth Wording
depth_ok = ("The primary GNN benchmark configuration uses two message-passing layers for all three architectures; a separate depth ablation evaluates one-, two-, and three-layer variants." in tex)
checks.append(("5. Primary GNN Depth Wording", depth_ok, "Primary configuration 2 layers, separate depth ablation 1, 2, 3 layers"))

# 6. Remove Unsupported Causal Claims
causal_bad = ("darknet market shutdown" in tex.lower() or "takedown" in tex.lower() or "law enforcement intervention" in tex.lower() or "macroeconomic interventions" in tex.lower())
causal_safe = ("The sharp change in illicit prevalence around timestep 43 coincides with a substantial reduction" in tex and "cause of this regime change cannot be established" in tex)
checks.append(("6. Remove Unsupported Causal Claims (t=43)", causal_safe and not causal_bad, "Safe non-stationarity wording used without causal assertion"))

# 7. Error Analysis Model Label
err_label_ok = ("validation-selected GraphSAGE model using the Local 93-feature configuration" in tex and "Validation-Selected GraphSAGE (Local 93 Features" in tex)
checks.append(("7. Error Analysis Model Label (Table XI)", err_label_ok, "Explicitly identified GraphSAGE Local 93"))

# 8. Table VII Delta Sign Convention
delta_ok = ("+0.0390" in tex and "-0.1966" in tex and "-0.1310" in tex and "\\Delta\\text{F1} = \\text{Retained} - \\text{Removed}" in tex)
checks.append(("8. Table VII Delta Sign Convention", delta_ok, "Delta = Retained - Removed (+0.0390, -0.1966, -0.1310)"))

# 9. Standardize Terminology
term_bad = ("fraudulent entity" in tex.lower() or "fraudulent entities" in tex.lower())
checks.append(("9. Standardize Terminology", not term_bad, "Uses illicit transaction detection / illicit transaction"))

# 10. Qualify Production Claims
prod_ok = ("establishing production readiness in live compliance systems requires additional evaluation" in tex)
prod_bad = ("offer strong detection capability, low inference latency, and high parameter stability.\n\n\\begin{table*}" in tex)
checks.append(("10. Remove / Qualify Production Claims", prod_ok and not prod_bad, "Properly qualified within benchmark boundaries"))

# 11. Author Info
author_ok = ("\\author{\\IEEEauthorblockN{Khushi}" in tex and "Indira Gandhi Delhi Technical University for Women" in tex)
checks.append(("11. Author Information", author_ok, "Khushi, IGDTUW preserved"))

print("=" * 80)
print("CORRECTION PASS VERIFICATION RESULTS:")
print("=" * 80)
all_pass = True
for name, status, details in checks:
    tag = "[PASS]" if status else "[FAIL]"
    if not status: all_pass = False
    print(f"{tag} {name}")
    print(f"       Details: {details}")

print("=" * 80)
print(f"OVERALL STATUS: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
print("=" * 80)
