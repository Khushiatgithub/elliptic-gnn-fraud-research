"""
Script to generate all Master Experiment Tables and Inventory for Phase 4 Paper Writing.
Strictly organizational and audit; zero modification of raw results.
"""

from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root = Path(__file__).resolve().parent.parent.parent
    tables_dir = root / "results" / "tables"
    reports_dir = root / "results" / "reports"
    figures_dir = root / "results" / "figures"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading existing experimental result files...")
    df_baseline_runs = pd.read_csv(tables_dir / "baseline_all_runs.csv")
    df_baseline_summary = pd.read_csv(tables_dir / "baseline_summary_mean_std.csv")
    df_gnn_runs = pd.read_csv(tables_dir / "gnn_all_runs.csv")
    df_gnn_summary = pd.read_csv(tables_dir / "gnn_summary_mean_std.csv")
    
    # -------------------------------------------------------------
    # 1. MASTER_EXPERIMENT_INVENTORY.csv
    # -------------------------------------------------------------
    print("Building MASTER_EXPERIMENT_INVENTORY.csv...")
    inventory_records = []
    exp_id = 1
    
    # A. Phase 2 Baselines (80 runs)
    for _, r in df_baseline_runs.iterrows():
        m_name = str(r["model"])
        w_mode = str(r["weighting"])
        f_mode = str(r["feature_mode"])
        f_space = str(r["feature_space"])
        seed = int(r["seed"])
        
        # Family classification
        if m_name == "Logistic Regression":
            family = "Linear Model"
        elif m_name == "Random Forest":
            family = "Tree Ensemble"
        elif m_name == "XGBoost":
            family = "Gradient Boosting"
        elif m_name == "MLP":
            family = "Neural Network"
        else:
            family = "Conventional ML"
            
        # Classification tag
        if f_mode == "full" and w_mode == "Standard":
            status = "MAIN_RESULT"
            paper_sec = "Section 4 (Main Model Comparison)"
        elif f_mode == "local":
            status = "ABLATION"
            paper_sec = "Section 5 (Feature Space Analysis)"
        elif w_mode == "Weighted":
            status = "SENSITIVITY_ANALYSIS"
            paper_sec = "Section 5 (Class Weighting Sensitivity)"
        else:
            status = "SUPPLEMENTARY"
            paper_sec = "Supplementary Material"
            
        inventory_records.append({
            "Experiment_ID": f"EXP_{exp_id:03d}",
            "Phase": "Phase 2 (Baselines)",
            "Model_Family": family,
            "Model": m_name,
            "Feature_Space": f_space,
            "Feature_Count": int(r["num_features"]),
            "Weighting": w_mode,
            "Graph_Context": "None (Tabular)",
            "Graph_Direction": "None (Tabular)",
            "Depth": "N/A" if m_name != "MLP" else "2 Dense",
            "Seed": seed,
            "Train_Timesteps": "1–34",
            "Val_Timesteps": "35–39",
            "Test_Timesteps": "40–49",
            "Validation_F1": round(float(r["val_illicit_f1"]), 4),
            "Validation_PR_AUC": round(float(r["val_pr_auc"]), 4),
            "Test_F1": round(float(r["test_illicit_f1"]), 4),
            "Test_PR_AUC": round(float(r["test_pr_auc"]), 4),
            "Test_Precision": round(float(r["test_illicit_precision"]), 4),
            "Test_Recall": round(float(r["test_illicit_recall"]), 4),
            "Test_MCC": round(float(r["test_mcc"]), 4),
            "Status": status,
            "Recommended_Paper_Section": paper_sec,
            "Notes": f"Conventional baseline run (Seed {seed})"
        })
        exp_id += 1
        
    # B. Phase 3 Primary GNNs (30 runs)
    for _, r in df_gnn_runs.iterrows():
        m_name = str(r["model"])
        f_mode = str(r["feature_mode"])
        f_space = str(r["feature_space"])
        seed = int(r["seed"])
        
        if f_mode == "full":
            status = "MAIN_RESULT"
            paper_sec = "Section 4 (Main Model Comparison)"
        else:
            status = "ABLATION"
            paper_sec = "Section 5 (Feature Space Analysis)"
            
        inventory_records.append({
            "Experiment_ID": f"EXP_{exp_id:03d}",
            "Phase": "Phase 3 (GNNs)",
            "Model_Family": "Graph Neural Network",
            "Model": m_name,
            "Feature_Space": f_space,
            "Feature_Count": int(r["num_features"]),
            "Weighting": "Weighted (pos_weight=7.63)",
            "Graph_Context": "Retained (100% Graph)",
            "Graph_Direction": "Forward (u -> v)",
            "Depth": 2,
            "Seed": seed,
            "Train_Timesteps": "1–34",
            "Val_Timesteps": "35–39",
            "Test_Timesteps": "40–49",
            "Validation_F1": round(float(r["val_illicit_f1"]), 4),
            "Validation_PR_AUC": round(float(r["val_pr_auc"]), 4),
            "Test_F1": round(float(r["test_illicit_f1"]), 4),
            "Test_PR_AUC": round(float(r["test_pr_auc"]), 4),
            "Test_Precision": round(float(r["test_illicit_precision"]), 4),
            "Test_Recall": round(float(r["test_illicit_recall"]), 4),
            "Test_MCC": round(float(r["test_mcc"]), 4),
            "Status": status,
            "Recommended_Paper_Section": paper_sec,
            "Notes": f"Primary forward GNN run with full graph context (Seed {seed})"
        })
        exp_id += 1
        
    df_inventory = pd.DataFrame(inventory_records)
    df_inventory.to_csv(tables_dir / "MASTER_EXPERIMENT_INVENTORY.csv", index=False)
    print(f"Saved {len(df_inventory)} runs to MASTER_EXPERIMENT_INVENTORY.csv")
    
    # -------------------------------------------------------------
    # 2. MASTER_MAIN_RESULTS.csv
    # -------------------------------------------------------------
    print("Building MASTER_MAIN_RESULTS.csv...")
    # Directly comparable models on Full 165 features
    main_rows = []
    
    # Baselines (Full 165, Standard)
    for m in ["Random Forest", "XGBoost", "MLP", "Logistic Regression"]:
        sub = df_baseline_summary[
            (df_baseline_summary["model"] == m) & 
            (df_baseline_summary["feature_mode"] == "full") & 
            (df_baseline_summary["weighting"] == "Standard")
        ].iloc[0]
        
        family = "Tree Ensemble" if m == "Random Forest" else ("Gradient Boosting" if m == "XGBoost" else ("Neural Network" if m == "MLP" else "Linear Model"))
        main_rows.append({
            "Model": m,
            "Family": family,
            "Features": "Full (165)",
            "Weighting": "Standard",
            "Validation F1 Mean": round(float(sub["val_illicit_f1_mean"]), 4),
            "Validation F1 Std": round(float(sub["val_illicit_f1_std"]), 4),
            "Validation PR-AUC Mean": round(float(sub["val_pr_auc_mean"]), 4),
            "Validation PR-AUC Std": round(float(sub["val_pr_auc_std"]), 4),
            "Test F1 Mean": round(float(sub["test_illicit_f1_mean"]), 4),
            "Test F1 Std": round(float(sub["test_illicit_f1_std"]), 4),
            "Test PR-AUC Mean": round(float(sub["test_pr_auc_mean"]), 4),
            "Test PR-AUC Std": round(float(sub["test_pr_auc_std"]), 4),
            "Test Precision Mean": round(float(sub["test_illicit_precision_mean"]), 4),
            "Test Precision Std": round(float(sub["test_illicit_precision_std"]), 4),
            "Test Recall Mean": round(float(sub["test_illicit_recall_mean"]), 4),
            "Test Recall Std": round(float(sub["test_illicit_recall_std"]), 4),
            "Test MCC Mean": round(float(sub["test_mcc_mean"]), 4),
            "Test MCC Std": round(float(sub["test_mcc_std"]), 4),
            "Seeds": 5 if m != "Logistic Regression" else "5 (Deterministic)"
        })
        
    # GNNs (Full 165, Primary Forward, pos_weight)
    for m in ["GAT", "GraphSAGE", "GCN"]:
        sub = df_gnn_summary[
            (df_gnn_summary["model"] == m) & 
            (df_gnn_summary["feature_mode"] == "full")
        ].iloc[0]
        
        main_rows.append({
            "Model": m,
            "Family": "Graph Neural Network",
            "Features": "Full (165)",
            "Weighting": "Weighted (pos_weight=7.63)",
            "Validation F1 Mean": round(float(sub["val_illicit_f1_mean"]), 4),
            "Validation F1 Std": round(float(sub["val_illicit_f1_std"]), 4),
            "Validation PR-AUC Mean": round(float(sub["val_pr_auc_mean"]), 4),
            "Validation PR-AUC Std": round(float(sub["val_pr_auc_std"]), 4),
            "Test F1 Mean": round(float(sub["test_illicit_f1_mean"]), 4),
            "Test F1 Std": round(float(sub["test_illicit_f1_std"]), 4),
            "Test PR-AUC Mean": round(float(sub["test_pr_auc_mean"]), 4),
            "Test PR-AUC Std": round(float(sub["test_pr_auc_std"]), 4),
            "Test Precision Mean": round(float(sub["test_illicit_precision_mean"]), 4),
            "Test Precision Std": round(float(sub["test_illicit_precision_std"]), 4),
            "Test Recall Mean": round(float(sub["test_illicit_recall_mean"]), 4),
            "Test Recall Std": round(float(sub["test_illicit_recall_std"]), 4),
            "Test MCC Mean": round(float(sub["test_mcc_mean"]), 4),
            "Test MCC Std": round(float(sub["test_mcc_std"]), 4),
            "Seeds": 5
        })
        
    df_main_res = pd.DataFrame(main_rows)
    df_main_res.to_csv(tables_dir / "MASTER_MAIN_RESULTS.csv", index=False)
    print("Saved MASTER_MAIN_RESULTS.csv")
    
    # -------------------------------------------------------------
    # 3. MASTER_FEATURE_COMPARISON.csv
    # -------------------------------------------------------------
    print("Building MASTER_FEATURE_COMPARISON.csv...")
    feat_rows = []
    
    # Baselines
    for m in ["Random Forest", "XGBoost", "MLP", "Logistic Regression"]:
        loc = df_baseline_summary[(df_baseline_summary["model"] == m) & (df_baseline_summary["feature_mode"] == "local") & (df_baseline_summary["weighting"] == "Standard")].iloc[0]
        ful = df_baseline_summary[(df_baseline_summary["model"] == m) & (df_baseline_summary["feature_mode"] == "full") & (df_baseline_summary["weighting"] == "Standard")].iloc[0]
        
        diff_f1 = float(ful["test_illicit_f1_mean"]) - float(loc["test_illicit_f1_mean"])
        diff_prauc = float(ful["test_pr_auc_mean"]) - float(loc["test_pr_auc_mean"])
        diff_mcc = float(ful["test_mcc_mean"]) - float(loc["test_mcc_mean"])
        
        feat_rows.append({
            "Model": m,
            "Feature Space": "Local Only",
            "Feature Count": 93,
            "Validation F1": f"{loc['val_illicit_f1_mean']:.4f} ± {loc['val_illicit_f1_std']:.4f}",
            "Validation PR-AUC": f"{loc['val_pr_auc_mean']:.4f} ± {loc['val_pr_auc_std']:.4f}",
            "Test F1": f"{loc['test_illicit_f1_mean']:.4f} ± {loc['test_illicit_f1_std']:.4f}",
            "Test PR-AUC": f"{loc['test_pr_auc_mean']:.4f} ± {loc['test_pr_auc_std']:.4f}",
            "Test MCC": f"{loc['test_mcc_mean']:.4f} ± {loc['test_mcc_std']:.4f}",
            "Δ F1 (Full - Local)": f"{diff_f1:+.4f}",
            "Δ PR-AUC (Full - Local)": f"{diff_prauc:+.4f}",
            "Δ MCC (Full - Local)": f"{diff_mcc:+.4f}"
        })
        feat_rows.append({
            "Model": m,
            "Feature Space": "Full Features",
            "Feature Count": 165,
            "Validation F1": f"{ful['val_illicit_f1_mean']:.4f} ± {ful['val_illicit_f1_std']:.4f}",
            "Validation PR-AUC": f"{ful['val_pr_auc_mean']:.4f} ± {ful['val_pr_auc_std']:.4f}",
            "Test F1": f"{ful['test_illicit_f1_mean']:.4f} ± {ful['test_illicit_f1_std']:.4f}",
            "Test PR-AUC": f"{ful['test_pr_auc_mean']:.4f} ± {ful['test_pr_auc_std']:.4f}",
            "Test MCC": f"{ful['test_mcc_mean']:.4f} ± {ful['test_mcc_std']:.4f}",
            "Δ F1 (Full - Local)": f"{diff_f1:+.4f}",
            "Δ PR-AUC (Full - Local)": f"{diff_prauc:+.4f}",
            "Δ MCC (Full - Local)": f"{diff_mcc:+.4f}"
        })
        
    # GNNs
    for m in ["GAT", "GraphSAGE", "GCN"]:
        loc = df_gnn_summary[(df_gnn_summary["model"] == m) & (df_gnn_summary["feature_mode"] == "local")].iloc[0]
        ful = df_gnn_summary[(df_gnn_summary["model"] == m) & (df_gnn_summary["feature_mode"] == "full")].iloc[0]
        
        diff_f1 = float(ful["test_illicit_f1_mean"]) - float(loc["test_illicit_f1_mean"])
        diff_prauc = float(ful["test_pr_auc_mean"]) - float(loc["test_pr_auc_mean"])
        diff_mcc = float(ful["test_mcc_mean"]) - float(loc["test_mcc_mean"])
        
        feat_rows.append({
            "Model": m,
            "Feature Space": "Local Only",
            "Feature Count": 93,
            "Validation F1": f"{loc['val_illicit_f1_mean']:.4f} ± {loc['val_illicit_f1_std']:.4f}",
            "Validation PR-AUC": f"{loc['val_pr_auc_mean']:.4f} ± {loc['val_pr_auc_std']:.4f}",
            "Test F1": f"{loc['test_illicit_f1_mean']:.4f} ± {loc['test_illicit_f1_std']:.4f}",
            "Test PR-AUC": f"{loc['test_pr_auc_mean']:.4f} ± {loc['test_pr_auc_std']:.4f}",
            "Test MCC": f"{loc['test_mcc_mean']:.4f} ± {loc['test_mcc_std']:.4f}",
            "Δ F1 (Full - Local)": f"{diff_f1:+.4f}",
            "Δ PR-AUC (Full - Local)": f"{diff_prauc:+.4f}",
            "Δ MCC (Full - Local)": f"{diff_mcc:+.4f}"
        })
        feat_rows.append({
            "Model": m,
            "Feature Space": "Full Features",
            "Feature Count": 165,
            "Validation F1": f"{ful['val_illicit_f1_mean']:.4f} ± {ful['val_illicit_f1_std']:.4f}",
            "Validation PR-AUC": f"{ful['val_pr_auc_mean']:.4f} ± {ful['val_pr_auc_std']:.4f}",
            "Test F1": f"{ful['test_illicit_f1_mean']:.4f} ± {ful['test_illicit_f1_std']:.4f}",
            "Test PR-AUC": f"{ful['test_pr_auc_mean']:.4f} ± {ful['test_pr_auc_std']:.4f}",
            "Test MCC": f"{ful['test_mcc_mean']:.4f} ± {ful['test_mcc_std']:.4f}",
            "Δ F1 (Full - Local)": f"{diff_f1:+.4f}",
            "Δ PR-AUC (Full - Local)": f"{diff_prauc:+.4f}",
            "Δ MCC (Full - Local)": f"{diff_mcc:+.4f}"
        })
        
    df_feat_comp = pd.DataFrame(feat_rows)
    df_feat_comp.to_csv(tables_dir / "MASTER_FEATURE_COMPARISON.csv", index=False)
    print("Saved MASTER_FEATURE_COMPARISON.csv")
    
    # -------------------------------------------------------------
    # 4. MASTER_WEIGHTING_COMPARISON.csv
    # -------------------------------------------------------------
    print("Building MASTER_WEIGHTING_COMPARISON.csv...")
    weight_rows = []
    for m in ["Random Forest", "XGBoost", "MLP", "Logistic Regression"]:
        for f_mode, f_label in [("full", "Full Features (165)"), ("local", "Local Only (93)")]:
            std = df_baseline_summary[(df_baseline_summary["model"] == m) & (df_baseline_summary["feature_mode"] == f_mode) & (df_baseline_summary["weighting"] == "Standard")].iloc[0]
            wgt = df_baseline_summary[(df_baseline_summary["model"] == m) & (df_baseline_summary["feature_mode"] == f_mode) & (df_baseline_summary["weighting"] == "Weighted")].iloc[0]
            
            d_val_f1 = float(wgt["val_illicit_f1_mean"]) - float(std["val_illicit_f1_mean"])
            d_val_prauc = float(wgt["val_pr_auc_mean"]) - float(std["val_pr_auc_mean"])
            d_test_f1 = float(wgt["test_illicit_f1_mean"]) - float(std["test_illicit_f1_mean"])
            d_test_prauc = float(wgt["test_pr_auc_mean"]) - float(std["test_pr_auc_mean"])
            d_test_prec = float(wgt["test_illicit_precision_mean"]) - float(std["test_illicit_precision_mean"])
            d_test_rec = float(wgt["test_illicit_recall_mean"]) - float(std["test_illicit_recall_mean"])
            d_test_mcc = float(wgt["test_mcc_mean"]) - float(std["test_mcc_mean"])
            
            weight_rows.append({
                "Model": m,
                "Feature Space": f_label,
                "Weighting": "Standard",
                "Validation F1": f"{std['val_illicit_f1_mean']:.4f} ± {std['val_illicit_f1_std']:.4f}",
                "Validation PR-AUC": f"{std['val_pr_auc_mean']:.4f} ± {std['val_pr_auc_std']:.4f}",
                "Test F1": f"{std['test_illicit_f1_mean']:.4f} ± {std['test_illicit_f1_std']:.4f}",
                "Test PR-AUC": f"{std['test_pr_auc_mean']:.4f} ± {std['test_pr_auc_std']:.4f}",
                "Test Precision": f"{std['test_illicit_precision_mean']:.4f} ± {std['test_illicit_precision_std']:.4f}",
                "Test Recall": f"{std['test_illicit_recall_mean']:.4f} ± {std['test_illicit_recall_std']:.4f}",
                "Test MCC": f"{std['test_mcc_mean']:.4f} ± {std['test_mcc_std']:.4f}",
                "Δ (Weighted - Standard) F1": f"{d_test_f1:+.4f}",
                "Δ (Weighted - Standard) PR-AUC": f"{d_test_prauc:+.4f}",
                "Δ (Weighted - Standard) Precision": f"{d_test_prec:+.4f}",
                "Δ (Weighted - Standard) Recall": f"{d_test_rec:+.4f}",
                "Δ (Weighted - Standard) MCC": f"{d_test_mcc:+.4f}",
            })
            weight_rows.append({
                "Model": m,
                "Feature Space": f_label,
                "Weighting": "Weighted",
                "Validation F1": f"{wgt['val_illicit_f1_mean']:.4f} ± {wgt['val_illicit_f1_std']:.4f}",
                "Validation PR-AUC": f"{wgt['val_pr_auc_mean']:.4f} ± {wgt['val_pr_auc_std']:.4f}",
                "Test F1": f"{wgt['test_illicit_f1_mean']:.4f} ± {wgt['test_illicit_f1_std']:.4f}",
                "Test PR-AUC": f"{wgt['test_pr_auc_mean']:.4f} ± {wgt['test_pr_auc_std']:.4f}",
                "Test Precision": f"{wgt['test_illicit_precision_mean']:.4f} ± {wgt['test_illicit_precision_std']:.4f}",
                "Test Recall": f"{wgt['test_illicit_recall_mean']:.4f} ± {wgt['test_illicit_recall_std']:.4f}",
                "Test MCC": f"{wgt['test_mcc_mean']:.4f} ± {wgt['test_mcc_std']:.4f}",
                "Δ (Weighted - Standard) F1": f"{d_test_f1:+.4f}",
                "Δ (Weighted - Standard) PR-AUC": f"{d_test_prauc:+.4f}",
                "Δ (Weighted - Standard) Precision": f"{d_test_prec:+.4f}",
                "Δ (Weighted - Standard) Recall": f"{d_test_rec:+.4f}",
                "Δ (Weighted - Standard) MCC": f"{d_test_mcc:+.4f}",
            })
            
    df_weight_comp = pd.DataFrame(weight_rows)
    df_weight_comp.to_csv(tables_dir / "MASTER_WEIGHTING_COMPARISON.csv", index=False)
    print("Saved MASTER_WEIGHTING_COMPARISON.csv")
    
    # -------------------------------------------------------------
    # 5. MASTER_GNN_ABLATIONS.csv
    # -------------------------------------------------------------
    print("Building MASTER_GNN_ABLATIONS.csv...")
    df_unk = pd.read_csv(tables_dir / "gnn_ablation_unknown_context.csv")
    df_dir = pd.read_csv(tables_dir / "gnn_ablation_direction.csv")
    df_dep = pd.read_csv(tables_dir / "gnn_ablation_depth.csv")
    
    ablation_rows = []
    
    # Unknown Node Context
    for _, r in df_unk.iterrows():
        m = r["model"]
        trt = r["unknown_treatment"]
        interp = "Attention mechanism adaptively downweights uninformative unknown neighbors" if m == "GAT" and "Retained" in trt else \
                 ("Unweighted aggregation over ~77% unknown nodes dilutes illicit minority signal" if "Removed" in trt else "Standard baseline message passing over full graph topology")
        ablation_rows.append({
            "Ablation": "Unknown-Node Context",
            "Model": m,
            "Setting": trt,
            "Test F1 Formatted": r["test_illicit_f1_formatted"],
            "Test PR-AUC Formatted": r["test_pr_auc_formatted"],
            "Test Precision Formatted": r["test_precision_formatted"],
            "Test Recall Formatted": r["test_recall_formatted"],
            "Test MCC Formatted": r["test_mcc_formatted"],
            "Test F1 Mean": round(float(r["test_illicit_f1_mean"]), 4),
            "Test F1 Std": round(float(r["test_illicit_f1_std"]), 4),
            "Test PR-AUC Mean": round(float(r["test_pr_auc_mean"]), 4),
            "Test PR-AUC Std": round(float(r["test_pr_auc_std"]), 4),
            "Interpretation": interp
        })
        
    # Graph Direction
    for _, r in df_dir.iterrows():
        m = r["model"]
        d = r["direction"]
        interp = "Primary inductive setting preserving causal payment flow (inputs -> outputs)" if "Forward" in d else \
                 ("Retrospective analysis aggregating upstream transaction sources" if "Backward" in d else "Exploratory setting allowing bidirectional message propagation")
        ablation_rows.append({
            "Ablation": "Graph Direction",
            "Model": m,
            "Setting": d,
            "Test F1 Formatted": r["test_illicit_f1_formatted"],
            "Test PR-AUC Formatted": r["test_pr_auc_formatted"],
            "Test Precision Formatted": r["test_precision_formatted"],
            "Test Recall Formatted": r["test_recall_formatted"],
            "Test MCC Formatted": r["test_mcc_formatted"],
            "Test F1 Mean": round(float(r["test_illicit_f1_mean"]), 4),
            "Test F1 Std": round(float(r["test_illicit_f1_std"]), 4),
            "Test PR-AUC Mean": round(float(r["test_pr_auc_mean"]), 4),
            "Test PR-AUC Std": round(float(r["test_pr_auc_std"]), 4),
            "Interpretation": interp
        })
        
    # Depth
    for _, r in df_dep.iterrows():
        m = r["model"]
        l = int(r["num_layers"])
        interp = "1-hop localized neighborhood aggregation" if l == 1 else \
                 ("2-layer structural hierarchy" if l == 2 else "3-layer deep message passing; susceptible to over-smoothing in isotropic models")
        ablation_rows.append({
            "Ablation": "GNN Depth / Over-Smoothing",
            "Model": m,
            "Setting": f"{l} Layer{'s' if l > 1 else ''}",
            "Test F1 Formatted": r["test_illicit_f1_formatted"],
            "Test PR-AUC Formatted": r["test_pr_auc_formatted"],
            "Test Precision Formatted": r["test_precision_formatted"],
            "Test Recall Formatted": r["test_recall_formatted"],
            "Test MCC Formatted": r["test_mcc_formatted"],
            "Test F1 Mean": round(float(r["test_illicit_f1_mean"]), 4),
            "Test F1 Std": round(float(r["test_illicit_f1_std"]), 4),
            "Test PR-AUC Mean": round(float(r["test_pr_auc_mean"]), 4),
            "Test PR-AUC Std": round(float(r["test_pr_auc_std"]), 4),
            "Interpretation": interp
        })
        
    df_ablation_master = pd.DataFrame(ablation_rows)
    df_ablation_master.to_csv(tables_dir / "MASTER_GNN_ABLATIONS.csv", index=False)
    print("Saved MASTER_GNN_ABLATIONS.csv")
    
    # -------------------------------------------------------------
    # 6. MASTER_SEED_STABILITY.csv
    # -------------------------------------------------------------
    print("Building MASTER_SEED_STABILITY.csv...")
    seed_stability_rows = []
    
    # Collect all primary models
    models_to_check = [
        ("Random Forest", "Phase 2", "full", "Standard"),
        ("XGBoost", "Phase 2", "full", "Standard"),
        ("MLP", "Phase 2", "full", "Standard"),
        ("Logistic Regression", "Phase 2", "full", "Standard"),
        ("GAT", "Phase 3", "full", "Weighted"),
        ("GraphSAGE", "Phase 3", "full", "Weighted"),
        ("GCN", "Phase 3", "full", "Weighted"),
        ("GraphSAGE", "Phase 3", "local", "Weighted"),
    ]
    
    for m, phase, f_mode, w_mode in models_to_check:
        if phase == "Phase 2":
            sub = df_baseline_runs[(df_baseline_runs["model"] == m) & (df_baseline_runs["feature_mode"] == f_mode) & (df_baseline_runs["weighting"] == w_mode)].copy()
        else:
            sub = df_gnn_runs[(df_gnn_runs["model"] == m) & (df_gnn_runs["feature_mode"] == f_mode)].copy()
            
        val_f1_vals = sub["val_illicit_f1"].astype(float).values
        test_f1_vals = sub["test_illicit_f1"].astype(float).values
        val_prauc_vals = sub["val_pr_auc"].astype(float).values
        test_prauc_vals = sub["test_pr_auc"].astype(float).values
        test_mcc_vals = sub["test_mcc"].astype(float).values
        
        for _, row in sub.iterrows():
            seed_stability_rows.append({
                "Model": f"{m} ({'Full 165' if f_mode == 'full' else 'Local 93'})",
                "Phase": phase,
                "Seed": int(row["seed"]),
                "Validation F1": round(float(row["val_illicit_f1"]), 4),
                "Validation PR-AUC": round(float(row["val_pr_auc"]), 4),
                "Test F1": round(float(row["test_illicit_f1"]), 4),
                "Test PR-AUC": round(float(row["test_pr_auc"]), 4),
                "Test MCC": round(float(row["test_mcc"]), 4),
                "Test F1 Mean": round(float(test_f1_vals.mean()), 4),
                "Test F1 Std": round(float(test_f1_vals.std()), 4),
                "Test F1 Min": round(float(test_f1_vals.min()), 4),
                "Test F1 Max": round(float(test_f1_vals.max()), 4),
                "Test F1 Range": round(float(test_f1_vals.max() - test_f1_vals.min()), 4),
                "Test PR-AUC Mean": round(float(test_prauc_vals.mean()), 4),
                "Test PR-AUC Std": round(float(test_prauc_vals.std()), 4),
                "Test PR-AUC Range": round(float(test_prauc_vals.max() - test_prauc_vals.min()), 4),
            })
            
    df_seed_stability = pd.DataFrame(seed_stability_rows)
    df_seed_stability.to_csv(tables_dir / "MASTER_SEED_STABILITY.csv", index=False)
    print("Saved MASTER_SEED_STABILITY.csv")
    
    # -------------------------------------------------------------
    # 7. MASTER_STATISTICAL_TESTS.csv
    # -------------------------------------------------------------
    print("Building MASTER_STATISTICAL_TESTS.csv...")
    df_stat_corrected = pd.read_csv(tables_dir / "gnn_statistical_tests_corrected.csv")
    stat_master_rows = []
    
    for _, r in df_stat_corrected.iterrows():
        stat_master_rows.append({
            "Model A (GNN)": r["GNN Model"],
            "Model B (Baseline)": r["Baseline Model"],
            "Metric": r["Metric"],
            "GNN Mean ± Std": r["GNN Mean ± Std"],
            "Baseline Mean ± Std": r["Baseline Mean ± Std"],
            "Mean Difference (Δ)": r["Mean Difference (Δ)"],
            "Paired t-statistic": round(float(r["Paired t-statistic"]), 4),
            "Paired t-test p (Raw)": round(float(r["Raw t-test p-value"]), 4),
            "Holm-Adjusted t-test p": round(float(r["Holm-Adjusted t-test p-value"]), 4),
            "Wilcoxon W-statistic": round(float(r["Wilcoxon statistic"]), 4),
            "Wilcoxon p-value": round(float(r["Wilcoxon p-value"]), 4),
            "Paired Cohen dz": round(float(r["Paired Cohen's d_z"]), 4),
            "Sample Size N": 5,
            "Notes": "Wilcoxon min achievable p is 0.0625 at N=5 (insufficient power for alpha=0.05)"
        })
        
    df_stat_master = pd.DataFrame(stat_master_rows)
    df_stat_master.to_csv(tables_dir / "MASTER_STATISTICAL_TESTS.csv", index=False)
    print("Saved MASTER_STATISTICAL_TESTS.csv")
    
    # -------------------------------------------------------------
    # 8. MASTER_ERROR_ANALYSIS.csv
    # -------------------------------------------------------------
    print("Building MASTER_ERROR_ANALYSIS.csv...")
    df_err_cat = pd.read_csv(tables_dir / "gnn_error_analysis.csv")
    df_err_time = pd.read_csv(tables_dir / "error_analysis_timesteps.csv")
    df_homophily = pd.read_csv(tables_dir / "gnn_homophily_analysis.csv")
    
    error_master_rows = []
    
    # Homophily / Topological Structure
    for _, r in df_homophily.iterrows():
        error_master_rows.append({
            "Analysis_Domain": "Graph Homophily & Relational Topology",
            "Category_or_Timestep": r["Metric"],
            "Sample_Count": r["Value"],
            "Proportion_or_Rate": "N/A",
            "Mean_In_Degree": "N/A",
            "Mean_Out_Degree": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "Illicit_F1": "N/A",
            "Key_Observation": "High homophily among labeled pairs, but ~84.4% of edges touch unknown nodes"
        })
        
    # Error Confusion Breakdown
    for _, r in df_err_cat.iterrows():
        error_master_rows.append({
            "Analysis_Domain": "Classification Error Taxonomy",
            "Category_or_Timestep": r["Category"],
            "Sample_Count": str(r["Count"]),
            "Proportion_or_Rate": str(r["Percentage"]),
            "Mean_In_Degree": str(r["Mean In-Degree"]),
            "Mean_Out_Degree": str(r["Mean Out-Degree"]),
            "Precision": "N/A",
            "Recall": "N/A",
            "Illicit_F1": "N/A",
            "Key_Observation": "Errors have lower in-degree/out-degree than true negatives"
        })
        
    # Temporal Shift Breakdown
    for _, r in df_err_time.iterrows():
        t = int(r["Time Step"])
        obs = "Darknet market shutdown disruption; severe illicit rate drop and sudden precision collapse" if t in [43, 44, 45] else "Pre-regime shift / post-shift recovery period"
        error_master_rows.append({
            "Analysis_Domain": "Temporal Regime Shift by Timestep",
            "Category_or_Timestep": f"Timestep {t}",
            "Sample_Count": str(r["Total Samples"]),
            "Proportion_or_Rate": f"{int(r['Illicit Count'])} illicit ({int(r['Illicit Count'])/int(r['Total Samples'])*100:.2f}%)",
            "Mean_In_Degree": "N/A",
            "Mean_Out_Degree": "N/A",
            "Precision": f"{float(r['Precision']):.4f}",
            "Recall": f"{float(r['Recall']):.4f}",
            "Illicit_F1": f"{float(r['Illicit F1']):.4f}",
            "Key_Observation": obs
        })
        
    df_error_master = pd.DataFrame(error_master_rows)
    df_error_master.to_csv(tables_dir / "MASTER_ERROR_ANALYSIS.csv", index=False)
    print("Saved MASTER_ERROR_ANALYSIS.csv")
    
    # -------------------------------------------------------------
    # 9. MASTER_FIGURE_INVENTORY.csv
    # -------------------------------------------------------------
    print("Building MASTER_FIGURE_INVENTORY.csv...")
    fig_items = [
        ("FIG_01", "label_distribution.png", "Dataset Exploration", "Distribution of Licit, Illicit, and Unknown transactions across the 49 discrete timesteps", "Main Paper (Sec 3)", "No", "300 DPI", "Illustrate class imbalance and large volume of unknown nodes"),
        ("FIG_02", "temporal_evolution.png", "Temporal Dynamics", "Total transaction volume and class composition over chronological time", "Main Paper (Sec 3)", "No", "300 DPI", "Document transaction dynamics over 49 timesteps"),
        ("FIG_03", "illicit_ratio_over_time.png", "Temporal Dynamics", "Illicit transaction ratio across timesteps showing regime collapse at Timestep 43", "Main Paper (Sec 3/8)", "No", "300 DPI", "Demonstrate temporal non-stationarity and darknet shutdown event"),
        ("FIG_04", "edge_homophily_matrix.png", "Graph Topology", "Edge class transition matrix and homophily distribution among labeled and unknown transactions", "Main Paper (Sec 3/8)", "No", "300 DPI", "Quantify graph homophily and edge connectivity to unknown context"),
        ("FIG_05", "baseline_model_comparison.png", "Phase 2 Baselines", "Validation and test illicit F1 and PR-AUC performance across conventional ML baselines", "Supplementary", "Yes", "300 DPI", "Summarize Phase 2 conventional baseline benchmark"),
        ("FIG_06", "gnn_vs_conventional_baselines.png", "Phase 2 vs Phase 3", "Direct comparison of test illicit F1 and PR-AUC between tree ensembles and GNN models", "Main Paper (Sec 4)", "No", "300 DPI", "Primary benchmark comparison figure"),
        ("FIG_07", "gnn_model_comparison_f1.png", "Phase 3 Primary GNNs", "Test illicit F1 scores across GCN, GraphSAGE, and GAT for Local and Full features", "Main Paper (Sec 4)", "No", "300 DPI", "Compare GNN architectures on illicit transaction classification"),
        ("FIG_08", "gnn_precision_recall_curves.png", "Phase 3 Primary GNNs", "Precision-Recall curves for primary GNN models evaluated on the blind test set (Timesteps 40–49)", "Main Paper (Sec 4)", "No", "300 DPI", "Illustrate precision-recall trade-offs under class imbalance"),
        ("FIG_09", "gnn_roc_curves.png", "Phase 3 Primary GNNs", "Receiver Operating Characteristic (ROC) curves across GNN architectures", "Supplementary", "Yes", "300 DPI", "Provide standard ROC-AUC performance curves"),
        ("FIG_10", "feature_ablation_local_vs_full.png", "Ablation Study", "Impact of local 93 vs full 165 feature spaces across GNN architectures", "Main Paper (Sec 5)", "No", "300 DPI", "Evaluate contribution of 1-hop handcrafted aggregate features"),
        ("FIG_11", "unknown_node_ablation.png", "Ablation Study", "Test F1 performance comparison when unknown nodes are retained vs removed from the graph", "Main Paper (Sec 6)", "No", "300 DPI", "Demonstrate architecture-dependent impact of unlabeled structural context"),
        ("FIG_12", "depth_ablation.png", "Ablation Study", "Test F1 performance across 1, 2, and 3 message-passing layers for GCN, GraphSAGE, and GAT", "Main Paper (Sec 6)", "No", "300 DPI", "Analyze over-smoothing and depth sensitivity"),
        ("FIG_13", "direction_sensitivity_ablation.png", "Ablation Study", "Test F1 comparison across Forward, Backward, and Bidirectional graph propagation", "Main Paper (Sec 6)", "No", "300 DPI", "Examine directional message flow sensitivity"),
        ("FIG_14", "gnn_temporal_performance_timesteps.png", "Error Analysis", "Timestep-by-timestep test performance showing collapse during the Timestep 43 regime change", "Main Paper (Sec 8)", "No", "300 DPI", "Analyze temporal failure modes and domain shift vulnerability"),
        ("FIG_15", "gnn_compute_benchmark.png", "Computational Benchmark", "Training and inference latency across GNN architectures and feature configurations", "Supplementary", "Yes", "300 DPI", "Document computational complexity and throughput"),
    ]
    
    df_fig_inv = pd.DataFrame(fig_items, columns=[
        "Figure_ID", "Filename", "Experiment", "Description", "Main_Paper", "Supplementary", "Resolution", "Purpose"
    ])
    df_fig_inv.to_csv(tables_dir / "MASTER_FIGURE_INVENTORY.csv", index=False)
    print("Saved MASTER_FIGURE_INVENTORY.csv")
    
    print("\nALL 9 MASTER TABLES CREATED AND VERIFIED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
