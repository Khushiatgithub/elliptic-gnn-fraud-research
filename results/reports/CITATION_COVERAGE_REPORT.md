# CITATION COVERAGE REPORT

## Overview
This report provides an exhaustive claim-by-claim and category-by-category verification of scholarly citation coverage across all sections of the research paper:

*"Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning"*

## Domain Category Coverage Matrix (Categories A through AB)

| Category | Domain / Topic | Primary References | Citation Keys | Status |
|---|---|---|---|---|
| **A** | Bitcoin / Blockchain Background | Foley et al. (2019), Meiklejohn et al. (2013), Möser et al. (2013) | `foley2019sex, meiklejohn2013fistful, moser2013inquiry` | **PASS** |
| **B** | Bitcoin Transaction Graph Characteristics | Meiklejohn et al. (2013), Reid & Harrigan (2013) | `meiklejohn2013fistful, reid2013analysis` | **PASS** |
| **C** | Illicit Transaction Detection Literature | Foley et al. (2019), Harlev et al. (2018), Hu et al. (2019), Wang et al. (2021) | `foley2019sex, harlev2018breaking, hu2019transaction, wang2021graph` | **PASS** |
| **D** | Elliptic Dataset Description | Weber et al. (2019) | `weber2019elliptic` | **PASS** |
| **E** | Dataset Provenance & Benchmark Release | Weber et al. (2019) | `weber2019elliptic` | **PASS** |
| **F** | Existing Bitcoin AML / Forensic Studies | Foley et al. (2019), Möser et al. (2013), Harlev et al. (2018), Hu et al. (2019), Weber et al. (2019) | `foley2019sex, moser2013inquiry, harlev2018breaking, hu2019transaction, weber2019elliptic` | **PASS** |
| **G** | Graph Neural Network Methodology | Bronstein et al. (2017), Wu et al. (2020), Errica et al. (2020) | `bronstein2017geometric, wu2020comprehensive, errica2020fair` | **PASS** |
| **H** | Graph Convolutional Networks (GCN) | Kipf & Welling (2017) | `kipf2017semi` | **PASS** |
| **I** | Graph Sample and Aggregate (GraphSAGE) | Hamilton et al. (2017) | `hamilton2017inductive` | **PASS** |
| **J** | Graph Attention Networks (GAT) | Veličković et al. (2018) | `velickovic2018graph` | **PASS** |
| **K** | Conventional Tabular ML Benchmarks | Breiman (2001), Chen & Guestrin (2016), Grinsztajn et al. (2022), Shwartz-Ziv & Armon (2022) | `breiman2001random, chen2016xgboost, grinsztajn2022why, shwartz2022tabular` | **PASS** |
| **L** | Random Forest Classifier | Breiman (2001) | `breiman2001random` | **PASS** |
| **M** | XGBoost Gradient Boosting | Chen & Guestrin (2016) | `chen2016xgboost` | **PASS** |
| **N** | Logistic Regression Baseline Standard | Demšar (2006) | `demsar2006statistical` | **PASS** |
| **O** | Multilayer Perceptron (MLP) Tabular Baselines | Grinsztajn et al. (2022), Shwartz-Ziv & Armon (2022) | `grinsztajn2022why, shwartz2022tabular` | **PASS** |
| **P** | Class Imbalance in Machine Learning | He & Garcia (2009), Saito & Rehmsmeier (2015) | `he2009learning, saito2015precision` | **PASS** |
| **Q** | Precision-Recall AUC (PR-AUC) | Davis & Goadrich (2006), Saito & Rehmsmeier (2015) | `davis2006relationship, saito2015precision` | **PASS** |
| **R** | ROC-AUC Skew Sensitivity | Davis & Goadrich (2006), Saito & Rehmsmeier (2015) | `davis2006relationship, saito2015precision` | **PASS** |
| **S** | Temporal Distribution Shift | Gama et al. (2014), Quiñonero-Candela et al. (2009), Arp et al. (2022) | `gama2014survey, quinonero2009dataset, arp2022dos` | **PASS** |
| **T** | Concept Drift & Non-Stationarity | Gama et al. (2014), Quiñonero-Candela et al. (2009) | `gama2014survey, quinonero2009dataset` | **PASS** |
| **U** | Graph Learning & Financial Fraud Surveys | Bronstein et al. (2017), Wu et al. (2020), Wang et al. (2021) | `bronstein2017geometric, wu2020comprehensive, wang2021graph` | **PASS** |
| **V** | Temporal / Dynamic GNN Literature | Xu et al. (2020), Rossi et al. (2020) | `xu2020inductive, rossi2020temporal` | **PASS** |
| **W** | Over-smoothing & GNN Depth Dynamics | Li et al. (2018), Oono & Suzuki (2020), Chen et al. (2020) | `li2018deeper, oono2020graph, chen2020measuring` | **PASS** |
| **X** | Statistical Testing for ML Models | Demšar (2006), Wilcoxon (1945), Holm (1979) | `demsar2006statistical, wilcoxon1945individual, holm1979simple` | **PASS** |
| **Y** | Wilcoxon Signed-Rank Test | Wilcoxon (1945) | `wilcoxon1945individual` | **PASS** |
| **Z** | Holm-Bonferroni Step-Down Procedure | Holm (1979) | `holm1979simple` | **PASS** |
| **AA** | Cohen's d & Effect Size Standards | Cohen (1988), Lakens (2013) | `cohen1988statistical, lakens2013calculating` | **PASS** |
| **AB** | Methodological Best Practices & Security ML | Errica et al. (2020), Arp et al. (2022), Grinsztajn et al. (2022) | `errica2020fair, arp2022dos, grinsztajn2022why` | **PASS** |

## Coverage Audit Summary
- **Total Externally Sourced Claim Categories**: 28 / 28 (100.0% coverage)
- **Supported Claim Categories**: 28 (100.0%)
- **Unsupported Claim Categories**: 0 (0.0%)
- **Citation Overreach Cases**: 0 (All claims properly bounded; empirical findings separated from external citations)
- **Missing Citations**: 0
- **Bibliography Orphan Count**: 0
- **Unresolved Citation Callouts**: 0

## Boundary Rules Compliance
1. **Zero External Citations on Experimental Findings**: Internal empirical results (e.g., RF test F1 = 0.7247, GCN test F1 = 0.2589, N=5 paired test p >= 0.0625) contain zero external citations and are reported purely as experimental facts.
2. **Attribution of Prior Literature**: Prior benchmark metrics (Weber et al., 2019) and foundational architectural definitions are strictly cited with exact attribution.
3. **Future Work Framing**: Temporal dynamic architectures (TGAT, TGN) are explicitly cited under Future Directions as unexamined future work rather than evaluated architectures.
