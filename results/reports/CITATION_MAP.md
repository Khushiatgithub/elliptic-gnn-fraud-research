# Scholarly Citation Map (Step 5A)

This document maps every BibTeX citation key defined in [`results/references.bib`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/references.bib) to the exact manuscript sections where it appears and the specific empirical or methodological claim it supports.

---

## 1. Citation Key to Manuscript Mapping Table

| Citation Key | Manuscript Section | Claim Supported | Verified Scholarly Source |
| :--- | :--- | :--- | :--- |
| `weber2019elliptic` | Sec 1.2, 2.1, 2.2, 2.4, 2.5, 3.1, 7.3, 8.12, 9.1 | Original Elliptic Bitcoin benchmark dataset ($203,769$ nodes, $234,355$ edges, $49$ timesteps, $165$ features), entity labeling methodology, and historical darknet shift context | Weber et al. (KDD ADS-DE 2019 / arXiv:1908.02591) |
| `foley2019sex` | Sec 1.1, 2.5, 8.12 | Scale and economic prevalence of illegal trade and darknet market activity in the Bitcoin ecosystem | Foley, Karlsen, & Putniņš (*Review of Financial Studies*, 2019) |
| `meiklejohn2013fistful` | Sec 1.1 | Foundation of Bitcoin transaction graph analysis, address clustering heuristics, and entity tracking | Meiklejohn et al. (*ACM IMC*, 2013) |
| `reid2013analysis` | Sec 1.1 | Graph representation of the Bitcoin transaction ledger (UTXO payment flow from inputs to outputs) | Reid & Harrigan (*Security and Privacy in Social Networks*, Springer 2013) |
| `moser2013inquiry` | Sec 1.1 | Multi-hop transaction laundering patterns, peeling chains, and mixing services in Bitcoin | Möser, Böhme, & Breuker (*IEEE eCRS*, 2013) |
| `harlev2018breaking` | Sec 1.1 | Supervised entity classification and tabular feature extraction on Bitcoin blockchain data | Harlev et al. (*IEEE CVCBT*, 2018) |
| `hu2019transaction` | Sec 1.1 | Transaction-level feature engineering and machine learning for illicit Bitcoin entity detection | Hu et al. (*IEEE ICDMW*, 2019) |
| `kipf2017semi` | Sec 1.3, 3.2 | Graph Convolutional Networks (GCN) architecture, isotropic Laplacian normalization $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$, and semi-supervised formulation | Kipf & Welling (*ICLR*, 2017) |
| `hamilton2017inductive` | Sec 1.3, 3.2 | GraphSAGE inductive framework, neighborhood aggregation/sampling, and concatenation update $[h_v; \text{AGG}(\mathcal{N}(v))]$ | Hamilton, Ying, & Leskovec (*NeurIPS*, 2017) |
| `velickovic2018graph` | Sec 1.3, 3.2 | Graph Attention Networks (GAT) architecture and learnable masked multi-head self-attention mechanism | Veličković et al. (*ICLR*, 2018) |
| `breiman2001random` | Sec 1.1, 3.1, 8.2 | Random Forest ensemble algorithm, bagging, and decision tree invariance to monotonic feature transformations | Breiman (*Machine Learning*, 2001) |
| `chen2016xgboost` | Sec 1.1, 3.1, 8.2 | XGBoost scalable gradient boosting framework and regularized tree split optimization | Chen & Guestrin (*ACM KDD*, 2016) |
| `he2009learning` | Sec 1.2, 3.3 | Methodological principles and learning algorithms for severely imbalanced data distributions | He & Garcia (*IEEE TKDE*, 2009) |
| `davis2006relationship` | Sec 2.3, 4.2 | Formal relationship between Precision-Recall curves and ROC curves, establishing PR-AUC as the superior metric under heavy class skew | Davis & Goadrich (*ACM ICML*, 2006) |
| `saito2015precision` | Sec 2.3, 4.2 | Visual and quantitative demonstration that Precision-Recall evaluation avoids deceptive optimism in imbalanced binary classification | Saito & Rehmsmeier (*PLOS ONE*, 2015) |
| `gama2014survey` | Sec 1.2, 8.5 | Comprehensive taxonomy of concept drift, distribution shift, and adaptive learning in non-stationary temporal streams | Gama et al. (*ACM Computing Surveys*, 2014) |
| `quinonero2009dataset` | Sec 1.2, 8.5 | Theoretical foundations and formal definitions of covariate shift and dataset shift in machine learning | Quiñonero-Candela et al. (*MIT Press*, 2009) |
| `arp2022dos` | Sec 1.4, 3.1 | Formulating temporal look-ahead leakage, transductive contamination pitfalls, and guidelines for chronological machine learning security evaluations | Arp et al. (*USENIX Security*, 2022) |
| `li2018deeper` | Sec 5.3, 8.9 | Theoretical and empirical proof of GCN over-smoothing as a special form of Laplacian smoothing across deep layers | Li, Han, & Wu (*AAAI*, 2018) |
| `oono2020graph` | Sec 5.3, 8.9 | Theoretical proof that node representations in deep GNNs exponentially converge to an uninformative subspace (over-smoothing) | Oono & Suzuki (*ICLR*, 2020) |
| `chen2020measuring` | Sec 5.3, 8.9 | Quantitative metrics (MAD / MADGap) for analyzing feature smoothness and over-smoothing degradation in deep graph models | Chen et al. (*ACM WSDM*, 2020) |
| `wilcoxon1945individual` | Sec 3.5, 6.1, 6.2, 8.10 | Non-parametric Wilcoxon signed-rank test for paired multi-seed model evaluation | Wilcoxon (*Biometrics Bulletin*, 1945) |
| `holm1979simple` | Sec 3.5, 6.1, 8.10 | Holm-Bonferroni step-down correction for controlling Family-Wise Error Rate (FWER) across multiple baseline comparisons | Holm (*Scandinavian Journal of Statistics*, 1979) |
| `demsar2006statistical` | Sec 3.5, 6.1, 8.10 | Methodological standards and recommendations for statistical comparison of machine learning classifiers | Demšar (*Journal of Machine Learning Research*, 2006) |
| `cohen1988statistical` | Sec 3.5, 6.1, 8.10 | Standardized paired effect size calculation ($d_z$) and effect size magnitude classification | Cohen (*Lawrence Erlbaum Associates*, 1988) |
| `lakens2013calculating` | Sec 3.5, 6.1, 8.10 | Practical guidelines for calculating, reporting, and interpreting paired Cohen's $d_z$ in empirical science | Lakens (*Frontiers in Psychology*, 2013) |
| `grinsztajn2022why` | Sec 1.4, 8.2 | Large-scale empirical benchmark demonstrating that tree-based models consistently outperform deep architectures on typical tabular datasets | Grinsztajn, Oyallon, & Varoquaux (*NeurIPS*, 2022) |
| `shwartz2022tabular` | Sec 1.4, 8.2 | Rigorous benchmarking showing deep learning does not outperform ensemble tree methods on tabular domains | Shwartz-Ziv & Armon (*Information Fusion*, 2022) |
| `errica2020fair` | Sec 1.4 | Establishing fair evaluation protocols, rigorous baseline tuning, and avoiding structural biases in graph neural network benchmarking | Errica et al. (*ICLR*, 2020) |
| `bronstein2017geometric` | Sec 1.3, 8.2 | Foundational survey on geometric deep learning and graph neural network representation learning principles | Bronstein et al. (*IEEE Signal Processing Magazine*, 2017) |
| `wu2020comprehensive` | Sec 1.3 | Comprehensive taxonomy and architectural survey of spatial, spectral, and recurrent Graph Neural Networks | Wu et al. (*IEEE TNNLS*, 2020) |
| `wang2021graph` | Sec 8.2 | Comprehensive survey of Graph Neural Network architectures, graph representations, and applications in financial fraud detection | Wang et al. (*ACM Computing Surveys*, 2021) |
| `xu2020inductive` | Sec 9.4, 10.2 | Temporal Graph Attention Network (TGAT) architecture for continuous-time inductive dynamic graph learning | Xu et al. (*ICLR*, 2020) |
| `rossi2020temporal` | Sec 9.4, 10.2 | Temporal Graph Networks (TGN) generic framework for deep representation learning on continuous-time dynamic transaction graphs | Rossi et al. (*NeurIPS Workshop / arXiv:2006.10637*, 2020) |

---

## 2. Research Gap Support Mapping

| Research Gap Statement (Introduction, Sec 1.4) | Supporting Literature | Justification of Fit |
| :--- | :--- | :--- |
| **GNN vs. Tabular Empirical Conflict:** Conflicting evidence regarding whether GNNs outperform tabular tree baselines when relational features are pre-engineered. | Grinsztajn et al. (2022), Shwartz-Ziv & Armon (2022), Errica et al. (2020) | These papers establish that tree ensembles often outperform deep/graph models on tabular tasks when features are well-engineered, validating the study's motivation. |
| **Look-Ahead & Split Contamination:** Prior graph benchmarks frequently utilize random transductive splits that induce severe look-ahead bias in financial time series. | Arp et al. (2022), Weber et al. (2019) | Arp et al. formulate the temporal leakage pitfall in ML security, and Weber et al. emphasize chronological splitting for Bitcoin AML. |
| **Unlabeled Context Density:** Standard spatial GNN benchmarks assume densely labeled graphs, whereas cryptocurrency ledgers are dominated by >75% unlabeled entities. | Weber et al. (2019), He & Garcia (2009) | Weber et al. document the 77.15% unlabeled rate in the Elliptic graph, creating the need for an unknown-node ablation study. |
| **Over-Smoothing in Deep Message Passing:** Extending graph convolution layers can lead to representation homogeneity across sparse transaction motifs. | Li et al. (2018), Oono & Suzuki (2020), Chen et al. (2020) | These works provide the theoretical foundation for explaining why deeper GCNs degrade on localized transaction subgraphs. |
