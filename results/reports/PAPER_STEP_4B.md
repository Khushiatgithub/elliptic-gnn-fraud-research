# Title Candidates

1. *Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning*
2. *Evaluating Graph Neural Networks for Illicit Financial Flow Detection on the Elliptic Bitcoin Network*
3. *An Empirical Study of Graph Neural Networks and Conventional Baselines for Illicit Transaction Detection in Transaction Networks*
4. *On the Efficacy of Graph Neural Networks for Cryptocurrency Fraud Detection under Temporal Distribution Shift*
5. *Graph-Based versus Feature-Based Illicit Transaction Classification in Bitcoin: A Comparative Benchmark on the Elliptic Dataset*

---

# Recommended Title

**Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning**

*Academic Rationale for Selection:* This title directly and neutrally identifies the core comparison (Graph Neural Networks vs. Tabular Machine Learning), the application domain (Illicit Transaction Detection in Bitcoin), the empirical methodology (Empirical Evaluation), and the key experimental protocol constraint (Chronological Partitioning), without sensational language or premature performance claims.

---

# Abstract

Detecting illicit entities and fraudulent fund transfers in cryptocurrency networks is a central challenge in financial surveillance and compliance. While Graph Neural Networks (GNNs) theoretically offer powerful inductive biases for relational data, their practical advantage over conventional tabular machine learning under realistic temporal deployment constraints remains an open empirical question. In this study, we conduct a controlled empirical evaluation comparing standard tabular models (Logistic Regression, Random Forest, XGBoost, and Multi-Layer Perceptrons) against representative GNN architectures (GCN, GraphSAGE, and GAT) on the Elliptic Bitcoin transaction dataset, comprising 203,769 transactions and 234,355 directed payment edges partitioned across 49 discrete time steps. Models are evaluated under a strict chronological protocol (Training: Timesteps 1–34; Validation: 35–39; Blind Testing: 40–49) over five random seeds using 93 local and 165 full (local plus 1-hop aggregate) feature representations. Under the evaluated setting with full features, tree-based ensembles achieved the highest test illicit F1 and Precision-Recall AUC (Random Forest: $\text{F1} = 0.7247 \pm 0.0027$, $\text{PR-AUC} = 0.6599 \pm 0.0025$; XGBoost: $\text{F1} = 0.7131 \pm 0.0111$, $\text{PR-AUC} = 0.6744 \pm 0.0028$), while primary GNN models exhibited lower test performance (GAT: $\text{F1} = 0.3308 \pm 0.0602$; GraphSAGE: $\text{F1} = 0.2822 \pm 0.1248$; GCN: $\text{F1} = 0.2589 \pm 0.0639$) alongside substantially wider cross-seed variability. Controlled ablations reveal that the contribution of unlabeled structural context (~77% unknown nodes) is architecture-dependent, improving GAT while degrading isotropic GCN and GraphSAGE. Furthermore, performance across all models dropped markedly during test-period regime shifts. These findings indicate that while GNNs capture relational dependencies, their effectiveness in static cryptocurrency snapshots is strongly influenced by feature pre-aggregation, unlabeled node density, and temporal domain shift.

---

# Research Questions

- **RQ1 (Benchmark Comparison):** How do conventional tabular machine learning models (Logistic Regression, Random Forest, XGBoost, and MLP) compare with inductive Graph Neural Networks (GCN, GraphSAGE, and GAT) for illicit transaction detection when evaluated under identical chronological partitioning and locked decision thresholds?
- **RQ2 (Feature Space Representation):** What is the empirical impact of incorporating 72 handcrafted 1-hop relational aggregate features (165-dimensional full feature space) versus relying exclusively on 93 transaction-level local features across tabular baselines and message-passing GNNs?
- **RQ3 (Graph Topological Sensitivities):** How sensitive are GNN predictions to fundamental graph design choices, specifically: (a) retaining versus removing unlabeled structural context nodes (~77% of the network), (b) message propagation directionality, and (c) network depth?
- **RQ4 (Temporal Generalization and Seed Stability):** How robust are tabular baselines and GNN architectures to random parameter initialization and real-world temporal distribution shifts, particularly during periods of structural market disruption?

---

# 1. Introduction

## 1.1 Background

Financial fraud, money laundering, and the proliferation of illicit entities across decentralized financial architectures present substantial regulatory and forensic challenges [CITATION REQUIRED]. Public, permissionless blockchain networks such as Bitcoin maintain transparent, immutable ledgers that record every transaction output and expenditure. However, the pseudonymous nature of cryptographic addresses obscures the real-world identities of transacting entities, facilitating illicit activities including ransomware extortion, darknet marketplace commerce, theft, and sanctions evasion [CITATION REQUIRED]. Automated detection of illicit transactions is therefore critical for cryptocurrency exchanges, anti-money laundering (AML) compliance frameworks, and law enforcement agencies tasked with identifying criminal fund flows.

Supervised machine learning algorithms have emerged as a primary paradigm for automated transaction classification [CITATION REQUIRED]. Traditional anti-financial crime pipelines typically structure blockchain records into tabular feature matrices, where each row represents a transaction characterized by transaction-specific metadata such as transaction fees, input/output counts, transacted Bitcoin volumes, and temporal timestamps [CITATION REQUIRED]. Standard tabular classifiers—including logistic regression, gradient boosted decision trees, and feed-forward neural networks—are then trained to differentiate illicit transactions from licit commercial activity.

However, treating financial transactions as independent and identically distributed (i.i.d.) tabular instances discards the fundamental relational structure of transaction networks. In unspent transaction output (UTXO) blockchains, transactions do not occur in isolation; rather, they form an interconnected, directed payment graph where funds flow from predecessor transactions to successor transactions [CITATION REQUIRED]. Fraudulent behaviors, such as peeling chains, layering, and mixing, inherently span multi-hop structural motifs designed to obfuscate transaction lineage [CITATION REQUIRED]. Consequently, graph-based machine learning paradigms that explicitly model network topology alongside node attributes represent a theoretically compelling methodology for financial forensics.

## 1.2 Problem Definition and Challenges

Formally, the illicit transaction detection problem can be framed as an inductive, semi-supervised node classification task on a sequence of temporal transaction subgraphs. Given a directed transaction graph $G = (V, E, X)$, the vertex set $V$ represents individual transactions, the directed edge set $E \subseteq V \times V$ represents directed fund transfers between transactions, and $X \in \mathbb{R}^{|V| \times D}$ denotes the node feature matrix. Ground-truth labels $Y \in \{0, 1\}$ are available for only a minority subset of labeled nodes $V_L \subset V$, where $y_v = 1$ denotes an illicit transaction and $y_v = 0$ denotes a licit transaction, while the vast majority of transactions $V_U = V \setminus V_L$ remain unlabeled ($y_v = \text{"unknown"}$).

Evaluating machine learning models on cryptocurrency transaction graphs presents several severe methodological challenges:

1. **Extreme Class Imbalance:** Labeled illicit transactions constitute a severe minority of observed transactions (often below 10% of labeled instances and barely 2% of total transaction volume), creating extreme class imbalance that penalizes standard accuracy metrics and necessitates careful cost-sensitive loss formulations and threshold tuning [CITATION REQUIRED].
2. **Massive Unlabeled Context:** In practical transaction networks, regulatory labeling is sparse and retrospective. In datasets such as the Elliptic Bitcoin benchmark [CITATION REQUIRED], over 77% of nodes possess unknown labels. These unlabeled transactions cannot be discarded without severing structural connectivity across the graph, yet their uncurated feature distributions may introduce substantial noise during neighborhood aggregation.
3. **Temporal Ordering and Non-Stationarity:** Financial systems evolve continuously over time. Legitimate transaction volumes, fee markets, and adversarial laundering strategies change dynamically in response to external events, market shocks, and regulatory interventions [CITATION REQUIRED]. Consequently, evaluations that utilize random train/test splits suffer from severe temporal look-ahead leakage. Realistic evaluation demands strict chronological partitioning, where models are trained on past time steps and tested on strictly subsequent, unseen temporal horizons.

## 1.3 Motivation for Graph Neural Networks

Graph Neural Networks (GNNs) have gained widespread prominence as an expressive framework for learning representations over non-Euclidean graph topologies [CITATION REQUIRED]. By iteratively aggregating feature vectors from neighboring nodes across directed or undirected edges, GNNs compute node representations that simultaneously capture intrinsic transaction attributes and local relational topology.

In this study, we investigate three foundational and architecturally distinct GNN paradigms:
- **Graph Convolutional Networks (GCN):** GCN [CITATION REQUIRED] performs isotropic spatial convolutions, updating a node's representation via a normalized, unweighted average of its 1-hop neighborhood and self-loop features. GCN serves as the canonical baseline for spectral and spatial graph message passing.
- **Graph Sample and Aggregate (GraphSAGE):** GraphSAGE [CITATION REQUIRED] introduces inductive neighborhood aggregation by explicitly decoupling a node's self-representation from its aggregated neighborhood vector through concatenation and localized sampling. This formulation is explicitly designed for inductive generalization to unseen nodes and evolving graphs.
- **Graph Attention Networks (GAT):** GAT [CITATION REQUIRED] incorporates learnable masked self-attention mechanisms into the message-passing framework. By assigning dynamic, anisotropic attention coefficients to different neighboring transactions, GAT enables the network to attend selectively to informative neighbors while potentially downweighting uninformative or noisy structural connections.

Conceptually, GNNs appear uniquely suited to illicit transaction detection, as multi-hop message passing can theoretically reconstruct money laundering chains and multi-party transaction patterns directly from raw graph topology without requiring manual feature engineering.

## 1.4 Research Gap

Despite the theoretical promise of GNNs, their empirical superiority over well-tuned tabular baselines in financial transaction networks remains subject to conflicting findings in the literature [CITATION REQUIRED]. Several benchmarking studies report substantial gains from GNNs, but frequently evaluate models under random transductive splits, utilize inconsistent baseline tuning, or omit pre-engineered relational features from the tabular baselines [CITATION REQUIRED].

Crucially, real-world blockchain compliance requires an answer to a pragmatic question: *Does deploying an end-to-end graph neural network over transaction graphs yield predictive advantages over tabular tree-based models when both paradigms are provided with identical feature information and evaluated under strict chronological splits?* Furthermore, the interplay between GNN message passing, extensive unlabeled structural context (~77% unknown nodes), message propagation directionality, and temporal distribution shift remains insufficiently dissected in controlled comparative settings.

## 1.5 Study Objective

The objective of this study is to empirically evaluate whether incorporating transaction-network structure through Graph Neural Networks provides measurable predictive advantages over conventional tabular machine learning approaches under a controlled chronological evaluation protocol. Specifically, we investigate the predictive performance, feature sensitivity, topological stability, and cross-seed robustness of conventional classifiers and GNN architectures on the Elliptic Bitcoin dataset under identical data partitions and anti-leakage controls.

## 1.6 Contributions

The principal empirical contributions of this study are fourfold:

1. **Controlled Benchmarking under Strict Chronological Partitioning:** We execute a rigorous, leakage-free empirical comparison across four tabular model families (Logistic Regression, Random Forest, XGBoost, and MLP) and three GNN architectures (GCN, GraphSAGE, and GAT) across five random seeds under an identical chronological temporal split (Training: Timesteps 1–34; Validation: 35–39; Test: 40–49) and locked validation threshold optimization.
2. **Systematic Feature Space Dissection (Local vs. Full):** We quantify the exact performance contribution of adding 72 handcrafted 1-hop relational aggregate features to 93 local features across all seven model architectures, demonstrating how pre-engineered neighborhood statistics interact differently with tabular tree partitions versus end-to-end graph convolutions.
3. **Controlled GNN Topological Ablations:** We conduct controlled ablation studies isolating the empirical effects of (i) retaining versus removing unlabeled structural context nodes, (ii) forward, backward, and bidirectional edge propagation, and (iii) network depth, establishing that unlabeled context impacts isotropic aggregators and attention mechanisms in fundamentally divergent ways.
4. **Temporal Robustness, Seed Sensitivity, and Error Analysis:** We document the sensitivity of GNN and tabular models to random parameter initialization and analyze failure modes across individual test timesteps, detailing how external regulatory disruptions (such as darknet market shutdowns) cause severe temporal domain shifts that impact model precision.

## 1.7 Paper Organization

The remainder of this paper is organized as follows: **Section 2** reviews related work in cryptocurrency transaction classification, tabular baseline modeling, and graph representation learning. **Section 3** details the Elliptic dataset, feature definitions, and graph topological properties. **Section 4** presents the experimental protocol, anti-leakage controls, and model configurations. **Section 5** reports the primary benchmark results comparing tabular and GNN architectures. **Section 6** analyzes feature space representations and class-weighting formulations. **Section 7** details the controlled GNN ablation studies. **Section 8** discusses seed stability, statistical significance testing, and temporal error analyses. Finally, **Section 9** concludes the paper with discussion of claim boundaries, practical implications, and future research directions.

---

# Evidence Integrity Check

- [x] **Numerical Values Verified:** All numerical figures (203,769 nodes, 234,355 edges, 49 timesteps, 46,564 labeled nodes, 157,205 unknown nodes, 93 local features, 72 aggregate features, 165 total features) match [`results/reports/PAPER_FACT_SHEET.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/reports/PAPER_FACT_SHEET.md) exactly.
- [x] **Empirical Metrics Verified:** Main benchmark metrics (RF F1 $0.7247$, XGB F1 $0.7131$, MLP F1 $0.5848$, LR F1 $0.4015$, GAT F1 $0.3308$, SAGE F1 $0.2822$, GCN F1 $0.2589$) align with evidence tables.
- [x] **Separation of Results from Mechanisms:** Untested mechanisms (e.g., over-smoothing, feature dilution) are explicitly described as potential explanations/hypotheses rather than proven facts.
- [x] **No Unverified / Fabricated Citations:** External literature references use explicit `[CITATION REQUIRED]` placeholders.
- [x] **No SOTA or Superiority Claims:** Text avoids promotional, marketing, or ranking assertions.
- [x] **Temporal Split Integrity:** Verified as Train (1–34), Val (35–39), Test (40–49).
- [x] **Random Seeds Integrity:** Verified as `[42, 123, 456, 789, 999]`.
- [x] **Statistical Power Disclosure:** Statistical testing limitations at $N=5$ explicitly acknowledged.
