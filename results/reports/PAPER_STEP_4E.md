# 5. GNN Ablation Studies

To understand how graph construction choices and message-passing hyper-parameters govern Graph Neural Network behavior, we perform controlled ablation experiments across three structural dimensions:
1. **Unknown-Label Neighborhood Context:** Evaluating the impact of retaining versus removing unlabeled intermediate transactions ($77.15\%$ of total nodes) from the message-passing topology.
2. **Graph Propagation Directionality:** Evaluating forward (causal payment flow), backward (retrospective fund source tracing), and bidirectional message propagation.
3. **Message-Passing Depth:** Evaluating receptive field radius across 1, 2, and 3 graph convolution layers.

All ablation experiments are conducted across the five standard evaluation seeds ($S = \{42, 123, 456, 789, 999\}$) using the chronological partition (Train: Timesteps 1–34; Validation: Timesteps 35–39; Blind Test: Timesteps 40–49) on the full 165-feature representation, unless explicitly noted otherwise.

---

## 5.1 Effect of Unknown-Label Nodes

In financial transaction networks, most entities are unannotated due to the pseudonymous nature of public ledgers. In the Elliptic dataset, $157,205$ out of $203,769$ nodes ($77.15\%$) lack ground-truth supervision. In our primary benchmark experiments, these unknown-label nodes are retained in the graph topology to propagate structural messages, but masked out from the supervised loss function:
$$\mathcal{L}_{\text{masked}} = -\frac{1}{|\mathcal{V}_{\text{train}}^{\text{labeled}}|} \sum_{v \in \mathcal{V}_{\text{train}}^{\text{labeled}}} \left[ w_{\text{pos}} y_v \log \hat{y}_v + (1 - y_v) \log (1 - \hat{y}_v) \right]$$

To evaluate whether unlabeled structural context aids or hinders representation learning, we compare this default topology ("Unknown Nodes Retained", $100\%$ graph) against an induced subgraph containing only ground-truth labeled nodes ("Unknown Nodes Removed", $22.85\%$ graph, $46,564$ nodes). Table 7 presents the resulting test performance across architectures.

```
+---------------------------------------------------------------------------------------------------+
| TABLE 7: Unknown-Node Context Ablation (Test Set, Timesteps 40–49, 165 Full Features)             |
+---------------+-----------------------+-----------------------+------------+----------------------+
| Architecture  | Unknown Retained (F1) | Unknown Removed (F1)  |   ΔF1      | Direction of Change  |
+---------------+-----------------------+-----------------------+------------+----------------------+
| GAT           | 0.3308 ± 0.0602       | 0.2918 ± 0.0521       | -0.0390    | Retained is Higher   |
| GCN           | 0.2589 ± 0.0639       | 0.4555 ± 0.0524       | +0.1966    | Removed is Higher    |
| GraphSAGE     | 0.2822 ± 0.1248       | 0.4132 ± 0.0514       | +0.1310    | Removed is Higher    |
+---------------+-----------------------+-----------------------+------------+----------------------+
```

The experimental results demonstrate that the effect of unlabeled structural context is **architecture-dependent**:

1. **GCN and GraphSAGE Show Higher F1 on Labeled-Only Subgraphs:**
   - For **GCN**, removing unknown nodes increased test illicit F1 from $0.2589 \pm 0.0639$ to $0.4555 \pm 0.0524$ ($\Delta\text{F1} = +0.1966$), accompanied by an increase in PR-AUC from $0.2202 \pm 0.0703$ to $0.3997 \pm 0.0788$ and precision from $0.2788$ to $0.5576$.
   - For **GraphSAGE**, removing unknown nodes increased test illicit F1 from $0.2822 \pm 0.1248$ to $0.4132 \pm 0.0514$ ($\Delta\text{F1} = +0.1310$), while improving PR-AUC from $0.2366 \pm 0.1386$ to $0.3326 \pm 0.0529$ and precision from $0.2695$ to $0.4321$.
   - Furthermore, seed-level dispersion was substantially reduced when unknown nodes were removed: the standard deviation across seeds decreased from $\sigma = 0.0639$ to $\sigma = 0.0524$ for GCN, and from $\sigma = 0.1248$ to $\sigma = 0.0514$ for GraphSAGE.

2. **GAT Achieves Higher F1 with Unknown Nodes Retained:**
   - In contrast, **GAT** produced a higher test illicit F1 when unknown nodes were retained in the graph ($0.3308 \pm 0.0602$) compared to when they were removed ($0.2918 \pm 0.0521$, $\Delta\text{F1} = -0.0390$).
   - When unknown nodes were removed, GAT precision dropped from $0.2680 \pm 0.0632$ to $0.2038 \pm 0.0653$, despite an increase in recall from $0.4412$ to $0.5736$.

### Scientific Interpretation and Methodological Cautions
These empirical findings highlight a structural trade-off. Removing unknown nodes isolates the $36,624$ edges connecting mutually labeled transactions (where labeled edge homophily is $95.37\%$), while discarding the $197,731$ edges ($84.37\%$) incident to unlabeled nodes. 

One plausible hypothesis is that isotropic mean aggregation (GCN) and uniform neighborhood sampling (GraphSAGE) aggregate features across high-degree unlabeled intermediaries without discriminative weighting, whereas parameterized self-attention (GAT) can dynamically downweight uninformative unknown neighbors while preserving structural reachability. However, because our experiments did not directly measure intermediate feature dispersion across individual message-passing steps, this mechanism represents a potential explanatory hypothesis rather than a proven causal fact.

From an operational perspective, discarding unknown nodes is impractical for prospective transaction screening: incoming real-time transactions naturally interface with unannotated wallets and addresses. Thus, while removing unlabeled nodes provides analytical insight into neighborhood composition, prospective GNN deployments must remain capable of operating over unannotated topological context.

---

## 5.2 Effect of Graph Propagation Direction

Bitcoin transaction graphs are directed acyclic graphs (DAGs) within each discrete timestep, reflecting the irreversible flow of funds from input transactions to output transactions ($u \to v$). We investigate the sensitivity of GNN architectures to edge orientation by evaluating three propagation schemes:
- **Forward Propagation ($u \to v$):** The primary causal configuration, where messages propagate strictly in the direction of transaction outputs (downstream fund flow).
- **Backward Propagation ($v \to u$):** Messages propagate against the flow of funds, allowing transactions to aggregate information from upstream funding sources.
- **Bidirectional Propagation ($u \leftrightarrow v$):** Messages propagate along both incoming and outgoing edges simultaneously, treating the graph as undirected.

Table 8 summarizes the empirical results under these three directional configurations on the full 165-feature representation.

```
+---------------------------------------------------------------------------------------------------+
| TABLE 8: Graph Directionality Ablation (Test Set, Timesteps 40–49, 165 Full Features)             |
+---------------+-----------------------+-----------------------+-----------------------+-----------+
| Architecture  | Forward (u -> v)      | Backward (v -> u)     | Bidirectional (u <->) | Δ(Bwd-Fwd)|
+---------------+-----------------------+-----------------------+-----------------------+-----------+
| GAT           | 0.3308 ± 0.0602       | 0.3938 ± 0.0706       | 0.4618 ± 0.0234       | +0.0630   |
| GCN           | 0.2589 ± 0.0639       | 0.3953 ± 0.0462       | 0.3142 ± 0.0571       | +0.1364   |
| GraphSAGE     | 0.2822 ± 0.1248       | 0.3213 ± 0.0558       | 0.3710 ± 0.1228       | +0.0391   |
+---------------+-----------------------+-----------------------+------------+----------------------+
```

### Empirical Observations
Across all three architectures, altering the message propagation direction yielded higher test F1-scores relative to the default forward baseline:
1. **Backward Propagation:**
   - For **GCN**, backward message passing increased test F1 from $0.2589 \pm 0.0639$ to $0.3953 \pm 0.0462$ ($\Delta\text{F1} = +0.1364$), while PR-AUC improved from $0.2202 \pm 0.0703$ to $0.3649 \pm 0.0398$ and precision rose from $0.2788$ to $0.4734$.
   - For **GAT**, backward propagation increased test F1 from $0.3308 \pm 0.0602$ to $0.3938 \pm 0.0706$ ($\Delta\text{F1} = +0.0630$), with PR-AUC increasing from $0.2667 \pm 0.0695$ to $0.4007 \pm 0.0388$.
   - For **GraphSAGE**, backward propagation increased test F1 from $0.2822 \pm 0.1248$ to $0.3213 \pm 0.0558$ ($\Delta\text{F1} = +0.0391$), with a notable reduction in seed variance ($\sigma$ decreased from $0.1248$ to $0.0558$).

2. **Bidirectional Propagation:**
   - **GAT** achieved its highest performance under bidirectional message passing, reaching a test F1 of $0.4618 \pm 0.0234$ ($\Delta\text{F1} = +0.1310$ over forward) and a PR-AUC of $0.4629 \pm 0.0252$.
   - **GraphSAGE** achieved a test F1 of $0.3710 \pm 0.1228$ under bidirectional propagation ($\Delta\text{F1} = +0.0888$ over forward).
   - **GCN** produced an intermediate test F1 of $0.3142 \pm 0.0571$ under bidirectional propagation ($\Delta\text{F1} = +0.0553$ over forward), which was lower than its backward setting ($0.3953$).

### Directionality and Operational Scope
These empirical results demonstrate that aggregating upstream provenance (backward propagation) provides stronger retrospective predictive signals than aggregating downstream fund destinations (forward propagation) within static historical graph snapshots. 

However, we emphasize that **forward propagation remains the primary inductive configuration** of this study. In live, streaming anti-money laundering (AML) operations, a newly broadcast transaction has known ancestor inputs but no downstream outputs at the exact moment of confirmation. Backward propagation within static batch snapshots incorporates downstream context that may not be fully available in real-time screening. Therefore, backward and bidirectional results should be viewed as retrospective topological sensitivity analyses rather than replacements for causal forward-flow evaluation.

---

## 5.3 Effect of Message-Passing Depth

To examine how expanding the structural receptive field influences node classification, we evaluated 1-layer, 2-layer, and 3-layer variants for each GNN architecture on the full 165-feature space. The results are reported in Table 9.

```
+---------------------------------------------------------------------------------------------------+
| TABLE 9: GNN Depth Ablation (Test Set, Timesteps 40–49, 165 Full Features)                        |
+---------------+-----------------------+-----------------------+-----------------------+-----------+
| Architecture  | 1 Layer (1-Hop)       | 2 Layers (2-Hop)      | 3 Layers (3-Hop)      | Profile   |
+---------------+-----------------------+-----------------------+-----------------------+-----------+
| GAT           | 0.2877 ± 0.0319       | 0.3308 ± 0.0602       | 0.2847 ± 0.0052       | Peak at 2 |
| GCN           | 0.2862 ± 0.0250       | 0.2589 ± 0.0639       | 0.2089 ± 0.0892       | Monotonic |
| GraphSAGE     | 0.3750 ± 0.1215       | 0.2822 ± 0.1248       | 0.3327 ± 0.1250       | Peak at 1 |
+---------------+-----------------------+-----------------------+-----------------------+-----------+
```

### Depth Dynamics by Architecture
The experimental results indicate distinct depth responses across the three architectures:

1. **GCN Exhibits Monotonic Performance Degradation:**
   - GCN test F1 decreased monotonically as depth increased from 1 to 3 layers: $0.2862 \pm 0.0250$ (1 Layer) $\longrightarrow$ $0.2589 \pm 0.0639$ (2 Layers) $\longrightarrow$ $0.2089 \pm 0.0892$ (3 Layers).
   - PR-AUC followed a matching downward trajectory ($0.2255 \to 0.2202 \to 0.1648$), while cross-seed standard deviation widened from $\sigma = 0.0250$ at 1 layer to $\sigma = 0.0892$ at 3 layers.
   - This monotonic decline is consistent with the known susceptibility of isotropic graph convolutions to feature over-smoothing and gradient attenuation across expanded neighborhoods `[CITATION REQUIRED]`, though feature distance metrics were not explicitly tracked layer-by-layer.

2. **GAT Peaks at Two Layers:**
   - GAT performance followed a non-monotonic curve: test F1 increased from $0.2877 \pm 0.0319$ (1 Layer) to a peak of $0.3308 \pm 0.0602$ (2 Layers, $\Delta\text{F1} = +0.0431$), before declining to $0.2847 \pm 0.0052$ at 3 layers ($\Delta\text{F1} = -0.0461$).
   - PR-AUC exhibited the same peak at 2 layers ($0.1936 \to 0.2667 \to 0.2362$). Notably, at 3 layers, GAT demonstrated exceptional cross-seed stability ($\sigma = 0.0052$), although at a reduced mean performance level.

3. **GraphSAGE Peaks at One Layer:**
   - GraphSAGE achieved its highest full-feature performance with a single aggregation layer ($0.3750 \pm 0.1215$ F1, $0.3471 \pm 0.1413$ PR-AUC).
   - Increasing depth to 2 layers resulted in a decline to $0.2822 \pm 0.1248$ F1 ($0.2366 \pm 0.1386$ PR-AUC), while 3 layers yielded an intermediate F1 of $0.3327 \pm 0.1250$ ($0.2916 \pm 0.1145$ PR-AUC).

These results indicate that deeper message passing does not systematically translate into higher holdout detection accuracy in transaction networks. Given that transaction graphs feature low average path lengths and localized relational motifs, shallow message-passing architectures (1–2 hops) captured structural signals more effectively than deeper configurations in our experiments.

---

# 6. Statistical Analysis and Robustness

## 6.1 Paired Statistical Comparison

To evaluate whether observed performance differences between model families are statistically significant, we conduct formal hypothesis testing across the five shared random seeds ($S = \{42, 123, 456, 789, 999\}$). 

### Methodological Protocol for Model Selection
To prevent post-hoc selection bias, statistical comparisons must not be conducted against the GNN configuration that achieved the highest holdout test score. Instead, the representative GNN model was selected based strictly on validation set performance during Phase 3 tuning:
- **Selected GNN Model:** **GraphSAGE on Local 93 Features** (Validation F1 $= 0.6060 \pm 0.2270$, Test F1 $= 0.4153 \pm 0.1914$, Test PR-AUC $= 0.3659 \pm 0.1859$).

We compare this validation-selected GNN against the primary conventional baselines on their representative full-feature configurations. For each pair, we report:
1. Mean paired difference: $\bar{\Delta} = \frac{1}{N} \sum_{s} (x_{\text{GNN}, s} - x_{\text{Baseline}, s})$
2. Two-tailed paired Student's $t$-test statistic and raw $p$-value ($p_{\text{raw}}$)
3. Holm-Bonferroni adjusted $p$-value ($p_{\text{Holm}}$) to control the Family-Wise Error Rate (FWER) across baseline comparisons
4. Non-parametric Wilcoxon signed-rank test statistic ($W$) and exact two-tailed $p$-value ($p_{\text{Wilcoxon}}$)
5. Paired Cohen's effect size: $d_z = \frac{\bar{\Delta}}{\text{SD}(\Delta)}$

Table 10 reports the complete statistical test battery for both Test Illicit F1 and Test PR-AUC.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 10: Paired Statistical Significance Tests (GraphSAGE Local 93 vs. Conventional Baselines, N = 5 Shared Seeds)   |
+-------------------+-----------------+----------+----------+----------+----------+-----------+----------+--------------+
| Metric / Baseline | Mean Diff (Δ)   | Paired t | p (Raw)  | p (Holm) | W-Stat   | p (Wilc.) | Cohen dz | FWER Signif. |
+-------------------+-----------------+----------+----------+----------+----------+-----------+----------+--------------+
| Test Illicit F1   |                 |          |          |          |          |           |          |              |
| vs. Random Forest | -0.3094         | -3.5917  | 0.0229   | 0.0917   | 0.0      | 0.0625    | -1.6062  | No (p >= .05)|
| vs. XGBoost       | -0.3044         | -3.5092  | 0.0247   | 0.0917   | 0.0      | 0.0625    | -1.5694  | No (p >= .05)|
| vs. MLP           | -0.1695         | -1.8872  | 0.1322   | 0.2644   | 3.0      | 0.3125    | -0.8440  | No (p >= .05)|
| vs. Logistic Reg. | +0.0673         | +0.7856  | 0.4760   | 0.4760   | 6.0      | 0.8125    | +0.3513  | No (p >= .05)|
+-------------------+-----------------+----------+----------+----------+----------+-----------+----------+--------------+
| Test PR-AUC       |                 |          |          |          |          |           |          |              |
| vs. Random Forest | -0.2940         | -3.5141  | 0.0246   | 0.0816   | 0.0      | 0.0625    | -1.5716  | No (p >= .05)|
| vs. XGBoost       | -0.3064         | -3.7246  | 0.0204   | 0.0816   | 0.0      | 0.0625    | -1.6657  | No (p >= .05)|
| vs. MLP           | -0.1456         | -1.6440  | 0.1755   | 0.3070   | 3.0      | 0.3125    | -0.7352  | No (p >= .05)|
| vs. Logistic Reg. | +0.1462         | +1.7584  | 0.1535   | 0.3070   | 3.0      | 0.3125    | +0.7864  | No (p >= .05)|
+-------------------+-----------------+----------+----------+----------+----------+-----------+----------+--------------+
```

### Interpretation of Test Statistics
1. **Tree Ensembles (Random Forest and XGBoost):**
   - In unadjusted paired $t$-tests, GraphSAGE produced lower test illicit F1 than Random Forest ($p_{\text{raw}} = 0.0229$) and XGBoost ($p_{\text{raw}} = 0.0247$), with large negative effect sizes ($d_z = -1.6062$ and $d_z = -1.5694$, respectively).
   - However, after applying the Holm-Bonferroni correction for multiple hypothesis testing across the four baseline comparisons, the adjusted $p$-values increased to $p_{\text{Holm}} = 0.0917$.
   - Concurrently, the exact two-sided Wilcoxon signed-rank test produced $p_{\text{Wilcoxon}} = 0.0625$ ($W = 0.0$).
   - Therefore, while the point estimates and effect sizes indicate a substantial performance difference, the comparisons do not reach statistical significance at the conventional $\alpha = 0.05$ threshold after multiple comparison correction.

2. **Multi-Layer Perceptron (MLP):**
   - The difference between GraphSAGE and MLP was not statistically significant in either raw or adjusted tests ($\Delta\text{F1} = -0.1695$, $t = -1.8872$, $p_{\text{raw}} = 0.1322$, $p_{\text{Holm}} = 0.2644$, $p_{\text{Wilcoxon}} = 0.3125$, $d_z = -0.8440$).

3. **Logistic Regression:**
   - GraphSAGE produced higher mean test F1 and PR-AUC than Logistic Regression ($\Delta\text{F1} = +0.0673$, $\Delta\text{PR-AUC} = +0.1462$), but the paired differences exhibited high variance and were not statistically significant ($p_{\text{raw}} = 0.4760$, $p_{\text{Holm}} = 0.4760$, $p_{\text{Wilcoxon}} = 0.8125$).

---

## 6.2 Statistical Power and Sample Size Constraints

A critical methodological consideration in interpreting Table 10 is the statistical power of small-sample seed batteries ($N = 5$):

### Mathematical Bound on Wilcoxon Signed-Rank Test
For a paired sample of size $N = 5$, there are $2^5 = 32$ possible signed-rank permutations. Under the sharp null hypothesis of symmetric paired differences around zero, the probability of observing the most extreme possible test statistic ($W = 0$, where all five paired differences share the same sign) is exactly:
$$p_{\min} = \frac{2}{2^5} = \frac{2}{32} = 0.0625$$

Consequently, **it is mathematically impossible for a two-sided Wilcoxon signed-rank test with $N = 5$ to achieve a $p$-value below the standard significance threshold of $\alpha = 0.05$**, regardless of the magnitude of the underlying effect. Even in the case where the baseline strictly outperformed the GNN on all five random seeds ($W = 0.0$), the test statistic is bounded at $p = 0.0625$.

### Implications for Empirical ML Benchmarking
This mathematical constraint carries important implications for empirical research in graph representation learning:
- Raw parametric $t$-test $p$-values should not be accepted without checking distributional assumptions, particularly given the high variance observed across GNN initialization seeds.
- Non-parametric tests provide robust protection against distributional skew, but require larger sample sizes ($N \ge 6$ for $p_{\min} = 0.03125$; $N \ge 10$ for robust power) to reach significance at $\alpha = 0.05$.
- Standardized effect sizes ($d_z = -1.6062$ for RF, $d_z = -1.5694$ for XGBoost) and confidence bounds should be reported alongside $p$-values to communicate practical effect magnitudes.

---

## 6.3 Statistical Analysis of PR-AUC

Analyzing Precision-Recall AUC (PR-AUC) under the same statistical framework confirms that ranking quality follows the patterns observed for discrete F1 classification:
- For **Random Forest**, the mean difference in PR-AUC was $\bar{\Delta} = -0.2940$ ($t = -3.5141$, $p_{\text{raw}} = 0.0246$, $p_{\text{Holm}} = 0.0816$, $p_{\text{Wilcoxon}} = 0.0625$, $d_z = -1.5716$).
- For **XGBoost**, the mean difference in PR-AUC was $\bar{\Delta} = -0.3064$ ($t = -3.7246$, $p_{\text{raw}} = 0.0204$, $p_{\text{Holm}} = 0.0816$, $p_{\text{Wilcoxon}} = 0.0625$, $d_z = -1.6657$).
- For **MLP**, the difference was $\bar{\Delta} = -0.1456$ ($t = -1.6440$, $p_{\text{raw}} = 0.1755$, $p_{\text{Holm}} = 0.3070$, $p_{\text{Wilcoxon}} = 0.3125$).
- For **Logistic Regression**, GraphSAGE produced higher mean PR-AUC by $+0.1462$ ($t = 1.7584$, $p_{\text{raw}} = 0.1535$, $p_{\text{Holm}} = 0.3070$, $p_{\text{Wilcoxon}} = 0.3125$).

Across both classification metrics (F1) and ranking metrics (PR-AUC), the statistical tests show consistent directional patterns: tree ensembles maintained higher sample means across all five evaluation seeds, but the statistical evidence does not meet formal significance thresholds under Holm-Bonferroni correction ($p \approx 0.08 - 0.09$) or Wilcoxon testing ($p = 0.0625$) due to small sample constraints.

---

## 6.4 Temporal Generalization and Non-Stationarity

A core dimension of model robustness is temporal stability across non-stationary distributions. In the Elliptic dataset, the underlying transaction environment undergoes substantial chronological shifts:
- **Training Horizon ($t \in [1, 34]$):** $N = 29,894$ labeled nodes, $11.58\%$ illicit prevalence (class imbalance $7.63 : 1$).
- **Validation Horizon ($t \in [35, 39]$):** $N = 5,486$ labeled nodes, $8.15\%$ illicit prevalence (class imbalance $11.27 : 1$).
- **Blind Test Horizon ($t \in [40, 49]$):** $N = 11,184$ labeled nodes, $5.69\%$ illicit prevalence (class imbalance $16.58 : 1$).

All evaluated architectures experienced performance degradation between validation and test intervals:
- Random Forest F1 dropped from $0.9489 \pm 0.0023$ to $0.7247 \pm 0.0027$ ($\Delta = -0.2242$).
- XGBoost F1 dropped from $0.9399 \pm 0.0053$ to $0.7131 \pm 0.0111$ ($\Delta = -0.2268$).
- MLP F1 dropped from $0.8212 \pm 0.0344$ to $0.5848 \pm 0.0194$ ($\Delta = -0.2364$).
- GAT (Full) F1 dropped from $0.6006 \pm 0.0982$ to $0.3308 \pm 0.0602$ ($\Delta = -0.2698$).
- GraphSAGE (Local) F1 dropped from $0.6060 \pm 0.2270$ to $0.4153 \pm 0.1914$ ($\Delta = -0.1907$).
- GCN (Full) F1 dropped from $0.4293 \pm 0.0923$ to $0.2589 \pm 0.0639$ ($\Delta = -0.1704$).

This systematic decline reflects the challenge of out-of-distribution temporal generalization. As illicit transaction prevalence drops by more than $50\%$ between training and test periods, models calibrated on historical time steps face a shifted prior probability landscape.

---

# 7. Error and Graph-Structural Analysis

To characterize where and why predictive models fail, we conduct an empirical error analysis on the blind test holdout ($N = 11,184$ labeled nodes across Timesteps 40–49). We examine topological degree distributions, graph homophily properties, and timestep-by-timestep error concentrations.

---

## 7.1 Error Taxonomy and Degree Disparity

Table 11 decomposes test set predictions into a four-category classification taxonomy and summarizes topological connectivity for each group.

```
+-----------------------------------------------------------------------------------------------------------------------+
| TABLE 11: Error and Graph-Structural Taxonomy (Holdout Test Set, N = 11,184 Nodes, Timesteps 40–49)                  |
+------------------------------------+--------------+------------+----------------+-----------------+-------------------+
| Category / Segment                 | Sample Count | Proportion | Mean In-Degree | Mean Out-Degree | Precision / F1    |
+------------------------------------+--------------+------------+----------------+-----------------+-------------------+
| Total Test Nodes                   | 11,184       | 100.00%    | 2.06           | 1.26            | —                 |
| True Positives (TP, Illicit Found) | 402          | 3.59%      | 0.87           | 0.70            | —                 |
| True Negatives (TN, Licit Correct) | 8,165        | 73.01%     | 2.38           | 1.34            | —                 |
| False Positives (FP, False Alarms) | 2,383        | 21.31%     | 1.23           | 1.13            | —                 |
| False Negatives (FN, Missed Fraud) | 234          | 2.09%      | 1.61           | 0.87            | —                 |
| High-Confidence FP (p >= 0.80)     | 0            | 0.00%      | 0.00           | 0.00            | —                 |
| High-Confidence FN (p <= 0.20)     | 0            | 0.00%      | 0.00           | 0.00            | —                 |
+------------------------------------+--------------+------------+----------------+-----------------+-------------------+
| Timestep Breakdown (Test Holdout)  |              |            |                |                 |                   |
| Timestep 40 (112 illicit, 9.25%)   | 1,211        | 10.83%     | —              | —               | P: 0.2090, F1: .31|
| Timestep 41 (116 illicit, 10.25%)  | 1,132        | 10.12%     | —              | —               | P: 0.2531, F1: .37|
| Timestep 42 (239 illicit, 11.10%)  | 2,154        | 19.26%     | —              | —               | P: 0.3800, F1: .52|
| Timestep 43 (24 illicit, 1.75%)    | 1,370        | 12.25%     | —              | —               | P: 0.0409, F1: .07|
| Timestep 44 (24 illicit, 1.51%)    | 1,591        | 14.23%     | —              | —               | P: 0.0181, F1: .04|
| Timestep 45 (5 illicit, 0.41%)     | 1,221        | 10.92%     | —              | —               | P: 0.0036, F1: .01|
| Timestep 46 (2 illicit, 0.28%)     | 712          | 6.37%      | —              | —               | P: 0.0087, F1: .02|
| Timestep 47 (22 illicit, 2.60%)    | 846          | 7.56%      | —              | —               | P: 0.0481, F1: .09|
| Timestep 48 (36 illicit, 7.64%)    | 471          | 4.21%      | —              | —               | P: 0.1600, F1: .19|
| Timestep 49 (56 illicit, 11.76%)   | 476          | 4.26%      | —              | —               | P: 0.1410, F1: .16|
+------------------------------------+--------------+------------+----------------+-----------------+-------------------+
```

### Topological Degree Characteristics
Comparing the graph connectivity of correctly classified versus misclassified nodes reveals structural patterns:
- **True Negatives (TN):** Exhibited the highest average graph connectivity, with a mean in-degree of $2.38$ and mean out-degree of $1.34$.
- **True Positives (TP):** Exhibited the lowest average connectivity, with a mean in-degree of $0.87$ and mean out-degree of $0.70$.
- **False Positives (FP):** Had an intermediate in-degree of $1.23$ and out-degree of $1.13$.
- **False Negatives (FN):** Exhibited an average in-degree of $1.61$ and out-degree of $0.87$.

These measurements show that illicit transactions in the test set possessed substantially fewer recorded edges than licit transactions. Consequently, GNN message passing over illicit nodes was constrained to sparser local neighborhoods, limiting the volume of relational information aggregated from adjacent transactions.

### Prediction Confidence Distributions
Notably, the test set contained **zero high-confidence false positives** ($\hat{p} \ge 0.80$) and **zero high-confidence false negatives** ($\hat{p} \le 0.20$). All classification errors occurred within the intermediate probability range ($0.20 < \hat{p} < 0.80$), indicating that misclassifications were concentrated in regions of moderate model uncertainty rather than extreme overconfident failures.

---

## 7.2 Graph Homophily and Relational Topology

A fundamental assumption underlying spatial GNN aggregation is relational homophily: nodes sharing an edge are assumed more likely to share the same class label. We quantify homophily across the full Elliptic graph ($|V| = 203,769$, $|E| = 234,355$):

### Dual Homophily Definitions and Disambiguation
It is essential to distinguish between **global labeled-edge homophily** and **local topological context**:
1. **Global Labeled Edge Homophily ($95.37\%$):**
   - Out of $234,355$ total directed edges, only $36,624$ edges ($15.63\%$) connect two labeled nodes.
   - Among these mutually labeled edges:
     - Licit $\to$ Licit: $33,930$ edges ($92.64\%$)
     - Illicit $\to$ Illicit: $998$ edges ($2.72\%$)
     - Licit $\to$ Illicit (Cross-Class): $781$ edges ($2.13\%$)
     - Illicit $\to$ Licit (Cross-Class): $915$ edges ($2.50\%$)
   - The proportion of same-class edges among mutually labeled pairs is:
$$h_{\text{labeled}} = \frac{33,930 + 998}{36,624} = \frac{34,928}{36,624} = 95.37\%$$

2. **Unlabeled Edge Context ($84.37\%$):**
   - In contrast to the labeled subgraph, $197,731$ edges ($84.37\%$) are incident to at least one unknown-label node.
   - For an average node in the graph, the local neighborhood is dominated by unlabeled transactions whose class alignment is unknown.
   - Therefore, while the *mutually labeled* core exhibits high homophily ($95.37\%$), real-time message passing primarily traverses unannotated edges.

This distinction explains why high global labeled homophily does not automatically yield strong GNN performance: message-passing aggregation must operate over the full graph where over $84\%$ of edge transitions touch unannotated nodes.

---

## 7.3 Temporal Error Patterns across Timesteps 40–49

Analyzing classification accuracy across individual test timesteps exposes a sharp temporal performance rupture:

1. **Pre-Shift Baseline (Timesteps 40–42):**
   - In Timesteps 40–42, illicit prevalence remained relatively high ($9.25\%$ in $t=40$, $10.25\%$ in $t=41$, $11.10\%$ in $t=42$).
   - Model performance remained robust, reaching an illicit F1 of $0.5234$ (Precision $= 0.3800$, Recall $= 0.8410$) at Timestep 42.

2. **Abrupt Performance Collapse (Timesteps 43–46):**
   - At **Timestep 43**, the proportion of illicit transactions dropped abruptly from $11.10\%$ to $1.75\%$ ($24$ illicit nodes out of $1,370$).
   - Concurrently, illicit F1 collapsed from $0.5234$ to $0.0718$, with precision falling to $0.0409$ (a false positive rate of over $95\%$ among flagged transactions).
   - This depressed regime persisted through Timestep 44 ($1.51\%$ illicit, F1 $= 0.0350$), Timestep 45 ($0.41\%$ illicit, F1 $= 0.0070$), and Timestep 46 ($0.28\%$ illicit, F1 $= 0.0171$).

3. **Partial Post-Shift Recovery (Timesteps 47–49):**
   - In Timesteps 47–49, illicit transaction volume partially rebounded ($2.60\%$ in $t=47$, $7.64\%$ in $t=48$, $11.76\%$ in $t=49$).
   - F1-scores recovered proportionally ($0.0861 \to 0.1860 \to 0.1642$), though precision remained lower than in the pre-shift regime ($0.1410$ at $t=49$).

### External Attribution Caution
In prior literature, the abrupt structural change observed at Timestep 43 has been attributed to a major darknet marketplace shutdown (e.g., the coordinated law enforcement seizure of AlphaBay and Hansa markets in mid-2017) `[CITATION REQUIRED]`. 

From an empirical standpoint, we confirm that a severe temporal distribution shift occurs at Timestep 43, characterized by an $84\%$ relative drop in illicit transaction prevalence ($11.10\% \to 1.75\%$) and a corresponding collapse in predictive precision. However, establishing a definitive causal link between this statistical shift and specific external legal actions requires independent off-chain verification that lies outside the scope of the transaction graph data alone.

---

# Tables

### Table 7: Unknown-Node Context Ablation (Test Set, Timesteps 40–49)
*Comparison of GNN architectures evaluated with unknown nodes retained ($100\%$ topology, loss-masked) versus unknown nodes removed ($22.85\%$ labeled-only subgraph) on full 165 features across 5 seeds. Metrics reported as Mean $\pm$ Std.*

| Model Architecture | Unknown Setting | Test Illicit F1 | Test PR-AUC | Test Precision | Test Recall | Test MCC | $\Delta\text{F1}$ (Retained - Removed) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GAT** | Retained ($100\%$ Graph) | $0.3308 \pm 0.0602$ | $0.2667 \pm 0.0695$ | $0.2680 \pm 0.0632$ | $0.4412 \pm 0.0281$ | $0.2899 \pm 0.0640$ | **$+0.0390$** |
| **GAT** | Removed (Labeled Only) | $0.2918 \pm 0.0521$ | $0.2316 \pm 0.0727$ | $0.2038 \pm 0.0653$ | $0.5736 \pm 0.0698$ | $0.2679 \pm 0.0430$ | — |
| **GCN** | Retained ($100\%$ Graph) | $0.2589 \pm 0.0639$ | $0.2202 \pm 0.0703$ | $0.2788 \pm 0.1255$ | $0.2884 \pm 0.0839$ | $0.2213 \pm 0.0766$ | **$-0.1966$** |
| **GCN** | Removed (Labeled Only) | $0.4555 \pm 0.0524$ | $0.3997 \pm 0.0788$ | $0.5576 \pm 0.1280$ | $0.3934 \pm 0.0301$ | $0.4381 \pm 0.0653$ | — |
| **GraphSAGE** | Retained ($100\%$ Graph) | $0.2822 \pm 0.1248$ | $0.2366 \pm 0.1386$ | $0.2695 \pm 0.1858$ | $0.3588 \pm 0.0756$ | $0.2391 \pm 0.1394$ | **$-0.1310$** |
| **GraphSAGE** | Removed (Labeled Only) | $0.4132 \pm 0.0514$ | $0.3326 \pm 0.0529$ | $0.4321 \pm 0.1439$ | $0.4261 \pm 0.0590$ | $0.3849 \pm 0.0630$ | — |

---

### Table 8: Graph Propagation Directionality Ablation (Test Set, Timesteps 40–49)
*Comparison of Forward ($u \to v$), Backward ($v \to u$), and Bidirectional ($u \leftrightarrow v$) message passing on full 165 features across 5 seeds. Metrics reported as Mean $\pm$ Std.*

| Architecture | Forward ($u \to v$) F1 | Backward ($v \to u$) F1 | Bidirectional ($u \leftrightarrow v$) F1 | $\Delta\text{F1}$ (Bwd - Fwd) | $\Delta\text{F1}$ (Bidir - Fwd) | Forward PR-AUC | Backward PR-AUC | Bidirectional PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GAT** | $0.3308 \pm 0.0602$ | $0.3938 \pm 0.0706$ | $0.4618 \pm 0.0234$ | **$+0.0630$** | **$+0.1310$** | $0.2667 \pm 0.0695$ | $0.4007 \pm 0.0388$ | $0.4629 \pm 0.0252$ |
| **GCN** | $0.2589 \pm 0.0639$ | $0.3953 \pm 0.0462$ | $0.3142 \pm 0.0571$ | **$+0.1364$** | **$+0.0553$** | $0.2202 \pm 0.0703$ | $0.3649 \pm 0.0398$ | $0.2707 \pm 0.0522$ |
| **GraphSAGE** | $0.2822 \pm 0.1248$ | $0.3213 \pm 0.0558$ | $0.3710 \pm 0.1228$ | **$+0.0391$** | **$+0.0888$** | $0.2366 \pm 0.1386$ | $0.2581 \pm 0.0993$ | $0.3595 \pm 0.1499$ |

---

### Table 9: GNN Message-Passing Depth Ablation (Test Set, Timesteps 40–49)
*Comparison of 1-layer, 2-layer, and 3-layer GNN architectures on full 165 features across 5 seeds. Metrics reported as Mean $\pm$ Std.*

| Architecture | 1 Layer (1-Hop) F1 | 2 Layers (2-Hop) F1 | 3 Layers (3-Hop) F1 | $\Delta\text{F1}$ (2L - 1L) | $\Delta\text{F1}$ (3L - 2L) | 1 Layer PR-AUC | 2 Layers PR-AUC | 3 Layers PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GAT** | $0.2877 \pm 0.0319$ | $0.3308 \pm 0.0602$ | $0.2847 \pm 0.0052$ | **$+0.0431$** | **$-0.0461$** | $0.1936 \pm 0.0284$ | $0.2667 \pm 0.0695$ | $0.2362 \pm 0.0126$ |
| **GCN** | $0.2862 \pm 0.0250$ | $0.2589 \pm 0.0639$ | $0.2089 \pm 0.0892$ | **$-0.0273$** | **$-0.0500$** | $0.2255 \pm 0.0527$ | $0.2202 \pm 0.0703$ | $0.1648 \pm 0.0931$ |
| **GraphSAGE** | $0.3750 \pm 0.1215$ | $0.2822 \pm 0.1248$ | $0.3327 \pm 0.1250$ | **$-0.0928$** | **$+0.0505$** | $0.3471 \pm 0.1413$ | $0.2366 \pm 0.1386$ | $0.2916 \pm 0.1145$ |

---

### Table 10: Statistical Hypothesis Tests and Effect Sizes (Holdout Test Set)
*Paired statistical comparisons between the validation-selected GNN architecture (GraphSAGE on Local 93 features) and representative baseline models across 5 shared seeds ($S = \{42, 123, 456, 789, 999\}$). $p_{\text{Holm}}$ controls FWER across baselines. $p_{\text{Wilcoxon}}$ is exact two-sided ($p_{\min} = 0.0625$ at $N=5$).*

| Evaluation Metric | Baseline Model Configuration | GNN Mean ± Std | Baseline Mean ± Std | Mean Diff ($\bar{\Delta}$) | Paired $t$ | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Wilcoxon $W$ | $p_{\text{Wilcoxon}}$ | Cohen's $d_z$ | FWER Signif. ($\alpha=.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Illicit F1** | **Random Forest** (Full, Std) | $0.4153 \pm 0.1914$ | $0.7247 \pm 0.0027$ | $-0.3094$ | $-3.5917$ | $0.0229$ | $0.0917$ | $0.0$ | $0.0625$ | $-1.6062$ | Not Significant |
| **Test Illicit F1** | **XGBoost** (Full, Weighted) | $0.4153 \pm 0.1914$ | $0.7197 \pm 0.0074$ | $-0.3044$ | $-3.5092$ | $0.0247$ | $0.0917$ | $0.0$ | $0.0625$ | $-1.5694$ | Not Significant |
| **Test Illicit F1** | **MLP** (Full, Std) | $0.4153 \pm 0.1914$ | $0.5848 \pm 0.0194$ | $-0.1695$ | $-1.8872$ | $0.1322$ | $0.2644$ | $3.0$ | $0.3125$ | $-0.8440$ | Not Significant |
| **Test Illicit F1** | **Logistic Regression** (Full, Wtd) | $0.4153 \pm 0.1914$ | $0.3480 \pm 0.0000$ | $+0.0673$ | $+0.7856$ | $0.4760$ | $0.4760$ | $6.0$ | $0.8125$ | $+0.3513$ | Not Significant |
| **Test PR-AUC** | **Random Forest** (Full, Std) | $0.3659 \pm 0.1859$ | $0.6599 \pm 0.0025$ | $-0.2940$ | $-3.5141$ | $0.0246$ | $0.0816$ | $0.0$ | $0.0625$ | $-1.5716$ | Not Significant |
| **Test PR-AUC** | **XGBoost** (Full, Weighted) | $0.3659 \pm 0.1859$ | $0.6723 \pm 0.0039$ | $-0.3064$ | $-3.7246$ | $0.0204$ | $0.0816$ | $0.0$ | $0.0625$ | $-1.6657$ | Not Significant |
| **Test PR-AUC** | **MLP** (Full, Std) | $0.3659 \pm 0.1859$ | $0.5115 \pm 0.0187$ | $-0.1456$ | $-1.6440$ | $0.1755$ | $0.3070$ | $3.0$ | $0.3125$ | $-0.7352$ | Not Significant |
| **Test PR-AUC** | **Logistic Regression** (Full, Wtd) | $0.3659 \pm 0.1859$ | $0.2197 \pm 0.0000$ | $+0.1462$ | $+1.7584$ | $0.1535$ | $0.3070$ | $3.0$ | $0.3125$ | $+0.7864$ | Not Significant |

---

### Table 11: Error Taxonomy and Graph-Structural Analysis (Holdout Test Set)
*Decomposition of test set nodes ($N = 11,184$, Timesteps 40–49) by prediction category, graph topology, and individual timestep dynamics.*

| Category / Timestep Segment | Sample Count ($N$) | Proportion (\%) | Mean In-Degree | Mean Out-Degree | Illicit Precision | Illicit Recall | Illicit F1 | Topological / Operational Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Test Set** | $11,184$ | $100.00\%$ | $2.06$ | $1.26$ | — | — | — | Full evaluation horizon ($t \in [40, 49]$) |
| **True Positives (TP)** | $402$ | $3.59\%$ | $0.87$ | $0.70$ | — | — | — | Correctly flagged illicit transactions |
| **True Negatives (TN)** | $8,165$ | $73.01\%$ | $2.38$ | $1.34$ | — | — | — | Correctly cleared licit transactions |
| **False Positives (FP)** | $2,383$ | $21.31\%$ | $1.23$ | $1.13$ | — | — | — | Licit transactions flagged as illicit |
| **False Negatives (FN)** | $234$ | $2.09\%$ | $1.61$ | $0.87$ | — | — | — | Missed illicit transactions |
| **High-Conf. FP ($\hat{p} \ge 0.80$)** | $0$ | $0.00\%$ | $0.00$ | $0.00$ | — | — | — | No extreme-confidence false alarms |
| **High-Conf. FN ($\hat{p} \le 0.20$)** | $0$ | $0.00\%$ | $0.00$ | $0.00$ | — | — | — | No extreme-confidence missed detections |
| **Timestep 40** | $1,211$ ($112$ illicit) | $10.83\%$ | — | — | $0.2090$ | $0.6250$ | $0.3132$ | Pre-shift stable regime ($9.25\%$ illicit) |
| **Timestep 41** | $1,132$ ($116$ illicit) | $10.12\%$ | — | — | $0.2531$ | $0.6983$ | $0.3716$ | Pre-shift stable regime ($10.25\%$ illicit) |
| **Timestep 42** | $2,154$ ($239$ illicit) | $19.26\%$ | — | — | $0.3800$ | $0.8410$ | $0.5234$ | Pre-shift peak activity ($11.10\%$ illicit) |
| **Timestep 43** | $1,370$ ($24$ illicit) | $12.25\%$ | — | — | $0.0409$ | $0.2917$ | $0.0718$ | Abrupt illicit collapse ($1.75\%$ illicit) |
| **Timestep 44** | $1,591$ ($14.23\%$) | $14.23\%$ | — | — | $0.0181$ | $0.5417$ | $0.0350$ | Depressed illicit regime ($1.51\%$ illicit) |
| **Timestep 45** | $1,221$ ($5$ illicit) | $10.92\%$ | — | — | $0.0036$ | $0.2000$ | $0.0070$ | Trough illicit regime ($0.41\%$ illicit) |
| **Timestep 46** | $712$ ($2$ illicit) | $6.37\%$ | — | — | $0.0087$ | $0.5000$ | $0.0171$ | Low-volume trough ($0.28\%$ illicit) |
| **Timestep 47** | $846$ ($22$ illicit) | $7.56\%$ | — | — | $0.0481$ | $0.4091$ | $0.0861$ | Early recovery period ($2.60\%$ illicit) |
| **Timestep 48** | $471$ ($36$ illicit) | $4.21\%$ | — | — | $0.1600$ | $0.2222$ | $0.1860$ | Post-shift rebound ($7.64\%$ illicit) |
| **Timestep 49** | $476$ ($56$ illicit) | $4.26\%$ | — | — | $0.1410$ | $0.1964$ | $0.1642$ | Post-shift rebound ($11.76\%$ illicit) |

---

# Figure Placement Plan

The following publication-grade figures (300 DPI) from the master inventory are designated for Sections 5, 6, and 7:

1. **Figure 6: Unknown-Node Context Ablation (Retained vs. Removed)**
   - **Source Filename:** [`results/figures/unknown_node_ablation.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/unknown_node_ablation.png)
   - **Section Placement:** Subsection 5.1 (immediately following Table 7).
   - **Scientific Demonstration:** Visualizes test illicit F1 when unknown nodes are retained in the graph versus removed from the topology, demonstrating the contrasting responses of GAT versus GCN and GraphSAGE.

2. **Figure 7: Directionality Sensitivity Ablation (Forward vs. Backward vs. Bidirectional)**
   - **Source Filename:** [`results/figures/direction_sensitivity_ablation.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/direction_sensitivity_ablation.png)
   - **Section Placement:** Subsection 5.2 (immediately following Table 8).
   - **Scientific Demonstration:** Illustrates the performance impact of reversing and bidirectionally expanding graph message propagation across GNN architectures.

3. **Figure 8: Message-Passing Depth Ablation (1, 2, and 3 Layers)**
   - **Source Filename:** [`results/figures/depth_ablation.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/depth_ablation.png)
   - **Section Placement:** Subsection 5.3 (immediately following Table 9).
   - **Scientific Demonstration:** Depicts the architectural trajectories of test F1 across 1-hop, 2-hop, and 3-hop receptive fields, highlighting monotonic GCN degradation versus GAT and GraphSAGE peaks.

4. **Figure 9: Cross-Seed Stability and Model Variance**
   - **Source Filename:** [`results/figures/gnn_model_comparison_f1.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_model_comparison_f1.png)
   - **Section Placement:** Subsection 6.1 (immediately following Table 10).
   - **Scientific Demonstration:** Compares the cross-seed variance distributions across models, illustrating the tight clustering of tree ensembles versus the wider performance dispersion of GNNs.

5. **Figure 10: Edge Homophily and Relational Class Transitions**
   - **Source Filename:** [`results/figures/edge_homophily_matrix.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/edge_homophily_matrix.png)
   - **Section Placement:** Subsection 7.2.
   - **Scientific Demonstration:** Illustrates the transition matrix of directed edges across licit, illicit, and unknown node categories, highlighting that $84.37\%$ of edge transitions involve unannotated nodes.

6. **Figure 11: Timestep-by-Timestep Performance and Prevalence Dynamics**
   - **Source Filename:** [`results/figures/gnn_temporal_performance_timesteps.png`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/figures/gnn_temporal_performance_timesteps.png)
   - **Section Placement:** Subsection 7.3 (immediately following Table 11).
   - **Scientific Demonstration:** Shows the chronological trajectory of test illicit F1 and ground-truth illicit prevalence across Timesteps 40–49, documenting the performance collapse at Timestep 43 and subsequent partial recovery.

---

# Evidence Integrity Check

- [x] **Ablation Values Verified:** All values in Table 7 (Unknown Nodes), Table 8 (Directionality), and Table 9 (Depth) match [`results/tables/MASTER_GNN_ABLATIONS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_GNN_ABLATIONS.csv) exactly.
- [x] **Statistical Values Verified:** All paired differences, $t$-statistics, raw $p$-values, Holm-adjusted $p$-values, Wilcoxon statistics, and Cohen's $d_z$ values in Table 10 match [`results/tables/MASTER_STATISTICAL_TESTS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_STATISTICAL_TESTS.csv) exactly.
- [x] **Error Taxonomy Values Verified:** All sample counts, in/out degrees, and timestep-level metrics in Table 11 match [`results/tables/MASTER_ERROR_ANALYSIS.csv`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/tables/MASTER_ERROR_ANALYSIS.csv) exactly.
- [x] **No Test-Based Model Selection:** The representative GNN evaluated in Table 10 (GraphSAGE Local 93) was pre-selected based on validation performance ($0.6060 \pm 0.2270$), not holdout test ranking.
- [x] **Small Sample Limitations Disclosed:** Explicitly noted that $N = 5$ seeds bounds the two-sided Wilcoxon signed-rank test at $p_{\min} = 0.0625$, preventing formal statistical significance at $\alpha = 0.05$.
- [x] **Homophily Disambiguation:** Clearly distinguished $95.37\%$ global labeled-edge homophily (on the $15.63\%$ mutually labeled subgraph) from the reality that $84.37\%$ of total edges connect to unlabeled nodes.
- [x] **Causal Attribution Cautions:** Directionality differences and depth variations are described as empirical observations without unmeasured causal assertions (e.g., over-smoothing and feature dilution framed as potential explanatory hypotheses).
- [x] **External Event Attribution Guardrail:** The Timestep 43 regime change is described statistically ($11.10\% \to 1.75\%$ illicit drop) with darknet marketplace attribution marked as requiring independent verification (`[CITATION REQUIRED]`).
- [x] **Temporal Split Consistency:** Verified as Train: Timesteps 1–34, Validation: Timesteps 35–39, Test: Timesteps 40–49.
