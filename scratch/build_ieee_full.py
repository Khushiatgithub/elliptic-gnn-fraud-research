# -*- coding: utf-8 -*-
import re, unicodedata, os

def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

with open('results/FINAL_MANUSCRIPT.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Author-Year to BibTeX Key and IEEE Number
key_to_ieee = {
    'foley2019sex': 1,
    'meiklejohn2013fistful': 2,
    'moser2013inquiry': 3,
    'harlev2018breaking': 4,
    'hu2019transaction': 5,
    'reid2013analysis': 6,
    'he2009learning': 7,
    'weber2019elliptic': 8,
    'gama2014survey': 9,
    'quinonero2009dataset': 10,
    'bronstein2017geometric': 11,
    'wu2020comprehensive': 12,
    'kipf2017semi': 13,
    'hamilton2017inductive': 14,
    'velickovic2018graph': 15,
    'grinsztajn2022why': 16,
    'shwartz2022tabular': 17,
    'errica2020fair': 18,
    'arp2022dos': 19,
    'davis2006relationship': 20,
    'saito2015precision': 21,
    'breiman2001random': 22,
    'chen2016xgboost': 23,
    'holm1979simple': 24,
    'wilcoxon1945individual': 25,
    'demsar2006statistical': 26,
    'cohen1988statistical': 27,
    'lakens2013calculating': 28,
    'li2018deeper': 29,
    'oono2020graph': 30,
    'chen2020measuring': 31,
    'wang2021graph': 32,
    'xu2020inductive': 33,
    'rossi2020temporal': 34
}

author_year_to_key = {
    'arp et al., 2022': 'arp2022dos',
    'breiman, 2001': 'breiman2001random',
    'bronstein et al., 2017': 'bronstein2017geometric',
    'chen and guestrin, 2016': 'chen2016xgboost',
    'chen et al., 2020': 'chen2020measuring',
    'cohen, 1988': 'cohen1988statistical',
    'davis and goadrich, 2006': 'davis2006relationship',
    'demsar, 2006': 'demsar2006statistical',
    'errica et al., 2020': 'errica2020fair',
    'foley et al., 2019': 'foley2019sex',
    'gama et al., 2014': 'gama2014survey',
    'grinsztajn et al., 2022': 'grinsztajn2022why',
    'hamilton et al., 2017': 'hamilton2017inductive',
    'harlev et al., 2018': 'harlev2018breaking',
    'he and garcia, 2009': 'he2009learning',
    'holm, 1979': 'holm1979simple',
    'hu et al., 2019': 'hu2019transaction',
    'kipf and welling, 2017': 'kipf2017semi',
    'lakens, 2013': 'lakens2013calculating',
    'li et al., 2018': 'li2018deeper',
    'meiklejohn et al., 2013': 'meiklejohn2013fistful',
    'moser et al., 2013': 'moser2013inquiry',
    'oono and suzuki, 2020': 'oono2020graph',
    'quinonero-candela et al., 2009': 'quinonero2009dataset',
    'reid and harrigan, 2013': 'reid2013analysis',
    'rossi et al., 2020': 'rossi2020temporal',
    'saito and rehmsmeier, 2015': 'saito2015precision',
    'shwartz-ziv and armon, 2022': 'shwartz2022tabular',
    'velickovic et al., 2018': 'velickovic2018graph',
    'wang et al., 2021': 'wang2021graph',
    'weber et al., 2019': 'weber2019elliptic',
    'wilcoxon, 1945': 'wilcoxon1945individual',
    'wu et al., 2020': 'wu2020comprehensive',
    'xu et al., 2020': 'xu2020inductive'
}

def format_ieee_citations_md(nums):
    nums = sorted(list(set(nums)))
    if not nums:
        return ''
    ranges = []
    start = nums[0]
    prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            if start == prev:
                ranges.append(f'[{start}]')
            elif prev == start + 1:
                ranges.append(f'[{start}], [{prev}]')
            else:
                ranges.append(f'[{start}]–[{prev}]')
            start = n
            prev = n
    if start == prev:
        ranges.append(f'[{start}]')
    elif prev == start + 1:
        ranges.append(f'[{start}], [{prev}]')
    else:
        ranges.append(f'[{start}]–[{prev}]')
    return ', '.join(ranges)

# Separate body and references
parts = text.split('# References')
body_text = parts[0]

# Function to replace in-text citations for Markdown
def replace_citations_in_md(t):
    def repl(m):
        content = m.group(1)
        if any(k in content for k in ['TABLE', 'Figure', 'd_{', 'u \to', 'u <->', 't \in', 'p >=', 'p <=', 'h_']):
            return m.group(0)
        chunks = content.split(';')
        nums = []
        for c in chunks:
            clean = normalize(c.strip().lower())
            matched_key = None
            for ay, k in author_year_to_key.items():
                if normalize(ay) == clean:
                    matched_key = k
                    break
            if matched_key:
                nums.append(key_to_ieee[matched_key])
            else:
                return m.group(0)
        return format_ieee_citations_md(nums)

    pat = r'\[([A-Z\u00C0-\u017F][^\]]*?\b\d{4}\b[^\]]*?)\]'
    return re.sub(pat, repl, t)

md_body = replace_citations_in_md(body_text)

# Replace Markdown Headers to IEEE roman numerals
replacements = [
    (r'^# Graph Neural Networks versus.*$', '# Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning\n\n**Anonymous Authors**  \n*Department of Computer Science and Engineering, Research Institute*  \n`{author1, author2, author3}@example.edu`'),
    (r'^## Abstract', '## Abstract'),
    (r'^## Keywords', '## Index Terms'),
    (r'^# 1\. Introduction', '# I. INTRODUCTION'),
    (r'^## 1\.1 Background', '## A. Background and Context'),
    (r'^## 1\.2 Problem Definition and Challenges', '## B. Problem Definition and Class Imbalance'),
    (r'^## 1\.3 Motivation for Graph Neural Networks', '## C. Motivation for Graph Representation Learning'),
    (r'^## 1\.4 Research Gap and Contributions', '## D. Research Gap and Primary Contributions'),
    (r'^## 1\.5 Paper Organization', '## E. Paper Organization'),
    (r'^# 2\. Dataset and Problem Formulation', '# II. DATASET AND PROBLEM FORMULATION'),
    (r'^## 2\.1 The Elliptic Bitcoin Benchmark', '## A. The Elliptic Bitcoin Benchmark'),
    (r'^## 2\.2 Entity Classes and Label Imbalance', '## B. Entity Classes and Label Imbalance'),
    (r'^## 2\.3 Evaluation Metrics under Class Skew', '## C. Evaluation Metrics under Class Skew'),
    (r'^## 2\.4 Feature Space Representation', '## D. Feature Space Representation'),
    (r'^## 2\.5 Chronological Partitioning Protocol', '## E. Chronological Partitioning Protocol'),
    (r'^# 3\. Methodology', '# III. METHODOLOGY'),
    (r'^## 3\.1 Baseline Model Architectures', '## A. Baseline Model Architectures'),
    (r'^## 3\.2 Graph Neural Network Architectures', '## B. Graph Neural Network Architectures'),
    (r'^## 3\.3 Loss Formulations and Cost-Sensitive Weighting', '## C. Loss Formulations and Cost-Sensitive Weighting'),
    (r'^## 3\.4 Training and Threshold Locking Protocol', '## D. Training and Threshold Locking Protocol'),
    (r'^## 3\.5 Statistical Significance Testing Protocol', '## E. Statistical Significance Testing Protocol'),
    (r'^# 4\. Results', '# IV. RESULTS'),
    (r'^## 4\.1 Main Comparison on the Full 165-Feature Representation', '## A. Main Comparison on Full Features'),
    (r'^## 4\.2 F1 and PR-AUC Behavior', '## B. F1 and PR-AUC Dynamics'),
    (r'^## 4\.3 Precision and Recall Dynamics', '## C. Precision-Recall Trade-offs'),
    (r'^## 4\.4 Local 93 Features versus Full 165 Features', '## D. Local versus Full Feature Representations'),
    (r'^## 4\.5 Standard versus Cost-Sensitive Loss Weighting', '## E. Standard versus Cost-Sensitive Loss Weighting'),
    (r'^## 4\.6 Validation-to-Test Generalization', '## F. Validation-to-Test Generalization Decay'),
    (r'^## 4\.7 Random-Seed Stability and Parameter Initialization Sensitivity', '## G. Random-Seed Stability and Initialization Variance'),
    (r'^# 5\. Ablation and Sensitivity Analysis', '# V. ABLATION AND SENSITIVITY ANALYSIS'),
    (r'^## 5\.1 Effect of Unknown-Label Nodes', '## A. Impact of Unlabeled Graph Context'),
    (r'^## 5\.2 Effect of Graph Propagation Direction', '## B. Graph Propagation Directionality'),
    (r'^## 5\.3 Effect of Message-Passing Depth', '## C. Message-Passing Layer Depth'),
    (r'^# 6\. Statistical Analysis', '# VI. STATISTICAL ANALYSIS'),
    (r'^## 6\.1 Paired Statistical Comparison', '## A. Paired Hypothesis Testing'),
    (r'^## 6\.2 Statistical Power and Sample Size Constraints', '## B. Statistical Power Bounds at N=5'),
    (r'^# 7\. Error and Graph-Structural Analysis', '# VII. ERROR AND GRAPH-STRUCTURAL ANALYSIS'),
    (r'^## 7\.1 Error Taxonomy and Degree Disparity', '## A. Error Taxonomy and Degree Disparity'),
    (r'^## 7\.2 Graph Homophily and Relational Topology', '## B. Graph Homophily and Relational Topology'),
    (r'^## 7\.3 Temporal Error Patterns across Timesteps 40–49', '## C. Temporal Error Patterns across Timesteps 40–49'),
    (r'^# 8\. Discussion', '# VIII. DISCUSSION'),
    (r'^## 8\.1 Overall Comparison Between GNNs and Tabular Models', '## A. Comparative Performance of GNNs versus Tabular Baselines'),
    (r'^## 8\.2 Why the Results Matter for Graph-Based Fraud Detection', '## B. Critical Evaluation of Relational Inductive Bias'),
    (r'^## 8\.3 Feature Representation Dynamics', '## C. Feature Representation Dynamics'),
    (r'^## 8\.4 Class Imbalance and Cost-Sensitive Loss Weighting', '## D. Class Imbalance and Threshold Tuning Dynamics'),
    (r'^## 8\.5 Temporal Generalization and Non-Stationarity', '## E. Temporal Distribution Shift and Concept Drift'),
    (r'^## 8\.6 GNN Seed Sensitivity and Optimization Variance', '## F. Seed Sensitivity and Optimization Landscape'),
    (r'^## 8\.7 Unknown-Node Structural Context', '## G. Structural Role of Unlabeled Transactions'),
    (r'^## 8\.8 Graph Propagation Directionality', '## H. Causal Directionality versus Retrospective Leakage'),
    (r'^## 8\.9 Message-Passing Depth', '## I. Over-Smoothing and Layer Depth Dynamics'),
    (r'^## 8\.10 Statistical Interpretation and Power Limitations', '## J. Statistical Power and Multiple Comparison Corrections'),
    (r'^## 8\.11 Error Analysis and Structural Connectivity', '## K. Structural Profiles of Model Misclassifications'),
    (r'^## 8\.12 Darknet Market Dynamics and Timestep 43', '## L. Darknet Market Dynamics and Timestep 43 Non-Stationarity'),
    (r'^## 8\.13 Practical Implications for Industry and Regulators', '## M. Practical Implications for Industry and Regulators'),
    (r'^# 9\. Limitations', '# IX. LIMITATIONS'),
    (r'^## 9\.1 Anonymized Feature Semantics', '## A. Anonymized Feature Semantics'),
    (r'^## 9\.2 High Proportion of Unlabeled Transactions', '## B. High Proportion of Unlabeled Transactions'),
    (r'^## 9\.3 Dynamic Concept Drift and Distribution Shift', '## C. Temporal Non-Stationarity and Regime Shifts'),
    (r'^## 9\.4 Statistical Power Constraints \(N=5\)', '## D. Statistical Power Constraints with Five Seeds'),
    (r'^## 9\.5 Static GNN Architectures', '## E. Static Architecture Limitations'),
    (r'^## 9\.6 Potential Artifacts of Homogeneous Graph Modeling', '## F. Homogeneous Graph Assumptions'),
    (r'^## 9\.7 Generalization Beyond the Elliptic Benchmark', '## G. Dataset-Specific Generalization Boundaries'),
    (r'^## 9\.8 Discrete Temporal Snapshots and Lack of Continuous-Time Graphs', '## H. Discrete Temporal Snapshots'),
    (r'^# 10\. Conclusion and Future Work', '# X. CONCLUSION AND FUTURE WORK'),
    (r'^## 10\.1 Conclusion', '## A. Conclusion'),
    (r'^## 10\.2 Future Work', '## B. Future Work'),
    (r'^# 11\. Research Question Synthesis and Empirical Answer Matrix', '# XI. RESEARCH QUESTION SYNTHESIS'),
    (r'^# 12\. Final Claim-Boundary Audit', '# XII. FINAL CLAIM-BOUNDARY AUDIT')
]

for pat, rep in replacements:
    md_body = re.sub(pat, rep, md_body, flags=re.MULTILINE)

# Replace table roman numerals in Markdown
table_roman = {
    'TABLE 1:': 'TABLE I:\n',
    'TABLE 2:': 'TABLE II:\n',
    'TABLE 3:': 'TABLE III:\n',
    'TABLE 4:': 'TABLE IV:\n',
    'TABLE 5:': 'TABLE V:\n',
    'TABLE 6:': 'TABLE VI:\n',
    'TABLE 7:': 'TABLE VII:\n',
    'TABLE 8:': 'TABLE VIII:\n',
    'TABLE 9:': 'TABLE IX:\n',
    'TABLE 10:': 'TABLE X:\n',
    'TABLE 11:': 'TABLE XI:\n',
    'TABLE 12:': 'TABLE XII:\n',
    'TABLE 13:': 'TABLE XIII:\n',
    'Table 1': 'Table I',
    'Table 2': 'Table II',
    'Table 3': 'Table III',
    'Table 4': 'Table IV',
    'Table 5': 'Table V',
    'Table 6': 'Table VI',
    'Table 7': 'Table VII',
    'Table 8': 'Table VIII',
    'Table 9': 'Table IX',
    'Table 10': 'Table X',
    'Table 11': 'Table XI',
    'Table 12': 'Table XII',
    'Table 13': 'Table XIII'
}

for k, v in table_roman.items():
    md_body = md_body.replace(k, v)

# Replace figure captions to Fig. 1. etc.
fig_repl = {
    '*(Figure 1:': '*(Fig. 1.',
    '*(Figure 2:': '*(Fig. 2.',
    '*(Figure 3:': '*(Fig. 3.',
    '*(Figure 4:': '*(Fig. 4.',
    '*(Figure 5:': '*(Fig. 5.',
    '*(Figure 6:': '*(Fig. 6.',
    '*(Figure 7:': '*(Fig. 7.',
    '*(Figure 8:': '*(Fig. 8.',
    '*(Figure 9:': '*(Fig. 9.',
    '*(Figure 10:': '*(Fig. 10.',
    '*(Figure 11:': '*(Fig. 11.',
    'Figure 1': 'Fig. 1',
    'Figure 2': 'Fig. 2',
    'Figure 3': 'Fig. 3',
    'Figure 4': 'Fig. 4',
    'Figure 5': 'Fig. 5',
    'Figure 6': 'Fig. 6',
    'Figure 7': 'Fig. 7',
    'Figure 8': 'Fig. 8',
    'Figure 9': 'Fig. 9',
    'Figure 10': 'Fig. 10',
    'Figure 11': 'Fig. 11'
}

for k, v in fig_repl.items():
    md_body = md_body.replace(k, v)

# Build IEEE References section
ieee_refs = [
    '[1] S. Foley, J. R. Karlsen, and T. J. Putniņš, "Sex, drugs, and bitcoin: How much illegal activity is financed through cryptocurrencies?" *The Review of Financial Studies*, vol. 32, no. 5, pp. 1798–1853, 2019, doi: 10.1093/rfs/hhz015.',
    '[2] S. Meiklejohn, M. Pomarole, G. Jordan, K. Levchenko, D. McCoy, G. M. Voelker, and S. Savage, "A fistful of bitcoins: Characterizing payments among men with no names," in *Proc. 2013 ACM SIGCOMM Conf. Internet Meas. (IMC)*, 2013, pp. 127–140, doi: 10.1145/2504730.2504747.',
    '[3] M. Möser, R. Böhme, and D. Breuker, "An inquiry into money laundering tools in the Bitcoin ecosystem," in *Proc. 2013 APWG eCrime Researchers Summit (eCRS)*, 2013, pp. 1–14, doi: 10.1109/eCRS.2013.6805780.',
    '[4] M. A. Harlev, H. Sun Yin, K. C. Langenheldt, R. Mukkamala, and R. Vatrapu, "Breaking bad: De-anonymising entity types on the Bitcoin blockchain," in *Proc. 2018 Crypto Valley Conf. Blockchain Technol. (CVCBT)*, 2018, pp. 8–18, doi: 10.1109/CVCBT.2018.00008.',
    '[5] Y. Hu, S. Sahoo, K. Wang, and M. Mayhew, "Transaction-based classification and detection of Bitcoin illicit entities," in *Proc. 2019 IEEE Int. Conf. Data Mining Workshops (ICDMW)*, 2019, pp. 83–90, doi: 10.1109/ICDMW.2019.00022.',
    '[6] F. Reid and M. Harrigan, "An analysis of anonymity in the Bitcoin system," in *Security and Privacy in Social Networks*, Y. Altshuler et al., Eds. New York, NY, USA: Springer, 2013, pp. 197–223, doi: 10.1007/978-1-4614-4139-7_10.',
    '[7] H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263–1284, Sep. 2009, doi: 10.1109/TKDE.2008.239.',
    '[8] M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. W. Shen, "Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for anomaly detection," in *KDD 2019 Workshop on Applied Data Science for Developing Economies (ADS-DE)*, 2019, arXiv:1908.02591.',
    '[9] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," *ACM Comput. Surv.*, vol. 46, no. 4, Art. no. 44, pp. 1–37, Apr. 2014, doi: 10.1145/2523813.',
    '[10] J. Quiñonero-Candela, M. Sugiyama, A. Schwaighofer, and N. D. Lawrence, Eds., *Dataset Shift in Machine Learning*. Cambridge, MA, USA: MIT Press, 2009.',
    '[11] M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst, "Geometric deep learning: Going beyond Euclidean data," *IEEE Signal Process. Mag.*, vol. 34, no. 4, pp. 18–42, Jul. 2017, doi: 10.1109/MSP.2017.2693418.',
    '[12] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu, "A comprehensive survey on graph neural networks," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 32, no. 1, pp. 4–24, Jan. 2021, doi: 10.1109/TNNLS.2020.2978386.',
    '[13] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. 5th Int. Conf. Learn. Representations (ICLR)*, 2017.',
    '[14] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 1024–1034.',
    '[15] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. 6th Int. Conf. Learn. Representations (ICLR)*, 2018.',
    '[16] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022, pp. 507–520.',
    '[17] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84–90, May 2022, doi: 10.1016/j.inffus.2021.11.011.',
    '[18] F. Errica, M. Podda, D. Bacciu, and A. Micheli, "A fair comparison of graph neural networks for graph classification," in *Proc. 8th Int. Conf. Learn. Representations (ICLR)*, 2020.',
    '[19] D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Dos Santos, L. Cavallaro, and K. Rieck, "Dos and don\'ts of machine learning in computer security," in *Proc. 31st USENIX Secur. Symp. (USENIX Security)*, 2022, pp. 3971–3988.',
    '[20] J. Davis and M. Goadrich, "The relationship between Precision-Recall and ROC curves," in *Proc. 23rd Int. Conf. Mach. Learn. (ICML)*, 2006, pp. 233–240, doi: 10.1145/1143844.1143874.',
    '[21] T. Saito and M. Rehmsmeier, "The Precision-Recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, Mar. 2015, doi: 10.1371/journal.pone.0118432.',
    '[22] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.',
    '[23] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.',
    '[24] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.',
    '[25] F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bull.*, vol. 1, no. 6, pp. 80–83, Dec. 1945, doi: 10.2307/3001968.',
    '[26] J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, Jan. 2006.',
    '[27] J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Hillsdale, NJ, USA: Lawrence Erlbaum Associates, 1988.',
    '[28] D. Lakens, "Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs," *Front. Psychol.*, vol. 4, Art. no. 863, Nov. 2013, doi: 10.3389/fpsyg.2013.00863.',
    '[29] Q. Li, Z. Han, and X.-M. Wu, "Deeper insights into graph convolutional networks for semi-supervised learning," in *Proc. 32nd AAAI Conf. Artif. Intell. (AAAI)*, 2018, pp. 3538–3545, doi: 10.1609/aaai.v32i1.11690.',
    '[30] K. Oono and T. Suzuki, "Graph neural networks exponentially lose expressive power for node classification," in *Proc. 8th Int. Conf. Learn. Representations (ICLR)*, 2020.',
    '[31] D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, and X. Sun, "Measuring and relieving over-smoothing problem for graph neural networks," in *Proc. 13th Int. Conf. Web Search Data Min. (WSDM)*, 2020, pp. 132–140, doi: 10.1145/3336191.3371783.',
    '[32] J. Wang, V. S. Sheng, C. Cortes, and M. Mohri, "A survey on graph neural networks for financial fraud detection," *ACM Comput. Surv.*, vol. 54, no. 11s, Art. no. 235, pp. 1–37, Nov. 2021, doi: 10.1145/3472753.',
    '[33] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, "Inductive representation learning on temporal graphs," in *Proc. 8th Int. Conf. Learn. Representations (ICLR)*, 2020.',
    '[34] E. Rossi, B. Zhou, F. Monti, F. Frasca, U. Alon, and M. M. Bronstein, "Temporal graph networks for deep learning on dynamic graphs," *arXiv preprint arXiv:2006.10637*, 2020.'
]

ieee_md_content = md_body.strip() + '\n\n---\n\n# References\n\n' + '\n'.join(ieee_refs) + '\n'

with open('results/IEEE_MANUSCRIPT.md', 'w', encoding='utf-8') as f:
    f.write(ieee_md_content)

print(f'Successfully generated results/IEEE_MANUSCRIPT.md ({len(ieee_md_content)} chars)')
