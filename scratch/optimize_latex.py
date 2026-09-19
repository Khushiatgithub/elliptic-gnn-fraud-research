# -*- coding: utf-8 -*-
"""
Polish and optimize results/latex/main.tex:
1. Fix remaining markdown **...** instances.
2. Ensure all single-column tables fit perfectly within \\columnwidth using \\resizebox{\\columnwidth}{!}{...}.
3. Add hyperref metadata for submission PDF.
"""

import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/latex/main.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# 1. Replace markdown **...**
tex = tex.replace('**Illicit-Class F1-Score**', r'\textbf{Illicit-Class F1-Score}')
tex = tex.replace('**Precision-Recall Area Under the Curve (PR-AUC)**', r'\textbf{Precision-Recall Area Under the Curve (PR-AUC)}')
tex = tex.replace('**Timestep 43**', r'\textbf{Timestep 43}')

# 2. Add hyperref metadata in preamble if not present
if 'hyperref' not in tex:
    meta = r"""\usepackage[hidelinks]{hyperref}
\hypersetup{
    pdftitle={Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning},
    pdfauthor={Anonymous Authors},
    pdfsubject={Blockchain Forensics and Graph Neural Networks},
    pdfkeywords={Bitcoin, illicit transaction detection, fraud detection, graph neural networks, GraphSAGE, GCN, GAT, tabular machine learning, temporal generalization, class imbalance}
}
"""
    tex = tex.replace(r'\usepackage{microtype}', "\\usepackage{microtype}\n" + meta)

# 3. For single-column tables (tab:model_spec, tab:feat_comp, tab:weight_comp, tab:seed_stability, tab:unknown_ablation, tab:direction_ablation, tab:depth_ablation, tab:error_taxonomy)
# Wrap tabular in \resizebox{\columnwidth}{!}{...} if not already wrapped
table_labels = [
    'tab:model_spec',
    'tab:feat_comp',
    'tab:weight_comp',
    'tab:seed_stability',
    'tab:unknown_ablation',
    'tab:direction_ablation',
    'tab:depth_ablation',
    'tab:error_taxonomy'
]

for lbl in table_labels:
    pattern = rf'(\\begin\{{table\}}[^\n]*\n\s*\\centering\n\s*\\caption\{{[^}}]+\}}\n\s*\\label\{{{lbl}\}}\n)(\s*\\begin\{{tabular\}}[\s\S]*?\\end\{{tabular\}})'
    m = re.search(pattern, tex)
    if m:
        header = m.group(1)
        tabular_code = m.group(2).strip()
        if r'\resizebox{\columnwidth}' not in tabular_code:
            new_code = header + "{\\small\n\\resizebox{\\columnwidth}{!}{\n" + tabular_code + "\n}\n}"
            tex = tex.replace(m.group(0), new_code)
            print(f"Wrapped table {lbl} with resizebox.")

with open('results/latex/main.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print("Saved updated results/latex/main.tex")
