# Final Freeze Audit Report: Research Manuscript

**Paper Title:** *Graph Neural Networks versus Tabular Machine Learning for Illicit Transaction Detection in Bitcoin: An Empirical Evaluation under Chronological Partitioning*  
**Author:** Khushi (*Department of Artificial Intelligence and Data Science, Indira Gandhi Delhi Technical University for Women*)  
**LaTeX Source:** [`results/latex/main.tex`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main.tex)  
**Final Production PDF:** [`results/latex/main_final.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main_final.pdf)  
**PDF Size:** 2,238,125 bytes  
**Page Count:** 11 pages  
**Audit Timestamp:** 2026-09-19T21:55:00+05:30  
**Overall Freeze Status:** **ALL CHECKS PASSED — 100% VERIFIED**

---

## 1. Audit Checklist & Verification Matrix

| # | Check Item | Status | Verification Details |
|---|---|:---:|---|
| 1 | **Absence of Old Phrasing** | **PASS** | Verified zero occurrences of obsolete wording (Abstract, Unknown-node ablation, and Conclusion). |
| 2 | **Presence of Polished Wording** | **PASS** | Verified exact integration of all 3 wording polish items. |
| 3 | **Numerical Value Integrity** | **PASS** | All 16 primary F1, PR-AUC, precision, recall, and MCC metrics match `MASTER_MAIN_RESULTS.csv` and sub-tables. |
| 4 | **References & Bibliography** | **PASS** | 34 verified entries in `references.bib`, 34 distinct keys cited and resolved in LaTeX without truncation. |
| 5 | **Citation Resolution** | **PASS** | Zero unresolved citations (`[?]` or `??`) in LaTeX source and generated output. |
| 6 | **Figure Rendering** | **PASS** | All 11 300-DPI publication figures verified on disk and rendered in PDF. |
| 7 | **Table Rendering** | **PASS** | All 12 IEEE formal tables (Tables I–XII) rendered with proper captions and alignment. |
| 8 | **PDF Generation & Health** | **PASS** | Successfully generated `main_final.pdf` (2,238,125 bytes, 11 pages) via Tectonic TeX Engine. |
| 9 | **Scientific Content Freeze** | **PASS** | Zero experiments altered, zero hyperparameters modified, splits and seed matrices 100% frozen. |

---

## 2. Verified Wording Modifications

### A. Abstract Polishing
- **Previous:** *"...operating on the premise that topological message passing over transaction networks will outperform conventional machine learning models."*
- **Polished:** *"...because they can incorporate relational information through topological message passing."*

### B. Section V-A Unknown-Node Ablation
- **Previous:** *"...reflecting noise injection during neighborhood aggregation."*
- **Polished:** *"...indicating that the effect of unlabeled graph context is architecture-dependent under the evaluated setup."*

### C. Section X-A Conclusion
- **Previous:** *"...In contrast, GNN performance proved highly sensitive to graph directionality, unlabeled context, layer depth, and parameter initialization."*
- **Polished:** *"...In contrast, GNN performance was highly sensitive to graph directionality, unlabeled context, layer depth, and parameter initialization under the evaluated setup."*

---

## 3. Final Artifact Deliverables

- **Production PDF:** [`results/latex/main_final.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main_final.pdf)
- **Corrected PDF:** [`results/latex/main_corrected.pdf`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main_corrected.pdf)
- **Authoritative LaTeX:** [`results/latex/main.tex`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/main.tex)
- **Authoritative Bibliography:** [`results/latex/references.bib`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/latex/references.bib)
- **Markdown Manuscripts:** [`results/FINAL_MANUSCRIPT.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/FINAL_MANUSCRIPT.md) & [`results/IEEE_MANUSCRIPT.md`](file:///c:/Users/hp/Desktop/internship%20research%20paper/elliptic-gnn-fraud-research/results/IEEE_MANUSCRIPT.md)
