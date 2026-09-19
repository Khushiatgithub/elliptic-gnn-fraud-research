# -*- coding: utf-8 -*-
"""
Audit extracted text from results/latex/main.pdf.
"""

import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/latex/pages/pdf_extracted_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("=" * 80)
print("1. CITATION CALLOUT AUDIT IN PDF")
print("=" * 80)
# Check for any '??' or unresolved citation indicators
question_marks = re.findall(r'\[\?+\]|\?{2,}', text)
print(f"Unresolved citations / references (??): {len(question_marks)} found -> {question_marks}")

# Check for all citation numbers [1] to [34]
cited_in_pdf = []
for i in range(1, 35):
    # look for [i] or [i, or , i] or -i] or –i]
    found = False
    if f"[{i}]" in text or f"[{i}," in text or f", {i}]" in text or f"–{i}]" in text or f"-{i}]" in text or f"[{i}–" in text or f"[{i}-" in text:
        found = True
    cited_in_pdf.append((i, found))

missing_cites = [i for i, f in cited_in_pdf if not f]
print(f"Citation numbers 1..34 present in PDF text: {len(cited_in_pdf) - len(missing_cites)} / 34")
if missing_cites:
    print(f"Missing citation callout numbers: {missing_cites}")
else:
    print("ALL 34 CITATIONS PRESENT IN PDF!")

print("\n" + "=" * 80)
print("2. SECTION HEADINGS AUDIT IN PDF")
print("=" * 80)
expected_sections = [
    "Introduction",
    "Dataset and Problem Formulation",
    "Methodology",
    "Results",
    "Ablation and Sensitivity Analysis",
    "Statistical Analysis",
    "Error and Graph-Structural Analysis",
    "Discussion",
    "Limitations",
    "Conclusion and Future Work",
    "References"
]

for sec in expected_sections:
    found = sec.lower() in text.lower()
    print(f"[{'PASS' if found else 'FAIL'}] Section: {sec}")

print("\n" + "=" * 80)
print("3. TABLE TITLES AUDIT IN PDF")
print("=" * 80)
for i in range(1, 13):
    # look for TABLE I, TABLE II, ...
    found = f"TABLE {i}" in text or f"TABLE I" in text or f"Table {i}" in text
    print(f"Table {i:2d} in text: {found}")

print("\n" + "=" * 80)
print("4. FIGURE CAPTIONS AUDIT IN PDF")
print("=" * 80)
for i in range(1, 12):
    found = f"Fig. {i}" in text or f"Figure {i}" in text or f"Fig. {i}:" in text
    print(f"Figure {i:2d} in text: {found}")
