# -*- coding: utf-8 -*-
"""
Deep structural audit of results/latex/main.tex.
"""

import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/latex/main.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

print("=" * 80)
print("1. LATEX PREAMBLE AUDIT")
print("=" * 80)
documentclass = re.findall(r'\\documentclass(\[[^\]]*\])?\{([^}]+)\}', tex)
print(f"Documentclass: {documentclass}")

packages = re.findall(r'\\usepackage(\[[^\]]*\])?\{([^}]+)\}', tex)
print(f"Total packages ({len(packages)}):")
for opt, pkg in packages:
    print(f"  - {pkg} (options: {opt if opt else 'none'})")

print("\n" + "=" * 80)
print("2. EQUATION AUDIT")
print("=" * 80)
eqs = re.findall(r'\\begin\{equation\}([\s\S]*?)\\end\{equation\}', tex)
print(f"Total equations found: {len(eqs)}")
for i, eq in enumerate(eqs, 1):
    label_m = re.search(r'\\label\{([^}]+)\}', eq)
    label = label_m.group(1) if label_m else "NO_LABEL"
    eq_clean = re.sub(r'\\label\{[^}]+\}', '', eq).strip().replace('\n', ' ')
    print(f"  Eq ({i:2d}) [{label}]: {eq_clean[:60]}...")

print("\n" + "=" * 80)
print("3. TABLE AUDIT")
print("=" * 80)
tables = re.findall(r'\\begin\{(?:table|table\*)([\s\S]*?)\\end\{(?:table|table\*)\}', tex)
print(f"Total tables found: {len(tables)}")
for i, tbl in enumerate(tables, 1):
    cap_m = re.search(r'\\caption\{([^}]+)\}', tbl)
    lbl_m = re.search(r'\\label\{([^}]+)\}', tbl)
    caption = cap_m.group(1) if cap_m else "NO_CAPTION"
    label = lbl_m.group(1) if lbl_m else "NO_LABEL"
    print(f"  Table {i:2d} [{label}]: {caption[:70]}...")

print("\n" + "=" * 80)
print("4. FIGURE AUDIT")
print("=" * 80)
figs = re.findall(r'\\begin\{(?:figure|figure\*)([\s\S]*?)\\end\{(?:figure|figure\*)\}', tex)
print(f"Total figures found in LaTeX: {len(figs)}")
for i, fig in enumerate(figs, 1):
    cap_m = re.search(r'\\caption\{([^}]+)\}', fig)
    lbl_m = re.search(r'\\label\{([^}]+)\}', fig)
    img_m = re.search(r'\\includegraphics(\[[^\]]*\])?\{([^}]+)\}', fig)
    caption = cap_m.group(1) if cap_m else "NO_CAPTION"
    label = lbl_m.group(1) if lbl_m else "NO_LABEL"
    img_path = img_m.group(2) if img_m else "NO_GRAPHIC"
    print(f"  Figure {i:2d} [{label}]: Graphic = {img_path} | Caption = {caption[:50]}...")
