# -*- coding: utf-8 -*-
"""
Deep claim-to-citation audit and coverage verification script.
"""

import re, unicodedata, json

with open('results/IEEE_MANUSCRIPT.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

with open('results/latex/main.tex', 'r', encoding='utf-8') as f:
    tex_text = f.read()

with open('results/references.bib', 'r', encoding='utf-8') as f:
    bib_text = f.read()

# 1. Parse all BibTeX keys
bib_keys = re.findall(r'@\w+\{([^,]+),', bib_text)
print(f"Total BibTeX entries: {len(bib_keys)}")

# 2. Extract citations in latex/main.tex
tex_cites = re.findall(r'\\cite\{([^}]+)\}', tex_text)
tex_keys_used = set()
for c in tex_cites:
    for k in c.split(','):
        tex_keys_used.add(k.strip())

print(f"Total unique keys cited in main.tex: {len(tex_keys_used)}")

# 3. Check for any missing or unused keys
missing_in_bib = tex_keys_used - set(bib_keys)
unused_in_tex = set(bib_keys) - tex_keys_used
print(f"Missing in BibTeX: {missing_in_bib}")
print(f"Unused in LaTeX: {unused_in_tex}")

# 4. Check in-text citations in IEEE_MANUSCRIPT.md
md_cites = re.findall(r'\[(\d+(?:[–,\s]+\d+)*)\]', md_text.split('# References')[0])
print(f"Total citation callout instances in IEEE_MANUSCRIPT.md: {len(md_cites)}")

# 5. Extract all numbers used in MD citations
md_nums_used = set()
for c in md_cites:
    # Handle ranges like [4]–[7] or lists [1], [2]
    parts = re.split(r'[,–\s]+', c)
    for p in parts:
        if p.isdigit():
            md_nums_used.add(int(p))

print(f"Numbers used in IEEE_MANUSCRIPT.md: {sorted(list(md_nums_used))}")
print(f"Min number: {min(md_nums_used)}, Max number: {max(md_nums_used)}, Expected: 1 to 34")
print(f"All 1..34 present: {set(range(1, 35)) == md_nums_used}")

# 6. Check order of appearance in main.tex vs bibliography
tex_cites_order = []
seen = set()
for c in tex_cites:
    for k in c.split(','):
        k_clean = k.strip()
        if k_clean not in seen:
            seen.add(k_clean)
            tex_cites_order.append(k_clean)

print(f"\nOrder of first appearance in LaTeX main.tex ({len(tex_cites_order)} keys):")
for i, k in enumerate(tex_cites_order, 1):
    print(f"[{i:2d}] {k}")

# 7. Check for unresolved placeholders
placeholders_md = re.findall(r'\[CITATION REQUIRED[^\]]*\]', md_text)
placeholders_tex = re.findall(r'\[CITATION REQUIRED[^\]]*\]', tex_text)
print(f"\nPlaceholders in MD: {len(placeholders_md)}")
print(f"Placeholders in TeX: {len(placeholders_tex)}")
