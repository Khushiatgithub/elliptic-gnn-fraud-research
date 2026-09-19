# -*- coding: utf-8 -*-
"""
Map all citations across manuscript sections for Step 5F audit.
"""

import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/latex/main.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Split into sections
sections = re.split(r'\\section\{([^}]+)\}', tex)
print(f"Total sections found: {len(sections)//2}")

for i in range(1, len(sections), 2):
    sec_name = sections[i]
    sec_body = sections[i+1]
    cites = re.findall(r'\\cite\{([^}]+)\}', sec_body)
    sec_keys = []
    for c in cites:
        for k in c.split(','):
            sec_keys.append(k.strip())
    print(f"\nSection: {sec_name}")
    print(f"  Citations count: {len(sec_keys)}")
    print(f"  Unique keys ({len(set(sec_keys))}): {', '.join(sorted(list(set(sec_keys))))}")
