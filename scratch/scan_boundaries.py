# -*- coding: utf-8 -*-
"""
Scan manuscript for claim boundary words and generate:
1. results/reports/FINAL_SUBMISSION_MANIFEST.md
2. results/reports/STEP_5H_RESEARCH_INTEGRITY_AUDIT.md
"""

import re
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/latex/main.tex', 'r', encoding='utf-8') as f:
    tex_text = f.read()

with open('results/IEEE_MANUSCRIPT.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

# Scan for boundary words
boundary_words = [
    'proves', 'prove', 'always', 'never', 'universally', 'guarantees',
    'causes', 'solves', 'definitive', 'state-of-the-art', 'SOTA',
    'production-ready', 'deployment-ready', 'best model', 'worst model'
]

print("=== CLAIM-BOUNDARY WORD SCAN IN MAIN.TEX ===")
flagged_instances = []
for word in boundary_words:
    pattern = rf'\b{word}\b'
    matches = list(re.finditer(pattern, tex_text, re.IGNORECASE))
    for m in matches:
        start = max(0, m.start() - 40)
        end = min(len(tex_text), m.end() + 40)
        snippet = tex_text[start:end].replace('\n', ' ').strip()
        flagged_instances.append((word, snippet))
        print(f"Word: '{word:15s}' | Context: ...{snippet}...")

print(f"\nTotal scanned instances: {len(flagged_instances)}")
