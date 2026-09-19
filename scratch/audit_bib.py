# -*- coding: utf-8 -*-
"""
Audit all 34 BibTeX references in detail.
"""

import re, json

with open('results/references.bib', 'r', encoding='utf-8') as f:
    bib = f.read()

entries = []
blocks = re.split(r'\n@', bib)
for block in blocks:
    if not block.strip(): continue
    block = '@' + block if not block.startswith('@') else block
    m = re.match(r'@(\w+)\{([^,]+),\s*(.*)\}', block, re.DOTALL)
    if not m: continue
    etype, key, fields_raw = m.groups()
    fields = {}
    
    # Simple regex for BibTeX field parsing
    field_pattern = r'(\w+)\s*=\s*[\"{]([\s\S]*?)[\"}],?(?=\n\s*\w+\s*=|\n\s*\}|$)'
    for fm in re.finditer(field_pattern, fields_raw):
        fields[fm.group(1).lower()] = fm.group(2).strip()
    
    entries.append({'type': etype, 'key': key, 'fields': fields})

print(f"Parsed {len(entries)} entries successfully.\n")

# Check for required fields and validate quality
for i, e in enumerate(entries, 1):
    f = e['fields']
    key = e['key']
    author = f.get('author', 'MISSING')
    title = f.get('title', 'MISSING')
    year = f.get('year', 'MISSING')
    venue = f.get('journal', f.get('booktitle', f.get('publisher', 'MISSING')))
    doi = f.get('doi', f.get('url', f.get('eprint', 'N/A')))
    
    # Assess quality
    # A = Peer-reviewed journal / major conference (IEEE, ACM, NeurIPS, ICLR, USENIX, AAAI, PLOS)
    # B = Authoritative dataset / Workshop / Book (Springer, MIT Press)
    # C = Verified arXiv preprint with author attribution
    quality = 'A'
    if 'arxiv' in doi.lower() or 'arxiv' in venue.lower() or 'workshop' in venue.lower():
        quality = 'B' if 'workshop' in venue.lower() or 'elliptic' in key else 'C'
    elif 'press' in venue.lower() or 'springer' in venue.lower() or 'erlbaum' in venue.lower():
        quality = 'A' if 'springer' in venue.lower() else 'B'
    
    print(f"[{i:2d}] Key: {key:25s} | Author: {author.split(' and ')[0][:20]:20s} | Year: {year} | Quality: {quality} | Venue: {venue[:30]:30s} | DOI: {doi[:30]}")

# Check for duplicate titles or DOIs
titles = [e['fields'].get('title', '').lower() for e in entries if e['fields'].get('title')]
dois = [e['fields'].get('doi', '').lower() for e in entries if e['fields'].get('doi')]

print(f"\nUnique titles: {len(set(titles))} / {len(titles)}")
print(f"Unique DOIs: {len(set(dois))} / {len(dois)}")
