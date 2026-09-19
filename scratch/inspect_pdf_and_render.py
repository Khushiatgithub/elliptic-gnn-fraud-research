# -*- coding: utf-8 -*-
"""
Inspect results/latex/main.pdf using PyMuPDF and render all pages to PNGs for visual inspection.
"""

import os
import fitz  # PyMuPDF
import sys

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = os.path.abspath('scratch/pdf_out/main.pdf')
if not os.path.exists(pdf_path):
    pdf_path = os.path.abspath('results/latex/main.pdf')
out_dir = os.path.abspath('results/latex/pages')
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(pdf_path):
    print(f"PDF not found at {pdf_path}")
    sys.exit(1)

doc = fitz.open(pdf_path)
page_count = len(doc)
print(f"=== PDF METADATA & GENERAL INFO ===")
print(f"File: {pdf_path}")
print(f"File size: {os.path.getsize(pdf_path):,} bytes")
print(f"Total pages: {page_count}")
print(f"Metadata: {doc.metadata}")

print(f"\n=== RENDERING {page_count} PAGES TO PNG ===")
zoom = 2.0  # ~144-150 DPI for crisp visual inspection
mat = fitz.Matrix(zoom, zoom)

for page_idx in range(page_count):
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=mat)
    out_file = os.path.join(out_dir, f"page_{page_idx + 1:02d}.png")
    pix.save(out_file)
    print(f"Rendered Page {page_idx + 1:2d} -> {out_file} ({pix.width}x{pix.height})")

print("\n=== EXTRACTING FULL TEXT SUMMARY ===")
full_text = []
for page_idx in range(page_count):
    page = doc[page_idx]
    text = page.get_text()
    full_text.append(f"--- PAGE {page_idx + 1} ---\n{text}")
    print(f"Page {page_idx + 1:2d}: {len(text.split())} words | {len(text)} characters")

full_text_str = "\n".join(full_text)
text_dump_path = os.path.join(out_dir, 'pdf_extracted_text.txt')
with open(text_dump_path, 'w', encoding='utf-8') as f:
    f.write(full_text_str)
print(f"\nSaved extracted text to {text_dump_path}")
