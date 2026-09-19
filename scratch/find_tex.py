# -*- coding: utf-8 -*-
"""
Search for TeX distributions on Windows system and check python PDF packages.
"""

import os
import glob
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Search paths for TeX
search_patterns = [
    r"C:\Program Files\MiKTeX*\miktex\bin\x64\pdflatex.exe",
    r"C:\Program Files (x86)\MiKTeX*\miktex\bin\pdflatex.exe",
    r"C:\Users\*\AppData\Local\Programs\MiKTeX*\miktex\bin\x64\pdflatex.exe",
    r"C:\texlive\*\bin\windows\pdflatex.exe",
    r"C:\texlive\*\bin\win32\pdflatex.exe",
    r"C:\Users\*\AppData\Roaming\TinyTeX\bin\windows\pdflatex.exe",
    r"C:\Users\*\AppData\Local\TinyTeX\bin\windows\pdflatex.exe",
    r"C:\Program Files\LLVM\bin\*",
    r"C:\msys64\mingw64\bin\pdflatex.exe",
    r"C:\msys64\usr\bin\pdflatex.exe"
]

print("=== SEARCHING FOR SYSTEM LATEX INSTALLATIONS ===")
found_tex = []
for p in search_patterns:
    matches = glob.glob(p)
    if matches:
        found_tex.extend(matches)
        print(f"Found match: {matches}")

if not found_tex:
    print("No system TeX distribution found in standard installation directories.")

print("\n=== CHECKING PYTHON PDF / RENDERING LIBRARIES ===")
python_libs = ['fitz', 'pymupdf', 'pypdf', 'pdfplumber', 'reportlab', 'weasyprint', 'PIL', 'matplotlib', 'seaborn', 'pandas', 'numpy']
for lib in python_libs:
    try:
        mod = __import__(lib)
        ver = getattr(mod, '__version__', 'Installed')
        print(f"{lib:15s}: AVAILABLE (version: {ver})")
    except ImportError:
        print(f"{lib:15s}: NOT INSTALLED")
