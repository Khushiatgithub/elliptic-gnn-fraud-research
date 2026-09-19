# -*- coding: utf-8 -*-
"""
Check available LaTeX toolchain.
"""

import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

tools = ['pdflatex', 'xelatex', 'lualatex', 'bibtex', 'biber', 'latexmk', 'typst', 'python', 'gs', 'tesseract']
print("=== TOOLCHAIN AVAILABILITY ===")
for tool in tools:
    path = shutil.which(tool)
    version = "N/A"
    if path:
        try:
            res = subprocess.run([tool, '--version'], capture_output=True, text=True, timeout=5)
            version = res.stdout.splitlines()[0] if res.stdout else (res.stderr.splitlines()[0] if res.stderr else 'Found')
        except Exception as e:
            version = f"Error checking version: {e}"
        print(f"{tool:12s}: AVAILABLE ({path})\n              -> {version}")
    else:
        print(f"{tool:12s}: NOT FOUND")
