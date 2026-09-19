# -*- coding: utf-8 -*-
"""
Compile results/latex/main.tex using Tectonic.
"""

import os
import subprocess
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

tectonic_exe = os.path.abspath('scratch/bin/tectonic.exe')
latex_dir = os.path.abspath('results/latex')
main_tex = os.path.join(latex_dir, 'main.tex')

print(f"Compiling {main_tex} with Tectonic...")
print(f"Working directory: {latex_dir}")

out_temp = os.path.abspath('scratch/pdf_out')
os.makedirs(out_temp, exist_ok=True)

cmd = [tectonic_exe, '-X', 'compile', main_tex, '--outdir', out_temp, '--print']
print(f"Running command: {' '.join(cmd)}")

res = subprocess.run(cmd, cwd=latex_dir, capture_output=True, text=True)

print("\n=== COMPILATION STDOUT ===")
print(res.stdout)

print("\n=== COMPILATION STDERR ===")
print(res.stderr)

print(f"\nExit code: {res.returncode}")

temp_pdf = os.path.join(out_temp, 'main.pdf')
final_pdf = os.path.join(latex_dir, 'main.pdf')
if os.path.exists(temp_pdf):
    try:
        shutil.copy2(temp_pdf, final_pdf)
        print(f"\nSUCCESS: Generated and copied {final_pdf} (Size: {os.path.getsize(final_pdf)} bytes)")
    except Exception as e:
        print(f"\nCopied error: {e}. Output available at {temp_pdf}")
else:
    print(f"\nFAILED: {temp_pdf} was not created.")
