# -*- coding: utf-8 -*-
"""
Download and setup standalone Tectonic LaTeX engine.
"""

import os
import urllib.request
import zipfile
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

bin_dir = os.path.abspath('scratch/bin')
os.makedirs(bin_dir, exist_ok=True)
tectonic_exe = os.path.join(bin_dir, 'tectonic.exe')

if not os.path.exists(tectonic_exe):
    url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
    zip_path = os.path.join(bin_dir, 'tectonic.zip')
    print(f"Downloading Tectonic from {url}...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp, open(zip_path, 'wb') as out_f:
        out_f.write(resp.read())
    print("Download completed. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(bin_dir)
    print(f"Extracted to {bin_dir}")

print(f"Checking Tectonic executable: {tectonic_exe}")
res = subprocess.run([tectonic_exe, '--version'], capture_output=True, text=True)
print("Version:", res.stdout.strip())
