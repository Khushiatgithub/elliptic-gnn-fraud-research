# -*- coding: utf-8 -*-
"""
Fast streaming download of Tectonic.
"""

import os
import requests
import zipfile
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

bin_dir = os.path.abspath('scratch/bin')
os.makedirs(bin_dir, exist_ok=True)
tectonic_exe = os.path.join(bin_dir, 'tectonic.exe')
zip_path = os.path.join(bin_dir, 'tectonic.zip')

url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
print(f"Downloading Tectonic from {url}...")

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
r = requests.get(url, headers=headers, stream=True, timeout=30)
r.raise_for_status()

total_bytes = 0
with open(zip_path, 'wb') as f:
    for chunk in r.iter_content(chunk_size=65536):
        if chunk:
            f.write(chunk)
            total_bytes += len(chunk)
            if total_bytes % (1024 * 1024 * 5) < 65536:
                print(f"Downloaded {total_bytes / (1024*1024):.1f} MB...")

print(f"Downloaded total {total_bytes / (1024*1024):.2f} MB. Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(bin_dir)

print(f"Extraction complete. Verifying executable at {tectonic_exe}...")
res = subprocess.run([tectonic_exe, '--version'], capture_output=True, text=True)
print("Tectonic Version:", res.stdout.strip())
