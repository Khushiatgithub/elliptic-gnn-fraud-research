# -*- coding: utf-8 -*-
"""
Find LaTeX syntax issues (backticks, unescaped underscores outside math mode, etc.) in main.tex.
"""

import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('results/latex/main.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in main.tex: {len(lines)}")

# Look for backticks `...`
backtick_issues = []
for i, line in enumerate(lines, 1):
    if '`' in line:
        backtick_issues.append((i, line.strip()))

print(f"\nLines with backticks (`): {len(backtick_issues)}")
for line_num, content in backtick_issues[:20]:
    print(f"Line {line_num:4d}: {content}")

# Check for unescaped underscores outside math mode
# A rough check: line contains _ but not \_ and not in $...$
underscore_issues = []
for i, line in enumerate(lines, 1):
    # remove math mode $...$ and \[...\] and \begin{equation}...\end{equation}
    line_no_comment = line.split('%')[0]
    line_no_escaped = line_no_comment.replace(r'\_', '')
    # remove inline math
    line_no_math = re.sub(r'\$[^\$]*\$', '', line_no_escaped)
    if '_' in line_no_math:
        underscore_issues.append((i, line_no_comment.strip()))

print(f"\nLines with unescaped underscores outside inline math: {len(underscore_issues)}")
for line_num, content in underscore_issues[:20]:
    print(f"Line {line_num:4d}: {content}")
