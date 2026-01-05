#!/usr/bin/env python3
"""Execute notebook code cells locally, excluding shell commands."""
import json
import sys
from pathlib import Path
import subprocess

nb_path = Path('新闻推荐系统-多路召回.ipynb')
with open(nb_path) as f:
    nb = json.load(f)

# Extract code cells, filtering out shell commands
exec_code = []
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        # Skip cells that are just shell commands (starting with !)
        if source.strip().startswith('!'):
            print(f"[Cell {idx}] Skipping shell command: {source[:50]}...")
            # Install packages via subprocess instead
            if 'pip install' in source:
                cmd_str = source.replace('!', '').strip()  # Remove ! prefix
                subprocess.run(cmd_str, shell=True)
            continue
        
        if source.strip():
            exec_code.append(f"# Cell {idx}\n{source}")

combined_code = '\n\n'.join(exec_code)

# Write and execute
temp_file = Path('_execute_notebook.py')
temp_file.write_text(combined_code)

print(f"Executing {len(exec_code)} code cells...")
print("=" * 60)

result = subprocess.run([sys.executable, str(temp_file)], timeout=7200)
sys.exit(result.returncode)
