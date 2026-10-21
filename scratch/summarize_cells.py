with open('scratch/verified_code_cells.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
cells = text.split('# ===================== CELL ')
for idx, c in enumerate(cells[1:], start=0):
    lines = c.split('\n')
    header = lines[0]
    first_few_code = [l.strip() for l in lines[1:10] if l.strip() and not l.strip().startswith('#')]
    # check functions or classes or main actions
    defs = [l.strip() for l in lines if l.strip().startswith('def ') or l.strip().startswith('class ') or 'pd.read' in l or 'main(' in l or 'argparse' in l]
    print(f"Cell {idx} ({header[:20]}): {len(lines)} lines | defs: {defs[:4]}")
