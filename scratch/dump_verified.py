import json

with open('Verified.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
with open('scratch/verified_code_cells.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n# ===================== CELL {i} ({cell.get('cell_type')}) =====================\n")
        src = ''.join(cell.get('source', []))
        out.write(src + '\n')

print("Wrote all code cells of Verified.ipynb to scratch/verified_code_cells.py")
