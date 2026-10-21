import json

with open('Final BSEE.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in Final BSEE: {len(nb['cells'])}")
with open('scratch/final_bsee_cells.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n# ===================== CELL {i} ({cell.get('cell_type')}) =====================\n")
        src = ''.join(cell.get('source', []))
        out.write(src + '\n')

print("Wrote all cells of Final BSEE.ipynb to scratch/final_bsee_cells.py")
