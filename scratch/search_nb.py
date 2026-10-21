import json, re

for nb_name in ['Final BSEE.ipynb', 'Verified.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f'Checking {nb_name}...')
    for idx, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        for pattern in ['29014', '29,014', '3172', '3,172', '22957', '22,957', 'audit_only', 'audit-only', 'panel builder', 'builder']:
            if pattern.lower() in src.lower():
                print(f"  [{idx}] found '{pattern}': {src.strip()[:100]}...")
