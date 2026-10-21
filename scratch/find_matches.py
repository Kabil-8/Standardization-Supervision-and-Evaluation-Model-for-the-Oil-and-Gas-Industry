import re

with open('scratch/verified_code_cells.py', 'r', encoding='utf-8') as f:
    text = f.read()

# find any mentions of 29014 or 22957 or 3172 in verified_code_cells.py
matches = re.findall(r'.{0,50}(?:29014|22957|3172|29,014|22,957|3,172).{0,50}', text)
print(f"Total regex matches: {len(matches)}")
for m in matches[:20]:
    print(repr(m.strip()))
