with open('scratch/full_revised_text.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '7. Conclusion' in l:
        for k in range(i, min(i+5, len(lines))):
            print(f"Line {k}: {lines[k].strip()[:140]}")
