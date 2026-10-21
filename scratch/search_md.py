import os, glob

with open('scratch/deep_search.py', 'r') as f:
    pass

# search in README.md or any other md files
for f in os.listdir('.'):
    if f.endswith('.md') or f.endswith('.txt'):
        with open(f, 'r', encoding='utf-8', errors='ignore') as infile:
            text = infile.read()
            for term in ['29014', '29,014', '3172', '3,172', '22957', '22,957', 'panel builder']:
                if term in text:
                    print(f"In {f}: found '{term}'")
