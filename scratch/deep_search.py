import os, glob, json, zipfile

print("Searching across all files in workspace...")

for root, dirs, files in os.walk('.'):
    # skip .git
    if '.git' in root:
        continue
    for f in files:
        path = os.path.join(root, f)
        if f.endswith('.zip'):
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    for name in z.namelist():
                        if any(name.endswith(ext) for ext in ['.py', '.json', '.txt', '.md', '.csv']):
                            try:
                                content = z.read(name).decode('utf-8', errors='ignore')
                                for term in ['29014', '29,014', '3172', '3,172', '22957', '22,957', 'panel builder', 'audit_only', 'audit-only']:
                                    if term in content:
                                        print(f"In ZIP {path} -> {name}: found '{term}'")
                            except:
                                pass
            except:
                pass
        elif any(f.endswith(ext) for ext in ['.py', '.ipynb', '.md', '.json', '.txt', '.csv']):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    for term in ['29014', '29,014', '3172', '3,172', 'panel builder', 'audit_only', 'audit-only']:
                        if term in content:
                            print(f"In File {path}: found '{term}'")
            except:
                pass
