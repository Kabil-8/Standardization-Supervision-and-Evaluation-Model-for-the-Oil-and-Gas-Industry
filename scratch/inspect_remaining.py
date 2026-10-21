import zipfile

with zipfile.ZipFile('ZIP files/bsee_remaining_evidence_audit.zip', 'r') as z:
    for name in z.namelist():
        if name.endswith('.csv') or name.endswith('.json'):
            print(f"=== {name} ===")
            content = z.read(name).decode('utf-8', errors='ignore')
            print(content[:1500])
