import zipfile

with zipfile.ZipFile('ZIP files/bsee_cohort_provenance_audit.zip', 'r') as z:
    for name in z.namelist():
        if name.endswith('.csv') or name.endswith('.json') or name.endswith('.txt'):
            print(f"=== {name} ===")
            content = z.read(name).decode('utf-8', errors='ignore')
            print(content[:1500])
