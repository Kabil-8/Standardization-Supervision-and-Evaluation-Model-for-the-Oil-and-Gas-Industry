import zipfile, os

zips = [f for f in os.listdir('ZIP files') if f.endswith('.zip')]
for zname in zips:
    zpath = os.path.join('ZIP files', zname)
    try:
        with zipfile.ZipFile(zpath, 'r') as z:
            names = z.namelist()
            matched = [n for n in names if any(k in n.lower() for k in ['builder', 'panel', 'reconstruction', 'risk_set', 'provenance', 'audit'])]
            if matched:
                print(f"=== {zname} ===")
                for m in matched[:10]:
                    print(f"  {m}")
    except Exception as e:
        pass
