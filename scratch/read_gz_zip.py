import zipfile, gzip, io, pandas as pd

with zipfile.ZipFile('ZIP files/bsee_priority1_cohort_provenance.zip', 'r') as z:
    for name in z.namelist():
        if 'exclusion' in name:
            print(f"Reading {name}")
            gz_bytes = z.read(name)
            with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz:
                df = pd.read_csv(gz, nrows=10)
                print(df.head(2))
                # read full to see length
            with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz:
                df_full = pd.read_csv(gz)
                print(f"Total rows in {name}: {len(df_full)}")
                if 'exclusion_reason' in df_full.columns:
                    print(df_full['exclusion_reason'].value_counts())
                elif 'reason' in df_full.columns:
                    print(df_full['reason'].value_counts())
                else:
                    print(df_full.columns)
