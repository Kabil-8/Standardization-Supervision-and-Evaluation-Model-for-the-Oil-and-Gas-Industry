import pandas as pd
import numpy as np

# Let's inspect bsee_priority1_cohort_provenance exclusion ledgers if any
import zipfile

with zipfile.ZipFile('ZIP files/bsee_priority1_cohort_provenance.zip', 'r') as z:
    for name in z.namelist():
        if 'exclusion' in name:
            print(f"Found {name}")
            df_ex = pd.read_csv(z.open(name), nrows=10)
            print(df_ex.head())
