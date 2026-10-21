import pandas as pd
import numpy as np

df = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)

# Let's check has_production_record
prod_0 = (df['has_production_record'] == 0.0).sum()
print("has_production_record == 0 in 31,065:", prod_0)

# Check removal date or install date
print("Removal date not null:", df['removal_date'].notna().sum())
print("Install date min/max:", df['install_date'].min(), df['install_date'].max())

# Let's check how 22,957 sequences can be obtained
# 22,957 is very close to 22,982 (2013-2023) or sequences with some criteria
# What if target year must be <= 2024 (i.e. feature year <= 2023)? Or what if platform must be installed before some date?
# Let's search over possible filter conditions:
for yr_min in [2011, 2012, 2013]:
    for yr_max in [2023, 2024]:
        for req_install in [True, False]:
            for req_removal in [True, False]:
                cond = (df['year'] >= yr_min) & (df['year'] <= yr_max)
                if req_removal:
                    cond = cond & (df['removal_date'].isna() | (pd.to_datetime(df['removal_date']).dt.year >= df['year']))
                sub = df[cond]
                # count sequences
                z = sub.set_index(['platform_id', 'year'])
                seq_count = 0
                for r in sub.itertuples():
                    keys = [(r.platform_id, r.year - k) for k in (2, 1, 0)]
                    if all(k in z.index for k in keys):
                        seq_count += 1
                if len(sub) == 29014 or seq_count == 22957 or sub['platform_id'].nunique() == 3172:
                    print(f"yr={yr_min}-{yr_max}, rem={req_removal} -> rows={len(sub)}, plts={sub['platform_id'].nunique()}, seqs={seq_count}")
