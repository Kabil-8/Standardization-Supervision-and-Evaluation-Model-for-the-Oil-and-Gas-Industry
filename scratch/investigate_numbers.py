import pandas as pd
import numpy as np

df = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)

print("Total in annual_risk_set:", len(df), "platforms:", df['platform_id'].nunique())

# Check years
for yr_start in [2011, 2012, 2013, 2014]:
    sub = df[df['year'] >= yr_start]
    print(f"Year >= {yr_start}: rows={len(sub)}, platforms={sub['platform_id'].nunique()}")

# Check install_date / removal_date / active / structure / etc.
# What condition gives exactly 29,014 rows and 3,172 platforms?
print("\nChecking combinations for 29,014 rows and 3,172 platforms...")
for yr_min in [2011, 2012]:
    for yr_max in [2023, 2024]:
        sub = df[(df['year'] >= yr_min) & (df['year'] <= yr_max)]
        if len(sub) == 29014:
            print(f"MATCH: year {yr_min}-{yr_max} has {len(sub)} rows, {sub['platform_id'].nunique()} platforms")

# Check if filtering platforms that have at least 3 years or installation date restrictions
# Let's check how many sequences can be made for each platform
z = df.set_index(['platform_id', 'year'])
seqs_all = []
for r in df.itertuples():
    keys = [(r.platform_id, r.year - k) for k in (2, 1, 0)]
    if all(k in z.index for k in keys):
        seqs_all.append((r.platform_id, r.year, r.target))

seqs_df = pd.DataFrame(seqs_all, columns=['platform_id', 'year', 'target'])
print(f"Total possible 3-year sequences across all years (2011-2024): {len(seqs_df)}")
for yr_min in [2013, 2014]:
    for yr_max in [2023, 2024]:
        s_sub = seqs_df[(seqs_df['year'] >= yr_min) & (seqs_df['year'] <= yr_max)]
        print(f"Sequences year {yr_min}-{yr_max}: {len(s_sub)}")
