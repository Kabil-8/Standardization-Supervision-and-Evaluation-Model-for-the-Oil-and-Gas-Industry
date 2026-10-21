import pandas as pd
import numpy as np

df = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)
plts_3172 = set(df[df['year'] >= 2012]['platform_id'].unique())
sub = df[df['platform_id'].isin(plts_3172)].copy()

print(f"Base sub_3172: {len(sub)} rows, {sub['platform_id'].nunique()} platforms")

# Check platform row counts
counts = sub.groupby('platform_id')['year'].count()
print("Min rows per platform:", counts.min(), "Max rows:", counts.max())

# Let's check: what if min rows per platform >= 2? >= 3?
for min_r in [2, 3, 4]:
    valid_p = counts[counts >= min_r].index
    s_filtered = sub[sub['platform_id'].isin(valid_p)]
    print(f"Min rows >= {min_r}: rows={len(s_filtered)}, platforms={s_filtered['platform_id'].nunique()}")

# What if year >= 2012 (27,672 rows) plus 2011 rows that have active_months > 0 or has_production_record?
s_prod_2011 = sub[(sub['year'] >= 2012) | ((sub['year'] == 2011) & (sub['has_production_record'] == 1.0))]
print(f"Year >= 2012 or (2011 & prod==1): rows={len(s_prod_2011)}, platforms={s_prod_2011['platform_id'].nunique()}")

# What if has_production_record == 1 at least once?
has_prod_any = sub.groupby('platform_id')['has_production_record'].max()
prod_platforms = has_prod_any[has_prod_any == 1.0].index
s_prod_only = sub[sub['platform_id'].isin(prod_platforms)]
print(f"Platforms with at least 1 production record: rows={len(s_prod_only)}, platforms={s_prod_only['platform_id'].nunique()}")

# What if we look at sequences:
# How many 3-year sequences (t-2, t-1, t) exist in sub?
z = sub.set_index(['platform_id', 'year'])
seqs = []
for r in sub.itertuples():
    keys = [(r.platform_id, r.year - 2), (r.platform_id, r.year - 1), (r.platform_id, r.year)]
    if all(k in z.index for k in keys):
        seqs.append((r.platform_id, r.year))

print(f"Total possible 3-year sequences in sub: {len(seqs)}")
seqs_df = pd.DataFrame(seqs, columns=['platform_id', 'year'])
print("Sequences by year in sub:")
print(seqs_df['year'].value_counts().sort_index())
print("Total sequences 2014-2024:", (seqs_df['year'].between(2014, 2024)).sum())
print("Total sequences 2013-2024:", (seqs_df['year'].between(2013, 2024)).sum())
