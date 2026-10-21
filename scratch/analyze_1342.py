import pandas as pd
import numpy as np

df = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)
plts_3172 = set(df[df['year'] >= 2012]['platform_id'].unique())
sub = df[df['platform_id'].isin(plts_3172)].copy()

# Let's check 2013 rows and sequences
z = sub.set_index(['platform_id', 'year'])
seqs_2013 = []
for r in sub[sub['year'] == 2013].itertuples():
    keys = [(r.platform_id, r.year - 2), (r.platform_id, r.year - 1), (r.platform_id, r.year)]
    if all(k in z.index for k in keys):
        seqs_2013.append(r)

df_2013_seqs = pd.DataFrame(seqs_2013)
print(f"Total 2013 sequences: {len(df_2013_seqs)}")

for col in ['has_production_record', 'active_months', 'inc_count', 'operator_id']:
    if col in df_2013_seqs.columns:
        print(f"Column {col} value counts in 2013 seqs:")
        print(df_2013_seqs[col].value_counts().head(5))

# Check has_production_record in 2013
prod_1_2013 = (df_2013_seqs['has_production_record'] == 1.0).sum()
print("2013 seqs with has_production_record == 1:", prod_1_2013)
prod_0_2013 = (df_2013_seqs['has_production_record'] == 0.0).sum()
print("2013 seqs with has_production_record == 0:", prod_0_2013)

# Check removal date in 2013
rem_notna_2013 = df_2013_seqs['removal_date'].notna().sum()
print("2013 seqs with removal_date notna:", rem_notna_2013)
rem_na_2013 = df_2013_seqs['removal_date'].isna().sum()
print("2013 seqs with removal_date isna:", rem_na_2013)

# Check installation date before some year:
df_2013_seqs['install_year'] = pd.to_datetime(df_2013_seqs['install_date']).dt.year
for y in range(1970, 2010, 5):
    c = (df_2013_seqs['install_year'] >= y).sum()
    if c == 1342 or abs(c - 1342) < 50:
        print(f"install_year >= {y}: {c}")
