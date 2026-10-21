import pandas as pd
import numpy as np

# Load annual risk set
df = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)

print(f"Total rows in annual_risk_set: {len(df)}")
print(f"Total unique platforms in annual_risk_set: {df['platform_id'].nunique()}")

# Let's inspect the platforms
# 3,172 platforms: which platforms are these?
# Year >= 2012 has exactly 3,172 platforms:
df_2012 = df[df['year'] >= 2012]
print(f"Year >= 2012: {len(df_2012)} rows, {df_2012['platform_id'].nunique()} platforms")

# If we take platforms that are active in 2012-2024 (3,172 platforms) and include their history back to 2011:
platforms_3172 = set(df_2012['platform_id'].unique())
df_from_3172 = df[df['platform_id'].isin(platforms_3172)]
print(f"All years for the 3,172 platforms: {len(df_from_3172)} rows, {df_from_3172['platform_id'].nunique()} platforms")

# What if we include installation date constraints or removal date constraints?
# Let's check:
print("Checking rows count:", len(df_from_3172))
# If 29,014 rows: 29014 is between 27672 (2012-2024) and 31065 (all).
# Specifically: 31065 - 29014 = 2051 rows dropped! Or 29014 - 27672 = 1342 rows from 2011!
# Look at that: 29,014 - 27,672 = 1,342!
# And earlier we noticed: 22,957 - 21,615 = 1,342!
# EXACTLY 1,342!
print(f"DIFFERENCE: 29,014 - 27,672 = {29014 - 27672}")
print(f"DIFFERENCE: 22,957 - 21,615 = {22957 - 21615}")

# In 2011: how many platforms are active out of the 3,172?
df_2011 = df[(df['year'] == 2011) & (df['platform_id'].isin(platforms_3172))]
print(f"Rows in 2011 for 3,172 platforms: {len(df_2011)}")

# In 2013: 3-year history requires 2011, 2012, 2013!
# In the notebook: sequences were generated only for year in 2014-2024 (11 feature years)!
# Year 2013 has sequences (2011, 2012, 2013) -> target in 2014!
# Let's check how many sequences exist for year 2013!
z = df_from_3172.set_index(['platform_id', 'year'])
seqs_2013 = []
for r in df_from_3172[df_from_3172['year'] == 2013].itertuples():
    keys = [(r.platform_id, r.year - k) for k in (2, 1, 0)] # 2011, 2012, 2013
    if all(k in z.index for k in keys):
        seqs_2013.append(r.platform_id)

print(f"Exact 3-year sequences for feature year 2013 (target 2014): {len(seqs_2013)}")
