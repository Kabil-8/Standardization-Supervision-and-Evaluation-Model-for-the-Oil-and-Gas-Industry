import pandas as pd
import numpy as np

df = pd.read_csv('tecr_net_corrected_extracted/tecr_net_corrected/annual_risk_set.csv', low_memory=False)

# Let's inspect install_date and removal_date
df['install_year'] = pd.to_datetime(df['install_date'], errors='coerce').dt.year
df['removal_year'] = pd.to_datetime(df['removal_date'], errors='coerce').dt.year

print("Platforms with install_date:", df['install_date'].notna().sum())
print("Platforms with removal_date:", df['removal_date'].notna().sum())

# How many platforms have install_year <= year and (removal_year is null or removal_year >= year)?
valid_existence = (df['install_year'] <= df['year']) & (df['removal_year'].isna() | (df['removal_year'] >= df['year']))
print("Valid existence rows:", valid_existence.sum())

# Let's inspect platforms active in 2012-2024:
# Exactly 3,172 platforms!
plts_3172 = df[df['year'] >= 2012]['platform_id'].unique()
print("Number of platforms active in 2012-2024:", len(plts_3172))

# How many rows do these 3,172 platforms have?
sub_3172 = df[df['platform_id'].isin(plts_3172)]
print("Total rows for these 3,172 platforms across 2011-2024:", len(sub_3172))

# If we drop platforms that were removed before 2012:
# What condition on sub_3172 gives exactly 29,014 rows?
# Total rows = 30,773. 30,773 - 29,014 = 1,759 rows dropped.
# What are the 1,759 rows?
# Could it be rows where removal_year < year? Or year == 2011 for platforms that didn't have 2012?
# Let's test:
for cond_name, mask in [
    ("removal_year >= year or na", (sub_3172['removal_year'].isna() | (sub_3172['removal_year'] >= sub_3172['year']))),
    ("year >= 2012", sub_3172['year'] >= 2012),
    ("year <= 2023", sub_3172['year'] <= 2023),
]:
    print(f"Filter {cond_name}: {mask.sum()} rows")
