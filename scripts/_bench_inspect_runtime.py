"""Inspect what stages contribute to the scripts/ throughput in tradeoff.csv."""

import pandas as pd

p = "/polyis-cache/caldot2-y05/evaluation/080_throughput/measurements/query_execution_overall.csv"
df = pd.read_csv(p)
print("columns:", df.columns.tolist())
print()
print("first 3 rows:")
print(df.head(3).to_string())
print()
for c in ["stage", "op", "step"]:
    if c in df.columns:
        print(f"unique {c}:", df[c].unique())
print()
# Look at sample_rate=16 + sortcython subset
mask = (df.classifier == "ShuffleNet05") & (df.tilesize == 60) & (df.sample_rate == 16) & (df.tracker == "sortcython")
print(f"rows for our sub-combo: {mask.sum()}")
print(df[mask].head(5).to_string())
