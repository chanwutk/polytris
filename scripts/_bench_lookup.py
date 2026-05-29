"""Look up scripts/ Polytris throughput from the existing tradeoff data."""

import pandas as pd

df = pd.read_csv("/polyis-cache/caldot2-y05/evaluation/090_tradeoff/tradeoff.csv")
poly = df[df.variant == "polytris"].sort_values("HOTA_HOTA").reset_index(drop=True)

print("=== Polytris Pareto front (scripts/p020-p060), caldot2-y05/test, 54000 frames ===")
print(f"{'HOTA':>6}  {'throughput_fps':>14}  {'time(s)':>8}  variant_id")
for _, row in poly.iterrows():
    print(f"{row.HOTA_HOTA:>6.4f}  {row.throughput_fps:>14.1f}  {row.time:>8.1f}  {row.variant_id}")

print()
naive = df[df.variant == "naive"]
n = naive.iloc[0]
print(f"Naive Reference:  HOTA={n.HOTA_HOTA:.4f}  throughput={n.throughput_fps:.1f} fps  time={n.time:.1f}s")

# Closest match to our benchmarked execution/ HOTA range (~0.21).
print()
print("=== Around HOTA=0.21 (matches our execution/ bench result) ===")
nearest = poly.iloc[(poly.HOTA_HOTA - 0.214).abs().argsort().head(3)]
for _, row in nearest.iterrows():
    print(f"HOTA={row.HOTA_HOTA:.4f}  throughput={row.throughput_fps:.1f} fps  variant_id={row.variant_id}")
