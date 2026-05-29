"""Side-by-side throughput comparison between scripts/ (from tradeoff data)
and execution/ (from pipeline-runtime data) using the same accounting method
that p201_compare_pareto visualizes."""

import json

import pandas as pd

# --- Scripts/ throughput from the canonical tradeoff data --------------------
tradeoff = pd.read_csv(
    "/polyis-cache/caldot2-y05/evaluation/090_tradeoff/tradeoff.csv"
)
poly = tradeoff[tradeoff.variant == "polytris"].copy()

# The closest sample_rate=16 + sortcython combo in the Pareto front.
target = "ShuffleNet05_60_16_r050_none_s100_sortcython"
scripts_row = poly[poly.variant_id == target].iloc[0]
print(f"Scripts/ combo (closest available in Pareto front): {target}")
print(f"  videoset:        {scripts_row.videoset}")
print(f"  frame_count:     {scripts_row.frame_count:,}")
print(f"  time (s):        {scripts_row.time:.3f}")
print(f"  throughput_fps:  {scripts_row.throughput_fps:.1f}")
print(f"  HOTA:            {scripts_row.HOTA_HOTA:.4f}")

# --- Execution/ throughput from the pipeline-runtime summary ------------------
with open(
    "/polyis-cache/caldot2-y05/pipeline-runtime/"
    "ShuffleNet05_60_16_040_r050_bl_s100_sortcython/runtime.jsonl"
) as f:
    last = list(f)[-1]
runtime = json.loads(last)

print(f"\nExecution/ combo: ShuffleNet05_60_16_040_r050_bl_s100_sortcython")
n_videos = runtime["num_videos"]
frame_count = n_videos * 900   # caldot2-y05 videos are 900 frames each
print(f"  videoset:        valid")
print(f"  frame_count:     {frame_count:,}")
print(f"  wall_clock_s:    {runtime['elapsed_ms'] / 1000:.3f}")
per_stage = runtime["per_stage_active_ms"]
total_active_s = sum(per_stage.values()) / 1000
print(f"  sum_per_stage_active_s: {total_active_s:.3f}")
print(f"  wall_clock_throughput_fps:    {frame_count / (runtime['elapsed_ms']/1000):.1f}")
print(f"  compute_throughput_fps "
      f"(p201 methodology — frames / sum_per_stage_active): "
      f"{frame_count / total_active_s:.1f}")

print("\nNote: scripts/ throughput uses test (54k frames); execution/ uses valid (9k).")
print("Both use sample_rate=16 + sortcython; padding differs (none vs bl).")
