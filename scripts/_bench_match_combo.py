"""Find scripts/ runtime for the EXACT combo we benchmarked (sample_rate=16,
tilepadding=bl, threshold=0.4, relevance=0.5, sortcython) in the query CSV."""

import pandas as pd

p = "/polyis-cache/caldot2-y05/evaluation/080_throughput/measurements/query_execution_overall.csv"
df = pd.read_csv(p)

# Filter to our combo's variant_id.
combo = "ShuffleNet05_60_16_040_r050_bl_s100_sortcython"
match = df[df.variant_id == combo]
print(f"rows for combo {combo}: {len(match)}")
if len(match) > 0:
    print("stages present:", match.stage.unique())
    print("videoset:", match.videoset.unique())
    # Per-video total time across pipeline stages.
    pipeline_stages = ['020_exec_classify', '022_exec_prune_polyominoes',
                       '030_exec_compress', '040_exec_detect',
                       '050_exec_uncompress', '060_exec_track']
    pipe = match[match.stage.isin(pipeline_stages)]
    per_video = pipe.groupby(['video', 'videoset'])['time'].sum().reset_index()
    print(f"\nPer-video pipeline compute time (sum across stages):")
    print(per_video.to_string())
    print()
    total_time = per_video['time'].sum()
    n_vid = len(per_video)
    print(f"Total pipeline compute time across {n_vid} videos: {total_time:.2f}s")
    # caldot2-y05 videos are 900 frames each.
    total_frames = n_vid * 900
    print(f"Total frames: {total_frames}")
    print(f"Throughput (compute basis): {total_frames / total_time:.1f} fps")
    print()
    # Per-stage breakdown
    print("Per-stage compute time (summed across all videos):")
    for stage in pipeline_stages:
        s_time = match[match.stage == stage]['time'].sum()
        print(f"  {stage:<30}  {s_time:>8.3f}s")
else:
    print("Not in query data either")
