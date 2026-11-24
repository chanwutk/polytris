import os
import json
import numpy as np
import pandas as pd

def reconstruct_tracks_json(results_file, tracks_file, output_dir, video_name):
    """
    Reconstruct predicted tracks from resolved_tuple and motion patterns,
    and save in JSON format compatible with HOTA/OTIF:
    one entry per frame, each frame is a list of detection dicts.
    Handles motion patterns shorter or longer than frame interval.
    """
    # Load JSON files
    with open(results_file, "r") as f:
        resolved_tuple = json.load(f)

    with open(tracks_file, "r") as f:
        tracks = json.load(f)

    # Determine total number of frames
    max_frame = 0
    for value in resolved_tuple.values():
        start_frame, end_frame = value[0]
        max_frame = max(max_frame, end_frame)

    # Initialize list for each frame
    frame_detections = [[] for _ in range(max_frame)]

    for obj_id, value in resolved_tuple.items():
        start_frame, end_frame = value[0]  # frame interval [start, end]
        traj_idx = value[1]                # index into tracks.json

        motion_pattern = tracks[traj_idx]  # list of [x1,y1,x2,y2] per frame
        motion_pattern_np = np.array(motion_pattern)
        required_length = end_frame - start_frame

        # Handle case where motion pattern length differs from frame interval
        if len(motion_pattern_np) == required_length:
            selected_pattern = motion_pattern_np
        elif len(motion_pattern_np) < required_length:
            # Linear interpolation to expand pattern
            if len(motion_pattern_np) == 1:
                selected_pattern = np.repeat(motion_pattern_np, required_length, axis=0)
            else:
                interp_indices = np.linspace(0, len(motion_pattern_np)-1, required_length)
                selected_pattern = []
                for t in interp_indices:
                    lower = int(np.floor(t))
                    upper = int(np.ceil(t))
                    alpha = t - lower
                    if lower == upper:
                        selected_pattern.append(motion_pattern_np[lower])
                    else:
                        selected_pattern.append((1-alpha)*motion_pattern_np[lower] + alpha*motion_pattern_np[upper])
                selected_pattern = np.array(selected_pattern)
        else:
            # Truncate pattern if longer than interval
            selected_pattern = motion_pattern_np[:required_length]

        # Assign to frames
        for idx, bbox in enumerate(selected_pattern):
            frame_idx = start_frame + idx
            # Check if track_id already exists in this frame
            if any(d["track_id"] == int(obj_id) for d in frame_detections[frame_idx]):
                continue
            frame_detections[frame_idx].append({
                "left": float(bbox[0]),
                "top": float(bbox[1]),
                "right": float(bbox[2]),
                "bottom": float(bbox[3]),
                "class": "car",
                "score": 1.0,
                "track_id": int(obj_id)
            })

    # Save as a single JSON file per video
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}.json")
    with open(output_file, "w") as f:
        json.dump(frame_detections, f, indent=2)

    print(f"Video {video_name} JSON saved to {output_file}")

## save tracks
results_dir = "/data/alenachao/data/leap-dataset/outputs/parsed_results/amsterdam-y5"
output_dir = "/sota-results/amsterdam-y5/leap/"

video_files = sorted([f.split("_results.json")[0] for f in os.listdir(results_dir) if f.endswith("_results.json")])

for video in video_files:
    results_path = os.path.join(results_dir, f"{video}_results.json")
    tracks_path  = os.path.join(results_dir, f"{video}_tracks.json")
    
    reconstruct_tracks_json(results_path, tracks_path, output_dir, video)

## also save runtime data
runtime_path =  os.path.join(results_dir, f"speed.tsv")
runtime_output_dir = "/sota-results/amsterdam-y5/leap_amsterdam-y5.csv"

df = pd.read_csv(runtime_path, sep="\t")

# 2. Sort descending by 'video_name'
df = df.sort_values(by="video_name")

# 3. Add a 'total' row (sum numeric columns)
total_row = df.select_dtypes("number").sum()
total_row["video_name"] = "total"
df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

df.to_csv(runtime_output_dir, index=False)

print(f"Processed TSV saved to {runtime_output_dir}")