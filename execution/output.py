"""Result persistence: tracking output + aggregated pipeline runtime."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

from polyis.io import cache
from polyis.utilities import build_param_str, save_tracking_results

from execution.config import PipelineConfig
from execution.messages import StageTiming, TrackingResult


def _output_param_str(config: PipelineConfig) -> str:
    """Build the param_str used by cache.exec to locate output paths.

    Mirrors the convention used by scripts/p060 so downstream evaluation
    scripts read both paths transparently.
    """
    return build_param_str(
        classifier=config.classifier,
        tilesize=config.tile_size,
        sample_rate=config.sample_rate,
        tilepadding=config.tilepadding,
        canvas_scale=config.canvas_scale,
        tracker=config.tracker,
        tracking_accuracy_threshold=config.tracking_accuracy_threshold,
        relevance_threshold=config.relevance_threshold,
    )


def save_tracking_result(result: TrackingResult, config: PipelineConfig) -> None:
    """Write one video's tracking.jsonl to the same path as scripts/p060."""
    param_str = _output_param_str(config)
    output_path = cache.exec(
        config.dataset, 'ucomp-tracks', result.video,
        param_str, 'tracking.jsonl',
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_tracking_results(result.frame_tracks, output_path)


def save_pipeline_runtime(
    *,
    config: PipelineConfig,
    elapsed_ms: float,
    num_videos: int,
    timings: Iterable[StageTiming],
    per_video_complete_ts: dict[str, float],
) -> str:
    """Write the aggregated pipeline runtime summary; return the file path."""
    # Sum per-stage worker-active time across all videos.
    per_stage_total_ms: dict[str, float] = defaultdict(float)
    per_stage_video_count: dict[str, int] = defaultdict(int)
    for t in timings:
        per_stage_total_ms[t.stage] += t.duration_ms
        per_stage_video_count[t.stage] += 1

    param_str = _output_param_str(config)
    summary = {
        'config': {**{k: getattr(config, k) for k in config.__dataclass_fields__.keys()}},
        'param_str': param_str,
        'elapsed_ms': elapsed_ms,
        'num_videos': num_videos,
        'per_stage_active_ms': dict(per_stage_total_ms),
        'per_stage_video_count': dict(per_stage_video_count),
        'per_video_complete_ts': per_video_complete_ts,
    }

    summary_dir = cache.root(config.dataset, 'pipeline-runtime', param_str)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(str(summary_dir), 'runtime.jsonl')
    with open(summary_path, 'a') as f:
        f.write(json.dumps(summary, default=str) + '\n')
    return summary_path
