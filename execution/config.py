"""Pipeline configuration container.

A single immutable ``PipelineConfig`` instance is constructed from CLI args
and passed to every stage worker (threads via direct argument, processes via
pickled kwargs).
"""

from __future__ import annotations

from dataclasses import dataclass

from polyis.utilities import TilePadding


@dataclass(frozen=True)
class PipelineConfig:
    # ---- Algorithmic parameters (required CLI flags) ----
    dataset: str
    videoset: str
    classifier: str
    tile_size: int
    sample_rate: int
    tilepadding: TilePadding
    canvas_scale: float
    tracker: str
    tracking_accuracy_threshold: float | None
    relevance_threshold: float

    # ---- Resource parameters (CLI flags with defaults) ----
    classify_gpu: int
    detect_gpu: int
    prune_workers: int
    compress_workers: int
    max_videos_in_flight: int

    # ---- Behavior flags ----
    no_interpolate: bool
    warmup: bool

    @property
    def use_prune(self) -> bool:
        """Whether the Prune stage should be wired in."""
        return self.tracking_accuracy_threshold is not None
