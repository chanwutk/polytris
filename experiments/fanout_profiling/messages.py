"""
Messages between profiling stages (experiments/fanout_profiling/PLAN.md 10A).

Every message carries the video, a Branch (the parameters chosen so far, in pipeline order), and the
StageTiming records of every stage that produced it.  Pixels never travel: frames and canvases stay in shared
blocks referenced by handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from tetris.utilities import build_param_str

if TYPE_CHECKING:
    # Types only: importing operators loads the detector libraries, which need a GPU driver.
    from execution import operators as ops

# Branch keys in the order stages add them, and the build_param_str argument each maps to.
PARAM_KEYS = {
    'classifier': 'classifier', 'tile_size': 'tilesize', 'sample_rate': 'sample_rate',
    'relevance_threshold': 'relevance_threshold', 'mistrack_tolerance': 'tracking_accuracy_threshold',
    'tile_padding': 'tilepadding', 'canvas_scale': 'canvas_scale', 'tracker': 'tracker',
}


@dataclass(frozen=True)
class Branch:
    """The parameters chosen so far, as ordered (name, value) pairs; immutable and picklable."""
    params: tuple[tuple[str, Any], ...] = ()

    def with_(self, **values: Any) -> 'Branch':
        """This branch plus new parameters (each name may be set once)."""
        present = dict(self.params)
        for k in values:
            if k not in PARAM_KEYS:
                raise KeyError(f'unknown branch parameter {k!r}')
            if k in present:
                raise KeyError(f'branch parameter {k!r} is already set to {present[k]!r}')
        return Branch(self.params + tuple(values.items()))

    def get(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)

    def __getitem__(self, name: str) -> Any:
        return dict(self.params)[name]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def param_str(self, *names: str) -> str:
        """The scripts' directory name for this branch, using only `names` (the ones that stage's output depends on).

        A name whose value is None (the unpruned branch's tolerance) is left out, as build_param_str leaves it out.
        """
        d = self.as_dict()
        missing = [n for n in names if n not in d]
        if missing:
            raise KeyError(f'branch has no {missing}; it has {list(d)}')
        return build_param_str(**{PARAM_KEYS[n]: d[n] for n in ('classifier', 'tile_size', *names)})


class StageTiming(NamedTuple):
    """One stage's work for a branch: wall-clock start and end (time.time()), and the seconds charged to it.

    charged is the end minus start, except where work is shared: classify charges each sample rate only for
    its frames (PLAN 1A), and decode charges 0 (its time is recorded but its charging is undecided, PLAN 1).
    """
    stage: str
    start: float
    end: float
    charged: float


@dataclass(frozen=True)
class Decoded:
    video: str                      # path; the name is os.path.basename
    branch: Branch
    timings: tuple[StageTiming, ...]
    decoded: ops.Decoded            # every frame (sample rate 1)


@dataclass(frozen=True)
class Scored:
    """Classify -> prune and prune -> pack: scores (or pruned 0/255 grids) for one sample rate's frames."""
    video: str
    branch: Branch
    timings: tuple[StageTiming, ...]
    decoded: ops.Decoded            # this sample rate's sampled/partner over the all-frames block
    scores: np.ndarray              # uint8 [len(decoded.sampled), grid_h, grid_w]
    ilp: dict | None = None         # prune only: the solve's status, solve_ms, and kept tiles


@dataclass(frozen=True)
class Packed:
    video: str
    branch: Branch
    timings: tuple[StageTiming, ...]
    packed: ops.Packed


@dataclass(frozen=True)
class Detected:
    video: str
    branch: Branch
    timings: tuple[StageTiming, ...]
    width: int
    height: int
    frame_count: int
    detections: dict[int, list[list[float]]]


@dataclass(frozen=True)
class Tracked:
    video: str
    branch: Branch
    timings: tuple[StageTiming, ...]
    frame_tracks: dict[int, list[list[float]]]


@dataclass(frozen=True)
class Evaluated:
    """The pipeline's output: one (configuration, video) with its raw TrackEval result and stage timings."""
    video: str
    branch: Branch
    timings: tuple[StageTiming, ...]
    metrics: dict[str, Any] = field(default_factory=dict)   # TrackEval per-sequence result, per metric
