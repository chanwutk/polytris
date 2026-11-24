"""
Load and validate a profiling sweep config (profiling/PLAN.md 9A).

One YAML per dataset, with no unknown keys allowed:

  dataset, videoset, output_root     the dataset id, the split to sweep (normally 'valid'), and where every
                                     saved result goes (cache-shaped; must not contain the reference tracks)
  params        a list of values per swept parameter, plus the fixed classifier, tile size, canvas format
  artifacts     explicit paths: classifier, never-relevant bitmap, one max-rate table per tracker, detector
  evaluation    where each video's reference tracks are: <reference_dir>/<video>/<reference_file>
  system        queue sizes, videos in flight, and per-stage procs / gpus / savers

The strict field checks and the detector spec are execution2.config's, so both configs share one copy.
"""

import os
from dataclasses import dataclass
from typing import Any

import yaml

from execution.config import (CANVAS_FORMATS, MISTRACK_TOLERANCES, ConfigError, Detector, _check_type, _detector,
                               _existing, _keys)
from polyis.utilities import TILEPADDING_MODES

STAGES = ('decode', 'classify', 'prune', 'pack', 'detect', 'track', 'evaluate')
GPU_STAGES = ('classify', 'detect')


@dataclass(frozen=True)
class Params:
    classifier: str
    tile_size: int
    canvas_format: str
    sample_rates: tuple[int, ...]
    relevance_thresholds: tuple[float, ...]
    mistrack_tolerances: tuple[float | None, ...]   # None = the branch without pruning
    tile_paddings: tuple[str, ...]
    canvas_scales: tuple[float, ...]
    trackers: tuple[str, ...]
    prune_time_limit_s: float = 0.1       # p022's solve limit; raise only for equivalence tests
    reduce_search_space: bool = False     # run one tolerance per identical-ILP group (profiling/reduction.py);
                                          # off by default so a sweep runs every tolerance, as the scripts do


@dataclass(frozen=True)
class Artifacts:
    classifier_weights: str
    compilation_benchmark: str
    never_relevant_bitmap: str
    max_rate_tables: dict[str, str]       # tracker -> path; empty when no tolerance is swept
    detector: Detector


@dataclass(frozen=True)
class Evaluation:
    reference_dir: str
    reference_file: str


@dataclass(frozen=True)
class StageSystem:
    procs: int
    gpus: tuple[int, ...]                 # empty for CPU stages
    savers: int


@dataclass(frozen=True)
class System:
    queue_maxsize: int
    save_queue_maxsize: int
    max_videos_in_flight: int
    stages: dict[str, StageSystem]


@dataclass(frozen=True)
class SweepConfig:
    dataset: str
    videoset: str
    output_root: str
    params: Params
    artifacts: Artifacts
    evaluation: Evaluation
    system: System

    @property
    def uses_prune(self) -> bool:
        """Whether any branch prunes (some tolerance is set)."""
        return any(t is not None for t in self.params.mistrack_tolerances)


def _values(data: dict, key: str, kind: Any, where: str, allow_none: bool = False) -> tuple:
    """A non-empty list of distinct values of `kind` (None allowed when allow_none)."""
    raw = data[key]
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f'{where}.{key}: expected a non-empty list')
    values = tuple(None if (allow_none and v is None) else _check_type(v, kind, f'{where}.{key}[{i}]')
                   for i, v in enumerate(raw))
    if len(set(values)) != len(values):
        raise ConfigError(f'{where}.{key}: values must be distinct, got {list(values)}')
    return values


def _params(data: Any) -> Params:
    data = _keys(data, {'classifier', 'tile_size', 'canvas_format', 'sample_rates', 'relevance_thresholds',
                        'mistrack_tolerances', 'tile_paddings', 'canvas_scales', 'trackers'}, 'params',
                 {'prune_time_limit_s', 'reduce_search_space'})
    p = Params(
        classifier=_check_type(data['classifier'], str, 'params.classifier'),
        tile_size=_check_type(data['tile_size'], int, 'params.tile_size'),
        canvas_format=_check_type(data['canvas_format'], str, 'params.canvas_format'),
        sample_rates=_values(data, 'sample_rates', int, 'params'),
        relevance_thresholds=_values(data, 'relevance_thresholds', float, 'params'),
        mistrack_tolerances=_values(data, 'mistrack_tolerances', float, 'params', allow_none=True),
        tile_paddings=_values(data, 'tile_paddings', str, 'params'),
        canvas_scales=_values(data, 'canvas_scales', float, 'params'),
        trackers=_values(data, 'trackers', str, 'params'),
        prune_time_limit_s=_check_type(data.get('prune_time_limit_s', 0.1), float, 'params.prune_time_limit_s'),
        reduce_search_space=data.get('reduce_search_space', False),
    )
    # _check_type rejects bools (they are ints), so this flag is checked on its own.
    if not isinstance(p.reduce_search_space, bool):
        raise ConfigError(f'params.reduce_search_space: expected bool, got {type(p.reduce_search_space).__name__}')
    if p.tile_size < 1:
        raise ConfigError(f'params.tile_size: must be >= 1, got {p.tile_size}')
    if p.prune_time_limit_s <= 0:
        raise ConfigError(f'params.prune_time_limit_s: must be > 0, got {p.prune_time_limit_s}')
    if p.canvas_format not in CANVAS_FORMATS:
        raise ConfigError(f'params.canvas_format: must be one of {CANVAS_FORMATS}, got {p.canvas_format!r}')
    for s in p.sample_rates:
        if s < 1:
            raise ConfigError(f'params.sample_rates: must be >= 1, got {s}')
    for t in p.relevance_thresholds:
        if not 0.0 <= t <= 1.0:
            raise ConfigError(f'params.relevance_thresholds: must be in [0, 1], got {t}')
    for t in p.mistrack_tolerances:
        if t is not None and t not in MISTRACK_TOLERANCES:
            raise ConfigError(f'params.mistrack_tolerances: each must be null or one of {MISTRACK_TOLERANCES}, got {t}')
    for padding in p.tile_paddings:
        if padding not in TILEPADDING_MODES:
            raise ConfigError(f'params.tile_paddings: each must be one of {sorted(TILEPADDING_MODES)}, got {padding!r}')
    for c in p.canvas_scales:
        if c <= 0:
            raise ConfigError(f'params.canvas_scales: must be > 0, got {c}')
    return p


def _artifacts(data: Any, params: Params) -> Artifacts:
    data = _keys(data, {'classifier_weights', 'compilation_benchmark', 'never_relevant_bitmap', 'detector'},
                 'artifacts', {'max_rate_tables'})
    uses_prune = any(t is not None for t in params.mistrack_tolerances)
    tables = data.get('max_rate_tables') or {}
    if uses_prune:
        # Every tracker can be paired with every tolerance, so each needs its own table.
        tables = _keys(tables, set(params.trackers), 'artifacts.max_rate_tables')
        tables = {t: _existing(path, f'artifacts.max_rate_tables.{t}') for t, path in tables.items()}
    elif tables:
        raise ConfigError('artifacts.max_rate_tables: must be omitted when no mistrack tolerance is swept')
    return Artifacts(
        classifier_weights=_existing(data['classifier_weights'], 'artifacts.classifier_weights'),
        compilation_benchmark=_existing(data['compilation_benchmark'], 'artifacts.compilation_benchmark'),
        never_relevant_bitmap=_existing(data['never_relevant_bitmap'], 'artifacts.never_relevant_bitmap'),
        max_rate_tables=tables,
        detector=_detector(data['detector']),
    )


def _evaluation(data: Any) -> Evaluation:
    data = _keys(data, {'reference_dir', 'reference_file'}, 'evaluation')
    return Evaluation(reference_dir=_existing(data['reference_dir'], 'evaluation.reference_dir'),
                      reference_file=_check_type(data['reference_file'], str, 'evaluation.reference_file'))


def _system(data: Any, uses_prune: bool) -> System:
    data = _keys(data, {'queue_maxsize', 'save_queue_maxsize', 'max_videos_in_flight', 'stages'}, 'system')
    ints = {k: _check_type(data[k], int, f'system.{k}') for k in ('queue_maxsize', 'save_queue_maxsize',
                                                                  'max_videos_in_flight')}
    for k, v in ints.items():
        if v < 1:
            raise ConfigError(f'system.{k}: must be >= 1, got {v}')
    required = set(STAGES) if uses_prune else set(STAGES) - {'prune'}
    raw = _keys(data['stages'], required, 'system.stages')
    stages = {}
    for name in STAGES:
        if name not in raw:
            continue
        where = f'system.stages.{name}'
        entry = _keys(raw[name], {'procs'}, where, {'gpus', 'savers'})
        procs = _check_type(entry['procs'], int, f'{where}.procs')
        savers = _check_type(entry.get('savers', 1), int, f'{where}.savers')
        gpus_raw = entry.get('gpus')
        if (gpus_raw is not None) != (name in GPU_STAGES):
            raise ConfigError(f'{where}.gpus: {"required" if name in GPU_STAGES else "not allowed"} for {name}')
        gpus = ()
        if gpus_raw is not None:
            if not isinstance(gpus_raw, list) or not gpus_raw:
                raise ConfigError(f'{where}.gpus: expected a non-empty list of GPU indices')
            gpus = tuple(_check_type(g, int, f'{where}.gpus[{i}]') for i, g in enumerate(gpus_raw))
            if any(g < 0 for g in gpus):
                raise ConfigError(f'{where}.gpus: indices must be >= 0, got {list(gpus)}')
        if procs < 1 or savers < 1:
            raise ConfigError(f'{where}: procs and savers must be >= 1, got procs={procs} savers={savers}')
        stages[name] = StageSystem(procs, gpus, savers)
    # One prune worker: the Gurobi web license allows 2 concurrent sessions, shared with other users (PLAN 16B).
    if uses_prune and stages['prune'].procs != 1:
        raise ConfigError('system.stages.prune.procs: must be 1 (Gurobi license sessions)')
    return System(stages=stages, **ints)


def parse_sweep_config(data: Any) -> SweepConfig:
    """Validate an already-parsed sweep YAML document."""
    data = _keys(data, {'dataset', 'videoset', 'output_root', 'params', 'artifacts', 'evaluation', 'system'}, 'config')
    params = _params(data['params'])
    evaluation = _evaluation(data['evaluation'])
    output_root = os.path.abspath(_check_type(data['output_root'], str, 'output_root'))
    # Saved results must never land where the reference tracks are read from.
    reference = os.path.abspath(evaluation.reference_dir)
    if os.path.commonpath([output_root, reference]) in (output_root, reference):
        raise ConfigError(f'output_root: {output_root} and the reference tracks ({reference}) contain one another; '
                          f'use a separate directory so the sweep cannot overwrite existing results')
    return SweepConfig(
        dataset=_check_type(data['dataset'], str, 'dataset'),
        videoset=_check_type(data['videoset'], str, 'videoset'),
        output_root=output_root,
        params=params,
        artifacts=_artifacts(data['artifacts'], params),
        evaluation=evaluation,
        system=_system(data['system'], any(t is not None for t in params.mistrack_tolerances)),
    )


def load_sweep_config(path: str) -> SweepConfig:
    """Read and validate the sweep YAML file at path."""
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f'{path}: invalid YAML: {e}') from e
    return parse_sweep_config(data)
