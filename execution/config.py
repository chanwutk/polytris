"""
Load and validate the execution --config YAML (PLAN.md "Config").

Three sections, all required, with no unknown keys allowed:
  params     Table 1 parameters plus tile_size and canvas_scale.
  artifacts  Explicit paths to every trained artifact; each must exist.
  system     Bounded-queue size and per-stage process count / GPU.

Any problem raises ConfigError naming the offending key, before any process starts.
"""

import os
import types
import typing
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    # Typeshed-only protocol for "any dataclass"; _flat's bound is evaluated lazily, so no runtime import.
    from _typeshed import DataclassInstance

from tetris.utilities import TILEPADDING_MODES

# p022's accuracy-threshold axis of max_rate_table.npy; mistrack_tolerance must be one of these.
# experiments/tests/execution/test_config_matches_scripts.py checks this stays equal to p022's _ALL_ACCURACY_THRESHOLDS.
MISTRACK_TOLERANCES = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)

# Stage names in pipeline order; prune runs only when mistrack_tolerance is set.
STAGES = ('decode', 'classify', 'prune', 'pack', 'detect', 'track')
GPU_STAGES = ('classify', 'detect')

# How pack hands canvases to detect: 'jpeg' reproduces p030's JPEG write + p040's read (same pixels the
# paper's runs detected on); 'raw' passes the drawn canvas unchanged.
CANVAS_FORMATS = ('jpeg', 'raw')

# Required fields per detector type, mirroring tetris.models.detector.get_detector.
DETECTOR_FIELDS: dict[str, dict[str, type]] = {
    'retina': {'config_path': str},
    'yolov3': {'config_path': str, 'model_path': str, 'detector_label': str, 'width': int, 'height': int},
    'ultralytics': {'model_path': str},
    'torchvision': {'model_path': str, 'model_type': str},
}


class ConfigError(ValueError):
    """The config file is malformed or names something that does not exist."""


@dataclass(frozen=True)
class Params:
    sample_rate: int
    relevance_threshold: float
    mistrack_tolerance: float | None
    tile_padding: str
    tracker: str
    tile_size: int
    canvas_scale: float
    canvas_format: str


@dataclass(frozen=True)
class Classifier:
    name: str
    weights: str
    compilation_benchmark: str


@dataclass(frozen=True)
class Detector:
    type: str
    options: dict[str, Any]


@dataclass(frozen=True)
class Artifacts:
    classifier: Classifier
    never_relevant_bitmap: str
    max_rate_table: str | None
    detector: Detector


@dataclass(frozen=True)
class StageSystem:
    procs: int
    gpu: int | None


@dataclass(frozen=True)
class System:
    queue_maxsize: int
    stages: dict[str, StageSystem]


@dataclass(frozen=True)
class Config:
    params: Params
    artifacts: Artifacts
    system: System

    @property
    def use_prune(self) -> bool:
        """Prune runs only when a mistrack tolerance is configured."""
        return self.params.mistrack_tolerance is not None


def _check_type(value: Any, expected: Any, where: str) -> Any:
    """Return value if it matches a simple annotation (int, float, str, X | None); else raise."""
    # Optional annotations: accept None, otherwise check the non-None member.
    if isinstance(expected, types.UnionType):
        if value is None:
            return None
        (expected,) = [t for t in typing.get_args(expected) if t is not type(None)]
    # bool is an int subclass; never accept it where a number is meant.
    if isinstance(value, bool):
        raise ConfigError(f'{where}: expected {expected.__name__}, got bool')
    # Ints are valid floats (YAML writes 1 for 1.0); normalize to float.
    if expected is float and isinstance(value, int):
        return float(value)
    if not isinstance(value, expected):
        raise ConfigError(f'{where}: expected {expected.__name__}, got {type(value).__name__}')
    return value


def _keys(data: Any, required: AbstractSet[str], where: str, optional: AbstractSet[str] = frozenset()) -> dict:
    """Check data is a mapping with exactly the required keys plus any of the optional ones."""
    if not isinstance(data, dict):
        raise ConfigError(f'{where}: expected a mapping, got {type(data).__name__}')
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ConfigError(f'{where}: missing keys {sorted(missing)}')
    if unknown:
        raise ConfigError(f'{where}: unknown keys {sorted(unknown)}')
    return data


def _flat[T: DataclassInstance](cls: type[T], data: Any, where: str, optional: AbstractSet[str] = frozenset()) -> T:
    """Build a dataclass whose fields are all simple types; optional fields default to None."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    data = _keys(data, names - optional, where, optional)
    return cls(**{n: _check_type(data.get(n), hints[n], f'{where}.{n}') for n in names})


def _existing(path: Any, where: str) -> str:
    """A path string that exists on disk."""
    path = _check_type(path, str, where)
    if not os.path.exists(path):
        raise ConfigError(f'{where}: {path} does not exist')
    return path


def _params(data: Any) -> Params:
    p = _flat(Params, data, 'params')
    # Range checks for every Table 1 knob and fixed knob.
    if p.sample_rate < 1:
        raise ConfigError(f'params.sample_rate: must be >= 1, got {p.sample_rate}')
    if not 0.0 <= p.relevance_threshold <= 1.0:
        raise ConfigError(f'params.relevance_threshold: must be in [0, 1], got {p.relevance_threshold}')
    if p.mistrack_tolerance is not None and p.mistrack_tolerance not in MISTRACK_TOLERANCES:
        raise ConfigError(f'params.mistrack_tolerance: must be null or one of {MISTRACK_TOLERANCES}, '
                          f'got {p.mistrack_tolerance}')
    if p.tile_padding not in TILEPADDING_MODES:
        raise ConfigError(f'params.tile_padding: must be one of {sorted(TILEPADDING_MODES)}, got {p.tile_padding!r}')
    if p.tile_size < 1:
        raise ConfigError(f'params.tile_size: must be >= 1, got {p.tile_size}')
    if p.canvas_scale <= 0:
        raise ConfigError(f'params.canvas_scale: must be > 0, got {p.canvas_scale}')
    if p.canvas_format not in CANVAS_FORMATS:
        raise ConfigError(f'params.canvas_format: must be one of {CANVAS_FORMATS}, got {p.canvas_format!r}')
    return p


def _detector(data: Any) -> Detector:
    if not isinstance(data, dict) or not isinstance(data.get('type'), str):
        raise ConfigError('artifacts.detector: expected a mapping with a string "type"')
    if data['type'] not in DETECTOR_FIELDS:
        raise ConfigError(f'artifacts.detector.type: must be one of {sorted(DETECTOR_FIELDS)}, got {data["type"]!r}')
    spec = DETECTOR_FIELDS[data['type']]
    _keys(data, {'type', *spec}, 'artifacts.detector')
    options = {}
    for name, expected in spec.items():
        where = f'artifacts.detector.{name}'
        # Fields ending in _path must exist; the rest are plain typed values.
        options[name] = _existing(data[name], where) if name.endswith('_path') else _check_type(data[name], expected, where)
    return Detector(type=data['type'], options=options)


def _artifacts(data: Any, use_prune: bool) -> Artifacts:
    data = _keys(data, {'classifier', 'never_relevant_bitmap', 'detector'}, 'artifacts', {'max_rate_table'})
    cls = _keys(data['classifier'], {'name', 'weights', 'compilation_benchmark'}, 'artifacts.classifier')
    classifier = Classifier(
        name=_check_type(cls['name'], str, 'artifacts.classifier.name'),
        weights=_existing(cls['weights'], 'artifacts.classifier.weights'),
        compilation_benchmark=_existing(cls['compilation_benchmark'], 'artifacts.classifier.compilation_benchmark'),
    )
    # The max-rate table is required exactly when pruning is on.
    max_rate_table = data.get('max_rate_table')
    if use_prune and max_rate_table is None:
        raise ConfigError('artifacts.max_rate_table: required when params.mistrack_tolerance is set')
    if not use_prune and max_rate_table is not None:
        raise ConfigError('artifacts.max_rate_table: must be omitted when params.mistrack_tolerance is null')
    return Artifacts(
        classifier=classifier,
        never_relevant_bitmap=_existing(data['never_relevant_bitmap'], 'artifacts.never_relevant_bitmap'),
        max_rate_table=_existing(max_rate_table, 'artifacts.max_rate_table') if use_prune else None,
        detector=_detector(data['detector']),
    )


def _system(data: Any, use_prune: bool) -> System:
    data = _keys(data, {'queue_maxsize', 'stages'}, 'system')
    queue_maxsize = _check_type(data['queue_maxsize'], int, 'system.queue_maxsize')
    if queue_maxsize < 1:
        raise ConfigError(f'system.queue_maxsize: must be >= 1, got {queue_maxsize}')
    # Prune's entry is required only when it runs; it must be absent otherwise.
    required = set(STAGES) if use_prune else set(STAGES) - {'prune'}
    raw = _keys(data['stages'], required, 'system.stages')
    stages = {}
    for name in STAGES:
        if name not in raw:
            continue
        where = f'system.stages.{name}'
        s = _flat(StageSystem, raw[name], where, optional={'gpu'})
        if s.procs < 1:
            raise ConfigError(f'{where}.procs: must be >= 1, got {s.procs}')
        if (s.gpu is not None) != (name in GPU_STAGES):
            raise ConfigError(f'{where}.gpu: {"required" if name in GPU_STAGES else "not allowed"} for {name}')
        if s.gpu is not None and s.gpu < 0:
            raise ConfigError(f'{where}.gpu: must be >= 0, got {s.gpu}')
        stages[name] = s
    # shortcut: one prune worker until the ILP is redesigned (PLAN.md issue 17).
    if use_prune and stages['prune'].procs != 1:
        raise ConfigError('system.stages.prune.procs: must be 1 (multi-worker pruning awaits the ILP redesign)')
    return System(queue_maxsize=queue_maxsize, stages=stages)


def parse_config(data: Any) -> Config:
    """Validate an already-parsed YAML document."""
    data = _keys(data, {'params', 'artifacts', 'system'}, 'config')
    params = _params(data['params'])
    use_prune = params.mistrack_tolerance is not None
    return Config(params=params, artifacts=_artifacts(data['artifacts'], use_prune),
                  system=_system(data['system'], use_prune))


def load_config(path: str) -> Config:
    """Read and validate the YAML file at path."""
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f'{path}: invalid YAML: {e}') from e
    return parse_config(data)
