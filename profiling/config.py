"""
Load a profiling sweep config and expand it into one execution config per configuration (profiling/README.md).

One YAML per dataset, with no unknown keys allowed:

  dataset, videoset   the dataset id and the split to profile (normally 'valid')
  output_root         results go to <output_root>/<dataset>/profiling/ (and, with --save, the stages' cache-shaped
                      outputs to <output_root>/<dataset>/execution/); must not contain the reference tracks
  params              a non-empty list of distinct values per swept parameter (mistrack_tolerances may hold null:
                      no pruning), the fixed classifier, tile_size and canvas_format, and dedup_pruning (default true)
  artifacts           classifier_weights, compilation_benchmark, never_relevant_bitmap, max_rate_tables (tracker ->
                      path; required exactly when a tolerance is swept), detector (as in an execution config)
  evaluation          each video's reference tracks: <reference_dir>/<video file>/<reference_file>
  system              execution's system section, used for every configuration (prune's entry is dropped for
                      configurations without pruning, as execution requires)
  configurations      optional: ids of the grid's configurations to run (default: all), e.g. a validation sweep's
                      Pareto configurations on the test split (experiments/tools/profiling/make_test_configs.py)

Every configuration's execution config is validated by execution.config.parse_config when the sweep loads, so the
field rules exist once and a bad sweep fails before anything runs.
"""

import dataclasses
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from typing import Any

import yaml

from execution.config import ConfigError, _check_type, _existing, _keys, parse_config
from tetris.utilities import build_param_str

# Swept parameters in the order configurations are enumerated, with the execution params key each one sets.
SWEPT = (('sample_rates', 'sample_rate'), ('relevance_thresholds', 'relevance_threshold'),
         ('mistrack_tolerances', 'mistrack_tolerance'), ('tile_paddings', 'tile_padding'),
         ('canvas_scales', 'canvas_scale'), ('trackers', 'tracker'))


@dataclass(frozen=True)
class Params:
    classifier: str
    tile_size: int
    canvas_format: str
    sample_rates: tuple[int, ...]
    relevance_thresholds: tuple[float, ...]
    mistrack_tolerances: tuple[float | None, ...]   # None = no pruning
    tile_paddings: tuple[str, ...]
    canvas_scales: tuple[float, ...]
    trackers: tuple[str, ...]
    dedup_pruning: bool = True                       # run one tolerance per identical-ILP group (profiling/reduction.py)


@dataclass(frozen=True)
class Artifacts:
    classifier_weights: str
    compilation_benchmark: str
    never_relevant_bitmap: str
    max_rate_tables: dict[str, str]                 # tracker -> path; empty when no tolerance is swept
    detector: dict[str, Any]                        # execution's detector spec, validated through parse_config


@dataclass(frozen=True)
class Evaluation:
    reference_dir: str
    reference_file: str


@dataclass(frozen=True)
class SweepConfig:
    dataset: str
    videoset: str
    output_root: str
    params: Params
    artifacts: Artifacts
    evaluation: Evaluation
    system: dict[str, Any]                          # execution's system section, as written
    raw: dict[str, Any]                             # the parsed YAML document, for the sweep's hash
    selected: frozenset[str] | None = None          # ids of the grid's configurations to run; None = all

    @property
    def profiling_dir(self) -> str:
        """<output_root>/<dataset>/profiling: every run, result, and aggregate of this sweep."""
        return os.path.join(self.output_root, self.dataset, 'profiling')

    @property
    def save_root(self) -> str:
        """Where execution --save puts the stages' outputs: cache-shaped under <output_root>/<dataset>."""
        return os.path.join(self.output_root, self.dataset)


@dataclass(frozen=True)
class Configuration:
    id: str                                         # the scripts' 060 directory name; unique within a sweep
    params: dict[str, Any]                          # classifier, tile_size, and one value per swept parameter
    execution: dict[str, Any]                       # the execution config document


def configuration_id(params: dict[str, Any]) -> str:
    """The scripts' tracks directory name (p060), which names every swept parameter."""
    return build_param_str(classifier=params['classifier'], tilesize=params['tile_size'],
                           sample_rate=params['sample_rate'], tracking_accuracy_threshold=params['mistrack_tolerance'],
                           relevance_threshold=params['relevance_threshold'], tilepadding=params['tile_padding'],
                           canvas_scale=params['canvas_scale'], tracker=params['tracker'])


def execution_config(params: dict[str, Any], artifacts: Artifacts, canvas_format: str, system: dict) -> dict:
    """One configuration's execution config document (execution/config.py's three sections)."""
    tolerance = params['mistrack_tolerance']
    doc_artifacts = {
        'classifier': {'name': params['classifier'], 'weights': artifacts.classifier_weights,
                       'compilation_benchmark': artifacts.compilation_benchmark},
        'never_relevant_bitmap': artifacts.never_relevant_bitmap,
        'detector': dict(artifacts.detector),
    }
    stages = {name: dict(entry) for name, entry in system['stages'].items()}
    # Pruning needs the tracker's table and a prune stage; without pruning both must be absent.
    if tolerance is not None:
        doc_artifacts['max_rate_table'] = artifacts.max_rate_tables[params['tracker']]
    else:
        stages.pop('prune', None)
    doc_params = {k: params[k] for k in ('sample_rate', 'relevance_threshold', 'mistrack_tolerance', 'tile_padding',
                                         'tracker', 'tile_size', 'canvas_scale')}
    return {'params': {**doc_params, 'canvas_format': canvas_format}, 'artifacts': doc_artifacts,
            'system': {**system, 'stages': stages}}


def configurations(cfg: SweepConfig) -> list[Configuration]:
    """The sweep's configurations: the selected ones (or the whole grid), always in grid order."""
    # Grid order, not the listed order, so dedup's choice of which configuration runs never depends on the listing.
    return [c for c in grid(cfg) if cfg.selected is None or c.id in cfg.selected]


def grid(cfg: SweepConfig) -> list[Configuration]:
    """Every configuration of the grid, in enumeration order (SWEPT's order, first parameter slowest)."""
    out = []
    lists = [getattr(cfg.params, name) for name, _ in SWEPT]
    for values in itertools.product(*lists):
        params = {'classifier': cfg.params.classifier, 'tile_size': cfg.params.tile_size,
                  **{key: v for (_, key), v in zip(SWEPT, values)}}
        out.append(Configuration(configuration_id(params), params,
                                 execution_config(params, cfg.artifacts, cfg.params.canvas_format, cfg.system)))
    return out


def sweep_hash(cfg: SweepConfig) -> str:
    """sha256 of the sweep document (canonical JSON): a changed sweep cannot resume into an old output."""
    return hashlib.sha256(json.dumps(cfg.raw, sort_keys=True).encode()).hexdigest()


def _values(data: dict, key: str, kind: Any, allow_none: bool = False) -> tuple:
    """A non-empty list of distinct values of `kind` (None allowed when allow_none)."""
    raw = data[key]
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f'params.{key}: expected a non-empty list')
    values = tuple(None if (allow_none and v is None) else _check_type(v, kind, f'params.{key}[{i}]')
                   for i, v in enumerate(raw))
    if len(set(values)) != len(values):
        raise ConfigError(f'params.{key}: values must be distinct, got {list(values)}')
    return values


def _params(data: Any) -> Params:
    data = _keys(data, {'classifier', 'tile_size', 'canvas_format', *(name for name, _ in SWEPT)}, 'params',
                 {'dedup_pruning'})
    dedup = data.get('dedup_pruning', True)
    # _check_type rejects bools (they are ints), so the flag is checked on its own.
    if not isinstance(dedup, bool):
        raise ConfigError(f'params.dedup_pruning: expected bool, got {type(dedup).__name__}')
    # Range checks happen once per configuration in execution's parse_config.
    return Params(
        classifier=_check_type(data['classifier'], str, 'params.classifier'),
        tile_size=_check_type(data['tile_size'], int, 'params.tile_size'),
        canvas_format=_check_type(data['canvas_format'], str, 'params.canvas_format'),
        sample_rates=_values(data, 'sample_rates', int),
        relevance_thresholds=_values(data, 'relevance_thresholds', float),
        mistrack_tolerances=_values(data, 'mistrack_tolerances', float, allow_none=True),
        tile_paddings=_values(data, 'tile_paddings', str),
        canvas_scales=_values(data, 'canvas_scales', float),
        trackers=_values(data, 'trackers', str),
        dedup_pruning=dedup,
    )


def _artifacts(data: Any, params: Params) -> Artifacts:
    data = _keys(data, {'classifier_weights', 'compilation_benchmark', 'never_relevant_bitmap', 'detector'},
                 'artifacts', {'max_rate_tables'})
    uses_prune = any(t is not None for t in params.mistrack_tolerances)
    tables = data.get('max_rate_tables') or {}
    if uses_prune:
        # Every tracker pairs with every tolerance, so each needs its own table.
        tables = _keys(tables, set(params.trackers), 'artifacts.max_rate_tables')
        tables = {t: _existing(path, f'artifacts.max_rate_tables.{t}') for t, path in tables.items()}
    elif tables:
        raise ConfigError('artifacts.max_rate_tables: must be omitted when no mistrack tolerance is swept')
    if not isinstance(data['detector'], dict):
        raise ConfigError('artifacts.detector: expected a mapping')
    return Artifacts(
        classifier_weights=_existing(data['classifier_weights'], 'artifacts.classifier_weights'),
        compilation_benchmark=_existing(data['compilation_benchmark'], 'artifacts.compilation_benchmark'),
        never_relevant_bitmap=_existing(data['never_relevant_bitmap'], 'artifacts.never_relevant_bitmap'),
        max_rate_tables=tables,
        detector=data['detector'],
    )


def _selected(data: Any, cfg: SweepConfig) -> frozenset[str]:
    """A non-empty list of distinct ids, each a configuration of the grid."""
    if not isinstance(data, list) or not data:
        raise ConfigError('configurations: expected a non-empty list of configuration ids')
    ids = [_check_type(v, str, f'configurations[{i}]') for i, v in enumerate(data)]
    if len(set(ids)) != len(ids):
        raise ConfigError('configurations: ids must be distinct')
    unknown = sorted(set(ids) - {c.id for c in grid(cfg)})
    if unknown:
        raise ConfigError(f'configurations: {len(unknown)} ids are not configurations of the grid, e.g. {unknown[0]}')
    return frozenset(ids)


def _evaluation(data: Any) -> Evaluation:
    data = _keys(data, {'reference_dir', 'reference_file'}, 'evaluation')
    return Evaluation(reference_dir=_existing(data['reference_dir'], 'evaluation.reference_dir'),
                      reference_file=_check_type(data['reference_file'], str, 'evaluation.reference_file'))


def parse_sweep_config(data: Any) -> SweepConfig:
    """Validate a parsed sweep document, including every configuration's execution config."""
    data = _keys(data, {'dataset', 'videoset', 'output_root', 'params', 'artifacts', 'evaluation', 'system'}, 'config',
                 {'configurations'})
    params = _params(data['params'])
    evaluation = _evaluation(data['evaluation'])
    output_root = os.path.abspath(_check_type(data['output_root'], str, 'output_root'))
    # Results (and saved stage outputs) must never land where the reference tracks are read from.
    reference = os.path.abspath(evaluation.reference_dir)
    if os.path.commonpath([output_root, reference]) in (output_root, reference):
        raise ConfigError(f'output_root: {output_root} and the reference tracks ({reference}) contain one another; '
                          f'use a separate directory so the sweep cannot overwrite existing results')
    if not isinstance(data['system'], dict) or not isinstance(data['system'].get('stages'), dict):
        raise ConfigError('system: expected execution\'s system section (queue_maxsize and stages)')
    cfg = SweepConfig(
        dataset=_check_type(data['dataset'], str, 'dataset'),
        videoset=_check_type(data['videoset'], str, 'videoset'),
        output_root=output_root,
        params=params,
        artifacts=_artifacts(data['artifacts'], params),
        evaluation=evaluation,
        system=data['system'],
        raw=data,
    )
    if 'configurations' in data:
        cfg = dataclasses.replace(cfg, selected=_selected(data['configurations'], cfg))
    # Every configuration must be a valid execution config; report the first bad one by its id.
    for c in configurations(cfg):
        try:
            parse_config(c.execution)
        except ConfigError as e:
            raise ConfigError(f'configuration {c.id}: {e}') from e
    return cfg


def load_sweep_config(path: str) -> SweepConfig:
    """Read and validate the sweep YAML file at path."""
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f'{path}: invalid YAML: {e}') from e
    return parse_sweep_config(data)
