"""Render revision tables and figures from the saved measurement snapshots.

Run in ace's polyis container:
  python /polytris-paper/scripts/generate_revision_figures.py
No experiment cache or reported operating point is modified.
Generated TeX contains numeric macros and table rows; prose and table wrappers live in content/.
"""
import hashlib
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import re

import duckdb
import altair as alt
import numpy as np

from leap_hota import reconstruction_scores


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'scripts/data/revision_evidence'
OUT = ROOT / 'figures/generated'
DATASETS = {'ams-y05': 'Amsterdam', 'caldot1-y05': 'CalDoT1', 'caldot2-y05': 'CalDoT2',
            'jnc0': 'B3D1', 'jnc2': 'B3D2', 'jnc6': 'B3D3', 'jnc7': 'B3D4'}
COLORS = {'tetris': '#008080', 'otif': '#d18400', 'leap': '#7556a6', 'reference': '#505050'}
NAMES = {'tetris': 'Tetris', 'otif': 'OTIF', 'leap': 'LEAP', 'reference': 'Reference'}
QUERIES = ('presence', 'count', 'topk')
# §7.3 query baselines, in table order; NoScope answers frame selection only.
BASELINES = ['otif', 'leap', 'seiden', 'noscope']
# Pixel height of one dataset's three preparation bars and their margins.
PREP_BAND = 16


def read(name):
    return json.loads((DATA / f'{name}.json').read_text())


def table_rows(name, rows):
    text = '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows) + '\n'
    (OUT / f'revision_{name}_rows.tex').write_text(text)


def project_video_hours(hours, preparation_h, throughput_fps, video_fps=15):
    assert preparation_h >= 0 and throughput_fps > 0 and video_fps > 0
    return np.maximum(0, np.asarray(hours) - preparation_h) * throughput_fps / video_fps


def query_comparison(gain_pp):
    if gain_pp >= 0:
        return 'best'
    if gain_pp >= -1:
        return 'comparable'
    if gain_pp >= -5:
        return 'within5pp'
    return 'lower'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Verify provenance and the unit conversion before rendering any reported values.
    for source in read('sources'):
        assert hashlib.sha256((DATA / source['file']).read_bytes()).hexdigest() == source['sha256']
    np.testing.assert_allclose(project_video_hours([0, 1, 2], 1, 15), [0, 0, 1])
    np.testing.assert_allclose(project_video_hours([0, 1], 0, 30, 30), [0, 1])
    assert [query_comparison(v) for v in (1, 1e-12, 0, -1e-12, -1, -1.1, -5, -5.1)] == [
        'best', 'best', 'best', 'comparable',
        'comparable', 'within5pp', 'within5pp', 'lower']
    con = duckdb.connect()
    con.execute('SET enable_progress_bar = false')
    numbers = dict(revisionDatasetCount=str(len(DATASETS)))
    matched, shares = read('classifier_matched'), read('operator_share')
    timing_rows = []
    for label in DATASETS.values():
        d = matched['datasets'][label]
        timing_rows.append([label, f"{d['runs']['classifier']['fps']:.0f}",
                            f"{d['runs']['detector']['fps']:.1f}",
                            f"{shares['datasets'][label]['operators']['classify']['share_pct']:.1f}"])
    sizes = read('ilp_sizes')['sizes']
    dataset_sizes = [r for r in sizes if r['level'] == 'dataset' and r['split'] == 'test']
    rows = [[r['label'], f"{r['variables_max']:,}", f"{r['constraints_max']:,}",
             f"{r['build_ms_median']:.1f}", f"{r['solve_ms_median']:.1f}"]
            for r in dataset_sizes]
    assert len(rows) == 7
    ilp_by_label = {r[0]: r[1:] for r in rows}
    combined = [r + ilp_by_label[r[0]] for r in timing_rows]
    table_rows('operator_costs', combined)
    ratios = matched['summary']['ratio_of_means']
    classifier = [d['runs']['classifier']['fps'] for d in matched['datasets'].values()]
    detector = [d['runs']['detector']['fps'] for d in matched['datasets'].values()]
    # The introduction uses the original full-split export, not the later matched timing sample.
    reference_row = next(line for line in (ROOT / 'figures/generated/naive_runtime_breakdown_det_track_table.tex').read_text().splitlines()
                         if line.startswith('Detection time proportion'))
    reference_shares = [float(value) for value in re.findall(r'([0-9.]+)\\%', reference_row)]
    assert len(reference_shares) == len(DATASETS)
    timing = read('ilp_timing')
    numbers.update(
        revisionReferenceDetectorShareMinPct=f'{min(reference_shares):.1f}',
        revisionReferenceDetectorShareMaxPct=f'{max(reference_shares):.1f}',
        revisionClassifierSpeedupMin=f'{ratios["as_deployed"]["min"]:.0f}',
        revisionClassifierSpeedupMax=f'{ratios["as_deployed"]["max"]:.0f}',
        revisionClassifierEqualBatchSpeedupMin=f'{ratios["same_batch_16"]["min"]:.0f}',
        revisionClassifierEqualBatchSpeedupMax=f'{ratios["same_batch_16"]["max"]:.0f}',
        revisionClassifierWithCopySpeedupMin=f'{ratios["as_deployed_with_rgb_copy"]["min"]:.0f}',
        revisionClassifierWithCopySpeedupMax=f'{ratios["as_deployed_with_rgb_copy"]["max"]:.0f}',
        revisionClassifierFpsMin=f'{min(classifier):,.0f}',
        revisionClassifierFpsMax=f'{max(classifier):,.0f}',
        revisionDetectorFpsMin=f'{min(detector):.1f}',
        revisionDetectorFpsMax=f'{max(detector):.1f}',
        revisionClassifierShareMinPct=f'{shares["summary"]["classify"]["min_pct"]:.0f}',
        revisionClassifierShareMaxPct=f'{shares["summary"]["classify"]["max_pct"]:.0f}',
        revisionClassifierShareMedianPct=f'{shares["summary"]["classify"]["median_pct"]:.1f}',
        revisionIlpGridTilesMin=str(min(r['tiles_per_frame'] for r in dataset_sizes)),
        revisionIlpGridTilesMax=str(max(r['tiles_per_frame'] for r in dataset_sizes)),
        revisionIlpVariablesMin=f'{min(r["variables_max"] for r in dataset_sizes):,}',
        revisionIlpVariablesMax=f'{max(r["variables_max"] for r in dataset_sizes):,}',
        revisionIlpConstraintsMin=f'{min(r["constraints_max"] for r in dataset_sizes):,}',
        revisionIlpConstraintsMax=f'{max(r["constraints_max"] for r in dataset_sizes):,}',
        revisionIlpRuns=f'{timing["all"]["runs"]:,}',
        revisionIlpBuildMedianMs=f'{timing["all"]["build_ms"]["median"]:.1f}',
        revisionIlpSolveMedianMs=f'{timing["all"]["solve_ms"]["median"]:.1f}',
        revisionIlpCombinedNinetyFifthMs=f'{timing["all"]["build_plus_solve_ms"]["p95"]:.1f}',
        revisionIlpSolveBudgetSeconds=f'{timing["solve_budget"]["default_seconds"]:g}',
        revisionIlpReachesBudgetPct=f'{timing["all"]["solve_reaches_budget"]["percent"]:.1f}')
    # Ground-truth tiles, no ILP, s=16, per dataset and padding.
    all_packing = {(r['label'], r['padding']): r for r in read('packing')['rows']}
    # The P=0/P=1 macros use the two paddings run on every dataset.
    packing = {key: r for key, r in all_packing.items() if key[1] in ('none', 'square')}
    assert len(packing) == 2 * len(DATASETS)
    # Sparse scenes (Amsterdam, CalDoT) vs. dense B3D: accuracy changes, density, and packing cost per padding.
    sparse = ('Amsterdam', 'CalDoT1', 'CalDoT2')
    dense = ('B3D1', 'B3D2', 'B3D3', 'B3D4')
    # The paper reports precision and recall (the reviewer did not ask for mAP).
    metrics = ('precision', 'recall')

    def change(label, padding, m):
        # Packed minus original on retained content, in percentage points.
        r = packing[(label, padding)]
        return r[f'retained_packed_{m}_pct'] - r[f'retained_original_{m}_pct']

    def values(labels, padding, field, scale=1):
        return [scale * packing[(label, padding)][field] for label in labels]

    def span(name, xs, fmt):
        numbers.update({f'{name}Min': format(min(xs), fmt), f'{name}Max': format(max(xs), fmt)})

    span('revisionPackingSparseNoPadRecallDrop', [-change(d, 'none', 'recall') for d in sparse], '.1f')
    span('revisionPackingSparseObjects', values(sparse, 'none', 'objects_per_frame'), '.1f')
    span('revisionPackingDenseObjects', values(dense, 'none', 'objects_per_frame'), '.1f')
    span('revisionPackingSparseTiles', values(sparse, 'none', 'occupied_fraction_of_grid', 100), '.1f')
    span('revisionPackingDenseTiles', values(dense, 'none', 'occupied_fraction_of_grid', 100), '.1f')
    for group, labels in [('Sparse', sparse), ('Dense', dense)]:
        span(f'revisionPacking{group}NoPadEfficacy', values(labels, 'none', 'packing_efficacy', 100), '.0f')
        span(f'revisionPacking{group}PadEfficacy', values(labels, 'square', 'packing_efficacy', 100), '.0f')
        span(f'revisionPacking{group}Merge', [packing[(d, 'none')]['polyominoes_per_frame'] /
                                              packing[(d, 'square')]['polyominoes_per_frame'] for d in labels], '.1f')
    span('revisionPackingDensePadTilesPerPoly', values(dense, 'square', 'tiles_per_polyomino'), '.0f')
    # The one-sided P=0.5 padding of the 5% HOTA-loss configurations (Amsterdam and CalDoT; B3D deploys P=0).
    deployed = {label: r for (label, padding), r in all_packing.items() if padding in ('tr', 'bl')}
    assert sorted(deployed) == sorted(sparse)

    def deployed_change(label, m):
        r = deployed[label]
        return r[f'retained_packed_{m}_pct'] - r[f'retained_original_{m}_pct']

    # Packed tiles (relevant plus padding) relative to P=0, and as a share of the frame area, per padding.
    def tiles(label, padding):
        r = deployed[label] if padding == 'deployed' else packing[(label, padding)]
        return r['occupied_fraction_of_grid']

    span('revisionPackingDeployedTileRatio', [tiles(d, 'deployed') / tiles(d, 'none') for d in sparse], '.1f')
    span('revisionPackingPadTileRatio', [tiles(d, 'square') / tiles(d, 'none') for d in DATASETS.values()], '.1f')
    span('revisionPackingSparsePadTiles', [100 * tiles(d, 'square') for d in sparse], '.0f')
    span('revisionPackingSparseDeployedEfficacy', [100 * deployed[d]['packing_efficacy'] for d in sparse], '.0f')
    span('revisionPackingDensePadTiles', [100 * tiles(d, 'square') for d in dense], '.0f')
    # Every metric at each dataset's 5% padding: P=0.5 on the sparse datasets, P=0 on B3D.
    numbers.update(
        revisionPackingDeployedMaxChange=f"{max([abs(deployed_change(d, m)) for d in sparse for m in metrics] + [abs(change(d, 'none', m)) for d in dense for m in metrics]):.1f}",
        revisionPackingDeployedRecallMaxChange=f"{max(abs(deployed_change(d, 'recall')) for d in sparse):.1f}",
        revisionPackingDeployedPrecisionDropPp=f"{max(-deployed_change(d, 'precision') for d in sparse):.1f}")
    numbers.update(
        revisionPackingSparsePadRecallMaxChange=f"{max(abs(change(d, 'square', 'recall')) for d in sparse):.1f}",
        revisionPackingDenseMaxChange=f"{max(abs(change(d, 'none', m)) for d in dense for m in metrics):.1f}",
        revisionPackingPadPrecisionDropPp=f"{max(-change(d, 'square', 'precision') for d in DATASETS.values()):.1f}")
    leap = read('leap_rerun')['datasets']
    leap_scores = reconstruction_scores(DATA, leap)
    assert all(leap[label]['hota']['linear']['HOTA'] > leap[label]['hota']['pattern']['HOTA']
               for label in ('CalDoT1', 'CalDoT2', 'Amsterdam'))
    durations = read('leap_duration_variability')['rows']
    pooled = next(r for r in durations if r['label'] == 'B3D pooled')
    assert pooled['tracks'] == sum(r['tracks'] for r in durations if r['label'] in ('B3D1', 'B3D2', 'B3D3', 'B3D4'))
    # The paper reports every B3D track, with clipped tracks measured within their clip.
    pooled = next(r for r in durations if r['label'] == 'B3D pooled, all tracks')
    numbers.update(revisionLeapBThreeDVisibilityTenSeconds=f'{pooled["p10_s"]:.1f}',
                   revisionLeapBThreeDVisibilityNinetySeconds=f'{pooled["p90_s"]:.1f}')
    diagnostics = read('leap_sampling_diagnostics')['datasets']
    c1, c2, ams, b3 = (diagnostics[label] for label in ('CalDoT1', 'CalDoT2', 'Amsterdam', 'B3D3'))
    coverage = c2['coverage']
    numbers.update(
        revisionLeapCalDoTTwoMedianSamplingSeconds=f'{c2["sampling"]["gap_p10_p50_p90_s"][1]:.0f}',
        revisionLeapCalDoTTwoFewerThanTwoSamplesPct=f'{100 * (coverage["zero_sample_exposure"] + coverage["one_sample_exposure"]) / coverage["gt_tracks"]:.1f}')
    second = read('leap_second_observation')
    assert second['source_sha256'] == c2['provenance']['source_sha256']
    cohort, pairs = second['cohort'], second['first_two_detections']
    assert cohort['observed_vehicles'] == cohort['observed_once'] + cohort['observed_at_least_twice']
    assert pairs['pairs'] == cohort['observed_at_least_twice']
    numbers.update(revisionLeapCalDoTTwoObservedOncePct=f'{100*cohort["observed_once"]/cohort["observed_vehicles"]:.1f}',
                   revisionLeapCalDoTTwoFirstLinkFailurePct=f'{100*pairs["different_identity"]/pairs["pairs"]:.1f}')
    assert c1['provenance']['mask_enabled'] is False and c2['provenance']['mask_enabled'] is False
    assert not c1['continuity_by_stationarity'][0]['passes_stationary_rule']
    numbers.update(
        revisionLeapCalDoTOneSampledRecallPct=f'{100*c1["matching"]["matched_observations"]/c1["matching"]["gt_boxes_on_sampled_frames"]:.1f}',
        revisionLeapCalDoTTwoSampledRecallPct=f'{100*c2["matching"]["matched_observations"]/c2["matching"]["gt_boxes_on_sampled_frames"]:.1f}',
        revisionLeapCalDoTOneMovingContinuityPct=f'{100*c1["continuity_by_stationarity"][0]["same_id_share"]:.1f}',
        revisionLeapCalDoTOneEarlyExitPct=f'{100*c1["first_sample_exit"]["exits_before_next_sample_share"]:.1f}',
        revisionLeapCalDoTTwoEarlyExitPct=f'{100*c2["first_sample_exit"]["exits_before_next_sample_share"]:.1f}',
        revisionLeapCalDoTOneNearStationaryPct=f'{100*c1["oracle_interpolation"]["motion"]["near_stationary_box_share"]:.1f}',
        revisionLeapCalDoTTwoNearStationaryPct=f'{100*c2["oracle_interpolation"]["motion"]["near_stationary_box_share"]:.1f}',
        revisionLeapCalDoTOneOracleBoxMatchPct=f'{100*c1["oracle_interpolation"]["iou_ge_half_share"]:.1f}',
        revisionLeapCalDoTTwoOracleBoxMatchPct=f'{100*c2["oracle_interpolation"]["iou_ge_half_share"]:.1f}',
        revisionLeapCalDoTTwoTrueCenterBoxMatchPct=f'{100*c2["oracle_interpolation"]["center_and_size"][1]["iou_ge_half_share"]:.1f}',
        revisionLeapCalDoTTwoPatternAnchorMatchPct=f'{100*c2["pattern_export_anchor_fit"]["iou_ge_half_share_all_observations"]:.1f}',
        revisionLeapAmsterdamOracleBoxMatchPct=f'{100*ams["oracle_interpolation"]["iou_ge_half_share"]:.1f}',
        revisionLeapBThreeDThreeContinuityPct=f'{100*b3["association"]["same_id_share"]:.1f}',
        revisionLeapBThreeDThreeEligiblePatterns=str(b3['patterns']['eligible_patterns']),
        revisionLeapBThreeDThreePatterns=str(b3['patterns']['patterns']))
    sparse = leap['B3D3']['classes']['observations_per_object']
    intervals = read('native_intervals')
    numbers.update(revisionLeapBThreeDThreeObjectsOnce=f'{sparse["objects_observed_once"]:,}',
                   revisionLeapBThreeDThreeObjects=f'{sparse["objects"]:,}',
                   revisionBThreeDClipCount=str(sum(r['clips'] for k, r in intervals.items() if k.startswith('B3D'))),
                   revisionBThreeDFrameCount=f'{sum(r["frames"] for k, r in intervals.items() if k.startswith("B3D")):,}')
    rows = [[label] + [r'\revpending{' + f"{leap[label]['hota'][k]['HOTA']:.3f}" + '}'
                      if 'matched_detector' in leap[label] else f"{leap[label]['hota'][k]['HOTA']:.3f}"
                      for k in ('raw', 'linear', 'pattern')]
            for label in DATASETS.values()]
    table_rows('leap_reconstruction', rows)
    labels = list(DATASETS.values())
    table_rows('leap_gap_interpolation', [
        [name] + [rf'\textbf{{{100 * leap_scores[label][method]:.1f}}}'
                  if leap_scores[label][method] == max(leap_scores[label].values())
                  else f'{100 * leap_scores[label][method]:.1f}' for label in labels]
        for name, method in [('(i)', 'linear'), ('(ii)', 'pattern'), ('(iii)', 'anchored')]])
    configs = read('extraction_configs')
    for dataset, label in DATASETS.items():
        leap_point, = (point for point in configs[dataset] if point['system'] == 'leap')
        leap_point['hota'] = max(leap_scores[label].values())
    assert read('native_query_configs')['count_metric'] == 'MAPE'
    assert read('native_query_configs')['selection_metric'] == 'selection_query_1'
    assert read('native_query_configs')['topk_metric'] == 'top_k_query_1'
    # Every system's operating points per (dataset, query): (fps, score, config).
    points = {}
    for r in read('native_query_configs')['records']:
        if r['system'] in ('tetris', 'otif', 'leap'):
            for query in QUERIES:
                points.setdefault((r['dataset'], query, r['system']), []).append((r['fps'], r[query], r['config']))
    for ds in DATASETS:
        # Seiden: one index per clip, query, and budget fraction 1/k, a single run (seed 0) as in Seiden's experiments.
        runs = {}
        for r in json.loads((DATA / 'seiden' / f'{ds}.json').read_text())['records']:
            if r['seed'] == 0:
                runs.setdefault((r['query'], r['denominator']), []).append(r)
        for (query, k), rs in runs.items():
            assert len(rs) == 1, f'{ds} Seiden {query} 1/{k}: expected one seed-0 run'
            fps = float(np.mean([r['fps'] for r in rs]))
            scores = ({'presence': [r['selection']['f1'] for r in rs]} if query == 'presence' else
                      {'count': [r['aggregation']['mape'] for r in rs], 'topk': [r['topk'] for r in rs]})
            for metric, values in scores.items():
                points.setdefault((ds, metric, 'seiden'), []).append((fps, float(np.mean(values)), f'1/{k}'))
        # NoScope answers frame selection only, on the datasets it was trained for.
        noscope = DATA / 'noscope' / f'{ds}.json'
        if noscope.exists():
            points[ds, 'presence', 'noscope'] = [(r['fps'], r['selection']['f1'], f"target {r['target']}")
                                                 for r in json.loads(noscope.read_text())['records']]

    def quality(query, score):
        return -score if query == 'count' else score

    def pareto(query, candidates):
        """Operating points no other point matches or beats in both FPS and the query's own score."""
        return [p for p in candidates if not any(
            q[0] >= p[0] and quality(query, q[1]) >= quality(query, p[1])
            and (q[0] > p[0] or quality(query, q[1]) > quality(query, p[1])) for q in candidates)]

    # LEAP has one configuration. OTIF, Seiden, and NoScope take the Pareto point closest to LEAP in FPS; Tetris takes
    # its slowest Pareto point faster than all of them. Each (dataset, query) is selected independently.
    faster = []
    for ds in DATASETS:
        for query in QUERIES:
            (leap_point,) = points[ds, query, 'leap']
            chosen = {'leap': leap_point}
            for system in (s for s in BASELINES if s != 'leap'):
                if (ds, query, system) in points:
                    chosen[system] = min(pareto(query, points[ds, query, system]),
                                         key=lambda p: (abs(p[0] - leap_point[0]), -quality(query, p[1]), p[2]))
            floor = max(p[0] for p in chosen.values())
            chosen['tetris'] = min((p for p in pareto(query, points[ds, query, 'tetris']) if p[0] > floor),
                                   key=lambda p: (p[0], -quality(query, p[1]), p[2]))
            faster += [dict(dataset=ds, label=DATASETS[ds], query=query, system=system, config=p[2], score=p[1],
                            fps=p[0]) for system, p in chosen.items()]
    lookup = {(r['dataset'], r['query'], r['system']): r for r in faster}
    order = ['caldot1-y05', 'caldot2-y05', 'jnc0', 'jnc2', 'jnc6', 'jnc7', 'ams-y05']
    rows = []
    for ds in order:
        cells = []
        for query in QUERIES:
            values = []
            # NoScope answers selection only; its count and top-10 slots stay '--'.
            for system in ['tetris', *BASELINES]:
                if (ds, query, system) in lookup:
                    value = 100 * lookup[ds, query, system]['score']
                    values.append(format(value, '.0f' if round(value, 1) >= 100 else '.1f'))
                else:
                    values.append('--')
            best = (min if query == 'count' else max)(float(v) for v in values if v != '--')
            values = [r'\textbf{' + v + '}' if v != '--' and float(v) == best else v for v in values]
            cells += [v + (' /' if i < len(values) - 1 else '') for i, v in enumerate(values)]
        rows.append([DATASETS[ds], *cells])
    table_rows('native_queries_faster_by_metric', rows)
    def compare(baselines):
        """Group every (dataset, query) by Tetris's gain over the best of `baselines` there."""
        groups = {key: [] for key in ('best', 'comparable', 'within5pp', 'lower')}
        for ds in order:
            for query in QUERIES:
                baseline = (min if query == 'count' else max)(
                    (lookup[ds, query, system] for system in baselines if (ds, query, system) in lookup),
                    key=lambda r: r['score'])
                # Decimal scores avoid inflating an exact ten-point gap through binary subtraction.
                gain = 100 * (Decimal(str(lookup[ds, query, 'tetris']['score'])) - Decimal(str(baseline['score'])))
                gain = -gain if query == 'count' else gain
                groups[query_comparison(gain)].append(dict(
                    dataset=DATASETS[ds], query=query, baseline=baseline['system'], gain_pp=float(gain)))
        total = len(order) * len(QUERIES)
        assert sum(map(len, groups.values())) == total

        def max_loss(key):
            return float(max((Decimal(str(-r['gain_pp'])) for r in groups[key]), default=Decimal(0)).quantize(
                Decimal('0.1'), rounding=ROUND_CEILING))
        summary = dict(best_pct=100 * len(groups['best']) / total,
                       comparable_pct=100 * len(groups['comparable']) / total,
                       within5_pct=100 * len(groups['within5pp']) / total,
                       within5_max_gap_pp=max_loss('within5pp'),
                       lower_pct=100 * len(groups['lower']) / total,
                       lower_max_gap_pp=max_loss('lower'))
        # The cases more than 5 points behind, largest first: the prose can name each one.
        lower = sorted((Decimal(str(-r['gain_pp'])) for r in groups['lower']), reverse=True)
        ceil = lambda v: float(v.quantize(Decimal('0.1'), rounding=ROUND_CEILING))
        summary.update(comparable_count=len(groups['comparable']), lower_gaps_pp=[ceil(v) for v in lower])
        return groups, summary

    # Against the other track-materialization systems, then against every system.
    comparisons = {'Query': compare(['otif', 'leap']), 'QueryAll': compare(BASELINES)}
    # §7.3's all-systems sentence has no within-1-point clause and names the two cases more than 5 points behind.
    all_summary = comparisons['QueryAll'][1]
    assert all_summary['comparable_count'] == 0, 'all-systems prose omits the within-1-point clause'
    assert len(all_summary['lower_gaps_pp']) == 2, 'all-systems prose names exactly two cases more than 5 points behind'
    numbers.update(revisionQueryAllLowerSecondGapPp=f"{all_summary['lower_gaps_pp'][1]:g}")
    # Tetris against LEAP alone (R1-O2): the response says Tetris is never less accurate than LEAP.
    # Positive gain means Tetris is more accurate than LEAP (lower MAPE for counting, higher score otherwise).
    leap_gains = [(lookup[ds, q, 'leap']['score'] - lookup[ds, q, 'tetris']['score']) * (1 if q == 'count' else -1)
                  for ds in order for q in QUERIES]
    assert all(g >= 0 for g in leap_gains), 'Tetris is less accurate than LEAP in some case'
    numbers.update(revisionQueryLeapCases=str(len(leap_gains)),
                   revisionQueryLeapBetterCount=str(sum(g > 0 for g in leap_gains)))
    # Where NoScope's selected B3D point is below near-perfect F1, its throughput cost of the strictest (0%) target.
    strict_costs = []
    for ds in ('jnc0', 'jnc2', 'jnc6', 'jnc7'):
        chosen = lookup[ds, 'presence', 'noscope']
        if chosen['score'] < .99:
            (strict,) = (p for p in points[ds, 'presence', 'noscope'] if p[2] == 'target 0.0')
            assert strict[1] >= .99, f'{ds}: NoScope\'s strictest target is not near-perfect'
            strict_costs.append(100 * (1 - strict[0] / chosen['fps']))
    assert strict_costs
    numbers.update(revisionNoScopeBThreeDStrictCostMin=f'{min(strict_costs):.0f}',
                   revisionNoScopeBThreeDStrictCostMax=f'{max(strict_costs):.0f}')
    (OUT / 'revision_query_faster_throughput_data.json').write_text(json.dumps(dict(
        source='scripts/data/revision_evidence/native_query_configs.json, seiden/, noscope/',
        source_sha256=hashlib.sha256((DATA / 'native_query_configs.json').read_bytes()).hexdigest(),
        selection='Per dataset and query, among each system\'s Pareto points for that query\'s own score: '
                  'LEAP\'s single configuration; OTIF, Seiden, and NoScope closest to LEAP in FPS; Tetris the slowest '
                  'faster than all of them. Ties prefer the better score, then configuration name.',
        count_metric='MAPE', selection_metric='selection_query_1', topk_metric='top_k_query_1', matches=faster,
        comparison_basis='Saved scores before display rounding; compare Tetris against the best of OTIF and LEAP '
                         '(track) and against the best other system (all) '
                         'for each dataset and query. Positive gain favors Tetris. Best means gain >= 0, including exact ties. '
                         'Nonzero losses of at most 1 pp are comparable; remaining losses are at most 5 pp or greater than 5 pp.',
        comparisons={'track' if prefix == 'Query' else 'all': dict(groups=groups, percentage_summary=summary)
                     for prefix, (groups, summary) in comparisons.items()}), indent=2) + '\n')
    for prefix, (_, summary) in comparisons.items():
        # One decimal, without a trailing ".0" (81.0 -> 81).
        pct = lambda v: f'{v:.1f}'.removesuffix('.0')
        numbers.update({f'revision{prefix}BestPct': pct(summary['best_pct']),
                        f'revision{prefix}ComparablePct': pct(summary['comparable_pct']),
                        f'revision{prefix}WithinFivePct': pct(summary['within5_pct']),
                        f'revision{prefix}WithinFiveMaxGapPp': f'{summary["within5_max_gap_pp"]:g}',
                        f'revision{prefix}LowerPct': pct(summary['lower_pct']),
                        f'revision{prefix}LowerMaxGapPp': f'{summary["lower_max_gap_pp"]:g}'})
    panels = []
    systems = ['tetris', 'otif', 'leap']
    query_titles = {'presence': 'Selection F1', 'count': 'Count MAPE', 'topk': 'Top-10 precision'}
    for query in ('presence', 'count', 'topk'):
        values = [r for r in faster if r['query'] == query and r['system'] in systems]
        domain = {'presence': [.8, 1], 'count': [0, 1.5], 'topk': [0, 1]}[query]
        ticks = {'presence': [.8, .9, 1], 'count': [0, .5, 1, 1.5], 'topk': [0, .5, 1]}[query]
        chart = alt.Chart(alt.Data(values=values)).mark_point(filled=True, size=45, opacity=.8).encode(
            x=alt.X('fps:Q', title='Throughput (FPS)' if query == 'topk' else None,
                    scale=alt.Scale(type='log', domain=[10, 1000]),
                    axis=alt.Axis(values=[10, 100, 1000], format='~s')),
            y=alt.Y('score:Q', title=query_titles[query], scale=alt.Scale(domain=domain),
                    axis=alt.Axis(values=ticks, format='.0%' if query in ('presence', 'count') else alt.Undefined)),
            color=alt.Color('system:N', scale=alt.Scale(domain=systems, range=['#4269d0', '#3ca951', '#ff8ab7']),
                            legend=alt.Legend(title=None, orient='top', direction='horizontal',
                                             labelExpr="{'tetris':'Tetris','otif':'OTIF','leap':'LEAP'}[datum.label]")),
            shape=alt.Shape('system:N', scale=alt.Scale(domain=systems, range=['diamond', 'circle', 'circle']), legend=None),
            tooltip=['label:N', 'system:N', 'config:N', alt.Tooltip('fps:Q', format='.1f'),
                     alt.Tooltip('score:Q', format='.4f')],
        ).properties(width=100, height=90).facet(column=alt.Column(
            'label:N', sort=[DATASETS[ds] for ds in order], title=None,
            header=alt.Header(labels=query == 'presence', labelFontWeight='bold')))
        panels.append(chart)
    # Altair defaults, as in the other figures, except for the plot border.
    alt.vconcat(*panels, spacing=12).resolve_scale(y='independent').configure_view(stroke=None).save(
        str(OUT / 'revision_query_faster_throughput.pdf'))
    gaps = read('gaps')['datasets']['jnc7']['trackers']['bytetrackcython']['tolerances']
    rows = [[f'{t:.1f}', f"{gaps[f'{t:.2f}']['relevant']['mean_gap']:.2f}"]
            for t in (0, .2, .4, .6)]
    table_rows('gap_table', [[r'Mean max. gap (frames)', *[r[1] for r in rows]]])
    # Preparation sums are work estimates, not observed serial elapsed time, all including I/O.
    # tetris/measurements/r9o2_preparation_total.py sums each system's measured components.
    preparation, overheads = read('preparation'), read('preparation_overheads')['rows']
    preparation_total = read('preparation_total')
    components = preparation_total['components']
    con.execute('CREATE TABLE components(dataset VARCHAR, system VARCHAR, phase VARCHAR, seconds DOUBLE)')
    con.executemany('INSERT INTO components VALUES (?,?,?,?)',
                    [(r['dataset'], r['system'], r['phase'], r['seconds']) for r in preparation_total['totals']])
    prep_rows = con.execute('SELECT dataset, system, sum(seconds) FILTER (WHERE phase=\'training\')/3600 AS training_h, '
                           'coalesce(sum(seconds) FILTER (WHERE phase=\'validation\'),0)/3600 AS validation_h, '
                           'sum(seconds) AS total_s FROM components GROUP BY ALL ORDER BY dataset,system').fetchall()
    prep = {(ds, system): dict(training_h=train, validation_h=valid, total_s=total)
            for ds, system, train, valid, total in prep_rows}
    for ds in DATASETS:
        prep[ds, 'reference'] = dict(training_h=0, validation_h=0, total_s=0)
    names = {'ams-y05': 'Amsterdam', 'caldot1-y05': 'CalDoTOne', 'caldot2-y05': 'CalDoTTwo'}
    for system in ('tetris', 'otif', 'leap'):
        hours = [prep[ds, system]['total_s'] / 3600 for ds in DATASETS]
        numbers.update({f'revision{system.title()}PreparationMinHours': f'{min(hours):.1f}',
                        f'revision{system.title()}PreparationMaxHours': f'{max(hours):.1f}'})
    validation_pct = [100 * prep[ds, 'tetris']['validation_h'] * 3600 / prep[ds, 'tetris']['total_s'] for ds in DATASETS]
    numbers.update(revisionTetrisValidationShareMinPct=f'{min(validation_pct):.0f}',
                   revisionTetrisValidationShareMaxPct=f'{max(validation_pct):.0f}')
    # Detector labeling of the training split (p011) and classifier training (p014), per dataset.
    stage_s = {}
    for r in components:
        if r['system'] == 'tetris':
            key = (r['dataset'], r['component'].split()[0])
            stage_s[key] = stage_s.get(key, 0) + r['seconds']
    detection_h = [stage_s[ds, 'p011'] / 3600 for ds in DATASETS]
    classifier_h = [stage_s[ds, 'p014'] / 3600 for ds in DATASETS]
    classifier_pct = [100 * stage_s[ds, 'p014'] / prep[ds, 'tetris']['total_s'] for ds in DATASETS]
    detection_pct = [100 * stage_s[ds, 'p011'] / prep[ds, 'tetris']['total_s'] for ds in DATASETS]
    numbers.update(revisionTetrisTrainingDetectionMinHours=f'{min(detection_h):.1f}',
                   revisionTetrisTrainingDetectionMaxHours=f'{max(detection_h):.1f}',
                   revisionTetrisClassifierTrainingMinHours=f'{min(classifier_h):.1f}',
                   revisionTetrisClassifierTrainingMaxHours=f'{max(classifier_h):.1f}',
                   revisionTetrisClassifierTrainingShareMinPct=f'{min(classifier_pct):.0f}',
                   revisionTetrisClassifierTrainingShareMaxPct=f'{max(classifier_pct):.0f}',
                   revisionTetrisTrainingDetectionShareMinPct=f'{min(detection_pct):.0f}',
                   revisionTetrisTrainingDetectionShareMaxPct=f'{max(detection_pct):.0f}')
    loading = {(r['dataset'], r['scenario']): r['total_s'] for r in overheads if r['boundary'] == 'instrumented'}
    repeated_minutes = [(loading[ds, 'per_video_and_configuration'] - loading[ds, 'once_per_dataset']) / 60
                        for ds in DATASETS]
    counts = {d['components']['validation_sweep']['configurations'] for d in preparation['datasets'].values()}
    assert len(counts) == 1
    numbers.update(revisionPreparationRepeatedLoadMinMinutes=f'{min(repeated_minutes):.0f}',
                   revisionPreparationRepeatedLoadMaxMinutes=f'{max(repeated_minutes):.0f}',
                   revisionValidationConfigurationCount=str(next(iter(counts))))
    # The same prepared frontier serves both operating points; it is never charged twice.
    selected = read('matched_points')['accuracy_matched']
    points = []
    for loss in ('0.05', '0.10'):
        for ds, label in DATASETS.items():
            pair = selected[loss]['per_dataset'][label]
            assert pair['prior']['system'] == 'otif'
            assert pair['tetris']['config'].startswith('ShuffleNet05_60_')
            for system, point in [('tetris', pair['tetris']), ('otif', pair['prior'])]:
                points.append({**point, 'dataset': ds, 'label': label, 'loss': loss, 'system': system})
            candidates = [r for r in configs[ds] if r['system'] == 'leap']
            assert len(candidates) == 1 and candidates[0]['hota'] <= pair['tetris']['hota']
            points.append(dict(dataset=ds, label=label, loss=loss, **candidates[0]))
            reference = [r for r in configs[ds] if r['system'] == 'reference']
            assert len(reference) == 1
            points.append(dict(dataset=ds, label=label, loss=loss, **reference[0]))
    con.execute('CREATE TABLE selected(dataset VARCHAR, loss VARCHAR, system VARCHAR, fps DOUBLE)')
    con.executemany('INSERT INTO selected VALUES (?,?,?,?)',
                    [(r['dataset'], r['loss'], r['system'], r['fps']) for r in points])
    con.execute('CREATE TABLE prep_totals(dataset VARCHAR, system VARCHAR, seconds DOUBLE)')
    con.executemany('INSERT INTO prep_totals VALUES (?,?,?)', [(d, s, r['total_s']) for (d, s), r in prep.items()])
    cross = con.execute("""
        SELECT t.dataset, t.loss, b.system AS baseline, t.fps AS tetris_fps, b.fps AS baseline_fps,
               p.seconds AS tetris_preparation_s, q.seconds AS baseline_preparation_s,
               CASE WHEN t.fps > b.fps AND p.seconds > q.seconds THEN
                    (p.seconds-q.seconds)/(900/b.fps-900/t.fps)
                    WHEN t.fps >= b.fps AND p.seconds <= q.seconds THEN 0 END AS break_even_video_minutes,
               CASE WHEN t.fps > b.fps AND p.seconds > q.seconds THEN
                    (t.fps*p.seconds-b.fps*q.seconds)/(t.fps-b.fps)/3600
                    WHEN t.fps >= b.fps AND p.seconds <= q.seconds THEN p.seconds/3600 END AS catch_up_elapsed_h
        FROM selected t JOIN selected b USING(dataset, loss)
        JOIN prep_totals p ON p.dataset=t.dataset AND p.system='tetris'
        JOIN prep_totals q ON q.dataset=b.dataset AND q.system=b.system
        WHERE t.system='tetris' AND b.system <> 'tetris'
        ORDER BY t.dataset,t.loss,b.system
    """)
    cross_rows = [dict(zip([d[0] for d in cross.description], r)) for r in cross.fetchall()]
    def crossover_h(fast, slow):
        # Elapsed hours after which `fast` has processed at least as much video as `slow`.
        if fast['fps'] <= slow['fps']:
            return None
        return max((fast['fps'] * fast['prep_s'] - slow['fps'] * slow['prep_s']) / (fast['fps'] - slow['fps']),
                   fast['prep_s']) / 3600
    assert crossover_h(dict(fps=2, prep_s=3600), dict(fps=1, prep_s=0)) == 2
    assert crossover_h(dict(fps=2, prep_s=3600), dict(fps=1, prep_s=7200)) == 1
    assert crossover_h(dict(fps=1, prep_s=0), dict(fps=2, prep_s=0)) is None
    runs = {}
    for ds in DATASETS:
        at5 = {r['system']: r for r in points if r['dataset'] == ds and r['loss'] == '0.05'}
        # LEAP never reaches the 5% bound, so it is compared against the fastest Tetris point that is more accurate.
        assert at5['leap']['hota'] < .95 * at5['reference']['hota']
        fastest = max((r for r in configs[ds] if r['system'] == 'tetris' and r['hota'] > at5['leap']['hota']),
                      key=lambda r: r['fps'])
        runs[ds] = {s: dict(fps=at5[s]['fps'], prep_s=prep[ds, s]['total_s'])
                    for s in ('tetris', 'otif', 'leap', 'reference') if (ds, s) in prep}
        runs[ds]['leapmatched'] = dict(fps=fastest['fps'], prep_s=prep[ds, 'tetris']['total_s'])
    # The prose assumes both baselines have preparation everywhere and names where OTIF meets the 5% bound.
    assert all('otif' in r and 'leap' in r for r in runs.values())
    assert [label for label, pair in selected['0.05']['per_dataset'].items() if pair['prior']['meets_bound']] == ['B3D3']
    pairs = [('tetris', 'reference'), ('otif', 'reference'), ('tetris', 'otif'),
             ('leapmatched', 'reference'), ('leap', 'reference'), ('leapmatched', 'leap')]
    tetris_crossings = []
    for fast, slow in pairs:
        key = f'{fast.title()}{slow.title()}'
        hours = {ds: crossover_h(r[fast], r[slow]) for ds, r in runs.items() if fast in r and slow in r}
        if fast == 'otif':
            # The paired OTIF configuration is slower than the reference only on CalDoT2.
            assert [ds for ds, h in hours.items() if h is None] == ['caldot2-y05']
            hours = {ds: h for ds, h in hours.items() if h is not None}
        ratios = [runs[ds][fast]['fps'] / runs[ds][slow]['fps'] for ds in hours]
        numbers.update({f'revisionCross{key}MinHours': f'{min(hours.values()):.1f}',
                        f'revisionCross{key}MaxHours': f'{max(hours.values()):.1f}',
                        f'revisionSpeedup{key}Min': f'{min(ratios):.1f}',
                        f'revisionSpeedup{key}Max': f'{max(ratios):.1f}'})
        if fast in ('tetris', 'leapmatched'):
            tetris_crossings.extend(hours.values())
    # The latest point at which Tetris overtakes any other system on any dataset.
    numbers['revisionCrossTetrisLatestHours'] = f'{max(tetris_crossings):.1f}'
    for ds, name in names.items():
        for loss, suffix in [('0.05', 'Five'), ('0.10', 'Ten')]:
            minutes = next(r['break_even_video_minutes'] for r in cross_rows
                           if r['dataset'] == ds and r['loss'] == loss and r['baseline'] == 'reference')
            numbers[f'revisionReferenceBreakEven{name}{suffix}VideoHours'] = f'{minutes / 60:.1f}'
    (OUT / 'revision_plot_data.json').write_text(json.dumps(dict(
        preparation=[dict(dataset=d, system=s, **r) for (d, s), r in prep.items()],
        components=components, operating_points=points, break_even=cross_rows,
        projection='max(0, elapsed_hours-preparation_hours)*throughput_fps/15', projection_units='video hours',
        note='Accounting projection; Tetris summed work and reconstructed baseline stage wall times.'), indent=2)+'\n')
    systems = [NAMES[s] for s in ('tetris', 'otif', 'leap')]
    phases = ['Training', 'Validation']
    records = []
    for ds in DATASETS:
        for system in [s for s in ('tetris', 'otif', 'leap') if (ds, s) in prep]:
            r = prep[ds, system]
            for phase, start, end in [(phases[0], 0, r['training_h']),
                                      (phases[1], r['training_h'], r['total_s'] / 3600)]:
                records.append(dict(dataset=DATASETS[ds], system=NAMES[system], phase=phase, start=start, end=end,
                                    total_h=r['total_s'] / 3600))
    base = alt.Chart(alt.Data(values=records)).encode(
        y=alt.Y('dataset:N', sort=list(DATASETS.values()), title=None, scale=alt.Scale(paddingInner=0, paddingOuter=0)),
        # Each dataset's band leaves equal margins above Tetris and below LEAP.
        yOffset=alt.YOffset('system:N', scale=alt.Scale(domain=systems, range=[1.75, PREP_BAND - 1.75])))
    # Separators sit on the boundary above every dataset but the first.
    separators = alt.Chart(alt.Data(values=[dict(dataset=label) for label in list(DATASETS.values())[1:]])).mark_rule(
        color='black', strokeWidth=.8).encode(y=alt.Y('dataset:N', sort=list(DATASETS.values()), bandPosition=0))
    bars = base.mark_bar().encode(
        x=alt.X('start:Q', title='Preparation (h)'),
        x2='end:Q', color=alt.Color('system:N', scale=alt.Scale(domain=systems,
            range=[COLORS[s] for s in ('tetris', 'otif', 'leap')]),
            legend=alt.Legend(title=None, orient='bottom-right', symbolType='square', fillColor='white',
                              padding=3, zindex=1)),
        opacity=alt.Opacity('phase:N', scale=alt.Scale(domain=phases, range=[.45, 1]),
            legend=alt.Legend(title=None, orient='bottom-right', symbolType='square', fillColor='white',
                              padding=3, zindex=1)),
    )
    # Altair defaults, as in the other figures, except for the plot border.
    (bars + separators).properties(width=290, height=len(DATASETS) * PREP_BAND).configure_view(stroke=None).save(
        str(OUT / 'revision_preparation.pdf'))
    # Display only the 5% operating point, its paired OTIF point, and the reference; LEAP misses the 5% bound.
    progress_systems = ('tetris', 'otif', 'reference')

    def progress_chart(title, curves, y_title, width, height, legend=alt.Legend(title=None, orient='top'),
                       x_domain=None, y_domain=None):
        # Automatic ranges end 10% past the last and highest crossover between any two curves.
        crossings = [(h, max(project_video_hours(h, curves[s]['prep_s'] / 3600, curves[s]['fps']) for s in (a, b)))
                     for a in progress_systems for b in progress_systems if a != b
                     if (h := crossover_h(curves[a], curves[b])) is not None]
        x_domain = x_domain or [0, 1.1 * max(h for h, _ in crossings)]
        hours = np.linspace(x_domain[0], x_domain[1], 301)
        top = 1.1 * max(v for _, v in crossings)
        records = [dict(hours=float(h), video_hours=float(v), system=NAMES[s]) for s in progress_systems
                   for h, v in zip(hours, project_video_hours(hours, curves[s]['prep_s'] / 3600, curves[s]['fps']))]
        return alt.Chart(alt.Data(values=records)).mark_line(strokeWidth=1.6, clip=True).encode(
            x=alt.X('hours:Q', title='Elapsed time (h)', scale=alt.Scale(domain=x_domain, nice=False, zero=False),
                    axis=alt.Axis(tickCount=4)),
            y=alt.Y('video_hours:Q', title='Video processed (h)' if y_title else None,
                    scale=alt.Scale(domain=y_domain or [0, float(top)], nice=False),
                    axis=alt.Axis(format='~s', tickCount=4)),
            color=alt.Color('system:N', scale=alt.Scale(domain=[NAMES[s] for s in progress_systems],
                            range=[COLORS[s] for s in progress_systems]),
                            legend=legend),
            order='hours:Q',
        ).properties(title=title, width=width, height=height)

    def save_progress(chart, name):
        # Altair defaults, as in the other figures, except for the plot border.
        chart.configure_view(stroke=None).save(str(OUT / name))

    panels = [progress_chart(label, runs[ds], i == 0, 100, 90) for i, (ds, label) in enumerate(DATASETS.items())]
    save_progress(alt.hconcat(*panels, spacing=12).resolve_scale(x='independent', y='independent'),
                  'revision_amortization.pdf')
    # Preparation is a plain mean; throughput averages runtime per video minute, i.e. the harmonic mean of FPS.
    average = {s: dict(prep_s=float(np.mean([runs[ds][s]['prep_s'] for ds in DATASETS])),
                       fps=900 / float(np.mean([900 / runs[ds][s]['fps'] for ds in DATASETS])))
               for s in progress_systems}
    assert abs(average['reference']['fps'] - len(DATASETS) / sum(1 / runs[ds]['reference']['fps'] for ds in DATASETS)) < 1e-9
    save_progress(progress_chart('', average, True, 110, len(DATASETS) * PREP_BAND,
                                 alt.Legend(title=None, orient='top-left', labelFontSize=10, symbolSize=60, rowPadding=0,
                                            offset=3, padding=3, fillColor='white', zindex=1),
                                 x_domain=[5, 20], y_domain=[0, 4000 / 60]), 'revision_amortization_average.pdf')
    assert all(name.isalpha() and np.isfinite(float(value.replace(',', ''))) for name, value in numbers.items())
    (OUT / 'revision_numbers.tex').write_text(
        '% Auto-generated by scripts/generate_revision_figures.py from saved measurement exports.\n'
        + ''.join(f'\\newcommand{{\\{name}}}{{\\autogen{{{value}}}}}\n' for name, value in numbers.items()))
    print('Wrote numeric macros, table rows, preparation/progress figures, and exact plot data.')


if __name__ == '__main__':
    main()
