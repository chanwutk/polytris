"""Use the existing renderer with LEAP's best verified reconstruction.

All output goes into the paper build directory. Cached experiments are read only.
"""
import json
import shutil
import sys
from pathlib import Path

import duckdb


def main():
    sys.path.insert(0, '/polyis')
    from evaluation import p201_compare_pareto as charts
    from evaluation import p203_compare_stats as stats
    from measurements.common import DATASET_LABELS

    root = Path(__file__).resolve().parents[1]
    out = root / 'build/revised-hota'
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((root / 'scripts/data/revision_evidence/leap_rerun.json').read_text())
    rows = [(dataset, variant, report['datasets'][label]['hota'][variant]['HOTA'])
            for dataset, label in DATASET_LABELS.items() for variant in ('raw', 'linear', 'pattern')]
    con = duckdb.connect()
    con.execute('CREATE TABLE variants(dataset VARCHAR, variant VARCHAR, hota DOUBLE)')
    con.executemany('INSERT INTO variants VALUES (?,?,?)', rows)
    best = dict(con.execute('SELECT dataset, max(hota) FROM variants GROUP BY dataset').fetchall())
    original_loader = charts.load_sota_tradeoff_data

    def load_prior(datasets, system):
        frame = original_loader(datasets, system)
        if system == 'leap' and not frame.empty:
            frame = frame.copy()
            frame['HOTA_HOTA'] = frame['dataset'].map(best)
            assert frame['HOTA_HOTA'].notna().all()
            for dataset, label in DATASET_LABELS.items():
                if 'extraction' in report['datasets'][label]:
                    extraction = report['datasets'][label]['extraction']
                    frame.loc[frame['dataset'] == dataset, 'time'] = extraction['compute_s']
        return frame

    charts.load_sota_tradeoff_data = load_prior
    stats.load_sota_tradeoff_data = load_prior
    tetris, reference = stats.load_polytris_and_naive_test_data(stats.DATASETS)
    prior = stats.load_prior_test_data(stats.DATASETS, stats.build_frame_count_lookup(tetris, reference))
    for dataset, label in DATASET_LABELS.items():
        if 'extraction' in report['datasets'][label]:
            row = prior['LEAP'].loc[prior['LEAP']['dataset'] == dataset].iloc[0]
            assert row['frame_count'] == report['datasets'][label]['extraction']['frames']
    (out / 'prior_points.json').write_text(json.dumps({
        system: json.loads(frame.to_json(orient='records')) for system, frame in prior.items()}, indent=2))
    pareto = stats.filter_pareto_by_dataset(tetris)
    summary, details = stats.build_threshold_reports(stats.DATASETS, pareto, reference, prior, stats.DEFAULT_THRESHOLDS)
    dominance = stats.build_dominance_detail_table(stats.DATASETS, pareto, stats.build_combined_prior_pareto(prior))
    global_summary = stats.build_dominance_global_summary(dominance)
    (out / 'summary.json').write_text(summary.to_json(orient='records', indent=2))
    (out / 'dominance.json').write_text(dominance.to_json(orient='records', indent=2))
    (out / 'dominance_global.json').write_text(global_summary.to_json(orient='records', indent=2))
    stats.save_tex_macros(summary, dominance, str(out / 'comparison_macros.tex'))
    reachable = dominance.dropna(subset=['hota_delta'])
    unreachable = dominance[dominance['hota_delta'].isna()]
    nonzero = unreachable[unreachable['prior_hota'] > 0]
    assert len(nonzero) == 1 and (reachable['hota_delta'] > 0).all()
    last = nonzero.iloc[0]
    fastest = pareto[pareto['dataset'] == last['dataset']].sort_values('throughput_fps').iloc[-1]
    best_row = reachable.loc[reachable['hota_delta'].idxmax()]
    macros = {
        'compareMaxHotaImprovementDataset': DATASET_LABELS[best_row['dataset']],
        'compareMatchedTputPriorConfigCount': str(len(dominance)),
        'compareMatchedTputReachableCount': str(len(reachable)),
        'compareMatchedTputUnreachableCount': str(len(unreachable)),
        'compareMatchedTputUnreachableZeroHotaCount': str((unreachable['prior_hota'] == 0).sum()),
        'compareMatchedTputUnreachableRestDataset': DATASET_LABELS[last['dataset']],
        'compareMatchedTputUnreachableRestSpeedup': f"{last['prior_throughput_fps'] / fastest['throughput_fps']:.1f}",
        'compareMatchedTputUnreachableRestHotaDrop': f"{fastest['HOTA_HOTA'] - last['prior_hota']:.2f}",
        'compareMatchedTputMedianHotaGain': f"{reachable['hota_delta'].median():.2f}",
        'compareMatchedTputPerDatasetMaxGainMin': f"{reachable.groupby('dataset')['hota_delta'].max().min():.2f}",
    }
    with (out / 'comparison_macros.tex').open('a') as f:
        for name, value in macros.items():
            f.write(f'\\newcommand{{\\{name}}}{{\\autogen{{{value}}}}}\n')
    for loss, detail in details.items():
        (out / f'details-{loss}.json').write_text(detail.to_json(orient='records', indent=2))
    # Redirect renderer output only; no canonical cache directory is written.
    charts.cache.summary = lambda *parts: out.joinpath(*parts)
    charts.PAPER_FIGURES_GENERATED_DIR = str(out / 'paper')
    charts.PRESENTATION_ASSETS_DIR = str(out / 'presentation')
    def save_chart(chart, directory, name):
        # Shorten the comparison rows without shrinking the main panels or fonts.
        if name.startswith('hota_hota_summary_one_row'):
            for row, comparison in zip(chart.vconcat[1:], ('(Ours/Other)', '(Ours-Other)')):
                row.spec.height = 70
                # Both line and point layers carry the axis title.
                for layer in row.spec.layer[:2]:
                    layer.encoding.y['title'][1] = comparison
                    if comparison == '(Ours/Other)':
                        layer.encoding.y['scale'] = charts.alt.Scale(type='log')
                        layer.encoding.y['axis'] = charts.alt.Axis(values=[0.1, 1, 10, 100], format='~g')
        chart.save(str(Path(directory) / f'{name}.pdf'))
        if name == 'hota_hota_summary_one_row':
            chart.save(str(Path(directory) / f'{name}.vl.json'))

    charts.save_chart = save_chart
    charts.visualize_all_datasets_tradeoffs_pareto(charts.DATASETS, log_scale=True)
    shutil.copyfile(out / 'paper/hota_hota_summary_one_row.pdf',
                    root / 'figures/generated/revision_hota_summary.pdf')
    print(summary.to_string(index=False))
    print(global_summary.to_string(index=False))


if __name__ == '__main__':
    main()
