"""Regenerate paper figures with LEAP's best verified reconstruction and runtime.

Cached experiments are read only; detailed reports go into the paper build directory.
Both the original and alternative Figure 8 are exported by default.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alternative', action='store_true',
                        help='Render the separate compact Figure 8 alternative only.')
    return parser.parse_args()


def alternative_summary(chart, alt):
    original = chart.to_dict()
    # Reclaim the row-title area while keeping the three rows aligned.
    for index, row in enumerate(chart.vconcat):
        row.title = alt.Undefined
        row.spec.width = 164
        row.facet = alt.FacetMapping(column=alt.Column(
            'dataset_display:N', sort=row.facet['sort'], title=None, header=None,
        ))
        row.columns = alt.Undefined
        # Avoid rounding independent domains to the same full decades.
        row.resolve['scale']['x'] = 'independent'
        for layer in row.spec.layer[:2]:
            layer.encoding.x['scale'] = alt.Scale(
                type='linear' if index == 1 else 'log', nice=False, zero=False, padding=4,
            )
            layer.encoding.x['axis'] = alt.Axis(
                labelOverlap='greedy', labelBound=True, labelFlush=True,
            )

    # Restore the original panel height with dataset names inside the top-right corner.
    chart.vconcat[0].spec.height = 122
    settings = [
        ('HOTA_HOTA', 'linear', "format(datum.HOTA_HOTA, '.1%')"),
        ('speedup_ratio', 'log', "format(datum.speedup_ratio, '.1f') + '×'"),
        ('accuracy_gain', 'linear', "format(datum.accuracy_gain, '+.2f')"),
    ]
    with duckdb.connect(config={'threads': 4}) as con:
        for index, (field, scale_type, label) in enumerate(settings):
            panel = chart.vconcat[index].spec
            data = original['datasets'][original['vconcat'][index]['data']['name']]
            if index == 0:
                low = 0.5
                high = con.execute("""
                    SELECT max(json_extract(value, '$.HOTA_HOTA')::DOUBLE)
                    FROM json_each(?)
                """, [json.dumps(data)]).fetchone()[0]
                assert low < high <= 1
                axis = alt.Axis(
                    format='.0%', values=[tick for tick in (0.5, 0.6, 0.7, 0.8, 0.9) if tick < high] + [high],
                    labelExpr=f"format(datum.value, datum.value === {high!r} ? '.1%' : '.0%')",
                    labelBaseline=alt.ExprRef(expr=f"datum.value === {high!r} ? 'top' : 'middle'"),
                )
            else:
                # Both comparison rows retain a shared y-axis based only on OTIF.
                low, high = con.execute("""
                    SELECT min(json_extract(value, ?)::DOUBLE),
                           max(json_extract(value, ?)::DOUBLE)
                    FROM json_each(?)
                    WHERE json_extract_string(value, '$.system') = 'OTIF'
                """, [f'$.{field}', f'$.{field}', json.dumps(data)]).fetchone()
                assert low is not None and low < high
                assert scale_type != 'log' or low > 0
                axis = alt.Axis(format='~g' if index == 1 else '.1f', tickCount=5)
            for layer in panel.layer:
                layer.mark.clip = True
                layer.encoding.y['scale'] = alt.Scale(
                    type=scale_type, domain=[low, high], nice=False, zero=False,
                )
                layer.encoding.y['axis'] = axis

            # Pixel offsets keep arrows inside the axes without expanding their domains.
            for direction, bound, operator in [('down', low, '<'), ('up', high, '>')]:
                tip, tail = ((panel.height - 4, panel.height - 17)
                             if direction == 'down' else (4, 17))
                outside = alt.Chart().transform_filter(
                    f"datum.system === 'LEAP' && datum.{field} {operator} {bound!r}",
                ).transform_calculate(leap_label=label).encode(
                    x=panel.layer[0].encoding.x,
                    color=alt.Color('system:N', scale=panel.layer[0].encoding.color['scale'], legend=None),
                )
                text = outside.mark_text(fontSize=11,
                                         align='right' if index == 0 else ('left' if index == 2 else 'center'),
                                         dx=-1 if index == 0 else (1 if index == 2 else 0),
                                         baseline='bottom' if direction == 'down' else 'top',
                                         dy=3 if index == 0 else (-3 if index == 2 else (-1 if direction == 'down' else 1))).encode(
                    y=alt.value(tail), text=alt.Text('leap_label:N'),
                )
                # Draw the outline beneath the pink glyphs to preserve their fill.
                outlined_text = text.copy(deep=True)
                outlined_text.mark['stroke'] = 'white'
                outlined_text.mark['strokeWidth'] = 2
                panel.layer.extend([
                    outside.mark_rule(strokeWidth=1.5, opacity=0.6).encode(
                        y=alt.value(tail), y2=alt.value(tip),
                    ),
                    outside.mark_point(shape=f'triangle-{direction}', size=55, filled=True,
                                       opacity=0.6).encode(y=alt.value(tip)),
                    outlined_text,
                    text,
                ])
    top = chart.vconcat[0].spec
    top.layer.append(alt.Chart().transform_aggregate(groupby=['dataset_display']).mark_text(
        align='right', baseline='top', fontSize=13, color='black',
    ).encode(
        x=alt.value(top.width - 4), y=alt.value(4), text=alt.Text('dataset_display:N'),
    ))
    # Validate layout changes without changing any source values.
    spec = chart.to_dict()
    assert len(spec['vconcat']) == 3
    assert spec['datasets'] == original['datasets']
    assert all('title' not in row for row in spec['vconcat'])
    assert all(row['spec']['width'] == 164 for row in spec['vconcat'])
    assert all(row['facet']['column']['header'] is None for row in spec['vconcat'])
    assert all(row['resolve']['scale']['x'] == 'independent' for row in spec['vconcat'])
    return chart


def main():
    args = parse_args()
    sys.path.insert(0, '/polyis')
    from evaluation import p201_compare_pareto as charts
    from evaluation import p203_compare_stats as stats
    from measurements.common import DATASET_LABELS

    root = Path(__file__).resolve().parents[1]
    out = root / ('build/figure8-alternative' if args.alternative else 'build/revised-hota')
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
        if args.alternative and name != 'hota_hota_summary_one_row':
            return
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
        if args.alternative:
            chart = alternative_summary(chart, charts.alt)
        chart.save(str(Path(directory) / f'{name}.pdf'))
        if name == 'hota_hota_summary_one_row':
            chart.save(str(Path(directory) / f'{name}.vl.json'))
            if not args.alternative:
                alternative_summary(chart.copy(deep=True), charts.alt).save(
                    str(Path(directory) / f'{name}_alternative.pdf'))

    charts.save_chart = save_chart
    charts.visualize_all_datasets_tradeoffs_pareto(charts.DATASETS, log_scale=True)
    if args.alternative:
        shutil.copyfile(out / 'paper/hota_hota_summary_one_row.pdf',
                        root / 'figures/generated/revision_hota_summary_alternative.pdf')
        return
    shutil.copyfile(out / 'paper/hota_hota_summary_one_row.pdf',
                    root / 'figures/generated/revision_hota_summary.pdf')
    shutil.copyfile(out / '101_compare_pareto/hota_hota_summary_one_row_alternative.pdf',
                    root / 'figures/generated/revision_hota_summary_alternative.pdf')
    for name in ('hota_hota_summary_one_row', 'hota_hota_summary_one_row_no_ablation'):
        shutil.copyfile(out / 'paper' / f'{name}.pdf', root / 'figures/generated' / f'{name}.pdf')
    for name in ('hota_hota_runtime_pareto_comparison', 'hota_hota_throughput_pareto_comparison',
                 'hota_hota_speedup_at_accuracy'):
        shutil.copyfile(out / '101_compare_pareto' / f'{name}.pdf', root / 'figures' / f'{name}.pdf')
    shutil.copyfile(out / '101_compare_pareto/hota_hota_accuracy_gain_at_throughput.pdf',
                    root / 'figures/hota_hota_accuracy_gain_at_runtime.pdf')
    print(summary.to_string(index=False))
    print(global_summary.to_string(index=False))


if __name__ == '__main__':
    main()
