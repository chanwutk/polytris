"""Inspection-only Altair figures from the saved LEAP sampling diagnostics."""
import argparse
import json
from pathlib import Path

import altair as alt
import duckdb

from measurements.common import DATASET_LABELS, query_dicts


ROOT = Path(__file__).resolve().parent.parent
ORDER = ['CalDoT1', 'CalDoT2', 'Amsterdam', 'B3D1', 'B3D2', 'B3D3', 'B3D4']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=ROOT / 'measurements-results/r1o2_leap_sampling_diagnostics')
    return parser.parse_args()


def main():
    args = parse_args()
    report = json.loads((args.results_dir / 'results.json').read_text())
    assert set(report['datasets']) == set(ORDER), 'Wait for every dataset to finish before plotting.'
    alt.data_transformers.disable_max_rows()
    con = duckdb.connect()
    # Keep the compact numeric overview reproducible from the same JSON used by the charts.
    con.execute("""COPY (SELECT key dataset,
        json_extract(value,'$.patterns.eligible_patterns')::INT eligible,
        json_extract(value,'$.patterns.patterns')::INT patterns,
        round(json_extract(value,'$.association.same_id_share')::DOUBLE*100,1) continuity_pct,
        round(json_extract(value,'$.oracle_interpolation.iou_ge_half_share')::DOUBLE*100,1) oracle_interp_pct,
        round(json_extract(value,'$.duration_uncensored_at_least_one_second.p10_p50_p90_s[0]')::DOUBLE,2) span_p10_s,
        round(json_extract(value,'$.duration_uncensored_at_least_one_second.p10_p50_p90_s[1]')::DOUBLE,2) span_p50_s,
        round(json_extract(value,'$.duration_uncensored_at_least_one_second.p10_p50_p90_s[2]')::DOUBLE,2) span_p90_s,
        json_extract(value,'$.provenance.excluded_out_of_clip_gt_boxes')::INT excluded_gt_boxes
        FROM json_each((SELECT datasets FROM read_json(?))) ORDER BY key)
        TO ? (HEADER,DELIMITER ',')""", [str(args.results_dir / 'results.json'), str(args.results_dir / 'summary.csv')])
    # Preserve the dataset identity when reading the compact per-track exports.
    paths = [(str(args.results_dir / f'{ds}-tracks.parquet'), label) for ds, label in DATASET_LABELS.items()]
    con.execute('CREATE TABLE inputs(filename VARCHAR,label VARCHAR)')
    con.executemany('INSERT INTO inputs VALUES (?,?)', paths)
    cdf_data = query_dicts(con, """WITH counts AS (
        SELECT label,(e-s+1)/15.0 AS seconds,count(*) n
        FROM read_parquet(?,filename=true) JOIN inputs USING(filename)
        WHERE NOT censored AND e-s+1>=15 GROUP BY label,seconds)
      SELECT label,seconds,sum(n) OVER(PARTITION BY label ORDER BY seconds)/sum(n) OVER(PARTITION BY label) fraction
      FROM counts ORDER BY label,seconds""", [[p for p, _ in paths]])
    duration = alt.Chart(alt.Data(values=cdf_data)).mark_line(interpolate='step-after').encode(
        x=alt.X('seconds:Q', title='Annotated time in view (seconds)', scale=alt.Scale(type='log'),
                axis=alt.Axis(values=[1, 2, 5, 10, 20, 40], format='~g')),
        y=alt.Y('fraction:Q', title='Fraction of tracks', axis=alt.Axis(format='.0%')),
        color=alt.Color('label:N', title='Dataset', sort=ORDER),
        order='seconds:Q', tooltip=['label:N', 'seconds:Q', alt.Tooltip('fraction:Q', format='.1%')]
    ).properties(width=650, height=250, title=alt.Title(
        'Variation in annotated time in view', subtitle=[
            'Complete tracks lasting at least one second; excludes tracks touching a clip boundary.',
            'Current automatic annotations; these are annotated spans, not verified physical dwell times.']))
    # Derive all displayed percentages from the result exports.
    rows = query_dicts(con, """SELECT key AS label,
        json_extract(value,'$.association.same_id_share')::DOUBLE continuity,
        json_extract(value,'$.oracle_interpolation.iou_ge_half_share')::DOUBLE interpolation,
        json_extract(value,'$.predicted_objects.observed_once')::DOUBLE/
          json_extract(value,'$.predicted_objects.objects')::DOUBLE single
        FROM json_each((SELECT datasets FROM read_json(?)))""", [str(args.results_dir / 'results.json')])
    panels = []
    for field, title in [('continuity', 'ID continuity'), ('interpolation', 'Ideal linear reconstruction'),
                         ('single', 'One-observation IDs')]:
        base = alt.Chart(alt.Data(values=rows)).encode(
            y=alt.Y('label:N', sort=ORDER, title=None),
            x=alt.X(field + ':Q', title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')))
        panels.append((base.mark_bar(color='#167d8d') + base.mark_text(align='left', dx=3).encode(
            text=alt.Text(field + ':Q', format='.1%'))).properties(width=200, height=210, title=title))
    comparison = alt.hconcat(*panels, spacing=35).properties(title=alt.Title(
        'Different failure modes in the saved runs', subtitle=[
            'ID continuity: matched consecutive-frame pairs retaining one LEAP ID.',
            'Ideal reconstruction: bracketed GT boxes with interpolated GT-box IoU >= 0.5.',
            'One-observation IDs: predicted LEAP IDs with one recorded detection.']))
    # Show all loaded patterns, including paths rejected by the configured endpoint regions.
    pattern_panels = {}
    inverse = {label: ds for ds, label in DATASET_LABELS.items()}
    for label in ORDER:
        ds = inverse[label]
        record = report['datasets'][label]
        cfg = record['provenance']['config']
        points = query_dicts(con, """SELECT *,start_region IS NOT NULL AND end_region IS NOT NULL eligible
          FROM read_parquet(?) ORDER BY pattern,point""", [str(args.results_dir / f'{ds}-patterns.parquet')])
        regions = [dict(left=r[0], top=r[1], right=r[2], bottom=r[3]) for r in cfg['stop_area']]
        base = alt.Chart(alt.Data(values=points)).encode(
            x=alt.X('x:Q', title='X (px)', scale=alt.Scale(domain=[0, cfg['w']], nice=False), axis=alt.Axis(tickCount=3)),
            y=alt.Y('y:Q', title='Y (px)', scale=alt.Scale(domain=[0, cfg['h']], reverse=True, nice=False),
                    axis=alt.Axis(tickCount=3)))
        boxes = alt.Chart(alt.Data(values=regions)).mark_rect(
            fill='#d94841', fillOpacity=0.12, stroke='#d94841', strokeWidth=1).encode(
                x='left:Q', x2='right:Q', y='top:Q', y2='bottom:Q')
        lines = base.mark_line(strokeWidth=1.3).encode(
            detail='pattern:N', order='point:Q',
            color=alt.condition('datum.eligible', alt.value('#15806b'), alt.value('#b0b0b0')),
            tooltip=['pattern:N', 'eligible:N', 'point:Q'])
        ends = record['pattern_endpoints']
        starts = alt.Chart(alt.Data(values=ends)).mark_point(shape='circle', size=35, color='black').encode(x='sx:Q', y='sy:Q')
        finishes = alt.Chart(alt.Data(values=ends)).mark_point(shape='cross', size=35, color='black').encode(x='ex:Q', y='ey:Q')
        width = 275 if not label.startswith('B3D') else 205
        pattern_panels[label] = (boxes + lines + starts + finishes).properties(
            width=width, height=width*cfg['h']/cfg['w'],
            title=f"{label}: {record['patterns']['eligible_patterns']}/{record['patterns']['patterns']} eligible")
    patterns = alt.vconcat(
        alt.hconcat(*(pattern_panels[x] for x in ORDER[:3]), spacing=20),
        alt.hconcat(*(pattern_panels[x] for x in ORDER[3:]), spacing=20), spacing=35).properties(
            title=alt.Title('Configured entry/exit regions reject many learned paths', subtitle=[
                'Red boxes: configured regions. Teal paths: eligible. Gray paths: rejected.',
                'Circle: first pattern point. Cross: last point. Both endpoints must lie in regions.']))
    # Save inspection artifacts separately; no manuscript figures or reported values are changed.
    for name, chart in [('time_in_view', duration), ('failure_modes', comparison), ('pattern_regions', patterns)]:
        chart.configure_axis(labelFontSize=10, titleFontSize=11).save(args.results_dir / f'{name}.pdf')
        chart.save(args.results_dir / f'{name}.html')
        print('wrote', args.results_dir / f'{name}.pdf', flush=True)
    con.close()


if __name__ == '__main__':
    main()
