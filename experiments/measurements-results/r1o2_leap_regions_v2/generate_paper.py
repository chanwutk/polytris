"""Export LEAP/LEAP2 numeric macros and table rows from saved comparison results."""
import argparse
from pathlib import Path

import duckdb


TOKENS = {'CalDoT1': 'CalDoTOne', 'CalDoT2': 'CalDoTTwo', 'Amsterdam': 'Amsterdam'}
METRICS = {'hota': 'Hota', 'fps': 'Fps', 'selection': 'Selection', 'count_mape': 'CountMape', 'topten': 'TopTen'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    con = duckdb.connect()
    con.execute('''CREATE TABLE records AS
        WITH versions AS (
            SELECT d.key dataset,v.role,v.suffix,v.label,
                   json_extract(d.value,'$.versions.'||v.role) stats,
                   json_extract(i.value,'$.reviewed_pattern_eligibility') eligibility
            FROM json_each(?::JSON,'$.datasets') d
            JOIN json_each(?::JSON,'$.datasets') i ON i.key=json_extract_string(d.value,'$.dataset')
            CROSS JOIN (VALUES ('original','Original','LEAP'),('regions_v2','Updated','LEAP2')) v(role,suffix,label)
        ) SELECT dataset,role,suffix,label,
            json_extract(stats,'$.best_hota')::DOUBLE*100 hota,
            json_extract(stats,'$.timing.fps')::DOUBLE fps,
            json_extract(stats,'$.native_queries.selection.f1')::DOUBLE*100 selection,
            json_extract(stats,'$.native_queries.aggregation.mape')::DOUBLE*100 count_mape,
            json_extract(stats,'$.native_queries.topk')::DOUBLE*100 topten,
            CASE WHEN role='original' THEN json_extract(eligibility,'$.eligible_before')
                 ELSE json_extract(eligibility,'$.eligible_after') END::INT eligible,
            json_extract(eligibility,'$.patterns')::INT patterns
        FROM versions''', [args.results.read_text(), args.inputs.read_text()])
    assert con.execute('SELECT count(*),count(DISTINCT dataset) FROM records').fetchone() == (6, 3)
    # Bold all ties at the displayed precision, using lower-is-better only for MAPE.
    metrics = con.execute('''WITH long_metrics AS (
        UNPIVOT records ON hota,fps,selection,count_mape,topten INTO NAME metric VALUE value
        ), shown AS (
        SELECT dataset,suffix,metric,value,CASE WHEN value>=100 THEN round(value) ELSE round(value,1) END displayed
        FROM long_metrics)
        SELECT *,displayed=CASE WHEN metric='count_mape' THEN min(displayed) OVER (PARTITION BY dataset,metric)
            ELSE max(displayed) OVER (PARTITION BY dataset,metric) END best
        FROM shown ORDER BY dataset,suffix,metric''').fetchall()
    macros, cells = [], {}
    for dataset, suffix, metric, value, displayed, best in metrics:
        name = 'revisionLeapTwo'+TOKENS[dataset]+suffix+METRICS[metric]
        formatted = f'{displayed:.0f}' if value >= 100 else f'{displayed:.1f}'
        macros.append(f'\\newcommand{{\\{name}}}{{\\autogen{{{formatted}}}}}')
        reference = '\\'+name+'{}'
        cells[dataset, suffix, metric] = '\\textbf{'+reference+'}' if best else reference
    rows = []
    for dataset, token in TOKENS.items():
        records = con.execute("SELECT suffix,label,eligible,patterns FROM records WHERE dataset=? ORDER BY role", [dataset]).fetchall()
        assert len(records) == 2 and records[0][0] == 'Original'
        total_name = 'revisionLeapTwo'+token+'Patterns'
        macros.append(f'\\newcommand{{\\{total_name}}}{{\\autogen{{{records[0][3]}}}}}')
        for suffix, label, eligible, total in records:
            eligible_name = 'revisionLeapTwo'+token+suffix+'Eligible'
            macros.append(f'\\newcommand{{\\{eligible_name}}}{{\\autogen{{{eligible}}}}}')
            row = [dataset if suffix == 'Original' else '', label,
                   '\\'+eligible_name+'{}/\\'+total_name+'{}']
            row += [cells[dataset, suffix, metric] for metric in METRICS]
            rows.append(' & '.join(row)+r' \\')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'leap2_numbers.tex').write_text('% Generated from the saved LEAP2 comparison; numeric macros only.\n'+'\n'.join(macros)+'\n')
    (args.output_dir/'leap2_comparison_rows.tex').write_text('% Generated table body only.\n'+'\n'.join(rows)+'\n')
    print(f'Wrote {len(macros)} numeric macros and {len(rows)} comparison rows.')


if __name__ == '__main__':
    main()
