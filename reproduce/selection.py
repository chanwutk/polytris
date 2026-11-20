"""Transfer scored validation configurations to the staged test pipeline."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('dataset')
    export.add_argument('output', type=Path)
    select = commands.add_parser('select')
    select.add_argument('dataset')
    select.add_argument('input', type=Path)
    select.add_argument('output', type=Path)
    select.add_argument('variant_id', nargs='?')
    return parser.parse_args()


def select_ids(report, dataset, variant_id, grid):
    if report['dataset'] != dataset or report['split'] != 'valid':
        raise ValueError(f'Expected validation results for {dataset}')

    rows = report['configurations']
    by_id = {row['variant_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate validation configuration IDs')
    selected = [variant_id] if variant_id is not None else [
        row['variant_id'] for row in rows if row['pareto'] is True
    ]
    if not selected:
        raise ValueError('No validation configurations selected for testing')
    for chosen in selected:
        if chosen not in by_id:
            raise ValueError(f'Configuration {chosen} is absent from the validation results')
        if chosen not in grid or by_id[chosen]['config'] != grid[chosen]:
            raise ValueError(f'Configuration {chosen} does not match the current sweep grid')
    return selected


def export_results(dataset, output):
    from evaluation.manifests import build_polytris_variant_manifest
    from polyis.pareto import PARETO_PARAM_COLS, load_pareto_params
    from polyis.utilities import load_tradeoff_data

    tradeoffs = load_tradeoff_data(dataset)
    valid = tradeoffs.loc[(tradeoffs['videoset'] == 'valid') & (tradeoffs['variant'] == 'polytris')]
    fields = ['throughput_fps', 'HOTA_HOTA']
    if valid.empty or valid[fields].isna().any().any() or valid['variant_id'].duplicated().any():
        raise ValueError(f'Invalid or incomplete validation results for {dataset}')
    expected = set(build_polytris_variant_manifest()['variant_id'])
    if (actual := set(valid['variant_id'])) != expected:
        raise ValueError(f'Validation results for {dataset} differ from the sweep grid '
                         f'({len(expected - actual)} missing, {len(actual - expected)} unexpected)')

    pareto = set(load_pareto_params(dataset)['variant_id'])
    if not pareto or not pareto.issubset(actual):
        raise ValueError(f'Validation Pareto set does not match the sweep for {dataset}')

    rows = json.loads(valid[[*PARETO_PARAM_COLS, *fields]].to_json(orient='records', double_precision=15))
    configurations = [
        {
            'variant_id': row['variant_id'],
            'config': {field: row[field] for field in PARETO_PARAM_COLS[1:]},
            'throughput_fps': row['throughput_fps'],
            'hota': row['HOTA_HOTA'],
            'pareto': row['variant_id'] in pareto,
        }
        for row in rows
    ]
    report = {'dataset': dataset, 'split': 'valid', 'configurations': configurations}
    with TemporaryDirectory(dir=output.parent) as scratch:
        temporary = Path(scratch) / output.name
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, output)


def main(args):
    if args.command == 'export':
        export_results(args.dataset, args.output)
        return

    from evaluation.manifests import build_polytris_variant_manifest
    from polyis.pareto import PARETO_PARAM_COLS

    report = json.loads(args.input.read_text())
    manifest = build_polytris_variant_manifest().set_index('variant_id')
    grid = json.loads(manifest[PARETO_PARAM_COLS[1:]].to_json(orient='index', double_precision=15))
    selected = select_ids(report, args.dataset, args.variant_id, grid)
    manifest.loc[selected].reset_index()[PARETO_PARAM_COLS].to_csv(args.output, index=False)


if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f'{args.command}: {error}') from None
