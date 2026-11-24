"""Rebuild the B3D parts of the paper's evidence snapshots after re-scoring on human ground truth.

Earlier sessions spliced B3D entries into three files by hand; this reproduces those splices:
- revision_evidence/leap_rerun.json: B3D entries come from measurements/r1o2_leap_otif_rerun.py output.
  Sections that do not depend on ground truth must be unchanged.
- revision_evidence/extraction_configs.json: each B3D list is measurements.common.system_configs(), with the
  LEAP row's HOTA replaced by the best of LEAP's raw/linear/pattern reconstructions from leap_rerun.json.
- measurements-results/r1o2_leap_sampling_unmasked/all_datasets.json: B3D entries come from
  measurements/r1o2_leap_sampling_diagnostics.py output.
Then every evidence file whose sources.json source is a plain measurements-results file is re-copied, and
sources.json gets the new sha256 of every changed evidence file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


# Paper labels of the B3D datasets.
B3D = {'jnc0': 'B3D1', 'jnc2': 'B3D2', 'jnc6': 'B3D3', 'jnc7': 'B3D4'}
# leap_rerun.json sections that do not depend on ground truth.
GT_INDEPENDENT = ('agreement', 'classes', 'clips', 'dataset', 'timing', 'video_identity')
# LEAP reconstructions whose best HOTA represents LEAP.
VARIANTS = ('raw', 'linear', 'pattern')
# Prefix of sources.json entries that are byte copies of a measurement output.
COPY_PREFIX = '../tetris/measurements-results/'


def parse_args():
    parser = argparse.ArgumentParser(description='Rebuild B3D evidence after re-scoring on human ground truth')
    parser.add_argument('--evidence', default='/polytris-paper/scripts/data/revision_evidence')
    parser.add_argument('--results', default='/polyis/measurements-results')
    parser.add_argument('--leap-rerun', required=True)
    parser.add_argument('--sampling', required=True)
    return parser.parse_args()


def read(path: Path):
    # Load a JSON file.
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    # Hash a file's bytes.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def splice_leap_rerun(old: dict, rerun: dict) -> dict:
    """leap_rerun.json with its B3D entries taken from the rerun, refusing any change outside ground-truth sections."""
    new = json.loads(json.dumps(old))
    for label in B3D.values():
        entry = rerun['datasets'][label]
        # The rerun must cover exactly the same sections.
        if sorted(entry) != sorted(old['datasets'][label]):
            raise ValueError(f'{label}: rerun sections {sorted(entry)} differ from {sorted(old["datasets"][label])}')
        # Only ground-truth sections may move.
        moved = [k for k in GT_INDEPENDENT if entry[k] != old['datasets'][label][k]]
        if moved:
            raise ValueError(f'{label}: ground-truth-independent sections changed: {moved}')
        new['datasets'][label] = entry
    return new


def best_leap_hota(entry: dict) -> float:
    # LEAP is represented by its best reconstruction, as in Fig. 8.
    return max(entry['hota'][v]['HOTA'] for v in VARIANTS)


def rebuild_extraction_configs(old: dict, configs: dict[str, list[dict]], leap_rerun: dict) -> dict:
    """extraction_configs.json with B3D lists rebuilt from system_configs rows."""
    new = dict(old)
    for dataset, label in B3D.items():
        rows = [dict(r) for r in configs[dataset]]
        leap = [r for r in rows if r['system'] == 'leap']
        if len(leap) != 1:
            raise ValueError(f'{dataset}: expected one LEAP row, found {len(leap)}')
        leap[0]['hota'] = best_leap_hota(leap_rerun['datasets'][label])
        # Every system and configuration must survive the rebuild.
        if sorted((r['system'], r['config']) for r in rows) != sorted((r['system'], r['config']) for r in old[dataset]):
            raise ValueError(f'{dataset}: configurations changed')
        new[dataset] = rows
    return new


def splice_sampling(old: dict, sampling: dict) -> dict:
    # all_datasets.json with B3D entries from the new diagnostics, other datasets untouched.
    new = json.loads(json.dumps(old))
    for label in B3D.values():
        new['datasets'][label] = sampling['datasets'][label]
    return new


def write(path: Path, value, indent) -> None:
    # Write JSON with the formatting the file already uses.
    path.write_text(json.dumps(value, indent=indent) + '\n')


def refresh_sources(evidence: Path, results: Path) -> list[str]:
    """Re-copy plain measurement-output snapshots and record every evidence file's sha256; returns changed files."""
    sources = read(evidence / 'sources.json')
    changed = []
    for entry in sources:
        target = evidence / entry['file']
        # Byte copies of a measurement output are refreshed from that output.
        if entry['source'].startswith(COPY_PREFIX):
            shutil.copyfile(results / entry['source'][len(COPY_PREFIX):], target)
        digest = sha256(target)
        if digest != entry['sha256']:
            entry['sha256'] = digest
            changed.append(entry['file'])
    write(evidence / 'sources.json', sources, 2)
    return changed


def main():
    args = parse_args()
    from measurements.common import system_configs
    evidence, results = Path(args.evidence), Path(args.results)
    # Validate every splice before writing anything.
    leap = splice_leap_rerun(read(evidence / 'leap_rerun.json'), read(Path(args.leap_rerun)))
    configs = rebuild_extraction_configs(read(evidence / 'extraction_configs.json'),
                                         {d: system_configs(d) for d in B3D}, leap)
    sampling_path = results / 'r1o2_leap_sampling_unmasked' / 'all_datasets.json'
    sampling = splice_sampling(read(sampling_path), read(Path(args.sampling)))
    write(evidence / 'leap_rerun.json', leap, 2)
    write(evidence / 'extraction_configs.json', configs, None)
    write(sampling_path, sampling, 2)
    for name in refresh_sources(evidence, results):
        print(f'updated {name}')


if __name__ == '__main__':
    main()
