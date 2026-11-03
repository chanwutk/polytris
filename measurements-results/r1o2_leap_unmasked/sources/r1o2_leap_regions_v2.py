"""Rerun LEAP extraction with reviewed regions, preserving matched-detector v1."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


SPEC = importlib.util.spec_from_file_location('matched_regions_base', Path(__file__).with_name('r1o2_leap_matched.py'))
MATCHED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATCHED)
BASE = MATCHED.BASE
ORIGINAL_RAW = MATCHED.RAW
ORIGINAL_PREP = MATCHED.SCRATCH / 'prep'
VERSION = BASE.TETRIS / 'measurements-results/r1o2_leap_regions_v2'
SCRATCH = BASE.ROOT / 'scratch/r1o2_leap_regions_v2'
RAW = VERSION / 'raw'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=list(MATCHED.DATASETS))
    parser.add_argument('--gpu', required=True, type=int, choices=list(BASE.GPU_CPUS))
    parser.add_argument('--run', action='store_true')
    return parser.parse_args()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    # Preserve the existing detector adapter and all mounts from the completed matched run.
    config, _ = MATCHED.configure(args.dataset)
    mounts = BASE.common_mounts()
    manifest = json.loads((VERSION / 'inputs.json').read_text())['datasets'][args.dataset]
    config_path = VERSION / 'configs' / config['config']
    assert file_sha(config_path) == manifest['config_sha256']
    assert {k for k in manifest['original_config'] if manifest['original_config'][k] != manifest['new_config'][k]} == {'stop_area'}
    assert manifest['original_config'].keys() == manifest['new_config'].keys()
    # Existing workspace links resolve through /scratch/prep; mount that preparation read-only.
    mounts += ['-v', f'{ORIGINAL_PREP}:/scratch/prep:ro',
               '-v', f'{config_path}:{BASE.LEAP_DATASET_MOUNT}/configs/{config["config"]}:ro']
    BASE.SCRATCH, BASE.RAW = SCRATCH, RAW
    BASE.common_mounts = lambda: mounts
    stage = SimpleNamespace(datasets=[args.dataset], gpu=args.gpu, run=args.run)
    if not args.run:
        BASE.leap(stage)
        return
    # Stop if any frozen input has changed since version creation.
    for path, expected in manifest['input_sha256'].items():
        assert file_sha(path) == expected, f'Input changed: {path}'
    out = RAW / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    BASE.write_json(out / 'manifest.json', manifest)
    if MATCHED.needs_extraction(out / 'leap_rerun_launch.json', False):
        BASE.leap(stage)
    launch = json.loads((out / 'leap_rerun_launch.json').read_text())
    assert launch['exit'] == 0 and not launch['contaminated'], 'Extraction failed or shared its GPU'
    # Verify the loaded configuration, library, and clip order, before exporting tracks.
    with gzip.open(out / 'leap_rerun_run.jsonl.gz', 'rt') as stream:
        actual = json.loads(next(stream))
    assert actual['config'] == manifest['new_config']
    assert actual['patterns_sha256'] == manifest['patterns_sha256']
    assert actual['video_files'] == manifest['video_files']
    parsed = Path(launch['work_dir']) / 'outputs/parsed_results' / config['full_name']
    # Use the same pattern reconstruction exporter as v1, writing only this version's output.
    command = ['docker', 'run', '--rm', '--cpus', '2', '-v', f'{SCRATCH}:/scratch',
        '-v', f'{MATCHED.ORIGINAL_SCRATCH}/inputs/scripts:/exporter:ro', '-v', f'{BASE.TETRIS}:/polyis',
        BASE.LEAP_IMAGE, 'python', '/polyis/measurements-gpu/r1o2_leap_otif_rerun_export.py',
        '--exporter', '/exporter/export-leap-outputs.py',
        '--parsed-dir', '/scratch/' + str(parsed.relative_to(SCRATCH)),
        '--out', '/polyis/' + str((out / 'leap_rerun_export.jsonl.gz').relative_to(BASE.TETRIS)),
        '--clip-negative-starts']
    subprocess.run(command, check=True)
    BASE.gpus(stage)
    # The v1 exports are immutable comparison inputs, not cache destinations.
    for path, expected in manifest['input_sha256'].items():
        assert file_sha(path) == expected, f'Input changed during extraction: {path}'
    BASE.write_json(out / 'complete.json', dict(dataset=args.dataset, gpu=args.gpu, extracted=True,
                                              exported=True, scored=False, version=VERSION.name))


if __name__ == '__main__':
    os.environ['DOCKER_HOST'] = f'unix:///run/user/{os.getuid()}/docker.sock'
    run(parse_args())
