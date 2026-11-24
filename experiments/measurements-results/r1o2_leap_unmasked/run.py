"""Reuse the verified reviewed-region launcher with image masking explicitly disabled."""
import gzip
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reviewed_launcher', ROOT/'measurements-gpu/r1o2_leap_regions_v2.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.VERSION = Path(__file__).resolve().parent
driver.SCRATCH = driver.BASE.ROOT/'scratch/r1o2_leap_unmasked'
driver.RAW = driver.VERSION/'raw'
original_leap = driver.BASE.leap


def unmasked(stage):
    stage.no_mask = True
    original_leap(stage)


if __name__ == '__main__':
    os.environ['DOCKER_HOST'] = f'unix:///run/user/{os.getuid()}/docker.sock'
    driver.BASE.leap = unmasked
    args = driver.parse_args()
    assert args.dataset in ('caldot1-y05','caldot2-y05')
    driver.run(args)
    if args.run:
        with gzip.open(driver.RAW/args.dataset/'leap_rerun_run.jsonl.gz','rt') as stream:
            actual = json.loads(next(stream))
        assert actual['mask_enabled'] is False and '--use_mask' in actual['leap_argv']
