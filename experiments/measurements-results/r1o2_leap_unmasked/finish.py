"""Finish native timing capture and CPU comparison after the two GPU queues."""
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parent
assert ROOT.name == 'r1o2_leap_unmasked'
INPUTS = json.loads((ROOT/'inputs.json').read_text())


def status(stage, **details):
    temporary = ROOT/'pipeline-status.tmp'
    temporary.write_text(json.dumps(dict(stage=stage, time=time.time(), **details), indent=2)+'\n')
    temporary.replace(ROOT/'pipeline-status.json')


try:
    deadline = time.monotonic()+7200
    while True:
        pending = [dataset for dataset in INPUTS['datasets'] if not (ROOT/f'raw/{dataset}/complete.json').exists()]
        if not pending:
            break
        status('extracting', pending=pending)
        for dataset in pending:
            alive = subprocess.run(['tmux', 'has-session', '-t', 'leap-unmasked-'+dataset], capture_output=True)
            assert alive.returncode == 0, f'Extraction queue stopped: {dataset}; see queue-{dataset}.log'
        assert time.monotonic() < deadline, 'Extraction exceeded the two-hour wait limit'
        time.sleep(20)
    # Native LEAP time files live on bombe's local disk; copy them before scoring on ace.
    (ROOT/'summary').mkdir(exist_ok=True)
    for dataset, entry in INPUTS['datasets'].items():
        record = json.loads((ROOT/f'raw/{dataset}/leap_rerun_launch.json').read_text())
        assert record['exit'] == 0 and not record['contaminated']
        source = Path(record['work_dir'])/'outputs/parsed_results'/entry['new_config']['full_name']/'speed.tsv'
        shutil.copyfile(source, ROOT/f'summary/speed-{dataset}.tsv')
    status('scoring')
    with (ROOT/'comparison.log').open('w') as log:
        subprocess.run(['ssh', 'ace', 'docker', 'exec', '-w', '/polyis', 'polyis', 'python',
                        'measurements-results/r1o2_leap_unmasked/score.py', '--workers', '2'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    status('complete', comparison='comparison/results.json')
except Exception as error:
    status('failed', error=str(error))
    raise
