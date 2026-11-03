"""Read-only hourly LEAP health checks; never launch agents or change remote work."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
INTERVAL = 3600
REMOTE = r"""
from pathlib import Path
import datetime,json,subprocess,time
scratch=Path('/data/cwkt/data/bombe-revision/scratch/r1o2_leap_matched')
results=Path('/work/cwkt/projects/tetris/measurements-results/r1o2_leap_matched')
reports=[]
for ds in ['caldot1-y05','caldot2-y05','ams-y05']:
    raw=results/'raw'/ds
    summary=results/'summary'/('results-'+ds+'.json')
    complete=(raw/'complete.json').exists() and summary.exists()
    if complete:
        complete=json.loads((raw/'complete.json').read_text()).get('scored') is True
        json.loads(summary.read_text())
    records=[]
    for p in sorted((results/'raw/prep/leap'/ds).glob('*.json')):
        if p.name.endswith('_compare.json'):continue
        d=json.loads(p.read_text())
        records.append(dict(stage=p.stem,exit=d.get('exit'),contaminated=d.get('contaminated')))
    launch=raw/'leap_rerun_launch.json'
    if launch.exists():
        d=json.loads(launch.read_text())
        records.append(dict(stage='extraction',exit=d.get('exit'),contaminated=d.get('contaminated')))
    live=subprocess.run(['tmux','has-session','-t','leap-matched-'+ds],capture_output=True).returncode==0
    logs=list((scratch/'prep/leap'/ds/'stdout').glob('*.log'))
    logs+=list((scratch/'prep/leap'/ds/'reid').glob('*/log.txt'))
    logs+=list((scratch/'runs/leap').glob(ds+'-*/stdout.log'))
    latest=max(logs,key=lambda p:p.stat().st_mtime) if logs else None
    tail=[]
    if latest:
        with latest.open('rb') as f:
            f.seek(max(0,latest.stat().st_size-4096))
            tail=f.read().decode(errors='replace').splitlines()[-5:]
    warnings=[r['stage']+' failed/contaminated' for r in records if r['exit']!=0 or r['contaminated']]
    if not complete and not live:warnings.append('queue session absent')
    if not complete and live and latest and time.time()-latest.stat().st_mtime>3600:
        warnings.append('no log progress for over an hour')
    reports.append(dict(dataset=ds,complete=complete,queue_active=live,stages=records,
        labels={split:len(list((scratch/'prep/leap'/ds/'labels').glob('*/*/'+split+'/label_*.txt'))) for split in ['train','test']},
        latest_log=str(latest) if latest else None,log_tail=tail,warnings=warnings))
print(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),datasets=reports,
    all_complete=all(r['complete'] for r in reports),automatic_repairs=False)))
"""


def probe():
    # The remote command only reads experiment records, logs, and queue status.
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'bombe', 'python3 -'],
                            input=REMOTE, text=True, capture_output=True, timeout=120, check=True)
    return json.loads(result.stdout)


def save_state(**fields):
    tmp = ROOT / 'state.tmp'
    tmp.write_text(json.dumps(dict(pid=os.getpid(), automatic_repairs=False, **fields), indent=2)+'\n')
    tmp.replace(ROOT / 'state.json')


if __name__ == '__main__':
    assert INTERVAL == 3600
    if '--once' in sys.argv:
        report=probe()
        assert {d['dataset'] for d in report['datasets']} == {'caldot1-y05','caldot2-y05','ams-y05'}
        print(json.dumps(report,indent=2))
        sys.exit(0)
    # Prevent idle sleep only while the requested local monitoring remains active.
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],
                     stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    next_tick=time.time()
    while not (ROOT/'stop').exists():
        if time.time()<next_tick:
            time.sleep(min(30,next_tick-time.time()))
            continue
        try:
            report=probe()
        except Exception as e:
            report=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),probe_error=str(e),all_complete=False)
        with (ROOT/'progress.jsonl').open('a') as log:log.write(json.dumps(report)+'\n')
        tmp=ROOT/'latest.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(ROOT/'latest.json')
        if report['all_complete']:
            save_state(status='completed');break
        next_tick=time.time()+INTERVAL
        save_state(status='waiting',next_check_utc=datetime.datetime.fromtimestamp(next_tick,datetime.timezone.utc).isoformat())
    else:
        save_state(status='stopped')
