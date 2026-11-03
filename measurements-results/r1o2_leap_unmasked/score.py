"""Score unmasked LEAP against the archived runs with identical reviewed regions."""
import argparse
import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys

import duckdb
import numpy as np

VERSION = Path(__file__).resolve().parent
ROOT = VERSION.parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.manifests import list_split_videos
from measurements import r1o2_leap_native_queries as queries
from measurements.common import DATASET_LABELS, output_path, write_json

PREVIOUS = VERSION.parent/'r1o2_leap_regions_v2'
spec=importlib.util.spec_from_file_location('reviewed_scorer',PREVIOUS/'score.py')
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.VERSION=VERSION
DATASETS=('caldot1-y05','caldot2-y05')


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    return parser.parse_args()


def main():
    args=parse_args()
    evaluator,evaluator_sha=helper.original_evaluator()
    baseline=json.loads((PREVIOUS/'comparison/results.json').read_text())
    inputs=json.loads((VERSION/'inputs.json').read_text())
    assert evaluator_sha==baseline['method']['evaluator_sha256']
    sections,tasks={},[]
    for dataset in DATASETS:
        label=DATASET_LABELS[dataset]
        assert json.loads((VERSION/f'raw/{dataset}/complete.json').read_text())['exported']
        import gzip
        with gzip.open(VERSION/f'raw/{dataset}/leap_rerun_run.jsonl.gz','rt') as stream:
            actual=json.loads(next(stream))
        assert actual['mask_enabled'] is False and '--use_mask' in actual['leap_argv']
        assert actual['config']==inputs['datasets'][dataset]['new_config']
        with duckdb.connect() as con:
            videos=list_split_videos(dataset,'test')
            queries.load_clips(con,dataset,videos)
            con.execute('ALTER TABLE clips ADD COLUMN leap_name VARCHAR')
            con.execute("UPDATE clips SET leap_name=regexp_extract(video,'te([0-9]+)',1)::INT::VARCHAR")
            queries.load_ground_truth(con,[(video,str(output_path(dataset,'ground truth','',video))) for video in videos])
            truth=con.execute('''SELECT coalesce(g.g,0)::BIGINT g FROM frames f LEFT JOIN gt_counts g
                ON g.video=f.video AND g.frame=f.frame AND g.variant='all' ORDER BY f.gidx''').fetchnumpy()['g']
            old=copy.deepcopy(baseline['datasets'][label]['versions']['regions_v2'])
            assert old['ground_truth_count_sha256']==hashlib.sha256(truth.tobytes()).hexdigest()
            reproduced=helper.native_scores(evaluator,truth,helper.load_state(con,PREVIOUS/f'raw/{dataset}/leap_rerun.jsonl.gz'))
            assert reproduced==old['native_queries'],(dataset,'masked query reproduction failed')
            predictions=helper.load_state(con,VERSION/f'raw/{dataset}/leap_rerun.jsonl.gz')
            fresh=dict(native_queries=helper.native_scores(evaluator,truth,predictions),
                timing=helper.timing(con,VERSION/f'summary/speed-{dataset}.tsv',
                                    VERSION/f'raw/{dataset}/leap_rerun_launch.json',len(truth)),
                ground_truth_count_sha256=hashlib.sha256(truth.tobytes()).hexdigest())
            sections[label]=dict(dataset=dataset,frames=len(truth),clips=len(videos),versions=dict(masked=old,unmasked=fresh))
            tasks+=helper.write_variants(con,dataset,videos,VERSION/f'raw/{dataset}/leap_rerun_export.jsonl.gz',VERSION/'tracks'/dataset)
        print(label+': archived masked queries reproduced; unmasked queries scored.',flush=True)
    with mp.Pool(max(1,min(args.workers,len(tasks)))) as pool:
        measured=pool.map(helper.run_trackeval,tasks)
    for label,section in sections.items():
        fresh=section['versions']['unmasked']
        fresh['hota']={row['variant']:{key:row['metrics']['HOTA'][key] for key in ('HOTA','DetA','AssA','LocA')}
                       for row in measured if row['dataset']==section['dataset']}
        assert set(fresh['hota'])=={'raw','linear','pattern'}
        fresh['best_reconstruction']=max(('linear','pattern'),key=lambda name:fresh['hota'][name]['HOTA'])
        fresh['best_hota']=fresh['hota'][fresh['best_reconstruction']]['HOTA']
        old=section['versions']['masked']
        section['change']=dict(hota=fresh['best_hota']-old['best_hota'],fps_ratio=fresh['timing']['fps']/old['timing']['fps'],
            selection_f1=fresh['native_queries']['selection']['f1']-old['native_queries']['selection']['f1'],
            count_mape=fresh['native_queries']['aggregation']['mape']-old['native_queries']['aggregation']['mape'],
            top10=fresh['native_queries']['topk']-old['native_queries']['topk'])
        print(label,json.dumps(section['change']),flush=True)
    # Preserve the earlier runs and their cache; every reconstructed track belongs to this version.
    (VERSION/'comparison').mkdir(exist_ok=True)
    write_json(VERSION/'comparison','results.json',dict(version=VERSION.name,baseline=PREVIOUS.name,
        method=dict(changed='mask disabled only; reviewed stop_area and learned artifacts unchanged',
                    evaluator_sha256=evaluator_sha,numpy=np.__version__,
                    queries='Original LEAP evaluator on pooled native intervals and shared ground-truth counts.',
                    throughput='Native inference_total - decode; one GPU per dataset.',
                    hota='Unchanged TrackEval and root ground truth; raw, linear, assigned-pattern reconstructions.'),datasets=sections))


if __name__=='__main__':
    main()
