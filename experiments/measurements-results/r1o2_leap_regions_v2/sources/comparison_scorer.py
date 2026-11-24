"""Compare reviewed LEAP regions with the completed matched-detector version."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import duckdb
import numpy as np

from evaluation.manifests import list_split_videos
from measurements.common import DATASET_LABELS, output_path, query_dicts, write_json
from measurements import r1o2_leap_native_queries as queries
from measurements.r1o2_leap_otif_rerun import RERUN_COLUMNS, run_trackeval, write_variants


ROOT = Path(__file__).resolve().parent.parent
VERSION = ROOT/'measurements-results/r1o2_leap_regions_v2'
BASELINE = ROOT/'measurements-results/r1o2_leap_matched'
DATASETS = ('caldot1-y05', 'caldot2-y05', 'ams-y05')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    return parser.parse_args()


def original_evaluator():
    # Execute only LEAP's evaluator class; its unrelated GPU/plotting imports are unnecessary.
    source = VERSION/'sources/evaluate.py'
    tree = ast.parse(source.read_text())
    node, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'evaluation']
    namespace = dict(np=np, Counter=Counter)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['evaluation'], hashlib.sha256(source.read_bytes()).hexdigest()


def native_scores(evaluator, ground_truth, predictions):
    # Supply the pooled counts to the original query methods, preserving their sorting and rounding.
    instance = evaluator.__new__(evaluator)
    instance.cfg = dict(start_frame=0, end_frame=len(ground_truth))
    instance.gt_result = ground_truth.tolist()
    instance.pred_result = predictions.tolist()
    f1, recall, precision, accuracy = instance.selection_query_1()
    mae, acc = instance.aggregation_query_1()
    return dict(selection=dict(f1=f1, recall=recall, precision=precision, accuracy=accuracy),
                aggregation=dict(mae=float(mae), acc=float(acc), mape=float(1-acc)),
                topk=float(instance.top_k_query_1()))


def load_state(con, path):
    # All nested fields are typed explicitly so matched-pattern dictionaries cannot change schema inference.
    con.execute(f'CREATE OR REPLACE TABLE states AS SELECT * FROM read_json(?, columns={RERUN_COLUMNS})', [str(path)])
    assert con.execute('SELECT count(*),count(DISTINCT video_name) FROM states').fetchone() == (60, 60)
    assert not con.execute('SELECT video_name FROM states ANTI JOIN clips ON video_name=leap_name').fetchall()
    con.execute('''CREATE OR REPLACE TABLE objects AS SELECT video_name,o.id,
        o."interval"[1] s,o."interval"[2] e,o.observations
        FROM states,unnest(objects) t(o)''')
    assert not con.execute('SELECT video_name,id FROM objects GROUP BY ALL HAVING count(*)>1').fetchall()
    con.execute('''CREATE OR REPLACE TABLE observations AS
        SELECT c.video,o.id,b[6]::BIGINT frame,b[1] x1,b[2] y1,b[3] x2,b[4] y2
        FROM objects o JOIN clips c ON c.leap_name=o.video_name,unnest(observations) t(b)''')
    # Frame generation clips native half-open intervals to each video's actual length.
    predictions = con.execute('''SELECT count(o.id)::BIGINT p FROM frames f
        JOIN clips c USING(video)
        LEFT JOIN objects o ON o.video_name=c.leap_name AND f.frame>=o.s AND f.frame<o.e
        GROUP BY f.gidx ORDER BY f.gidx''').fetchnumpy()['p']
    return predictions


def timing(con, path, launch_path, frames):
    # Match p140 and the paper: native inference time minus native decode time, one GPU per dataset.
    records = query_dicts(con, '''SELECT count(*) clips,count(DISTINCT video_name) names,
        sum(inference_total) inference_s,sum(decode) decode_s,
        sum(inference_total-decode) compute_s
        FROM read_csv(?,delim='\t',header=true)''', [str(path)])[0]
    assert records['clips'] == records['names'] == 60 and records['compute_s'] > 0
    launch = json.loads(launch_path.read_text())
    assert launch['exit'] == 0 and not launch['contaminated']
    return dict(records, fps=frames/records['compute_s'], gpu=launch['gpu'], cpus=launch['cpus'],
                contaminated=launch['contaminated'], image=launch['image'])


def main():
    args = parse_args()
    evaluator, evaluator_sha = original_evaluator()
    sections, tasks = {}, []
    for dataset in DATASETS:
        label = DATASET_LABELS[dataset]
        assert json.loads((VERSION/f'raw/{dataset}/complete.json').read_text())['exported']
        expected = json.loads((VERSION/f'baseline_summary/native-{dataset}.json').read_text())
        assert evaluator_sha == expected['evaluator_sha256'] and np.__version__ == expected['numpy']
        expected_query, = [row for row in expected['datasets'][label]['records'] if row['system'] == 'leap']
        baseline_hota = json.loads((VERSION/f'baseline_summary/results-{dataset}.json').read_text())['datasets'][label]['hota']
        with duckdb.connect() as con:
            videos = list_split_videos(dataset, 'test')
            queries.load_clips(con, dataset, videos)
            con.execute('ALTER TABLE clips ADD COLUMN leap_name VARCHAR')
            con.execute("UPDATE clips SET leap_name=regexp_extract(video,'te([0-9]+)',1)::INT::VARCHAR")
            queries.load_ground_truth(con, [(video, str(output_path(dataset, 'ground truth', '', video))) for video in videos])
            truth = con.execute('''SELECT coalesce(g.g,0)::BIGINT g FROM frames f LEFT JOIN gt_counts g
                ON g.video=f.video AND g.frame=f.frame AND g.variant='all' ORDER BY f.gidx''').fetchnumpy()['g']
            section = dict(dataset=dataset, frames=len(truth), clips=len(videos), versions={})
            for version, raw in [('original', BASELINE/'raw'), ('regions_v2', VERSION/'raw')]:
                predictions = load_state(con, raw/dataset/'leap_rerun.jsonl.gz')
                assert len(predictions) == len(truth)
                scores = native_scores(evaluator, truth, predictions)
                if version == 'original':
                    assert scores['selection'] == expected_query['selection']
                    assert abs(scores['aggregation']['mape']-expected_query['aggregation']['mape']) < 1e-10
                    assert scores['topk'] == expected_query['topk']
                    speed = VERSION/f'baseline_summary/speed-{dataset}.tsv'
                else:
                    speed = VERSION/f'summary/speed-{dataset}.tsv'
                    tasks += write_variants(con, dataset, videos, raw/dataset/'leap_rerun_export.jsonl.gz', VERSION/'tracks'/dataset)
                section['versions'][version] = dict(native_queries=scores,
                    timing=timing(con, speed, raw/dataset/'leap_rerun_launch.json', len(truth)),
                    ground_truth_count_sha256=hashlib.sha256(truth.tobytes()).hexdigest())
            section['versions']['original']['hota'] = {variant: baseline_hota[variant]
                                                       for variant in ('raw', 'linear', 'pattern')}
            sections[label] = section
        print(label+': original query results reproduced; v2 query scoring complete.', flush=True)
    # New reconstruction outputs remain in this version's track directory; the original cache is never written.
    with mp.Pool(max(1, min(args.workers, len(tasks)))) as pool:
        measured = pool.map(run_trackeval, tasks)
    for label, section in sections.items():
        new = section['versions']['regions_v2']
        new['hota'] = {row['variant']: {key: row['metrics']['HOTA'][key] for key in ('HOTA', 'DetA', 'AssA', 'LocA')}
                       for row in measured if row['dataset'] == section['dataset']}
        for value in section['versions'].values():
            value['best_reconstruction'] = max(('linear', 'pattern'), key=lambda name: value['hota'][name]['HOTA'])
            value['best_hota'] = value['hota'][value['best_reconstruction']]['HOTA']
        old = section['versions']['original']
        section['change'] = dict(hota=new['best_hota']-old['best_hota'],
            fps_ratio=new['timing']['fps']/old['timing']['fps'],
            selection_f1=new['native_queries']['selection']['f1']-old['native_queries']['selection']['f1'],
            count_mape=new['native_queries']['aggregation']['mape']-old['native_queries']['aggregation']['mape'],
            top10=new['native_queries']['topk']-old['native_queries']['topk'])
        print(label, json.dumps(section['change']), flush=True)
    output = VERSION/'comparison'
    output.mkdir(exist_ok=True)
    write_json(output, 'results.json', dict(version='regions_v2', baseline='r1o2_leap_matched',
        method=dict(changed='stop_area only; preparation reused', evaluator_sha256=evaluator_sha,
            queries='Original LEAP selection_query_1, aggregation_query_1, top_k_query_1 on pooled native intervals and shared ground-truth counts.',
            throughput='Native inference_total - decode; one GPU per dataset.',
            hota='Original v1 summaries; fresh v2 raw/linear/pattern TrackEval, unchanged root ground truth and p140 transform.',
            baseline_hota_precision='Original summaries retain four decimal places.'), datasets=sections))


if __name__ == '__main__':
    main()
