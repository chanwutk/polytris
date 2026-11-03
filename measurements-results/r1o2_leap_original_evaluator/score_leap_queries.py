"""Run LEAP's original evaluator on shared annotations and saved system outputs.

Run inside ace's polyis container. This performs CPU scoring only.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import duckdb
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetris-root', type=Path, default=Path('/polyis'))
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--native-intervals', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+')
    args = parser.parse_args()
    sys.path.insert(0, str(args.tetris_root))
    from evaluation.manifests import list_split_videos
    from measurements import r1o2_leap_native_queries as native
    from measurements.common import DATASET_LABELS, system_configs

    # Load the unmodified evaluation class, without LEAP's unrelated GPU/UI imports.
    source = args.evaluator.read_bytes()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'evaluation')
    namespace = dict(np=np, Counter=Counter, settings=SimpleNamespace(coco_names={'car': 2}))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(args.evaluator), 'exec'), namespace)
    evaluator = namespace['evaluation']
    check = evaluator('pseudo_label', 'car', dict(start_frame=0, end_frame=2))
    check.preprocess({0: [[1, 0, 0, 1, 1, 'car']], 1: []}, {'1': [[0, 1], -1, 2]})
    assert check.gt_result == check.pred_result == [1, 0]
    assert check.selection_query_1() == (1, 1, 1, 1)
    assert check.aggregation_query_1() == (0, 1) and check.top_k_query_1(k=1) == 1

    intervals = json.loads(args.native_intervals.read_text())
    report = dict(evaluator_sha256=hashlib.sha256(source).hexdigest(), numpy=np.__version__,
                  ground_truth_mode="LEAP preprocess(pseudo_label), shared 003_groundtruth boxes; all vehicles as car",
                  queries=['selection_query_1', 'aggregation_query_1', 'top_k_query_1'], datasets={})
    for dataset in args.datasets or list(DATASET_LABELS):
        label = DATASET_LABELS[dataset]
        print(f'{label}: loading saved outputs', flush=True)
        con = duckdb.connect()
        con.execute('SET enable_progress_bar=false')
        videos = list_split_videos(dataset, 'test')
        native.load_clips(con, dataset, videos)
        configs = [(r['system'], r['config'], r['fps'], r['hota']) for r in system_configs(dataset)]
        con.execute('CREATE TABLE configs(system VARCHAR, config VARCHAR, fps DOUBLE, hota DOUBLE)')
        con.executemany('INSERT INTO configs VALUES (?,?,?,?)', configs)
        gt_files, external = native.ground_truth_files(dataset, videos, None)
        assert not external
        con.execute('CREATE TABLE gt_files(video VARCHAR, filename VARCHAR)')
        con.executemany('INSERT INTO gt_files VALUES (?,?)', gt_files)
        labels = con.execute(f"""
            SELECT video, frame_idx, list(box ORDER BY box[1])
            FROM {native.TRACK_BOXES} JOIN gt_files USING(filename)
            GROUP BY video, frame_idx ORDER BY video, frame_idx
        """, [[f for _, f in gt_files]]).fetchall()
        by_video = {video: {} for video in videos}
        for video, frame, boxes in labels:
            by_video[video][frame] = [box + ['car'] for box in boxes]
        gt = []
        for video, n in con.execute('SELECT video,n FROM clips ORDER BY clip').fetchall():
            assert all(frame >= 0 for frame in by_video[video]), f'{video}: negative ground-truth frame'
            obj = evaluator('pseudo_label', 'car', dict(start_frame=0, end_frame=n))
            obj.preprocess({frame: by_video[video].get(frame, []) for frame in range(n)}, {})
            gt.extend(obj.gt_result)
        con.execute("""
            CREATE TABLE gt_counts AS
            SELECT 'all' AS variant, f.video, f.frame, v.g FROM frames f
            JOIN (SELECT unnest(range(?)) AS gidx, unnest(?) AS g) v USING(gidx)
        """, [len(gt), gt])
        native.load_counts(con, native.file_rows(dataset, videos, [r for r in configs if r[0] != 'leap']), [])
        # LEAP predictions come directly from native intervals, not recovered trajectories.
        raw = intervals[label]['source']
        con.execute("""
            CREATE TABLE native_clips AS
            SELECT printf('te%02d.mp4', regexp_extract(video_name, '([0-9]+)', 1)::INTEGER) AS video, objects
            FROM read_json(?, format='newline_delimited',
                 columns={'video_name':'VARCHAR', 'objects':'STRUCT(id BIGINT, "interval" BIGINT[])[]'})
        """, [raw])
        assert not con.execute("""
            SELECT video FROM clips FULL JOIN native_clips USING(video) GROUP BY video
            HAVING count(clips.video)<>1 OR count(native_clips.video)<>1
        """).fetchall(), f'{dataset}: mismatched native clips'
        con.execute('CREATE TABLE native_objects AS SELECT video,unnest(objects) AS object FROM native_clips')
        assert not con.execute('SELECT video,object.id FROM native_objects GROUP BY ALL HAVING count(*)>1').fetchall()
        con.execute("DELETE FROM counts WHERE system='leap'")
        con.execute("""
            INSERT INTO counts
            SELECT 'leap', 'param 0', f.video, f.frame, count(*)
            FROM native_objects o JOIN frames f USING(video)
            WHERE f.frame>=o.object.interval[1] AND f.frame<o.object.interval[2]
            GROUP BY f.video,f.frame
        """)
        native.densify(con)
        keys = con.execute('SELECT system,config FROM configs ORDER BY system,config').fetchall()
        values = con.execute('SELECT p FROM pred ORDER BY system,config,gidx').fetchnumpy()['p']
        assert len(values) == len(keys) * len(gt)
        records = []
        for index, ((system, config), predicted) in enumerate(zip(keys, values.reshape(len(keys), len(gt)))):
            obj = evaluator('pseudo_label', 'car', dict(start_frame=0, end_frame=len(gt)))
            obj.gt_result, obj.pred_result = gt, predicted.tolist()
            f1, recall, precision, accuracy = obj.selection_query_1()
            mae, acc = obj.aggregation_query_1()
            records.append(dict(system=system, config=config,
                                selection=dict(f1=f1, recall=recall, precision=precision, accuracy=accuracy),
                                aggregation=dict(mae=mae, acc=acc, mape=1-acc), topk=obj.top_k_query_1()))
            if (index + 1) % 10 == 0:
                print(f'{label}: scored {index+1}/{len(keys)}', flush=True)
        report['datasets'][label] = dict(dataset=dataset, frames=len(gt), clips=len(videos),
                                         ground_truth_files=gt_files, native_intervals=raw,
                                         ground_truth_count_sha256=hashlib.sha256(np.asarray(gt, dtype='<i8').tobytes()).hexdigest(),
                                         records=records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        con.close()
        print(f'{label}: saved {len(records)} original-evaluator results', flush=True)


if __name__ == '__main__':
    main()
