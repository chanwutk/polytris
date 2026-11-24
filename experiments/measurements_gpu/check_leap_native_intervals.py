"""Score LEAP's saved native intervals without recovering them from rendered tracks.

Uses existing query scorers and writes only paper evidence. No inference is run.
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetris-root', type=Path, default=Path('/polyis'))
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tetris_root))
    from evaluation.manifests import list_split_videos
    from measurements import r1o2_leap_native_queries as native
    from measurements.common import DATASET_LABELS, query_dicts

    root = Path(__file__).resolve().parents[1]
    raw = args.raw_dir or args.tetris_root / 'measurements-results/r1o2_leap_otif_rerun/raw'
    datasets = args.datasets or list(DATASET_LABELS)
    assert set(datasets) <= DATASET_LABELS.keys(), f'Unknown datasets: {datasets}'
    out = {}
    for dataset in datasets:
        label = DATASET_LABELS[dataset]
        con = duckdb.connect()
        con.execute('SET enable_progress_bar=false')
        videos = list_split_videos(dataset, 'test')
        native.load_clips(con, dataset, videos)
        files, external = native.ground_truth_files(dataset, videos, None)
        assert not external
        native.load_ground_truth(con, files)
        path = raw / dataset / 'leap_rerun.jsonl.gz'
        con.execute("""
            CREATE TABLE native_clips AS
            SELECT printf('te%02d.mp4', regexp_extract(video_name, '([0-9]+)', 1)::INTEGER) AS video,
                   objects
            FROM read_json(?, format='newline_delimited',
                 columns={'video_name':'VARCHAR', 'objects':'STRUCT(id BIGINT, "interval" BIGINT[])[]'})
        """, [str(path)])
        # Missing or duplicate clips must fail rather than silently become empty predictions.
        bad = con.execute("""
            SELECT video FROM clips FULL JOIN native_clips USING(video)
            GROUP BY video HAVING count(clips.video) <> 1 OR count(native_clips.video) <> 1
        """).fetchall()
        assert not bad, f'{dataset}: missing, unexpected, or duplicate native clips: {bad}'
        con.execute('CREATE TABLE native_objects AS SELECT video, unnest(objects) AS object FROM native_clips')
        con.execute("CREATE TABLE configs AS SELECT 'leap'::VARCHAR AS system, 'param 0'::VARCHAR AS config")
        con.execute("""
            CREATE TABLE counts AS
            SELECT 'leap' AS system, 'param 0' AS config, f.video, f.frame, count(DISTINCT o.object.id) AS cnt
            FROM native_objects o JOIN frames f USING (video)
            WHERE f.frame >= o.object.interval[1] AND f.frame < o.object.interval[2]
            GROUP BY ALL
        """)
        native.densify(con)
        scores = native.score_frames(con)
        topk, taus = native.score_topk(con)
        out[label] = dict(dataset=dataset, clips=len(videos),
                          frames=con.execute('SELECT count(*) FROM frames').fetchone()[0],
                          source=str(path), interval_rule='start <= frame < end, clipped to actual video length',
                          scores={r['variant']: dict(native.frame_metrics(r),
                                  topk=topk[r['variant'], 'leap', 'param 0'], tau=taus[r['variant']]) for r in scores},
                          intervals=query_dicts(con, 'SELECT count(*) AS objects, min(object.interval[1]) AS first, '
                                                    'max(object.interval[2]) AS last FROM native_objects'))
        con.close()
        print(label, out[label]['scores']['all'], flush=True)
    target = args.output or root / 'scripts/data/revision_evidence/native_intervals.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
