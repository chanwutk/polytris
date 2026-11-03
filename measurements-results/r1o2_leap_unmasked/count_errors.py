"""Decompose native count MAPE changes by annotated count, without rerunning inference."""
import importlib.util
import json
from pathlib import Path
import sys

import duckdb

VERSION=Path(__file__).resolve().parent
ROOT=VERSION.parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.manifests import list_split_videos
from measurements import r1o2_leap_native_queries as queries
from measurements.common import DATASET_LABELS,output_path,query_dicts

spec=importlib.util.spec_from_file_location('unmasked_scorer',VERSION/'score.py')
scorer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
comparison=json.loads((VERSION/'comparison/results.json').read_text())
report={}
for dataset in ('caldot1-y05','caldot2-y05'):
    label=DATASET_LABELS[dataset]
    with duckdb.connect() as con:
        queries.load_clips(con,dataset,list_split_videos(dataset,'test'))
        con.execute('ALTER TABLE clips ADD COLUMN leap_name VARCHAR')
        con.execute("UPDATE clips SET leap_name=regexp_extract(video,'te([0-9]+)',1)::INT::VARCHAR")
        queries.load_ground_truth(con,[(v,str(output_path(dataset,'ground truth','',v))) for v in list_split_videos(dataset,'test')])
        truth=con.execute('''SELECT coalesce(g.g,0)::BIGINT g FROM frames f LEFT JOIN gt_counts g
            ON g.video=f.video AND g.frame=f.frame AND g.variant='all' ORDER BY f.gidx''').fetchnumpy()['g']
        con.execute('CREATE TABLE points(version VARCHAR,g BIGINT,p BIGINT)')
        for version,directory in [('masked',VERSION.parent/'r1o2_leap_regions_v2'),('unmasked',VERSION)]:
            predicted=scorer.helper.load_state(con,directory/f'raw/{dataset}/leap_rerun.jsonl.gz')
            con.execute('INSERT INTO points SELECT ?,unnest(?::BIGINT[]),unnest(?::BIGINT[])',
                        [version,truth.tolist(),predicted.tolist()])
        con.execute('''CREATE TABLE errors AS SELECT *,abs(p-g) ae,
            CASE WHEN g=0 THEN (p>0)::DOUBLE ELSE abs(p-g)*1.0/g END ape,
            CASE WHEN g=0 THEN '0' WHEN g=1 THEN '1' WHEN g<=4 THEN '2-4' ELSE '5+' END bucket
            FROM points''')
        totals=query_dicts(con,'''SELECT version,count(*) frames,avg(g) mean_truth,avg(p) mean_predicted,
            avg(ae) mae,avg(ape) mape,avg((p>g)::DOUBLE) overcount_frame_share,
            avg((p<g)::DOUBLE) undercount_frame_share,
            avg(CASE WHEN p>g THEN ape ELSE 0 END) overcount_mape_contribution,
            avg(CASE WHEN p<g THEN ape ELSE 0 END) undercount_mape_contribution
            FROM errors GROUP BY version ORDER BY version''')
        for row in totals:
            expected=comparison['datasets'][label]['versions'][row['version']]['native_queries']['aggregation']
            assert abs(row['mape']-expected['mape'])<1e-12 and abs(row['mae']-expected['mae'])<1e-12
        bins=query_dicts(con,'''SELECT version,bucket,count(*) frames,avg(ae) mae,avg(ape) mape,
            sum(ape)/? mape_contribution,avg((p>g)::DOUBLE) overcount_share
            FROM errors GROUP BY version,bucket ORDER BY bucket,version''',[len(truth)])
        report[label]=dict(totals=totals,bins=bins)
        print(label,json.dumps(report[label]),flush=True)
(VERSION/'count_errors.json').write_text(json.dumps(report,indent=2)+'\n')
