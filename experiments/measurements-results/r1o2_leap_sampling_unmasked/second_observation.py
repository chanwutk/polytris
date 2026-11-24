"""Check CalDoT2's second sampling opportunity and first identity link from saved observations."""
import json
from pathlib import Path
import sys

import duckdb

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
from measurements import r1o2_leap_sampling_diagnostics as diagnostic
from measurements.common import query_dicts

saved = json.loads((OUT / 'all_datasets.json').read_text())['datasets']['CalDoT2']
with duckdb.connect() as con:
    con.execute('SET threads=4')
    provenance = diagnostic.load_dataset(con, 'caldot2-y05')
    assert provenance['source_sha256'] == saved['provenance']['source_sha256']
    con.execute('CREATE TABLE matches AS SELECT * FROM read_parquet(?)', [str(OUT / 'caldot2-y05-matches.parquet')])
    diagnostic.derive_tracks(con)
    cohort = query_dicts(con, '''SELECT count(*) eligible_vehicles,
        count(*) FILTER(observations>0) observed_vehicles,
        count(*) FILTER(observations=1) observed_once,
        count(*) FILTER(observations>=2) observed_at_least_twice
        FROM track_diagnostics WHERE NOT censored AND e-s+1>=15''')[0]
    previous = saved['coverage_uncensored_at_least_one_second']
    assert cohort['eligible_vehicles'] == previous['gt_tracks']
    assert cohort['observed_once'] == previous['one_matched_observation']
    assert cohort['observed_at_least_twice'] == previous['two_plus_matched_observations']
    assert cohort['observed_vehicles'] == previous['gt_tracks'] - previous['never_matched']
    con.execute('''CREATE TABLE first_followup AS WITH first_seen AS (
        SELECT video,gt_id,min(frame) first_frame,arg_min(id,frame) first_id
        FROM matches GROUP BY video,gt_id),
        schedule AS (SELECT video,frame,lead(frame) OVER(PARTITION BY video ORDER BY frame) next_sample FROM samples)
        SELECT f.*,s.next_sample,CASE
            WHEN next_sample IS NULL THEN 'no following sample'
            WHEN t.e<next_sample THEN 'exited before next sample'
            WHEN g.id IS NULL THEN 'no annotation at next sample'
            WHEN m.id IS NULL THEN 'present but detection missed'
            WHEN m.id<>first_id THEN 'detected with different identity'
            ELSE 'detected with same identity' END outcome
        FROM first_seen f JOIN gt_tracks t ON t.video=f.video AND t.id=f.gt_id
        JOIN schedule s ON s.video=f.video AND s.frame=f.first_frame
        LEFT JOIN gt g ON g.video=f.video AND g.id=f.gt_id AND g.frame=s.next_sample
        LEFT JOIN matches m ON m.video=f.video AND m.gt_id=f.gt_id AND m.frame=s.next_sample
        WHERE NOT t.censored AND t.e-t.s+1>=15''')
    outcomes = query_dicts(con, 'SELECT outcome,count(*) vehicles FROM first_followup GROUP BY outcome ORDER BY outcome')
    assert sum(r['vehicles'] for r in outcomes) == cohort['observed_vehicles']
    pairs = query_dicts(con, '''WITH paired AS (
        SELECT m.*,row_number() OVER w rn,lag(m.id) OVER w previous_id
        FROM matches m JOIN gt_tracks t ON t.video=m.video AND t.id=m.gt_id
        WHERE NOT t.censored AND t.e-t.s+1>=15
        WINDOW w AS (PARTITION BY m.video,m.gt_id ORDER BY m.frame))
        SELECT count(*) pairs,count(*) FILTER(id<>previous_id) different_identity
        FROM paired WHERE rn=2''')[0]
    assert pairs['pairs'] == cohort['observed_at_least_twice']
    report = dict(method='CalDoT2 unmasked run; annotated entries and exits inside the clip; visibility at least one second; matches require IoU>=0.5. Follow-up is the next actual sampled frame after the first matched detection.',
                  source_sha256=provenance['source_sha256'], cohort=cohort,
                  first_followup=outcomes, first_two_detections=pairs)
    (OUT / 'second_observation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
