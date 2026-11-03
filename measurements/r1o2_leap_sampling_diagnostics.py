"""Read-only LEAP diagnostics: sampled coverage, identity continuity, and pattern eligibility.

Ground-truth matches and interpolated ground-truth boxes are diagnostic oracles,
never inputs to the reported LEAP pipeline. Only this script's result directory is written.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.manifests import list_split_videos
from measurements.common import DATASET_LABELS, canonical_leap_result, output_path, query_dicts, write_json
from polyis.utilities import get_video_frame_count


ROOT = Path(__file__).resolve().parent.parent
MATCH_IOU = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_LABELS), default=list(DATASET_LABELS))
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'measurements-results/r1o2_leap_sampling_diagnostics')
    return parser.parse_args()


def box_iou(a, b):
    """Pairwise continuous-coordinate box IoU, including empty and degenerate boxes."""
    # Broadcast boxes rather than iterating over individual detections.
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    size = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                      - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = size.prod(axis=2)
    union = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)[:, None] \
        + np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def assign_boxes(predicted, ground_truth):
    # Prefer the maximum number of threshold-valid matches, then maximize their IoU.
    iou = box_iou(predicted, ground_truth)
    if not iou.size:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    valid = iou >= MATCH_IOU
    rows, columns = linear_sum_assignment(-(valid * (min(iou.shape) + 1 + iou)))
    keep = valid[rows, columns]
    return rows[keep], columns[keep], iou[rows[keep], columns[keep]]


def load_dataset(con, dataset):
    raw = raw_directory(dataset)
    state = raw / 'leap_rerun.jsonl.gz'
    run = raw / 'leap_rerun_run.jsonl.gz'
    with gzip.open(run, 'rt') as stream:
        metadata = json.loads(next(stream))
    cfg = metadata['config']
    assert cfg['fps'] == 15, (dataset, cfg['fps'])
    # Exact video lengths identify clip-censored trajectories without treating the last detection as the clip end.
    videos = list_split_videos(dataset, 'test')
    con.execute('CREATE TABLE clips(video VARCHAR, leap_name VARCHAR, frames BIGINT, filename VARCHAR)')
    con.executemany('INSERT INTO clips VALUES (?,?,?,?)', [
        (v, str(int(Path(v).stem[2:])), get_video_frame_count(dataset, v),
         str(output_path(dataset, 'ground truth', '', v))) for v in videos])
    files = [r[0] for r in con.execute('SELECT filename FROM clips').fetchall()]
    # Keep annotation identities so sampling opportunity can be separated from LEAP association.
    con.execute("""
        CREATE TABLE gt_unclipped AS
        SELECT c.video, frame_idx AS frame, b[1]::BIGINT AS id,
               b[2] x1, b[3] y1, b[4] x2, b[5] y2
        FROM read_json(?, filename=true, columns={'frame_idx':'BIGINT','tracks':'DOUBLE[][]'}) r
        JOIN clips c USING(filename), unnest(tracks) t(b)
    """, [files])
    # Historical B3D annotations include a few rows past the decoded clip; exclude them only in this diagnosis.
    con.execute('''CREATE TABLE gt AS SELECT g.* FROM gt_unclipped g JOIN clips c USING(video)
                   WHERE g.frame>=0 AND g.frame<c.frames''')
    out_of_clip = con.execute('SELECT (SELECT count(*) FROM gt_unclipped)-(SELECT count(*) FROM gt)').fetchone()[0]
    con.execute("""
        CREATE TABLE state AS SELECT * FROM read_json(?, columns={
            'video_name':'VARCHAR','frames_sampled':'BIGINT[]','patterns_sha256':'VARCHAR',
            'objects':'STRUCT(id BIGINT, "interval" BIGINT[], pattern BIGINT, matched JSON,
                              observations DOUBLE[][])[]'})
    """, [str(state)])
    assert con.execute('SELECT count(DISTINCT patterns_sha256),min(patterns_sha256) FROM state').fetchone() == (1, metadata['patterns_sha256'])
    con.execute("""
        CREATE TABLE samples AS
        SELECT c.video, unnest(frames_sampled) AS frame
        FROM state s JOIN clips c ON c.leap_name=s.video_name
    """)
    con.execute("""
        CREATE TABLE objects AS
        SELECT c.video, o.id, o."interval"[1] AS s, o."interval"[2] AS e,
               o.pattern, o.matched, o.observations
        FROM state s JOIN clips c ON c.leap_name=s.video_name, unnest(objects) t(o)
    """)
    con.execute("""
        CREATE TABLE observations AS
        SELECT video, id, b[6]::BIGINT AS frame, b[1] x1, b[2] y1, b[3] x2, b[4] y2
        FROM objects, unnest(observations) t(b)
    """)
    # Reject duplicate identities or times rather than inflating coverage statistics.
    for table, keys in [('gt', 'video,id,frame'), ('observations', 'video,id,frame'),
                        ('samples', 'video,frame'), ('objects', 'video,id')]:
        assert not con.execute(f'SELECT {keys} FROM {table} GROUP BY {keys} HAVING count(*)>1 LIMIT 1').fetchall(), table
    assert con.execute('SELECT count(*) FROM state').fetchone()[0] == len(videos)
    # Reproduce the runtime entry/exit eligibility test without modifying the configured regions.
    con.execute("""CREATE TABLE regions AS
        SELECT ordinality-1 AS region,b[1] l,b[2] t,b[3] r,b[4] b
        FROM unnest(?::DOUBLE[][]) WITH ORDINALITY t(b,ordinality)""", [cfg['stop_area']])
    con.execute("""CREATE TABLE pattern_endpoints AS
        SELECT ordinality-1 AS pattern,len(p) frames,
               (p[1][1]+p[1][3])/2 sx,(p[1][2]+p[1][4])/2 sy,
               (p[-1][1]+p[-1][3])/2 ex,(p[-1][2]+p[-1][4])/2 ey
        FROM unnest(?::DOUBLE[][][]) WITH ORDINALITY t(p,ordinality)""", [metadata['patterns']])
    con.execute("""CREATE TABLE pattern_routes AS
        SELECT p.*,
          (SELECT min(region) FROM regions WHERE sx BETWEEN l AND r AND sy BETWEEN t AND b) start_region,
          (SELECT min(region) FROM regions WHERE ex BETWEEN l AND r AND ey BETWEEN t AND b) end_region
        FROM pattern_endpoints p""")
    con.execute("""CREATE TABLE pattern_points AS
        SELECT ordinality-1 AS pattern, point_idx-1 AS point,(b[1]+b[3])/2 AS x,(b[2]+b[4])/2 AS y
        FROM unnest(?::DOUBLE[][][]) WITH ORDINALITY t(p,ordinality),
          unnest(p) WITH ORDINALITY t2(b,point_idx)""",
                [metadata['patterns']])
    return dict(config=cfg, arguments=metadata['leap_argv'],
                mask_enabled=metadata.get('mask_enabled', '--use_mask' not in metadata['leap_argv']),
                source=str(state), source_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
                pattern_sha256=metadata['patterns_sha256'], clips=len(videos), excluded_out_of_clip_gt_boxes=out_of_clip)


def match_observations(con):
    # Hungarian assignment has a separate matrix for each frame; aggregation stays in DuckDB.
    groups = con.execute("""
        SELECT p.video,p.frame,p.ids,p.boxes,g.ids,g.boxes FROM
          (SELECT video,frame,list(id ORDER BY id) ids,list([x1,y1,x2,y2] ORDER BY id) boxes
           FROM observations GROUP BY video,frame) p
        JOIN
          (SELECT video,frame,list(id ORDER BY id) ids,list([x1,y1,x2,y2] ORDER BY id) boxes
           FROM gt JOIN samples USING(video,frame) GROUP BY video,frame) g USING(video,frame)
        ORDER BY p.video,p.frame
    """).fetchall()
    matches = []
    for video, frame, pred_ids, pred_boxes, gt_ids, gt_boxes in groups:
        a, b, iou = assign_boxes(pred_boxes, gt_boxes)
        matches.extend(zip([video] * len(a), [frame] * len(a), np.asarray(pred_ids)[a].tolist(),
                           np.asarray(gt_ids)[b].tolist(), iou.tolist()))
    con.execute('CREATE TABLE matches(video VARCHAR,frame BIGINT,id BIGINT,gt_id BIGINT,iou DOUBLE)')
    if matches:
        con.executemany('INSERT INTO matches VALUES (?,?,?,?,?)', matches)


def derive_tracks(con):
    # Annotation gaps and clip boundaries are reported separately from elapsed track span.
    con.execute("""CREATE TABLE gt_tracks AS
        SELECT g.video,g.id,min(frame) s,max(frame) e,count(*) visible_frames,
               min(frame)=0 OR max(frame)=max(c.frames)-1 AS censored
        FROM gt g JOIN clips c USING(video) GROUP BY g.video,g.id""")
    con.execute("""CREATE TABLE track_diagnostics AS
        SELECT t.*,coalesce(x.sampled,0) sampled,coalesce(m.observations,0) observations,
          coalesce(m.predicted_ids,0) predicted_ids,m.first_observation,m.last_observation
        FROM gt_tracks t LEFT JOIN
          (SELECT g.video,g.id,count(*) sampled FROM gt g JOIN samples USING(video,frame) GROUP BY g.video,g.id) x
          USING(video,id) LEFT JOIN
          (SELECT video,gt_id AS id,count(*) observations,count(DISTINCT id) predicted_ids,
                  min(frame) first_observation,max(frame) last_observation
           FROM matches GROUP BY video,gt_id) m USING(video,id)""")


def oracle_interpolation(con):
    # Interpolate true boxes at sampled times to isolate lost motion from detector and identity errors.
    con.execute("""CREATE TABLE oracle_pairs AS
        SELECT video,id,frame AS s,lead(frame) OVER w AS e,[x1,y1,x2,y2] AS a,
          lead([x1,y1,x2,y2]) OVER w AS b
        FROM gt JOIN samples USING(video,frame)
        WINDOW w AS (PARTITION BY video,id ORDER BY frame)""")
    con.execute("""CREATE TABLE oracle_boxes AS
        SELECT g.video,g.id,g.frame,g.x1,g.y1,g.x2,g.y2,p.s,p.e,p.a,p.b,
          p.a[1]+(p.b[1]-p.a[1])*(g.frame-p.s)/(p.e-p.s) px1,
          p.a[2]+(p.b[2]-p.a[2])*(g.frame-p.s)/(p.e-p.s) py1,
          p.a[3]+(p.b[3]-p.a[3])*(g.frame-p.s)/(p.e-p.s) px2,
          p.a[4]+(p.b[4]-p.a[4])*(g.frame-p.s)/(p.e-p.s) py2
        FROM gt g JOIN oracle_pairs p ON g.video=p.video AND g.id=p.id
          AND g.frame>=p.s AND g.frame<p.e WHERE p.e>p.s""")
    # Continuous-coordinate IoU agrees with the one-to-one matcher's geometry.
    result = query_dicts(con, """WITH areas AS (
        SELECT *,greatest(0,least(x2,px2)-greatest(x1,px1))*
          greatest(0,least(y2,py2)-greatest(y1,py1)) AS inter,
          greatest(0,x2-x1)*greatest(0,y2-y1)+greatest(0,px2-px1)*greatest(0,py2-py1) AS total
        FROM oracle_boxes)
        SELECT count(*) evaluated_gt_boxes,avg(inter/nullif(total-inter,0)) mean_iou,
          avg((inter/nullif(total-inter,0)>=0.5)::DOUBLE) iou_ge_half_share
        FROM areas""")[0]
    # Distinguish bending away from the endpoint line from uneven progress along that line.
    result['motion'] = query_dicts(con, """WITH centers AS (
        SELECT *, (x1+x2)/2 cx,(y1+y2)/2 cy,(a[1]+a[3])/2 ax,(a[2]+a[4])/2 ay,
          (b[1]+b[3]-a[1]-a[3])/2 dx,(b[2]+b[4]-a[2]-a[4])/2 dy,
          sqrt((x2-x1)^2+(y2-y1)^2) diagonal FROM oracle_boxes),
      errors AS (SELECT *,dx^2+dy^2 chord_squared,
          abs((cx-ax)*dy-(cy-ay)*dx)/nullif(sqrt(dx^2+dy^2)*diagonal,0) perpendicular_in_box_diagonals,
          abs(((cx-ax)*dx+(cy-ay)*dy)/nullif(dx^2+dy^2,0)-(frame-s)*1.0/(e-s)) progress_error
        FROM centers)
      SELECT avg((chord_squared<100)::DOUBLE) near_stationary_box_share,
        avg(perpendicular_in_box_diagonals) FILTER(chord_squared>=100) mean_perpendicular_box_diagonals,
        quantile_cont(perpendicular_in_box_diagonals,0.9) FILTER(chord_squared>=100) p90_perpendicular_box_diagonals,
          avg(progress_error) FILTER(chord_squared>=100) mean_along_line_progress_error,
          avg(progress_error*sqrt(chord_squared)/nullif(diagonal,0)) FILTER(chord_squared>=100) mean_along_line_error_box_diagonals,
        quantile_cont(progress_error,0.9) FILTER(chord_squared>=100) p90_along_line_progress_error
      FROM errors""")[0]
    # Use true box sizes or true centers independently to identify which changes defeat interpolation.
    result['center_and_size'] = query_dicts(con, """WITH variants AS (
        SELECT g.*,v.name,v.p[1] px1v,v.p[2] py1v,v.p[3] px2v,v.p[4] py2v FROM oracle_boxes g,
        unnest([
          {'name':'interpolated_center_true_size','p':[(px1+px2-x2+x1)/2,(py1+py2-y2+y1)/2,
                                                    (px1+px2+x2-x1)/2,(py1+py2+y2-y1)/2]},
          {'name':'true_center_interpolated_size','p':[(x1+x2-px2+px1)/2,(y1+y2-py2+py1)/2,
                                                    (x1+x2+px2-px1)/2,(y1+y2+py2-py1)/2]}
        ]) t(v)), areas AS (
          SELECT *,greatest(0,least(x2,px2v)-greatest(x1,px1v))*
            greatest(0,least(y2,py2v)-greatest(y1,py1v)) inter,
            greatest(0,x2-x1)*greatest(0,y2-y1)+greatest(0,px2v-px1v)*greatest(0,py2v-py1v) total
          FROM variants)
        SELECT name,avg(inter/nullif(total-inter,0)) mean_iou,
          avg((inter/nullif(total-inter,0)>=0.5)::DOUBLE) iou_ge_half_share
        FROM areas GROUP BY name ORDER BY name""")
    return result


def raw_directory(dataset):
    # Diagnose the same run that supplies the paper's canonical LEAP point.
    selected = canonical_leap_result(dataset)
    if selected is not None:
        return selected['native_intervals'].parent
    kind = 'r1o2_leap_matched' if dataset.endswith('-y05') else 'r1o2_leap_otif_rerun'
    return ROOT / 'measurements-results' / kind / 'raw' / dataset


def pattern_anchors(con, dataset):
    # Compare the actual saved pattern export with its own observed boxes, before attributing error to skipped motion.
    source = raw_directory(dataset) / 'leap_rerun_export.jsonl.gz'
    con.execute("""CREATE TABLE pattern_boxes AS
        SELECT c.video,ordinality-1 AS frame,b.track_id AS id,b.left x1,b.top y1,b.right x2,b.bottom y2
        FROM read_json(?,columns={'video_name':'VARCHAR',
          'frames':'STRUCT("left" DOUBLE,"top" DOUBLE,"right" DOUBLE,"bottom" DOUBLE,track_id BIGINT)[][]'}) r
        JOIN clips c ON r.video_name=c.leap_name,
        unnest(r.frames) WITH ORDINALITY t(boxes,ordinality),unnest(boxes) t2(b)
        JOIN objects o ON c.video=o.video AND b.track_id=o.id
        WHERE ordinality-1>=greatest(o.s,0) AND ordinality-1<least(o.e,c.frames)""", [str(source)])
    return query_dicts(con, """WITH areas AS (
        SELECT o.video,o.id,o.frame,p.id IS NOT NULL AS present,
          greatest(0,least(o.x2,p.x2)-greatest(o.x1,p.x1))*
            greatest(0,least(o.y2,p.y2)-greatest(o.y1,p.y1)) inter,
          greatest(0,o.x2-o.x1)*greatest(0,o.y2-o.y1)+
            greatest(0,p.x2-p.x1)*greatest(0,p.y2-p.y1) total
        FROM observations o LEFT JOIN pattern_boxes p USING(video,id,frame))
      SELECT count(*) observations,count(*) FILTER(present) reconstructed_anchor_boxes,
        avg(inter/nullif(total-inter,0)) FILTER(present) mean_iou_when_present,
        avg((present AND inter/nullif(total-inter,0)>=0.5)::DOUBLE) iou_ge_half_share_all_observations
      FROM areas""")[0]


def continuity_by_stationarity(con, cfg):
    # Apply the runtime's actual stationary-overlap threshold to ground-truth-matched consecutive observations.
    return query_dicts(con, """WITH paired AS (
        SELECT m.*, [o.x1,o.y1,o.x2,o.y2] b,lag([o.x1,o.y1,o.x2,o.y2]) OVER w a,
          lag(m.id) OVER w previous_id,lag(m.frame) OVER w previous_frame,lag(m.iou) OVER w previous_gt_iou
        FROM matches m JOIN observations o USING(video,id,frame)
        WINDOW w AS (PARTITION BY m.video,m.gt_id ORDER BY m.frame)),
      schedule AS (SELECT video,frame,lag(frame) OVER(PARTITION BY video ORDER BY frame) previous_sample FROM samples),
      areas AS (SELECT p.*,greatest(0,least(a[3],b[3])-greatest(a[1],b[1]))*
          greatest(0,least(a[4],b[4])-greatest(a[2],b[2])) inter,
          greatest(0,a[3]-a[1])*greatest(0,a[4]-a[2])+
          greatest(0,b[3]-b[1])*greatest(0,b[4]-b[2]) total
        FROM paired p JOIN schedule USING(video,frame) WHERE previous_frame=previous_sample)
      SELECT coalesce(inter/nullif(total-inter,0)>?,false) passes_stationary_rule,
        count(*) pairs,count(*) FILTER(id=previous_id) same_id,
        avg((id=previous_id)::DOUBLE) same_id_share,
        count(*) FILTER(iou>=0.75 AND previous_gt_iou>=0.75) high_confidence_pairs,
        avg((id=previous_id)::DOUBLE) FILTER(iou>=0.75 AND previous_gt_iou>=0.75) high_confidence_same_id_share
      FROM areas GROUP BY passes_stationary_rule ORDER BY passes_stationary_rule""", [cfg['stop_iou_thresh']])


def summarize(con, fps, cfg):
    result = {}
    # Show both the complete annotated spans and spans not touching either clip boundary.
    for label, predicate in [('all', 'TRUE'), ('uncensored', 'NOT censored'),
                             ('uncensored_at_least_one_second', f'NOT censored AND e-s+1>={fps}')]:
        result['duration_' + label] = query_dicts(con, f"""SELECT count(*) tracks,
          avg((e-s+1)/{fps}.0) mean_s,quantile_cont((e-s+1)/{fps}.0,[0.1,0.5,0.9]) p10_p50_p90_s,
          stddev_pop(e-s+1)/avg(e-s+1) coefficient_of_variation,
          min(e-s+1)/{fps}.0 min_s,max(e-s+1)/{fps}.0 max_s,
          avg(visible_frames*1.0/(e-s+1)) mean_visible_fraction
          FROM gt_tracks WHERE {predicate}""")[0]
    result['sampling'] = query_dicts(con, f"""SELECT count(*) sampled_frames,
      quantile_cont(gap/{fps}.0,[0.1,0.5,0.9]) gap_p10_p50_p90_s,max(gap)/{fps}.0 max_gap_s
      FROM (SELECT frame-lag(frame) OVER(PARTITION BY video ORDER BY frame) gap FROM samples)""")[0]
    result['coverage'] = query_dicts(con, """SELECT count(*) gt_tracks,
      count(*) FILTER(sampled=0) zero_sample_exposure,count(*) FILTER(sampled=1) one_sample_exposure,
      count(*) FILTER(sampled>=2) two_plus_sample_exposure,
      count(*) FILTER(observations=0) never_matched,count(*) FILTER(observations=1) one_matched_observation,
      count(*) FILTER(observations>=2) two_plus_matched_observations,
      count(*) FILTER(observations>=2 AND predicted_ids>1) fragmented_among_reobserved,
      avg(predicted_ids) predicted_ids_per_gt_track,
      max(predicted_ids) max_predicted_ids_per_gt_track FROM track_diagnostics""")[0]
    result['coverage_uncensored_at_least_one_second'] = query_dicts(con, f"""SELECT count(*) gt_tracks,
      count(*) FILTER(sampled=0) zero_sample_exposure,count(*) FILTER(sampled=1) one_sample_exposure,
      count(*) FILTER(sampled>=2) two_plus_sample_exposure,
      count(*) FILTER(observations=0) never_matched,count(*) FILTER(observations=1) one_matched_observation,
      count(*) FILTER(observations>=2) two_plus_matched_observations,
      count(*) FILTER(observations>=2 AND predicted_ids>1) fragmented_among_reobserved
      FROM track_diagnostics WHERE NOT censored AND e-s+1>={fps}""")[0]
    result['first_sample_exit'] = query_dicts(con, f"""WITH first_samples AS (
        SELECT g.video,g.id,min(g.frame) frame FROM gt g JOIN samples USING(video,frame) GROUP BY g.video,g.id),
      schedule AS (SELECT video,frame,lead(frame) OVER(PARTITION BY video ORDER BY frame) next_sample FROM samples)
      SELECT count(*) tracks_with_next_sample,
        count(*) FILTER(e<next_sample) exits_before_next_sample,
        avg((e<next_sample)::DOUBLE) exits_before_next_sample_share
      FROM first_samples f JOIN schedule USING(video,frame) JOIN gt_tracks t USING(video,id)
      WHERE next_sample IS NOT NULL AND NOT censored AND e-s+1>={fps}""")[0]
    result['association'] = query_dicts(con, """WITH pairs AS (
        SELECT m.*,lag(id) OVER w AS previous_id,lag(s.frame) OVER w AS previous_frame
        FROM matches m JOIN samples s USING(video,frame)
        WINDOW w AS (PARTITION BY m.video,m.gt_id ORDER BY m.frame)),
      schedule AS (SELECT video,frame,lag(frame) OVER(PARTITION BY video ORDER BY frame) previous_sample FROM samples)
      SELECT count(*) matched_consecutive_frame_pairs,
        count(*) FILTER(id=previous_id) same_predicted_id,
        avg((id=previous_id)::DOUBLE) same_id_share
      FROM pairs JOIN schedule USING(video,frame) WHERE previous_frame=previous_sample""")[0]
    result['continuity_by_stationarity'] = continuity_by_stationarity(con, cfg)
    result['observed_box_dimensions'] = query_dicts(con, """SELECT
      quantile_cont(x2-x1,[0.1,0.5,0.9]) width_px_p10_p50_p90,
      quantile_cont(y2-y1,[0.1,0.5,0.9]) height_px_p10_p50_p90 FROM observations""")[0]
    result['matching'] = query_dicts(con, """SELECT
      (SELECT count(*) FROM observations) predicted_observations,
      (SELECT count(*) FROM gt JOIN samples USING(video,frame)) gt_boxes_on_sampled_frames,
      count(*) matched_observations,avg(iou) mean_matched_iou FROM matches""")[0]
    result['predicted_objects'] = query_dicts(con, """WITH n AS (
        SELECT o.*,len(observations) observations_n,list_contains(json_keys(matched),'[66, 66]') fallback FROM objects o)
      SELECT count(*) objects,count(*) FILTER(observations_n=1) observed_once,
        count(*) FILTER(fallback) fallback_pattern_objects,
        quantile_cont(e-s,[0.1,0.5,0.9]) interval_frames_p10_p50_p90 FROM n""")[0]
    result['patterns'] = query_dicts(con, """SELECT count(*) patterns,
      count(*) FILTER(start_region IS NOT NULL AND end_region IS NOT NULL) eligible_patterns,
      quantile_cont(frames,[0.1,0.5,0.9]) length_frames_p10_p50_p90 FROM pattern_routes""")[0]
    result['pattern_endpoints'] = query_dicts(con, 'SELECT * FROM pattern_routes ORDER BY pattern')
    # Separate geometric path coverage from rejection by configured endpoint regions.
    result['pattern_point_coverage'] = query_dicts(con, """WITH distances AS (
        SELECT o.video,o.id,o.frame,
          min(((o.x1+o.x2)/2-p.x)^2+((o.y1+o.y2)/2-p.y)^2) all_squared,
          min(((o.x1+o.x2)/2-p.x)^2+((o.y1+o.y2)/2-p.y)^2)
            FILTER(r.start_region IS NOT NULL AND r.end_region IS NOT NULL) eligible_squared
        FROM observations o CROSS JOIN pattern_points p JOIN pattern_routes r USING(pattern)
        GROUP BY o.video,o.id,o.frame)
      SELECT count(*) observations,avg((all_squared<?)::DOUBLE) near_any_pattern_share,
        avg(coalesce(eligible_squared<?,false)::DOUBLE) near_eligible_pattern_share,
        quantile_cont(sqrt(all_squared),[0.1,0.5,0.9]) nearest_any_point_px_p10_p50_p90
      FROM distances""", [cfg['dis_thresh'], cfg['dis_thresh']])[0]
    # This is the fraction of annotated detections bracketed by observations under perfect identities/detection.
    result['oracle_bracket_coverage'] = query_dicts(con, """SELECT count(*) covered_gt_boxes,
      (SELECT count(*) FROM gt) total_gt_boxes FROM gt g JOIN
        (SELECT video,id,min(g.frame) s,max(g.frame) e FROM gt g JOIN samples USING(video,frame) GROUP BY video,id) a
      ON g.video=a.video AND g.id=a.id AND g.frame BETWEEN a.s AND a.e""")[0]
    result['oracle_interpolation'] = oracle_interpolation(con)
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(method=dict(fps=15, matching_iou=MATCH_IOU,
        matching='one-to-one Hungarian, maximum valid cardinality then IoU, on sampled frames only',
        duration='annotated last-minus-first frame plus one; uncensored excludes clip boundaries',
        oracle='ground-truth identities and boxes only diagnose sampling/reconstruction, never replace baseline outputs'),
        datasets={})
    for dataset in args.datasets:
        with duckdb.connect() as con:
            con.execute('SET threads=4')
            provenance = load_dataset(con, dataset)
            match_observations(con)
            derive_tracks(con)
            result = summarize(con, provenance['config']['fps'], provenance['config'])
            result['pattern_export_anchor_fit'] = pattern_anchors(con, dataset)
            # Independent summaries must agree on the identity-link denominator and fallback behavior.
            assert sum(r['pairs'] for r in result['continuity_by_stationarity']) == result['association']['matched_consecutive_frame_pairs']
            assert sum(r['same_id'] for r in result['continuity_by_stationarity']) == result['association']['same_predicted_id']
            if not result['patterns']['eligible_patterns']:
                assert result['predicted_objects']['objects'] == result['predicted_objects']['fallback_pattern_objects']
            assert result['oracle_interpolation']['evaluated_gt_boxes'] <= result['oracle_bracket_coverage']['covered_gt_boxes']
            result['provenance'] = provenance
            report['datasets'][DATASET_LABELS[dataset]] = result
            # Export compact per-track diagnostics for reproducible duration and coverage plots.
            con.execute('COPY track_diagnostics TO ? (FORMAT PARQUET)',
                        [str(args.output_dir / f'{dataset}-tracks.parquet')])
            con.execute('COPY (SELECT p.*,r.start_region,r.end_region FROM pattern_points p JOIN pattern_routes r USING(pattern)) TO ? (FORMAT PARQUET)',
                        [str(args.output_dir / f'{dataset}-patterns.parquet')])
            write_json(args.output_dir, 'results.json', report)
            print(dataset, json.dumps({k: result[k] for k in ('duration_uncensored','coverage','association','patterns')}), flush=True)


if __name__ == '__main__':
    main()
