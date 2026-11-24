"""CPU-only checks of mask coverage, exit timing, and interpolation on the selected LEAP runs."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from measurements import r1o2_leap_sampling_diagnostics as diagnostic
from measurements.common import query_dicts

OUT = Path(__file__).resolve().parent
report = {}
for dataset, label in [('caldot1-y05','CalDoT1'),('caldot2-y05','CalDoT2'),('ams-y05','Amsterdam')]:
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        provenance = diagnostic.load_dataset(con, dataset)
        diagnostic.match_observations(con)
        diagnostic.derive_tracks(con)
        con.execute('''CREATE TABLE unambiguous AS
            SELECT video,id,min(gt_id) gt_id,count(*) matched_observations
            FROM matches GROUP BY video,id HAVING count(DISTINCT gt_id)=1''')
        # An exit is compared only for an annotated track whose last frame precedes the clip boundary.
        exits = query_dicts(con, '''WITH errors AS (
            SELECT m.*,o.s,o.e,t.s actual_s,t.e+1 actual_e,
                (o.e-t.e-1)/15.0 exit_error_s
            FROM unambiguous m JOIN objects o USING(video,id)
            JOIN gt_tracks t ON t.video=m.video AND t.id=m.gt_id
            JOIN clips c ON c.video=t.video WHERE t.e<c.frames-1)
            SELECT count(*) predicted_ids,count(DISTINCT (video,gt_id)) annotated_tracks,
                quantile_cont(exit_error_s,[0.1,0.5,0.9]) signed_exit_error_s_p10_p50_p90,
                median(abs(exit_error_s)) median_absolute_exit_error_s,
                avg((exit_error_s>1)::DOUBLE) over_one_second_late,
                avg((exit_error_s < -1)::DOUBLE) over_one_second_early
            FROM errors''')[0]
        # Save matches for reproducing case studies without changing any predicted identities.
        con.execute('COPY matches TO ? (FORMAT PARQUET)', [str(OUT/f'{dataset}-matches.parquet')])
        result = dict(provenance=provenance, predicted_exit_timing=exits)
        mask_path = OUT/(dataset.split('-')[0]+'_mask.jpg')
        mask_applied = provenance['mask_enabled'] and mask_path.exists()
        result['mask_applied'] = bool(mask_applied)
        if mask_applied:
            mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
            assert mask is not None and mask.shape==(provenance['config']['h'],provenance['config']['w'])
            # cv2.add uses nonzero mask pixels, including JPEG boundary artifacts.
            pixels = mask>0
            integral = np.pad(pixels.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
            rows = con.execute('SELECT video,frame,id,x1,y1,x2,y2 FROM gt ORDER BY video,frame,id').fetchall()
            coords = np.asarray([r[3:] for r in rows],dtype=float)
            h,w = mask.shape
            left=np.clip(np.floor(coords[:,0]).astype(int),0,w)
            top=np.clip(np.floor(coords[:,1]).astype(int),0,h)
            right=np.clip(np.ceil(coords[:,2]).astype(int),0,w)
            bottom=np.clip(np.ceil(coords[:,3]).astype(int),0,h)
            count=integral[bottom,right]-integral[top,right]-integral[bottom,left]+integral[top,left]
            cx=np.clip(((coords[:,0]+coords[:,2])/2).astype(int),0,w-1)
            cy=np.clip(((coords[:,1]+coords[:,3])/2).astype(int),0,h-1)
            con.execute('''CREATE TABLE mask_coverage AS SELECT unnest(?::VARCHAR[]) video,
                unnest(?::BIGINT[]) frame,unnest(?::BIGINT[]) id,
                unnest(?::BOOLEAN[]) center_kept,unnest(?::BOOLEAN[]) box_kept''',
                [[r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows],
                 pixels[cy,cx].tolist(),(count>0).tolist()])
            assert con.execute('SELECT count(*) FROM mask_coverage').fetchone()[0] == len(rows)
            result['mask'] = dict(path=str(mask_path),sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                boxes=query_dicts(con,'''WITH scope AS (
                    SELECT 'all annotated frames' scope,c.*,false sampled FROM mask_coverage c
                    UNION ALL SELECT 'recorded detector frames',c.*,true FROM mask_coverage c JOIN samples USING(video,frame))
                    SELECT scope,count(*) gt_boxes,avg((NOT center_kept)::DOUBLE) centers_masked_share,
                        avg((NOT box_kept)::DOUBLE) entire_boxes_masked_share,
                        count(*) FILTER(box_kept) boxes_with_visible_pixels FROM scope GROUP BY scope ORDER BY scope'''),
                sampled_recall=query_dicts(con,'''SELECT c.box_kept,count(*) gt_boxes,
                    count(m.id) matched_boxes,count(m.id)*1.0/count(*) matched_share
                    FROM mask_coverage c JOIN samples USING(video,frame)
                    LEFT JOIN matches m ON c.video=m.video AND c.frame=m.frame AND c.id=m.gt_id
                    GROUP BY c.box_kept ORDER BY c.box_kept'''))
        # Inspect log-confirmed background-filter skips separately from the exit-derived sample schedule.
        state = diagnostic.raw_directory(dataset)/'leap_rerun.jsonl.gz'
        result['background_filter'] = query_dicts(con,'''SELECT sum(timings.frames_filtered) filtered_frames,
            sum(len(frames_sampled)) detector_frames
            FROM read_json(?,columns={'timings':'STRUCT(frames_filtered BIGINT)','frames_sampled':'BIGINT[]'})''',[str(state)])[0]
        diagnostic.oracle_interpolation(con)
        # Restrict the geometry control to visible mask content so masked lanes cannot explain its failures.
        mask_join = 'JOIN mask_coverage USING(video,frame,id) WHERE center_kept' if mask_applied else ''
        result['visible_geometry'] = query_dicts(con,f'''WITH errors AS (
            SELECT g.*,greatest(0,least(x2,px2)-greatest(x1,px1))*
                greatest(0,least(y2,py2)-greatest(y1,py1)) inter,
                greatest(0,x2-x1)*greatest(0,y2-y1)+greatest(0,px2-px1)*greatest(0,py2-py1) total,
                ((a[1]+a[3]-b[1]-b[3])/2)^2+((a[2]+a[4]-b[2]-b[4])/2)^2 displacement_squared
            FROM oracle_boxes g {mask_join})
            SELECT displacement_squared<100 near_stationary,count(*) bracketed_boxes,
                avg(inter/nullif(total-inter,0)) mean_iou,
                avg((inter/nullif(total-inter,0)>=0.5)::DOUBLE) iou_ge_half_share
            FROM errors GROUP BY near_stationary ORDER BY near_stationary''')
        if dataset=='caldot2-y05':
            # Require a shared predicted ID and accurate endpoint detections to isolate in-between motion.
            result['linear_case'] = query_dicts(con,'''WITH areas AS (
                SELECT g.*,m1.id predicted_id,m1.iou start_iou,m2.iou end_iou,
                    greatest(0,least(g.x2,g.px2)-greatest(g.x1,g.px1))*
                    greatest(0,least(g.y2,g.py2)-greatest(g.y1,g.py1)) inter,
                    (g.x2-g.x1)*(g.y2-g.y1)+(g.px2-g.px1)*(g.py2-g.py1) total
                FROM oracle_boxes g JOIN matches m1 ON g.video=m1.video AND g.id=m1.gt_id AND g.s=m1.frame
                JOIN matches m2 ON g.video=m2.video AND g.id=m2.gt_id AND g.e=m2.frame AND m1.id=m2.id
                WHERE g.frame=floor((g.s+g.e)/2) AND g.e-g.s>=30 AND m1.iou>=0.75 AND m2.iou>=0.75
)
                SELECT *,inter/nullif(total-inter,0) interpolated_iou FROM areas
                ORDER BY interpolated_iou,e-s DESC,video,id LIMIT 1''')
        report[label] = result
        print(label,json.dumps({k:v for k,v in result.items() if k!='provenance'}),flush=True)
        (OUT/'explanation.json').write_text(json.dumps(report,indent=2)+'\n')
