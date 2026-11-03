"""Overlay the recorded LEAP stop_area rectangles on one real frame per dataset."""
import argparse
import base64
import json
from pathlib import Path

import altair as alt
import cv2

from evaluation.manifests import list_split_videos
from measurements.common import DATASET_LABELS
from polyis.utilities import DATASETS_DIR


ROOT = Path(__file__).resolve().parent.parent
ORDER = list(DATASET_LABELS.values())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=ROOT / 'measurements-results/r1o2_leap_sampling_diagnostics')
    return parser.parse_args()


def frame_chart(uri, record, width):
    # Image and rectangles share native pixel coordinates, including the downward-positive image Y axis.
    native_w, native_h = record['resolution']
    height = width*native_h/native_w
    xscale = alt.Scale(domain=[0, native_w], nice=False)
    yscale = alt.Scale(domain=[0, native_h], nice=False, reverse=True)
    image = alt.Chart(alt.Data(values=[dict(url=uri, x=native_w/2, y=native_h/2)])).mark_image(
        width=width, height=height).encode(
            url='url:N', x=alt.X('x:Q', scale=xscale, axis=None), y=alt.Y('y:Q', scale=yscale, axis=None))
    boxes = alt.Chart(alt.Data(values=record['rectangles'])).mark_rect(
        fill='#ff3030', fillOpacity=0.06, stroke='#ff3030', strokeWidth=3).encode(
            x=alt.X('left:Q', scale=xscale, axis=None), x2='right:Q',
            y=alt.Y('top:Q', scale=yscale, axis=None), y2='bottom:Q')
    labels = alt.Chart(alt.Data(values=record['rectangles'])).mark_text(
        align='left', baseline='top', dx=6, dy=6, fontSize=max(14, 22*width/native_w),
        fontWeight='bold', color='white', stroke='black', strokeWidth=0.6).encode(
            x=alt.X('left:Q', scale=xscale, axis=None), y=alt.Y('top:Q', scale=yscale, axis=None), text='label:N')
    return (image+boxes+labels).properties(width=width, height=height, title=alt.Title(
        record['dataset'], subtitle=f"{record['video']} | frame {record['frame']} (0-based) | {native_w} x {native_h}"))


def main():
    args = parse_args()
    report = json.loads((args.results_dir / 'results.json').read_text())
    inverse = {label: ds for ds, label in DATASET_LABELS.items()}
    output = args.results_dir / 'region_frames'
    output.mkdir(parents=True, exist_ok=True)
    records, previews = [], []
    for label in ORDER:
        ds = inverse[label]
        cfg = report['datasets'][label]['provenance']['config']
        # A middle frame from the first test clip shows the scene without selecting for any result.
        video = sorted(list_split_videos(ds, 'test'))[0]
        source = Path(DATASETS_DIR)/ds/'test'/video
        capture = cv2.VideoCapture(str(source))
        assert capture.isOpened(), source
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count>0
        frame = count//2
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, pixels = capture.read()
        capture.release()
        assert success and pixels is not None, (source, frame)
        h, w = pixels.shape[:2]
        assert (w, h)==(cfg['w'], cfg['h']), (ds, (w, h), (cfg['w'], cfg['h']))
        rectangles = [dict(label=f'R{i}', index=i, left=r[0], top=r[1], right=r[2], bottom=r[3])
                      for i, r in enumerate(cfg['stop_area'])]
        assert rectangles and all(0<=r['left']<r['right']<=w and 0<=r['top']<r['bottom']<=h for r in rectangles)
        original = output/f'{ds}-frame.png'
        assert cv2.imwrite(str(original), pixels)
        record = dict(dataset=label, dataset_id=ds, source_video=str(source), video=video,
                      frame=frame, resolution=[w, h], rectangles=rectangles)
        records.append(record)
        uri='data:image/png;base64,'+base64.b64encode(original.read_bytes()).decode('ascii')
        chart=frame_chart(uri, record, w)
        chart.save(output/f'{ds}-regions.png')
        chart.save(output/f'{ds}-regions.pdf')
        previews.append(frame_chart(uri, record, 420))
        print(label, video, frame, 'regions', len(rectangles), flush=True)
    # The overview and native-size pages contain identical rectangles; only presentation size differs.
    overview=alt.vconcat(*(alt.hconcat(*previews[i:i+3], spacing=18) for i in range(0,len(previews),3)),
                         spacing=22).properties(title=alt.Title(
        "LEAP's configured entry/exit rectangles on actual video frames",
        subtitle='Red rectangles and R0-R3 labels reproduce the stop_area values saved with each run.'))
    overview.save(output/'overview.png')
    overview.save(output/'overview.pdf')
    (output/'frames.json').write_text(json.dumps(records, indent=2)+'\n')
    assert len(records)==len(ORDER)


if __name__ == '__main__':
    main()
