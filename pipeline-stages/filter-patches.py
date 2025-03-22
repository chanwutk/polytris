import sys
import pathlib

# ROOT = pathlib.Path().absolute().parent
MODULES_PATH = pathlib.Path().absolute() / 'modules'
# sys.path.append(str(ROOT))
sys.path.append(str(MODULES_PATH))
sys.path.append(str(MODULES_PATH / 'detectron2'))

import time
import shutil
import multiprocessing as mp
from typing import NamedTuple
import torch
import os
import numpy as np
import json
import cv2
import argparse

from b3d.utils import parse_outputs, regionize_image
from b3d.external.sort import Sort
from b3d.external.nms import nms
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


CONFIG = './modules/b3d/configs/config_refined.json'
MASK = './masks.xml'

PIPELINE_DIR = 'pipeline-stages'
TRACK_RESULTS_DIR = os.path.join(PIPELINE_DIR, 'track-results')
VIDEO_DIR = os.path.join(PIPELINE_DIR, 'video-masked')


class Input(NamedTuple):
    video: str
    gpu: str


def hex_to_rgb(hex: str) -> tuple[int, int, int]:
    # hex in format #RRGGBB
    return int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)


colors_ = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
*colors, = map(hex_to_rgb, colors_)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Example detection and tracking script')
    parser.add_argument('-v', '--video', required=True,
                        help='Input video')
    # parser.add_argument('-c', '--config', required=True,
    #                     help='Detection model configuration')
    parser.add_argument('-g', '--gpu', required=True,
                        help='GPU device')
    # parser.add_argument('-m', '--mask', required=True,
    #                     help='Mask for the video')
    return parser.parse_args()


def main(args):
    videofilename = args.video.split('/')[-1]
    device = f'cuda:{args.gpu}'

    with torch.no_grad():
        fpd = open(os.path.join(TRACK_RESULTS_DIR,
                   f'{videofilename}.d.jsonl'), 'w')
        fpr = open(os.path.join(TRACK_RESULTS_DIR,
                   f'{videofilename}.r.jsonl'), 'w')
        fpp = open(os.path.join(TRACK_RESULTS_DIR,
                   f'{videofilename}.p.jsonl'), 'w')
        try:
            with open(CONFIG) as fp:
                config = json.load(fp)
            cfg = get_cfg()
            cfg.merge_from_file(os.path.join(
                './modules/detectron2/configs', config['config']))
            cfg.MODEL.WEIGHTS = config['weights']
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['num_classes']
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['score_threshold']
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config['score_threshold']
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
            cfg.MODEL.RETINANET.NMS_THRESH_TEST = config['nms_threshold']
            cfg.TEST.DETECTIONS_PER_IMAGE = config['detections_per_image']
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config['anchor_generator_sizes']
            cfg.MODEL.DEVICE = device
            predictor = DefaultPredictor(cfg)

            tracker = Sort(max_age=5)
            cap = cv2.VideoCapture(os.path.expanduser(args.video))
            trajectories = {}
            frame_index = 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_sep = frame_count // 5

            def log(action: str, start: int, end: int, **kwargs):
                start //= 1000
                end //= 1000
                fpp.write(json.dumps({
                    'fid': frame_index,
                    'action': action,
                    'time': end - start,
                    'start': start,
                    'end': end,
                    **kwargs
                }) + '\n')

            inference_time = []
            while cap.isOpened():
                print(f'{videofilename} -- GPU {device}' +
                      'Parsing frame {:d} / {:d}...'.format(frame_index, frame_count))
                start = time.time_ns()
                success, frame = cap.read()
                if not success:
                    break
                end = time.time_ns()
                log('read', start, end)

                start = time.time_ns()
                frame_masked = frame.copy()
                end = time.time_ns()
                log('mask', start, end)

                if frame_index % frame_sep == 0:
                    cv2.imwrite(os.path.join(TRACK_RESULTS_DIR, f'{
                                videofilename}.{frame_index}.0.jpg'), frame_masked)

                start = time.time_ns()
                image_regions = regionize_image(frame_masked)
                assert len(image_regions) == 1, len(image_regions)

                _image, _offset = image_regions[0]
                assert _offset == (0, 0), _offset
                frame_resolution = _image.shape
                # TODO: new predictor with GPU image transformation
                _outputs = predictor(_image)
                bboxes, scores, _ = parse_outputs(_outputs, _offset)

                nms_threshold = config['nms_threshold']
                nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
                detections = np.zeros((len(nms_bboxes), 5))
                detections[:, 0:4] = nms_bboxes
                detections[:, 4] = nms_scores
                end = time.time_ns()
                inference_time.append(end-start)
                log('detect', start, end, resolutions=frame_resolution, num_detections=len(
                    detections), num_regions=len(image_regions))

                fpd.write(json.dumps(
                    [frame_index, detections.tolist()]) + '\n')

                start = time.time_ns()
                tracked_objects = tracker.update(detections)
                end = time.time_ns()
                log('track', start, end, num_tracks=len(tracked_objects))

                rendering = []
                for tracked_object in tracked_objects:
                    tl = (int(tracked_object[0]), int(tracked_object[1]))
                    br = (int(tracked_object[2]), int(tracked_object[3]))
                    object_index = int(tracked_object[4])
                    if object_index not in trajectories:
                        trajectories[object_index] = []
                    trajectories[object_index].append([
                        frame_index, tl[0], tl[1], br[0], br[1]])
                    rendering.append([
                        object_index, tl[0], tl[1], br[0], br[1]])
                fpr.write(json.dumps([frame_index, rendering]) + '\n')

                if frame_index % 50 == 0:
                    fpd.flush()
                    fpr.flush()
                    fpp.flush()

                frame_masked_r = frame_masked.copy()
                frame_masked_d = frame_masked.copy()
                if frame_index % frame_sep == 0:
                    for object_index, tlx, tly, brx, bry in rendering:
                        cv2.rectangle(
                            frame_masked_r, (tlx, tly), (brx, bry), colors[object_index % len(colors)], 2)
                        cv2.putText(frame_masked_r, str(object_index), (tlx, tly),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[object_index % len(colors)], 2)
                    cv2.imwrite(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.{
                                frame_index}.r.jpg'), frame_masked_r)
                    for bbox in nms_bboxes:
                        tlx, tly, brx, bry = map(int, bbox)
                        cv2.rectangle(frame_masked_d, (tlx, tly),
                                      (brx, bry), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.{
                                frame_index}.d.jpg'), frame_masked_d)

                frame_index = frame_index + 1
            cap.release()
            cv2.destroyAllWindows()

            with open(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.t.json'), 'w') as fp:
                json.dump(trajectories, fp)
            with open(os.path.join(TRACK_RESULTS_DIR, f'{videofilename}.done'), 'w') as fp:
                fp.write('done')
            print(f'{videofilename} -- GPU {device} -- Inference time: {
                  sum(inference_time) / 1_000_000 / 512:.2f} ms')
        finally:
            fpd.close()
            fpr.close()
            fpp.close()


if __name__ == '__main__':
    # main(parse_args())

    if os.path.exists(TRACK_RESULTS_DIR):
        shutil.rmtree(TRACK_RESULTS_DIR)
    os.makedirs(TRACK_RESULTS_DIR)

    processes = []
    try:
        for idx, videofile in enumerate(os.listdir(VIDEO_DIR)):
            assert videofile.endswith('.mp4')
            if videofile.endswith('.x264.mp4'):
                continue
            if not videofile.startswith('jnc'):
                continue

            input = Input(os.path.join(VIDEO_DIR, videofile),
                          str(idx % torch.cuda.device_count()))
            # main(input)
            p = mp.Process(target=main, args=(input,))
            p.start()
            processes.append(p)
            print(f'Processed {videofile}')

        for p in processes:
            p.join()
            print(f'Joined {p}')

    finally:
        for p in processes:
            p.terminate()
            print(f'Terminated {p}')
