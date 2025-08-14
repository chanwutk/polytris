import os
import json
import multiprocessing as mp
import time

import cv2
import numpy as np
import torch

import sys
sys.path.append('/data/chanwutk/projects/polyis/modules/b3d')
sys.path.append('/data/chanwutk/projects/polyis/modules/detectron2')

from b3d.external.nms import nms
from b3d.external.sort import iou_batch, linear_assignment
from polyis.models.retinanet_b3d import get_detector


VIDEOS = '/data/chanwutk/projects/polyis/videos_crop'
VIDEOS_VALID = '/data/chanwutk/projects/polyis/videos_validate'

CONFIG = os.path.join('/data/chanwutk/projects/polyis/modules', 'b3d/b3d/configs/config_refined.json')


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Example masking script')
    parser.add_argument('-v', '--videos', required=False,
                        default=VIDEOS,
                        help='Video directory')
    parser.add_argument('-o', '--output', required=False,
                        default=VIDEOS_VALID,
                        help='Output directory')
    return parser.parse_args()


def process_video(files: str, videodir, outputdir, gpuIdx):
    with open(CONFIG) as fp:
        config = json.load(fp)
    predictor = get_detector(f'cuda:{gpuIdx}')

    if not os.path.exists(os.path.join(outputdir, files)):
        os.makedirs(os.path.join(outputdir, files))

    for file in sorted(os.listdir(os.path.join(videodir, files))):
        video_path = os.path.join(videodir, files, file)
        print(f'Processing {video_path}...')
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(os.path.join(outputdir, files, file), cv2.VideoWriter.fourcc(*'mp4v'), 15, (width, height))

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            print('Parsing frame {:d} / {:d}...'.format(idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            if not ret:
                break
            idx += 1

            if idx % 8 != 0:
                continue
        
            frame0 = frame.copy()

            start = time.time()
            frame1 = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
            frame1.to(predictor.cfg.MODEL.DEVICE)
            inputs = {"image": frame1, "height": height, "width": width}
            output = predictor.model([inputs])[0]
            bboxes = output['instances'].pred_boxes.tensor.detach().cpu().numpy()
            scores = output['instances'].scores.detach().cpu().numpy()
            # bboxes, scores, _ = parse_outputs(output, (0, 0))
            nms_threshold = config['nms_threshold']
            nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
            detections1 = np.zeros((len(nms_bboxes), 5))
            if len(nms_bboxes) > 0:
                detections1[:, 0:4] = nms_bboxes
                detections1[:, 4] = nms_scores
            end = time.time()
            time1 = end - start

            start = time.time()
            output = predictor(frame)
            # bboxes, scores, _ = parse_outputs(output, (0, 0))
            bboxes = output['instances'].pred_boxes.tensor.detach().cpu().numpy()
            scores = output['instances'].scores.detach().cpu().numpy()
            nms_threshold = config['nms_threshold']
            nms_bboxes, nms_scores = nms(bboxes, scores, nms_threshold)
            detections0 = np.zeros((len(nms_bboxes), 5))
            if len(nms_bboxes) > 0:
                detections0[:, 0:4] = nms_bboxes
                detections0[:, 4] = nms_scores
            end = time.time()
            time0 = end - start

            proportion = int(100 * time1 / time0)
            print(f"Detection time: {time0:.3f} {time1:.3f}")
            print("*" * proportion, "." * (100 - proportion), f"{proportion}%")
            
            ious = iou_batch(detections0[:, :4], detections1[:, :4])
            matched_indices = linear_assignment(-ious)
            unmatched_detections0 = []
            for d, det in enumerate(detections0):
                if(d not in matched_indices[:,0]):
                    unmatched_detections0.append(d)
            unmatched_detections1 = []
            for t, trk in enumerate(detections1):
                if(t not in matched_indices[:,1]):
                    unmatched_detections1.append(t)
            
            for d0, d1 in matched_indices:
                det0 = detections0[d0]
                det1 = detections1[d1]
                frame0 = cv2.rectangle(frame0, (int(det0[0]), int(det0[1])), (int(det0[2]), int(det0[3])), (0, int(255 * det0[4]), 0), 2)
                frame0 = cv2.rectangle(frame0, (int(det1[0]), int(det1[1])), (int(det1[2]), int(det1[3])), (0, 0, int(255 * det1[4])), 2)
            
            for d in unmatched_detections0:
                det0 = detections0[d]
                frame0 = cv2.rectangle(frame0, (int(det0[0]), int(det0[1])), (int(det0[2]), int(det0[3])), (int(255 * det0[4]), 0, 0), 2)
            
            for d in unmatched_detections1:
                det1 = detections1[d]
                frame0 = cv2.rectangle(frame0, (int(det1[0]), int(det1[1])), (int(det1[2]), int(det1[3])), (0, int(255 * det1[4]), int(255 * det1[4])), 2)
            
            writer.write(frame0)

        cap.release()


def main(args):
    videodir = args.videos
    outputdir = args.output


    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    
    processes: list[mp.Process] = []
    count = 0
    for file in os.listdir(videodir):
        process = mp.Process(target=process_video, args=(file, videodir, outputdir, count % torch.cuda.device_count()))
        process.start()
        processes.append(process)
        count += 1
    
    for process in processes:
        process.join()
        process.terminate()


if __name__ == '__main__':
    main(parse_args())