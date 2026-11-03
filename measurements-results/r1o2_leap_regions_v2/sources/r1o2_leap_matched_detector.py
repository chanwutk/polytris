"""Use Tetris's fine-tuned detector in LEAP's labeling and extraction interfaces.

Runs in the original LEAP environment, from its read-only checkout. Inference uses
the same Ultralytics calls and defaults as polyis/models/ultralytics.py. Only the
class identifier changes: our single car class 0 becomes LEAP's COCO car class 2.
"""
import os
import runpy
import sys
import time

import torch
from ultralytics import YOLO


class MatchedDetector:
    def __init__(self, model_path, input_size=640, device='0', *unused_thresholds):
        # Match Tetris's get_detector, including fusion and the model's default FP32 inference.
        self.model = YOLO(model_path, verbose=False)
        assert self.model.names == {0: 'car'}, f'Unexpected classes: {self.model.names}'
        self.model.fuse()
        self.model.to(f'cuda:{device}' if str(device).isdigit() else device)
        self.model.eval()

    def detect(self, image):
        # Match Tetris's detect: do not inherit LEAP's old COCO class filter or NMS thresholds.
        result = self.model(image, verbose=False)[0]
        if result.boxes is None:
            return image, []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        # LEAP uses the class for trajectory bookkeeping; evaluation remains a single class.
        return image, [[*box.tolist(), float(score), 2] for box, score in zip(boxes, scores)]


class MatchedLabelDetector(MatchedDetector):
    def __init__(self, cfg):
        super().__init__(cfg.MODEL.YOLO_WEIGHT)
        self.detactor_time = []

    def preprocess(self, cfg, size):
        # Ultralytics applies the same image preprocessing as Tetris inside its prediction call.
        pass

    def __call__(self, image, timer):
        # Preserve LEAP's labeling timer and the tensor shape its BYTETracker expects.
        self.img0 = image
        timer.tic()
        start = time.perf_counter()
        _, boxes = self.detect(image)
        self.detactor_time.append(time.perf_counter() - start)
        return (torch.tensor(boxes, dtype=torch.float32), True) if boxes else ([], False)


def label():
    # Repeat LEAP's multi-video entrypoint with the matched detector injected at its existing boundary.
    from video_labeling import video_tracker
    video_tracker.Detector = MatchedLabelDetector
    args = video_tracker.make_parser().parse_args()
    original_save_path = args.save_path
    for name in sorted(f for f in os.listdir(args.video_dir) if f.endswith('.mp4')):
        args.video_name = os.path.splitext(name)[0]
        args.video_path = os.path.join(args.video_dir, name)
        args.save_path = os.path.join(os.path.dirname(original_save_path), args.video_name,
                                      os.path.basename(original_save_path))
        os.makedirs(args.save_path, exist_ok=True)
        video_tracker.logger.info(f'Processing {name}')
        video_tracker.labeling_video(args)


if __name__ == '__main__':
    # Imports resolve to LEAP's checkout, just as in the existing rerun wrapper.
    sys.path.insert(0, os.getcwd())
    if len(sys.argv) > 1 and sys.argv[1] == 'label':
        del sys.argv[1]
        label()
    else:
        # Keep LEAP's sampler, matcher, exporter, and per-clip timers unchanged.
        import main as leap_main
        leap_main.Detector = MatchedDetector
        runpy.run_path('/polyis/measurements-gpu/r1o2_leap_otif_rerun_leap.py', run_name='__main__')
