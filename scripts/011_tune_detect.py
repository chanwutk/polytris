import argparse
import json
import os

import cv2

import minivan.models.retinanet_b3d

CACHE_DIR = '/minivan-cache'


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess video dataset')
    parser.add_argument('--dataset', required=False,
                        default='b3d',
                        help='Dataset name')
    parser.add_argument('--detector', required=False,
                        default='retina',
                        help='Detector name')
    return parser.parse_args()


def detect_retina(dataset_dir: str):
    detector = minivan.models.retinanet_b3d.get_detector(device='cuda:0')

    for video in os.listdir(dataset_dir):
        video_path = os.path.join(dataset_dir, video)
        if not os.path.isdir(video_path):
            continue

        print(f"Processing video {video_path}")

        for snippet in os.listdir(video_path):
            snippet_path = os.path.join(video_path, snippet)
            if not os.path.isfile(snippet_path) or not snippet.startswith('d_') or not snippet.endswith('.mp4'):
                continue

            # Process the snippet
            print(f"Processing {snippet_path}")

            meta = snippet.split('.')[0].split('_')
            start = int(meta[2])
            end = int(meta[3])

            with open(snippet_path[:-len('.mp4')] + '.jsonl', 'w') as f:
                # Read the video
                cap = cv2.VideoCapture(snippet_path)
                idx = start
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Detect objects in the frame
                    outputs = minivan.models.retinanet_b3d.detect(frame, detector)

                    f.write(json.dumps([idx, outputs[:, :4].tolist()]) + '\n')

                    idx += 1

                cap.release()


def main(args):
    dataset_dir = os.path.join(CACHE_DIR, args.dataset)
    detector = args.detector

    if detector == 'retina':
        detect_retina(dataset_dir)
    else:
        raise ValueError(f"Unknown detector: {detector}")


if __name__ == '__main__':
    main(parse_args())