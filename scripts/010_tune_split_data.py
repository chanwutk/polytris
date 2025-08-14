import argparse
import os

import cv2


SELECTIVITY = 0.05
CLASSIFIER_SIZES = [32, 64, 128]
PADDING_SIZES = [0, 1, 2]
DIFF_THRESHOLDS = [10, 20, 30]
DIFF_SCALE = [1, 2, 4]


DATASET_DIR = '/polyis-data/video-datasets-low'
CACHE_DIR = '/polyis-cache'


def parse_args():
    parser = argparse.ArgumentParser(description="Tune parameters for the model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='b3d',
        help="The dataset name.",
    )
    parser.add_argument(
        "--selectivity",
        type=float,
        default=SELECTIVITY,
        help="Selectivity parameter for tuning.",
    )
    parser.add_argument(
        "--num_snippets",
        type=int,
        default=10,
        help="Number of snippets to extract from the video for tuning.",
    )
    parser.add_argument(
        "--tracking_selectivity_multiplier",
        type=int,
        default=4,
        help="Multiplier for tracking selectivity.",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=DATASET_DIR,
        help="Directory containing the dataset.",
    )
    return parser.parse_args()


# todo: cut snippet of videos for tuning
# todo: run detector
# todo: split dataset
# todo: train classifier


def save_snippet(input_video, output_video, start, end):
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    width, height, fps = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )
    # print(input_video, output_video, start, end)

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))

    idx = start
    while cap.isOpened():
        ret, frame = cap.read()
        # print(idx)
        if not ret or idx > end:
            break
        writer.write(frame)
        idx += 1
    
    cap.release()
    writer.release()


def main(args):
    datasets_dir = args.datasets_dir
    dataset = args.dataset
    selectivity = args.selectivity

    for input_video in os.listdir(os.path.join(datasets_dir, dataset)):
        if not input_video.endswith(".mp4"):
            continue

        output_dir = os.path.join(dataset, input_video)
        input_video = os.path.join(datasets_dir, dataset, input_video)
        print(input_video)

        cap = cv2.VideoCapture(input_video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(num_frames)

        num_d_snippets = args.num_snippets
        num_t_snippets = num_d_snippets * args.tracking_selectivity_multiplier

        snippet_size = int(num_frames * selectivity / num_d_snippets)

        starts_d = [s * (num_frames // num_d_snippets) for s in range(num_d_snippets)]
        ends_d = [s + snippet_size for s in starts_d]

        starts_t = [s * (num_frames // num_t_snippets) for s in range(num_t_snippets)]
        ends_t = [s + snippet_size for s in starts_t]

        if not os.path.exists(os.path.join(CACHE_DIR, output_dir)):
            os.makedirs(os.path.join(CACHE_DIR, output_dir))

        for i, (start, end) in enumerate(zip(starts_d, ends_d)):
            save_snippet(input_video, os.path.join(CACHE_DIR, f"{output_dir}/d_{i}_{start}_{end}.mp4"), start, end)
        
        for i, (start, end) in enumerate(zip(starts_t, ends_t)):
            save_snippet(input_video, os.path.join(CACHE_DIR, f"{output_dir}/t_{i}_{start}_{end}.mp4"), start, end)


if __name__ == "__main__":
    main(parse_args())