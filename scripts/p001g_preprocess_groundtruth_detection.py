#!/usr/local/bin/python

import argparse
import os
import subprocess

from polyis.utilities import GC_CACHE, GC_DATA, GS_CACHE, GS_DATA, FakeQueue


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess groundtruth detection for a single video file')
    parser.add_argument('--video_file_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (used to auto-select detector)')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the output JSONL file will be saved')
    parser.add_argument('--gcp', action='store_true', help='Execute the code in GCP')
    
    return parser.parse_args()


def main(args):
    video_file_path = args.video_file_path
    dataset = args.dataset
    output_path = args.output_path
    gcp = args.gcp

    # Print execution mode
    if gcp:
        from scripts.p001_preprocess_groundtruth_detection import detect_objects
        detect_objects(video_file_path, dataset, output_path, 0, FakeQueue())
    else:
        print("Executing locally (not in GCP)")
        video_file = os.path.basename(video_file_path)
        # output_path = os.path.join(CACHE_DIR, dataset, 'execution', video_file, '000_groundtruth', 'detections.jsonl')

        # 1. create instance from ./gcp/create-from-image.py
        #   - run python ./gcp/create-from-image.py --system <system> --dataset <dataset> --video <video> --stage <stage> --step <step> --params <params>
        #   - system: polytris
        #   - dataset: <dataset>
        #   - video: <video_file>
        #   - stage: preprocess
        #   - step: GroundtruthDetection
        command = f"python ./gcp/create-from-image.py --system polytris --dataset {dataset} --video {video_file} --stage preprocess --step GroundtruthDetection"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        # 2. load data from Google Cloud Storage bucket gs://polytris to the instance
        instance_name = f"polytris-{dataset}-{video_file}-preprocess-GroundtruthDetection"
        command = f"gcloud compute ssh {instance_name} --command " \
            f"'mkdir -p {GC_DATA}/{dataset}'"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        command = f"gcloud compute ssh {instance_name} --command " \
            f"'gsutil -m cp {GS_DATA}/{dataset}/{video_file} {GC_DATA}/{dataset}/'"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # 3. copy the repository to the instance
        command = f"gcloud compute scp --recurse configs polyis scripts dock docker-compose.yml Dockerfile environment.yml pyproject.toml requirements.txt {instance_name}:/home/chanwutk/polyis"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        command = f"gcloud compute ssh {instance_name} --command " \
            f"'mkdir -p /home/chanwutk/polyis/modules'"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        command = f"gcloud compute scp --recurse modules/darknet {instance_name}:/home/chanwutk/polyis/modules"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # 4. execute the code in the instance
        #   - run sudo docker compose up --detach --build
        #   - run sudo docker exec -it bash python scripts/p001g_preprocess_groundtruth_detection.py --video_file_path <video_file_path> --dataset <dataset> --output_path <output_path> --gcp
        command = f"gcloud compute ssh {instance_name} --command " \
            f"'sudo docker compose up --detach --build'"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        command = f"gcloud compute ssh {instance_name} --command " \
            f"'sudo docker exec -it bash python scripts/p001g_preprocess_groundtruth_detection.py --video_file_path {video_file_path} --dataset {dataset} --output_path {output_path} --gcp'"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # 5. copy the results back to Google Cloud Storage bucket gs://polytris
        command = f"gcloud compute ssh {instance_name} --command " \
            f"'gcloud storage rsync {GC_CACHE}/{dataset}/execution/{video_file}/000_groundtruth " \
            f"{GS_CACHE}/{dataset}/execution/{video_file}/000_groundtruth/'"
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)


if __name__ == '__main__':
    main(parse_args())
