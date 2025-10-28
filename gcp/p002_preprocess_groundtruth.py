#!/usr/local/bin/python

import argparse
import subprocess

from polyis.utilities import GC_CACHE, GC_DATASETS_DIR, GS_CACHE, GS_DATASETS_DIR, FakeQueue


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess groundtruth detection for a single video file')
    parser.add_argument('func_name', type=str, help='Name of the function to execute')
    parser.add_argument('video_file_path', type=str, help='Path to the input video file')
    parser.add_argument('dataset', type=str, help='Name of the dataset (used to auto-select detector)')
    parser.add_argument('output_path', type=str, help='Path where the output JSONL file will be saved')
    parser.add_argument('--gcp', action='store_true', help='Execute the code in GCP')
    
    return parser.parse_args()


def main(args):
    video_file_path = args.video_file_path
    func_name = args.func_name
    assert func_name == 'detect_objects', f"Function name must be 'detect_objects', got {func_name}"
    dataset = args.dataset
    output_path = args.output_path
    gcp = args.gcp

    # Print execution mode
    if gcp:
        from scripts.p002_preprocess_groundtruth_detection import detect_objects
        from scripts.p003_preprocess_groundtruth_tracking import track

        detect_objects(video_file_path, dataset, output_path, 0, FakeQueue())
        track(video_file_path.split('/')[-1], 'sort', dataset, 0, FakeQueue())
    else:
        print("Executing locally (not in GCP)")
        video_file = video_file_path.split('/')[-1]

        # 1. create instance from ./gcp/create-from-image.py
        #   - run python ./gcp/create-from-image.py --system <system> --dataset <dataset> --video <video> --stage <stage> --step <step> --params <params>
        #   - system: polyis
        #   - dataset: <dataset>
        #   - video: <video_file>
        #   - stage: preprocess
        #   - step: GroundtruthDetection
        stage = 'prep'
        step = 'gtdet'
        instance_name = f"polyis-{dataset}-{video_file}-{stage}-{step}"

        try:
            command = f"python ./gcp/create-from-image.py --system polyis --dataset {dataset} --video {video_file} --stage {stage} --step {step}"
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

            # 2. load data from Google Cloud Storage bucket gs://polyis to the instance
            command = f"gcloud compute ssh {instance_name} --command " \
                f"'mkdir -p {GC_DATASETS_DIR}/{dataset}/train/{video_file}'"
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            command = f"gcloud compute ssh {instance_name} --command " \
                f"'gsutil -m cp {GS_DATASETS_DIR}/{dataset}/train/{video_file} {GC_DATASETS_DIR}/{dataset}/train/{video_file}'"
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
            #   - run sudo docker exec -it bash python scripts/p002g_preprocess_groundtruth_detection.py --gcp <video_file_path> <dataset> <output_path>
            command = f"gcloud compute ssh {instance_name} --command " \
                f"'sudo docker compose up --detach --build'"
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            command = f"gcloud compute ssh {instance_name} --command " \
                f"'sudo docker exec -it bash python scripts/p002g_preprocess_groundtruth_detection.py --gcp {video_file_path} {dataset} {output_path}'"
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            # 5. copy the results back to Google Cloud Storage bucket gs://polyis
            command = f"gcloud compute ssh {instance_name} --command " \
                f"'gcloud storage rsync {GC_CACHE}/{dataset}/execution/{video_file}/000_groundtruth " \
                f"{GS_CACHE}/{dataset}/execution/{video_file}/000_groundtruth/'"
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        except Exception as e:
            print(f"Error: {e}")
            raise e
        finally:
            # 6. delete the instance
            command = f"gcloud compute instances delete {instance_name}"
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)


if __name__ == '__main__':
    main(parse_args())
