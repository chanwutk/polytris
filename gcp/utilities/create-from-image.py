import argparse
import subprocess


create = """
gcloud compute instances create {name} \\
    --project=video-analytics-acceleration \\
    --zone=us-west2-c \\
    --machine-type=n1-highmem-8 \\
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \\
    --maintenance-policy=TERMINATE \\
    --provisioning-model=STANDARD \\
    --service-account=652788562293-compute@developer.gserviceaccount.com \\
    --scopes=https://www.googleapis.com/auth/cloud-platform \\
    --accelerator=count=1,type=nvidia-tesla-t4 \\
    --min-cpu-platform=Intel\\ Skylake \\
    --no-shielded-secure-boot \\
    --no-shielded-vtpm \\
    --no-shielded-integrity-monitoring \\
    --labels=goog-ec-src=vm_add-gcloud \\
    --reservation-affinity=any \\
    --threads-per-core=1 \\
    --visible-core-count=4 \\
    --source-machine-image=polyis
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--stage', type=str, default=None)
    parser.add_argument('--step', type=str, default=None)
    parser.add_argument('--params', nargs='+', default=[])
    args = parser.parse_args()
    values = []
    for value in [args.system, args.dataset, args.video, args.stage, args.step, *args.params]:
        if value is not None:
            values.append(value)
    name = '-'.join(values)
    command = create.format(name=name).strip()
    print(f"Executing: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Command executed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
