import argparse


create = """
gcloud compute instances create {system}-{dataset}-{video}-{stage}-{step}-{params} \\
    --project=video-analytics-acceleration \\
    --zone=us-central1-a \\
    --machine-type=n1-highmem-8 \\
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \\
    --metadata=enable-osconfig=TRUE \\
    --maintenance-policy=TERMINATE \\
    --provisioning-model=STANDARD \\
    --service-account=652788562293-compute@developer.gserviceaccount.com \\
    --scopes=https://www.googleapis.com/auth/cloud-platform \\
    --accelerator=count=1,type=nvidia-tesla-t4 \\
    --min-cpu-platform=Intel\\ Skylake \\
    --tags=http-server,https-server,lb-health-check \\
    --create-disk=auto-delete=yes,boot=yes,device-name=instance-20251023-230834,image=projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-conda,mode=rw,size=200,type=pd-ssd \\
    --no-shielded-secure-boot \\
    --no-shielded-vtpm \\
    --no-shielded-integrity-monitoring \\
    --labels=goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud \\
    --reservation-affinity=any \\
    --threads-per-core=1 \\
    --visible-core-count=4
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--stage', type=str, required=True)
    parser.add_argument('--step', type=str, required=True)
    parser.add_argument('--params', nargs='+', required=True)
    args = parser.parse_args()
    # Join params list into a string for the template
    args_dict = args.__dict__.copy()
    args_dict['params'] = '-'.join(args.params)
    print(create.format(**args_dict))


if __name__ == '__main__':
    main()
