#!/usr/bin/env python3
"""R1.O2 and R9.O2 on bombe: rerun LEAP's inference on our test clips, and time LEAP's preparation. Host-side.

Runs on the bombe host (python3 standard library only) from the shared checkout, and starts every GPU job through
measurements-gpu/bombe_container.sh (idle-GPU check, NUMA pinning, GPU sampling, contamination flag) with
BOMBE_IMAGE set to the LEAP image. Without --run it prints what it would do. Actions:

  setup   Build the LEAP image and the run inputs under SCRATCH (nothing outside it is written):
          - image bombe-revision-leap:scored-env: LEAP's own image as built on bombe from its Dockerfile
            (docker-config-leap-alenatunnel-leap) with /opt/conda/envs/leap replaced by the env of the container
            the scored runs used (manchester container alenatunnel-leap, 2026-03-16 to 2026-09-12; detectron2, yacs,
            fvcore, and faiss were installed into that running container, so the image alone cannot run LEAP).
            The env is SCRATCH/env/leap/leap-env.tar (`docker cp alenatunnel-leap:/opt/conda/envs/leap -`).
          - SCRATCH/inputs/leap-dataset: a copy of the scored workspace manchester:/data/alenachao/data/leap-dataset
            (configs, pseudo labels without their visualization videos, motion patterns, detector weights, final
            re-identification weights with their training logs, LEAP's run logs, its videos, and the scored
            parsed results), with SCRATCH/inputs/leap-dataset.source.md5 taken on manchester. LEAP reads labels
            from labels/<full_name>; CalDoT's are in labels/caldot1-y5 and caldot2-y5, so setup adds the aliases
            labels/caldot1 and labels/caldot2 (labels only feed LEAP's own evaluation printout and its
            visualizations, never its sampling, matching, or intervals).
          - SCRATCH/inputs/LEAP: a copy of LEAP's checkout (/work/alenachao/tunnel/work/LEAP, which the scored
            container mounted as /work/alenachao/LEAP) without logs/, outputs/, runs/, datasets/, data/videos/.
          - SCRATCH/videos/<dataset>/<split>/<N>.mp4: links to our clips under /polyis-data (te00.mp4 -> 0.mp4, as
            LEAP's rename_leap_videos.py named them), so LEAP sees the names and order of the scored runs.
          - raw/manifest.json: images, env provenance, and the checksum comparison of LEAP's videos with our clips.
  scored  Package the scored parsed results (SCRATCH/inputs/leap-dataset/outputs/parsed_results/<full_name>) as
          raw/<dataset>/leap_scored.jsonl.gz (one line per clip) and leap_scored_run.jsonl.gz (patterns, speed.tsv).
  leap    Per dataset, one container on --gpu runs measurements-gpu/r1o2_leap_otif_rerun_leap.py (LEAP's main.py
          loop plus the state export) under GNU time; writes raw/<dataset>/leap_rerun.jsonl.gz,
          leap_rerun_run.jsonl.gz, leap_rerun_launch.json (launcher summary, CPU set, contamination, GNU time),
          leap_rerun_samples.jsonl.gz, and LEAP's log and GNU time output as .log files.
  export  Rebuild the track exports (/sota-results/<dataset>/leap/<clip>.json format) of the rerun's and the scored
          parsed results with the scored runs' exporter (export-leap-outputs.py's reconstruct_tracks_json, run
          unchanged by measurements-gpu/r1o2_leap_otif_rerun_export.py in the LEAP image, CPU only):
          raw/<dataset>/leap_rerun_export.jsonl.gz and leap_scored_export.jsonl.gz.
  prep    Per --system, dataset, and step, time the system's preparation in one launcher container per step on --gpu
          under GNU time, writing only into SCRATCH/prep/<system>/<dataset>/. A step whose record exists with exit 0 is
          skipped (--force reruns it), so a killed queue resumes. Per step it writes raw/prep/<system>/<dataset>/
          <step>.json (launcher summary, CPU set, contamination, GNU time, command, output size),
          <step>_samples.jsonl.gz, and <step>_stdout.log.gz, plus the step's outputs next to the scored run's
          (<step>_compare.json[l.gz]) so the aggregation can check that the reconstruction reproduces them.
          - leap (LEAP_PREP_STEPS, in the LEAP image, the read-only checkout copy as working directory): pseudo labels
            of the train and test clips (README step 4; video_labeling/video_tracker.py's multi-video mode with
            weights/yolov5x.pt and no mask, or RetinaNet for B3D: the combination that reproduces the scored labels),
            the re-identification crops (step 6a, vehicle_reid2/generate_reid.py), re-identification training (step
            6b, vehicle_reid2/run.sh, 120 epochs from veriwild_bot_R50-ibn.pth as in the scored logs), and the
            motion-pattern clustering (track_cluster_updated.py). Step 7, distillation, was skipped in the scored runs.
          - otif (OTIF_PREP_STEPS, in bombe-revision-otif:scored-env, a snapshot of the exited container the scored
            OTIF runs used on bombe, with Alena's OTIF fork read-only at /otif): YOLOv5 or B3D RetinaNet detection of the train clips,
            IoU tracks, frame extraction, segmentation-proxy training (5 resolutions), RNN tracker training,
            detection of the validation clips, and the validation tuning (30_speed_accuracy.go ours-simple, 10
            iterations from OTIF_TUNE_START), on a copy-on-write clone of the scored data root
            (/data/alenachao/data/otif-dataset) with the outputs these steps regenerate removed, and each step's own
            output removed again before it runs. The first OTIF prep call builds OTIF_GOPATH and writes
            raw/prep/otif_env.json. OTIF's YOLOv3 training and detector-resolution sweep (README pre-processing) were
            not part of the scored y5 runs, whose detector is our fine-tuned YOLOv5.
  compare Rebuild the <step>_compare files of finished prep steps from their outputs (no container on a GPU).
  gpus    For every timed run on record (LEAP reruns and prep steps), write every GPU's utilization and memory during
          the run (the launcher's <run>.gpus.csv) as <record>_gpus.jsonl.gz next to its samples, so the aggregation
          can report what else ran on the host's GPUs (GPU 4 shares NUMA node 1, and GPUs 5 and 6 a PCIe switch).

raw/ is measurements-results/r1o2_leap_otif_rerun/raw/ in the shared checkout; measurements/r1o2_leap_otif_rerun.py
aggregates it.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


# Everything this track writes on bombe, and the shared checkout (the same files on ace and bombe).
ROOT = Path('/data/cwkt/data/bombe-revision')
SCRATCH = ROOT / 'scratch' / 'r1o2_leap_otif_rerun'
TETRIS = Path('/work/cwkt/projects/tetris')
RAW = TETRIS / 'measurements-results' / 'r1o2_leap_otif_rerun' / 'raw'
LAUNCHER = TETRIS / 'measurements-gpu' / 'bombe_container.sh'
RUNNER = '/polyis/measurements-gpu/r1o2_leap_otif_rerun_leap.py'
# The exporter that wrote the scored /sota-results/<dataset>/leap/<clip>.json files.
EXPORTER_SOURCE = Path('/work/alenachao/tunnel/work/scripts/export-leap-outputs.py')
# LEAP's image as built on bombe, and the image this track builds from it with the scored container's env.
LEAP_BASE_IMAGE = 'docker-config-leap-alenatunnel-leap:latest'
LEAP_IMAGE = 'bombe-revision-leap:scored-env'
DOCKERFILE = f'FROM {LEAP_BASE_IMAGE}\nRUN rm -rf /opt/conda/envs/leap\nADD leap-env.tar /opt/conda/envs/\n'
# Container paths: LEAP reads labels from this absolute path (settings.label_path), so the workspace copy is mounted
# there; LEAP's checkout copy is the working directory; GNU time comes from the host (the image has none).
LEAP_DATASET_MOUNT = '/data/alenachao/data/leap-dataset'
LEAP_CODE_MOUNT = '/leap'
GNU_TIME = '/opt/gnu-time'
# How long a queued step waits for its GPU when another user holds it.
BUSY_WAIT_S = 6 * 3600
# GPUs 0/2 use NUMA node 0 and 5/6 node 1; each queue gets its own 10 cores and their hyperthreads.
GPU_CPUS = {0: '0-9,40-49', 2: '10-19,50-59', 5: '20-29,60-69', 6: '30-39,70-79'}
# Per dataset (our name): the root dataset and its clip prefixes, LEAP's config file and full_name, the label
# directory in the scored workspace, LEAP's detector, and whether LEAP read motion patterns from the workspace's
# motion_patterns/ (the others use LEAP's default ./fixed_files/preprocessed). From the scored run logs.
LEAP_DATASETS = {
    'caldot1-y05': dict(root='caldot1', config='caldot1.yaml', full_name='caldot1', labels='caldot1-y5',
                        detector='yolov5', workspace_motion=False, reid='Caldot1'),
    'caldot2-y05': dict(root='caldot2', config='caldot2.yaml', full_name='caldot2', labels='caldot2-y5',
                        detector='yolov5', workspace_motion=False, reid='Caldot2'),
    'ams-y05': dict(root='ams', config='amsterdam.yaml', full_name='amsterdam-y5', labels='amsterdam-y5',
                    detector='yolov5', workspace_motion=True, reid='Amsterdam'),
    # B3D's pseudo labels and crops (scored: labels/jncN, datasets/jncN) use the same names as LEAP's config.
    'jnc0': dict(root='jnc0', config='jnc0.yaml', full_name='jnc0', labels='jnc0', detector='retinanet',
                 workspace_motion=True, reid='Jnc0'),
    'jnc2': dict(root='jnc2', config='jnc2.yaml', full_name='jnc2', labels='jnc2', detector='retinanet',
                 workspace_motion=True, reid='Jnc2'),
    'jnc6': dict(root='jnc6', config='jnc6.yaml', full_name='jnc6', labels='jnc6', detector='retinanet',
                 workspace_motion=True, reid='Jnc6'),
    'jnc7': dict(root='jnc7', config='jnc7.yaml', full_name='jnc7', labels='jnc7', detector='retinanet',
                 workspace_motion=True, reid='Jnc7'),
}
# Detector weights in the workspace (LEAP's --weight; RetinaNet reads its directory).
DETECTOR_WEIGHTS = {'yolov5': 'models/yolov5/yolov5x.pt', 'retinanet': 'models/retinanet/retinanet_b3d.pth'}
# Our split directories and their clip-name prefixes.
SPLIT_PREFIXES = {'test': 'te', 'train': 'tr'}
# LEAP's preparation, in order (README steps 4, 6a, 6b, and the motion-pattern clustering of track_cluster_updated.py;
# step 7, distillation, was skipped in the scored runs, which used the undistilled detector).
LEAP_PREP_STEPS = ('labels_train', 'labels_test', 'reid_data', 'reid_train', 'motion_patterns')
# OTIF's preparation, in order (README sections four and five for one detector, then the validation tuning).
OTIF_PREP_STEPS = ('detect_train', 'tracks_train', 'extract_frames', 'train_proxy', 'train_tracker', 'detect_valid',
                   'tune_valid')
PREP_STEPS = {'leap': LEAP_PREP_STEPS, 'otif': OTIF_PREP_STEPS}
# OTIF: its label (dataset directory) per dataset and the tracker thresholds; the scored y5 runs used our fine-tuned
# YOLOv5 at 704x480 on all three (detector_cfg of /sota-results/<label>/otif_<label>.csv) and, per the scored
# container's shell history (/root/.bash_history of OTIF_IMAGE, kept in raw/prep/otif_env.json), [0,1,1,1,1,1,1] for Amsterdam too (OTIF's README has [0,0,1,...] there).
OTIF_THRESHOLDS = '{"Thresholds":[0,1,1,1,1,1,1]}'
OTIF_DATASETS = {'caldot1-y05': dict(label='caldot1-y5', thresholds=OTIF_THRESHOLDS, detector='yolov5-704x480'),
                 'caldot2-y05': dict(label='caldot2-y5', thresholds=OTIF_THRESHOLDS, detector='yolov5-704x480'),
                 'ams-y05': dict(label='amsterdam-y5', thresholds=OTIF_THRESHOLDS, detector='yolov5-704x480'),
                 **{d: dict(label=d, thresholds=OTIF_THRESHOLDS, detector='retinanet-1080x720')
                    for d in ('jnc0', 'jnc2', 'jnc6', 'jnc7')}}
# The segmentation proxy the scored tuning started from (shell history: amsterdam-y5 and jncN ran 30_speed_accuracy with
# 640_352_-1; OTIF's README starts from 416_256_-1). The history's 92 lines do not reach back to the CalDoT y5 runs.
OTIF_TUNE_START = '640_352_-1'
# OTIF's 16_extract_frames.py tracks the train detections with `go run iou-tracker.go` in pipeline/, which has no go.mod
# and imports github.com/cpmech/gosl, which OTIF's image does not have; the call fails silently (eval.track ignores the
# exit status), leaving 0 good frames, so the scored runs sampled them from best-tracks left over from OTIF's own
# dataset (identical files in caldot1/ and caldot1-y5/, amsterdam/ and amsterdam-y5/). The step runs here in GOPATH mode
# against OTIF_GOPATH: the versions pipeline2/go.mod pins, copied from the image's module cache, and gosl v1.1.0
# (cloned; cgo off, which drops only gosl's METIS partitioning, unused by iou-tracker.go).
OTIF_GOPATH = SCRATCH / 'env' / 'otif' / 'gopath'
OTIF_GOPATH_ENV = 'export GO111MODULE=off GOFLAGS= CGO_ENABLED=0 GOPATH=/scratch/env/otif/gopath && '
OTIF_GOPATH_MODULES = {'github.com/mitroadmaps/gomapinfer': 'gomapinfer@v0.0.0-20210917033103-4e3dcc98a112',
                       'github.com/ajstarks/svgo': 'svgo@v0.0.0-20211024235047-1546f124cd8b',
                       'github.com/dhconnelly/rtreego': 'rtreego@v1.1.0', 'github.com/qedus/osmpbf': 'osmpbf@v1.2.0',
                       'google.golang.org/protobuf': 'protobuf@v1.26.0'}
OTIF_GOSL = ('https://github.com/cpmech/gosl', 'v1.1.0')
# The scored OTIF data root and code (Alena's fork, which adds the YOLOv5 and RetinaNet detectors), the scored
# validation curves, the image this track snapshots from the exited scored container, and the YOLOv3 configs OTIF's
# detector tradeoff can switch to (CalDoT's and the generic COCO model).
OTIF_SOURCE = Path('/data/alenachao/data/otif-dataset')
OTIF_CODE = Path('/work/alenachao/tunnel/work/otif')
OTIF_SCORED_CURVES = OTIF_CODE / 'pipeline2' / 'outputs'
OTIF_IMAGE = 'bombe-revision-otif:scored-env'
OTIF_YOLOV3 = ('caldot', 'generic')
# Outputs the OTIF steps regenerate, removed from the clone of the scored data root before the first step.
OTIF_REGENERATED = ('train/{detector}', 'train/tracks', 'train/seg-train', 'segmentation_models', 'tracker/rnn',
                    'valid/{detector}')


def parse_args():
    parser = argparse.ArgumentParser(description='R1.O2/R9.O2: LEAP rerun and preparation timing on bombe (dry run unless --run)')
    parser.add_argument('action', choices=['setup', 'scored', 'leap', 'export', 'prep', 'compare', 'gpus'])
    parser.add_argument('--datasets', nargs='+', default=list(LEAP_DATASETS), choices=list(LEAP_DATASETS))
    parser.add_argument('--system', choices=list(PREP_STEPS), default='leap')
    parser.add_argument('--steps', nargs='+', choices=sorted(set(LEAP_PREP_STEPS) | set(OTIF_PREP_STEPS)))
    parser.add_argument('--gpu', type=int, choices=list(GPU_CPUS))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--run', action='store_true')
    return parser.parse_args()


def sh(argv: list[str], run: bool, **kwargs) -> subprocess.CompletedProcess | None:
    """Print a command, and execute it (checked) only with --run."""
    # Every command is shown, so a dry run is the plan.
    print('+', shlex.join(argv), flush=True)
    return subprocess.run(argv, check=True, **kwargs) if run else None


def write_jsonl_gz(path: Path, rows: list[dict]):
    """Rows as gzipped JSON lines, written to a temporary name first so readers never see a partial file."""
    # The raw directory may not exist yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with gzip.open(tmp, 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    # Atomic replace.
    os.replace(tmp, path)


def write_json(path: Path, data):
    """One JSON file, written atomically."""
    # The raw directory may not exist yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Independent dataset queues also write shared metadata such as steps.json.
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    # Atomic replace.
    os.replace(tmp, path)


def clip_number(name: str) -> int:
    """The clip number of te07.mp4, tr12.mp4, or 7.mp4 (LEAP's names drop the prefix and leading zeros)."""
    # Digits of the stem only (the extension mp4 has a digit too).
    match = re.fullmatch(r'[a-z]*(\d+)', Path(name).stem)
    assert match, f'not a clip name: {name}'
    return int(match.group(1))


def read_md5(path: Path) -> dict[str, str]:
    """{relative path: md5} from an md5sum listing."""
    # md5sum writes '<hash>  <path>' per line.
    return {p: h for h, p in (line.rstrip('\n').split(maxsplit=1) for line in path.read_text().splitlines() if line)}


def video_identity() -> list[dict]:
    """Per dataset and split, LEAP's workspace videos against our clips by MD5 (LEAP's N.mp4 is our teNN/trNN)."""
    # Checksums taken on manchester at copy time, and of our mirrored clips (computed once, then reused).
    leap = read_md5(SCRATCH / 'inputs' / 'leap-dataset.source.md5')
    mirror_md5 = SCRATCH / 'inputs' / 'mirror-videos.md5'
    if not mirror_md5.exists():
        clips = sorted(str(p.relative_to(ROOT / 'mirror' / 'polyis-data' / 'datasets'))
                       for d in {c['root'] for c in LEAP_DATASETS.values()} for s in SPLIT_PREFIXES
                       for p in (ROOT / 'mirror' / 'polyis-data' / 'datasets' / d / s).glob('*.mp4'))
        out = subprocess.run(['md5sum', *clips], cwd=ROOT / 'mirror' / 'polyis-data' / 'datasets', check=True,
                             capture_output=True, text=True).stdout
        mirror_md5.write_text(out)
    mirror = read_md5(mirror_md5)
    # Our clips by (root, split, number).
    ours = {(p.split('/')[0], p.split('/')[1], clip_number(p.split('/')[2])): h for p, h in mirror.items()}
    rows = []
    for dataset, c in LEAP_DATASETS.items():
        for split in SPLIT_PREFIXES:
            # LEAP's videos of this dataset and split (videos/caldot1-y5 and caldot2-y5, which the CalDoT runs read,
            # are links to videos/caldot1 and caldot2; md5sum listed the link targets' own paths).
            prefix = f"videos/{c['full_name']}/{split}/"
            theirs = {clip_number(p[len(prefix):]): h for p, h in leap.items() if p.startswith(prefix)}
            same = sorted(n for n, h in theirs.items() if ours.get((c['root'], split, n)) == h)
            rows.append(dict(dataset=dataset, split=split, leap_dir=prefix, leap_videos=len(theirs),
                             our_videos=sum(1 for k in ours if k[:2] == (c['root'], split)), identical=len(same),
                             differing=sorted(set(theirs) - set(same))))
    return rows


def setup(args):
    """Image, label aliases, clip links, and the manifest (see the module docstring)."""
    env = SCRATCH / 'env' / 'leap'
    # The env tarball must have been copied from manchester first.
    assert (env / 'leap-env.tar').exists(), f'missing {env}/leap-env.tar (copy it from the scored container first)'
    # Build the image only when missing; the Dockerfile sits next to the tarball as the build context.
    present = subprocess.run(['docker', 'image', 'inspect', LEAP_IMAGE], capture_output=True).returncode == 0
    if not present:
        (env / 'Dockerfile').write_text(DOCKERFILE)
        sh(['docker', 'build', '-t', LEAP_IMAGE, str(env)], args.run)
    # Label aliases LEAP's label loader needs (labels/<full_name>), relative so they resolve inside the container.
    labels = SCRATCH / 'inputs' / 'leap-dataset' / 'labels'
    for c in LEAP_DATASETS.values():
        alias = labels / c['full_name']
        if c['full_name'] != c['labels'] and not alias.exists():
            print(f"+ ln -s {c['labels']} {alias}")
            if args.run:
                alias.symlink_to(c['labels'])
    # Clip links with LEAP's names, pointing at the container path of our mirrored clips.
    for dataset, c in LEAP_DATASETS.items():
        for split, prefix in SPLIT_PREFIXES.items():
            source = ROOT / 'mirror' / 'polyis-data' / 'datasets' / c['root'] / split
            target = SCRATCH / 'videos' / dataset / split
            for clip in sorted(source.glob(f'{prefix}*.mp4')):
                link = target / f'{clip_number(clip.name)}.mp4'
                if args.run and not link.is_symlink():
                    target.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(f"/polyis-data/datasets/{c['root']}/{split}/{clip.name}")
    # The manifest: images, env provenance, and the video identity check.
    if args.run:
        inspect = {name: subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}} {{.Created}}', name],
                                        capture_output=True, text=True).stdout.strip()
                   for name in (LEAP_BASE_IMAGE, LEAP_IMAGE)}
        container = json.loads((env / 'alenatunnel-leap.inspect.json').read_text())[0]
        manchester_image = json.loads((env / 'manchester-image.inspect.json').read_text())[0]
        write_json(RAW / 'manifest.json', dict(
            images=inspect, dockerfile=DOCKERFILE,
            scored_container=dict(host='manchester', name='alenatunnel-leap', created=container['Created'],
                                  started=container['State']['StartedAt'], finished=container['State']['FinishedAt'],
                                  image=container['Image'], manchester_image_created=manchester_image['Created'],
                                  mounts=[f"{m['Source']} -> {m['Destination']}" for m in container['Mounts']],
                                  env_tar_bytes=(env / 'leap-env.tar').stat().st_size),
            leap_datasets=LEAP_DATASETS, video_identity=video_identity(),
            label_aliases={c['full_name']: c['labels'] for c in LEAP_DATASETS.values() if c['full_name'] != c['labels']}))


def scored(args):
    """Package the scored parsed results per dataset as JSON lines (see the module docstring)."""
    for dataset, c in LEAP_DATASETS.items():
        if dataset not in args.datasets:
            continue
        source = SCRATCH / 'inputs' / 'leap-dataset' / 'outputs' / 'parsed_results' / c['full_name']
        # One record per clip: LEAP's parsed intervals and the hash of the patterns it saved for the clip.
        names = sorted(p.name[:-len('_results.json')] for p in source.glob('*_results.json'))
        clips, patterns = [], {}
        for name in names:
            results = json.loads((source / f'{name}_results.json').read_text())
            tracks = json.loads((source / f'{name}_tracks.json').read_text())
            digest = hashlib.sha256(json.dumps(tracks, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            patterns[digest] = tracks
            clips.append(dict(video_name=name, patterns_sha256=digest,
                              objects=[dict(id=int(k), interval=v[0], pattern=v[1], class_id=v[2])
                                       for k, v in sorted(results.items(), key=lambda kv: int(kv[0]))]))
        # speed.tsv rows in file order (CalDoT's hold two runs of every clip).
        lines = (source / 'speed.tsv').read_text().splitlines()
        header = lines[0].split('\t')
        speed = [dict(zip(header, line.split('\t')), row=i) for i, line in enumerate(lines[1:])]
        print(f'{dataset}: {len(clips)} scored clips, {len(patterns)} distinct pattern files, {len(speed)} speed rows')
        if args.run:
            write_jsonl_gz(RAW / dataset / 'leap_scored.jsonl.gz', clips)
            write_jsonl_gz(RAW / dataset / 'leap_scored_run.jsonl.gz',
                           [dict(source=str(source), patterns=patterns, speed=speed)])


def launch(gpu: int, docker_args: list[str], command: str, run: bool, image: str = LEAP_IMAGE) -> dict | None:
    """Run one command in an image (LEAP's by default) through the shared launcher; returns its summary
    (samples/<run>.json)."""
    # The launcher reads the image and the extra docker arguments from its environment.
    env = {**os.environ, 'BOMBE_IMAGE': image, 'BOMBE_DOCKER_ARGS': ' '.join(docker_args), 'BOMBE_CPU_SUBSET': GPU_CPUS[gpu]}
    print(f'+ BOMBE_IMAGE={image} BOMBE_CPU_SUBSET={GPU_CPUS[gpu]} BOMBE_DOCKER_ARGS={shlex.quote(env["BOMBE_DOCKER_ARGS"])} '
          f'{LAUNCHER} {gpu} {SCRATCH} {shlex.quote(command)}', flush=True)
    if not run:
        return None
    # The launcher refuses a busy GPU (exit 3, another user's job); wait for it, up to BUSY_WAIT_S, rather than fail.
    waited = 0
    while True:
        out = subprocess.run([str(LAUNCHER), str(gpu), str(SCRATCH), command], env=env, capture_output=True, text=True)
        if out.returncode != 3 or re.search(r'^run g\d', out.stdout, re.MULTILINE) or waited >= BUSY_WAIT_S:
            break
        print(f'GPU {gpu} busy, waiting ({waited} s so far): {out.stderr.strip()[-300:]}', flush=True)
        time.sleep(120)
        waited += 120
    # The launcher prints 'run <id> ...' first; its summary file is named by that id.
    print(out.stdout[-4000:], out.stderr[-4000:], sep='\n', flush=True)
    match = re.search(r'^run (g\d+-\d{8}T\d{6}) ', out.stdout, re.MULTILINE)
    assert match, f'launcher did not start (exit {out.returncode}): {out.stderr[-2000:]}'
    summary = json.loads((SCRATCH / 'samples' / f'{match.group(1)}.json').read_text())
    return dict(summary, launcher_stdout=out.stdout[-20000:], launcher_stderr=out.stderr[-20000:])


def common_mounts() -> list[str]:
    """The workspace copy at LEAP's label path, the checkout copy, and GNU time, all read-only."""
    return ['-v', f'{SCRATCH}/inputs/leap-dataset:{LEAP_DATASET_MOUNT}:ro', '-v', f'{SCRATCH}/inputs/LEAP:{LEAP_CODE_MOUNT}:ro',
            '-v', f'/usr/bin/time:{GNU_TIME}:ro', '-e', 'PYTHONPATH=']


def leap(args):
    """LEAP's inference per dataset on --gpu (see the module docstring)."""
    assert args.gpu is not None, '--gpu is required'
    for dataset in args.datasets:
        c = LEAP_DATASETS[dataset]
        # A fresh work directory per run (LEAP's outputs and logs), inside the scratch mount.
        stamp = time.strftime('%Y%m%dT%H%M%S')
        work = f'/scratch/runs/leap/{dataset}-{stamp}'
        out = RAW / dataset
        # The runner's arguments: the scored run's config, detector, re-identification weights, and motion patterns.
        runner = ['python', RUNNER, '--config', f"{LEAP_DATASET_MOUNT}/configs/{c['config']}",
                  '--weight', f"{LEAP_DATASET_MOUNT}/{DETECTOR_WEIGHTS[c['detector']]}",
                  '--reid-weight', f'{LEAP_DATASET_MOUNT}/logs', '--video-dir', f'/scratch/videos/{dataset}/test',
                  '--work-dir', work, '--export-prefix', f'/polyis/{out.relative_to(TETRIS)}/leap_rerun']
        runner += ['--motion-dir', f'{LEAP_DATASET_MOUNT}/motion_patterns'] if c['workspace_motion'] else []
        runner += ['--retinanet'] if c['detector'] == 'retinanet' else []
        runner += ['--no-mask'] if getattr(args, 'no_mask', False) else []
        # GNU time around the whole process; the working directory is LEAP's checkout copy.
        command = (f'mkdir -p {work} && cd {LEAP_CODE_MOUNT} && {GNU_TIME} -v -o {work}/time.log '
                   f'{shlex.join(runner)} > {work}/stdout.log 2>&1')
        summary = launch(args.gpu, common_mounts(), command, args.run)
        if summary is None:
            continue
        # Keep the logs, GNU time's report, and the GPU samples next to the export.
        host_work = SCRATCH / work.removeprefix('/scratch/')
        time_report = (host_work / 'time.log').read_text() if (host_work / 'time.log').exists() else ''
        samples = SCRATCH / 'samples' / f"{summary['run']}.jsonl"
        write_jsonl_gz(out / 'leap_rerun_samples.jsonl.gz',
                       [json.loads(line) for line in samples.read_text().splitlines() if line.strip()])
        (out / 'leap_rerun_time.log').write_text(time_report)
        (out / 'leap_rerun_stdout.log').write_bytes((host_work / 'stdout.log').read_bytes())
        leap_logs = sorted((host_work / 'logs').glob('*.log'))
        (out / 'leap_rerun_leap.log').write_bytes(b''.join(p.read_bytes() for p in leap_logs))
        write_json(out / 'leap_rerun_launch.json', dict(summary, dataset=dataset, work_dir=str(host_work),
                                                        command=command, gnu_time=time_report))
        print(f"{dataset}: exit {summary['exit']}, contaminated {summary['contaminated']}", flush=True)


def export(args):
    """Rebuild the track exports of the rerun's and the scored parsed results with the scored runs' exporter (CPU)."""
    # A copy of the exporter next to the other inputs (its source digest goes into every exported line).
    exporter = SCRATCH / 'inputs' / 'scripts' / 'export-leap-outputs.py'
    if args.run and not exporter.exists():
        exporter.parent.mkdir(parents=True, exist_ok=True)
        exporter.write_bytes(EXPORTER_SOURCE.read_bytes())
    for dataset in args.datasets:
        c = LEAP_DATASETS[dataset]
        # The rerun's parsed results are in the work directory its launch record names.
        launch_record = RAW / dataset / 'leap_rerun_launch.json'
        if not launch_record.exists():
            print(f'{dataset}: no rerun yet, skipped')
            continue
        work = Path(json.loads(launch_record.read_text())['work_dir'])
        sources = {'rerun': work / 'outputs' / 'parsed_results' / c['full_name'],
                   'scored': SCRATCH / 'inputs' / 'leap-dataset' / 'outputs' / 'parsed_results' / c['full_name']}
        for name, parsed in sources.items():
            # Host paths as the container sees them (the scratch directory is /scratch, the checkout /polyis).
            command = ['python', '/polyis/measurements-gpu/r1o2_leap_otif_rerun_export.py',
                       '--exporter', f"/scratch/{exporter.relative_to(SCRATCH)}",
                       '--parsed-dir', f"/scratch/{parsed.relative_to(SCRATCH)}",
                       '--out', f'/polyis/{(RAW / dataset / f"leap_{name}_export.jsonl.gz").relative_to(TETRIS)}']
            # CPU only and short, so no GPU and no launcher; a CPU quota keeps it light next to timed runs.
            sh(['docker', 'run', '--rm', '--cpus', '4', '-v', f'{SCRATCH}:/scratch', '-v', f'{TETRIS}:/polyis',
                '-e', 'PYTHONPATH=', LEAP_IMAGE, *command], args.run, capture_output=True)


def leap_prep_command(dataset: str, step: str) -> tuple[str, str]:
    """LEAP's command for one preparation step (run from the checkout copy) and the container path of its output."""
    c = LEAP_DATASETS[dataset]
    out = f'/scratch/prep/leap/{dataset}'
    # fastreid's dataset name: the config directory's name in lower case (run.sh capitalizes it back).
    reid = c['reid'].lower()
    if step in ('labels_train', 'labels_test'):
        split = step.removeprefix('labels_')
        # The detector that reproduces the scored labels: COCO YOLOv5x from weights/ (not video_labeling/'s own
        # yolov5x.pt) without a mask, or B3D's RetinaNet; multi-video mode writes <save_path's parent>/<clip>/<split>/.
        detector = ('--yolo_weight weights/yolov5x.pt' if c['detector'] == 'yolov5'
                    else f'--use_retinanet --retinanet_path {LEAP_DATASET_MOUNT}/models/retinanet')
        return (f'python video_labeling/video_tracker.py --video_dir /scratch/videos/{dataset}/{split} '
                f"--save_path {out}/labels/{c['labels']}/{split} --fps 15 {detector}", f"{out}/labels/{c['labels']}")
    if step == 'reid_data':
        # Train crops and the test gallery and queries, from our clips and the labels above.
        return (f'python vehicle_reid2/generate_reid.py {reid} --video_base /scratch/videos/{dataset} '
                f"--label_base {out}/labels/{c['labels']} --save_base {out}/datasets", f'{out}/datasets/{reid}')
    if step == 'reid_train':
        # The dataset's bagtricks_R50-ibn config on the one visible GPU, with the crops above as fastreid's root.
        return f'bash vehicle_reid2/run.sh {reid} "" 0 {out}/reid/{reid} {out}/datasets', f'{out}/reid/{reid}'
    # motion_patterns: clustering of the train labels' tracks, all of them (k=0, as the scored file names say). The
    # default --cluster_min_num 16 cannot give the scored odd cluster counts (3 for Amsterdam, 19 for jnc0, since k
    # steps by 2 from it) and gives none for Amsterdam's 12 tracks; 3 does (1 gives a single cluster). LEAP's
    # k-medoids starts from unseeded random centers, so the clusters themselves differ from run to run.
    return (f"python track_cluster_updated.py --video_name {c['labels']} --label_dir {out}/labels "
            f'--output_dir {out}/motion_patterns --k 0 --cluster_min_num 3', f"{out}/motion_patterns/{c['labels']}")


def otif_prep_command(dataset: str, step: str) -> tuple[str, str]:
    """OTIF's command for one preparation step and the container path of its output (the data root is /otif-data)."""
    label, thresholds = OTIF_DATASETS[dataset]['label'], shlex.quote(OTIF_DATASETS[dataset]['thresholds'])
    detector = OTIF_DATASETS[dataset]['detector']
    dims = detector.rsplit('-', 1)[1]
    data = f'/otif-data/dataset/{label}'
    commands = {
        # 10_detect.go picks YOLOv5 from the label's y5 suffix and reads /otif-data/yolov5/<label>/best.pt.
        'detect_train': (f'cd /otif/pipeline2 && go run 10_detect.go /otif-data/ {label} {data}/train/video/ '
                         f'{data}/train/{detector}/ {dims}', f'{data}/train/{detector}'),
        # iou-tracker2.go writes into an existing directory (the scored run's existed; the clone's was removed).
        'tracks_train': (f'mkdir -p {data}/train/tracks && cd /otif/pipeline2 && go run iou-tracker2.go /otif-data/ {label} '
                         f"{data}/train/video/ {data}/train/{detector}/ {data}/train/tracks/ iou '' {thresholds}",
                         f'{data}/train/tracks'),
        # GOPATH mode, so the step's `go run iou-tracker.go` builds (see OTIF_GOPATH).
        'extract_frames': (f'{OTIF_GOPATH_ENV}cd /otif/pipeline && python 16_extract_frames.py /otif-data/ {label} '
                           f'{detector}', f'{data}/train/seg-train'),
        'train_proxy': (f'cd /otif/pipeline && python 20_train_segmentation.py /otif-data/ {label} car',
                        f'{data}/segmentation_models'),
        # The fork's rnn/train.py reads train/tracks (the original reads the tracker split).
        'train_tracker': (f'cd /otif/rnn && python train.py /otif-data/ {label}', f'{data}/tracker/rnn'),
        'detect_valid': (f'cd /otif/pipeline2 && go run 10_detect.go /otif-data/ {label} {data}/valid/video/ '
                         f'{data}/valid/{detector}/ {dims}', f'{data}/valid/{detector}'),
        'tune_valid': (f'cd /otif/pipeline2 && go run 30_speed_accuracy.go /otif-data/ {label} ours-simple {detector} '
                       f'{OTIF_TUNE_START} {thresholds} 10 /scratch/prep/otif/{dataset}/{label}-valid-ours-simple.txt',
                       f'/scratch/prep/otif/{dataset}/{label}-valid-ours-simple.txt'),
    }
    return commands[step]


def otif_root(dataset: str, run: bool) -> Path:
    """The dataset's OTIF data root: a copy-on-write clone of the scored one without the outputs the steps regenerate."""
    label = OTIF_DATASETS[dataset]['label']
    root = SCRATCH / 'prep' / 'otif' / dataset / 'otif-dataset'
    # Built once; the marker lists what was removed from the clone.
    marker = root / 'clone.json'
    if marker.exists() or not run:
        return root
    # A clone without its marker was interrupted; cp -a into its existing directories would nest a second copy.
    if root.exists():
        assert root.is_relative_to(SCRATCH), f'not removing {root} outside {SCRATCH}'
        sh(['rm', '-rf', str(root)], run)
    # Clone the dataset, its detector weights, and the YOLOv3 configs (ZFS block cloning: no data is copied, and a
    # write to the clone never reaches the source).
    weights = Path('retinanet/b3d') if OTIF_DATASETS[dataset]['detector'].startswith('retinanet-') else Path('yolov5') / label
    pairs = [(OTIF_SOURCE / 'dataset' / label, root / 'dataset' / label), (OTIF_SOURCE / weights, root / weights)]
    pairs += [(OTIF_SOURCE / 'yolov3' / name, root / 'yolov3' / name) for name in OTIF_YOLOV3]
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        sh(['cp', '-a', '--reflink=always', str(source), str(target)], run)
    # Remove the regenerated outputs from the clone only.
    outputs = [p.format(detector=OTIF_DATASETS[dataset]['detector']) for p in OTIF_REGENERATED]
    removed = [p for p in outputs if (root / 'dataset' / label / p).exists()]
    for p in removed:
        sh(['rm', '-rf', str(root / 'dataset' / label / p)], run)
    write_json(marker, dict(source=str(OTIF_SOURCE), label=label, cloned=[str(s) for s, _ in pairs], removed=removed))
    return root


def otif_env(run: bool):
    """Build OTIF_GOPATH once (CPU only, in OTIF's image) and record it with the scored container's shell history in
    raw/prep/otif_env.json."""
    record = RAW / 'prep' / 'otif_env.json'
    if record.exists() or not run:
        return
    # Module-cache copies at pipeline2/go.mod's versions, then gosl at its tag; the commit and go version are printed.
    copies = ' && '.join(f'mkdir -p /g/src/{path} && cp -r /root/go/pkg/mod/{path.rsplit("/", 1)[0]}/{version}/. /g/src/{path}/'
                         for path, version in OTIF_GOPATH_MODULES.items())
    script = (f'set -e; rm -rf /g/src; {copies} && git clone -q --depth 1 --branch {OTIF_GOSL[1]} {OTIF_GOSL[0]} /g/src/github.com/cpmech/gosl '
              '2>/dev/null && chmod -R u+w /g && echo "gosl $(git -C /g/src/github.com/cpmech/gosl rev-parse HEAD)" '
              '&& echo "go $(go version)" && cd /otif/pipeline && GO111MODULE=off GOFLAGS= CGO_ENABLED=0 GOPATH=/g '
              'go build -o /dev/null iou-tracker.go && echo built')
    OTIF_GOPATH.mkdir(parents=True, exist_ok=True)
    out = sh(['docker', 'run', '--rm', '--cpus', '4', '--entrypoint', 'bash', '-v', f'{OTIF_CODE}:/otif:ro',
              '-v', f'{OTIF_GOPATH}:/g', OTIF_IMAGE, '-c', script], run, capture_output=True, text=True)
    # The scored container's shell history: the commands the scored y5 preparation ran.
    history = sh(['docker', 'run', '--rm', '--entrypoint', 'cat', OTIF_IMAGE, '/root/.bash_history'], run,
                 capture_output=True, text=True)
    image = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}} {{.Created}}', OTIF_IMAGE],
                           capture_output=True, text=True).stdout.strip()
    write_json(record, dict(image=image, gopath=str(OTIF_GOPATH), modules=OTIF_GOPATH_MODULES, gosl=OTIF_GOSL,
                            build_output=out.stdout.splitlines(), env=OTIF_GOPATH_ENV, tune_start=OTIF_TUNE_START,
                            thresholds=OTIF_THRESHOLDS, scored_shell_history=history.stdout.splitlines()))


def tree_size(path: Path) -> dict:
    """Files and bytes under a path (a file counts as one)."""
    # A missing output is reported as such.
    if not path.exists():
        return dict(exists=False, files=0, bytes=0)
    files = [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
    return dict(exists=True, files=len(files), bytes=sum(p.stat().st_size for p in files))


def compare_prep(system: str, dataset: str, step: str, host_out: Path, raw: Path, run: bool):
    """Package a finished step's outputs next to the scored run's, for the aggregation (labels, crops, re-id
    training, motion patterns, OTIF's validation curve)."""
    if not run:
        return
    scored_root = SCRATCH / 'inputs' / 'leap-dataset'
    if system == 'leap':
        c = LEAP_DATASETS[dataset]
        reid = c['reid'].lower()
    if system == 'leap' and step.startswith('labels_'):
        # Every label line of both runs, per clip (LEAP's N is our clip number), with the file's modification time
        # (the scored copy kept its times, so their spread is the scored labeling's duration).
        split = step.removeprefix('labels_')
        rows = [dict(source=source, clip=int(p.parent.parent.name), mtime=p.stat().st_mtime, lines=p.read_text().splitlines())
                for source, base in (('rerun', host_out), ('scored', scored_root / 'labels' / c['labels']))
                for p in sorted(base.glob(f'*/{split}/label_*.txt'))]
        write_jsonl_gz(raw / f'{step}_compare.jsonl.gz', rows)
    elif system == 'leap' and step == 'reid_data':
        # The crop file names (camera, object, class, box, and frame), per directory.
        rows = [dict(source=source, dir=d, names=sorted(p.name for p in (base / d).glob('*.jpg')))
                for source, base in (('rerun', host_out), ('scored', scored_root / 'datasets' / reid))
                for d in ('image_train', 'image_test', 'image_query')]
        write_jsonl_gz(raw / f'{step}_compare.jsonl.gz', rows)
    elif system == 'leap' and step == 'reid_train':
        # fastreid's final metrics and its training-time lines, rerun and scored (logs/<label dir> in the workspace).
        scored_logs = scored_root / 'logs' / ('amsterdam-y5' if reid == 'amsterdam' else reid)
        def summary(base: Path) -> dict:
            log = (base / 'log.txt').read_text() if (base / 'log.txt').exists() else ''
            metrics = (base / 'metrics.json').read_text().splitlines() if (base / 'metrics.json').exists() else []
            return dict(metrics=[json.loads(m) for m in metrics if m.strip()],
                        time_lines=[l for l in log.splitlines() if 'training speed' in l or 'Total training time' in l
                                    or 'Starting training' in l])
        write_json(raw / f'{step}_compare.json', dict(rerun=summary(host_out), scored=summary(scored_logs)))
    elif system == 'leap' and step == 'motion_patterns':
        # The clustered patterns as JSON (numpy object arrays, so read in the LEAP image), next to the scored ones:
        # the workspace's motion_patterns/<labels> for Amsterdam and B3D, LEAP's own fixed_files for CalDoT, whose
        # scored runs used the patterns shipped in LEAP's repository.
        scored_dir = (f"/data/alenachao/data/leap-dataset/motion_patterns/{c['labels']}" if c['workspace_motion']
                      else f"{LEAP_CODE_MOUNT}/fixed_files/preprocessed/{c['full_name']}")
        code = ('import glob, json, numpy as np, sys; '
                'load = lambda d: {f.split("/")[-1]: np.load(f, allow_pickle=True).tolist() for f in sorted(glob.glob(d + "/*_clustered.npy"))}; '
                'json.dump(dict(rerun=load(sys.argv[1]), scored=load(sys.argv[2]), scored_dir=sys.argv[2]), open(sys.argv[3], "w"), '
                'default=lambda v: v.tolist())')
        out = f"/polyis/{(raw / f'{step}_compare.json').relative_to(TETRIS)}"
        sh(['docker', 'run', '--rm', '--cpus', '2', *common_mounts(), '-v', f'{SCRATCH}:/scratch', '-v', f'{TETRIS}:/polyis',
            LEAP_IMAGE, 'python', '-c', code, f'/scratch/prep/leap/{dataset}/motion_patterns/{c["labels"]}', scored_dir, out],
           run, capture_output=True)
    elif system == 'otif' and step == 'tune_valid':
        # The validation curve, rerun and scored (pipeline2/outputs/<label>-valid-ours-simple.txt).
        label = OTIF_DATASETS[dataset]['label']
        scored = OTIF_SCORED_CURVES / f'{label}-valid-ours-simple.txt'
        write_json(raw / f'{step}_compare.json', dict(rerun=host_out.read_text().splitlines() if host_out.exists() else [],
                                                      scored=scored.read_text().splitlines(), scored_path=str(scored)))


def host_path(path: str, otif_data: Path | None) -> Path:
    """The host path of a prep container path (/scratch is SCRATCH; OTIF's /otif-data is the dataset's clone)."""
    if path.startswith('/scratch/'):
        return SCRATCH / path.removeprefix('/scratch/')
    assert path.startswith('/otif-data/') and otif_data is not None, f'not a prep path: {path}'
    return otif_data / path.removeprefix('/otif-data/')


def compare(args):
    """Rebuild the comparison files of every finished step (exit 0) of --system and --datasets, from its outputs."""
    for dataset in args.datasets:
        raw = RAW / 'prep' / args.system / dataset
        root = SCRATCH / 'prep' / 'otif' / dataset / 'otif-dataset' if args.system == 'otif' else None
        for step in args.steps or PREP_STEPS[args.system]:
            record = raw / f'{step}.json'
            # Only finished steps have outputs to compare.
            if not record.exists() or json.loads(record.read_text())['exit'] != 0:
                continue
            out = json.loads(record.read_text())['output']
            print(f'{args.system} {dataset} {step}: comparing {out}', flush=True)
            compare_prep(args.system, dataset, step, host_path(out, root), raw, args.run)


def prep(args):
    """Time one system's preparation per dataset and step on --gpu (see the module docstring)."""
    assert args.gpu is not None, '--gpu is required'
    steps = args.steps or list(PREP_STEPS[args.system])
    assert set(steps) <= set(PREP_STEPS[args.system]), f'{args.system} has no steps {set(steps) - set(PREP_STEPS[args.system])}'
    datasets = [d for d in args.datasets if args.system == 'leap' or d in OTIF_DATASETS]
    # The step lists, for the aggregation's completeness check (one source of truth).
    if args.run:
        write_json(RAW / 'prep' / 'steps.json', PREP_STEPS)
    for dataset in datasets:
        raw = RAW / 'prep' / args.system / dataset
        # The container's inputs: LEAP's workspace and checkout copies, or OTIF's data root clone and code.
        if args.system == 'leap':
            root = None
            image, mounts = LEAP_IMAGE, common_mounts() + ['-e', 'PYTHONPATH=/leap']
        else:
            # The GOPATH extract_frames builds against, then the dataset's data root clone.
            otif_env(args.run)
            root = otif_root(dataset, args.run)
            image = OTIF_IMAGE
            mounts = ['-v', f'{root}:/otif-data', '-v', f'{OTIF_CODE}:/otif:ro', '-v', f'/usr/bin/time:{GNU_TIME}:ro',
                      '-e', 'PYTHONPATH=/otif/python:']
            # The scored RetinaNet loader and its config hardcode this absolute root for config and weights.
            if OTIF_DATASETS[dataset]['detector'].startswith('retinanet-'):
                mounts += ['-v', f'{root}:/path/to/otif-dataset:ro']
        for step in steps:
            record = raw / f'{step}.json'
            # A finished step is kept unless --force.
            if record.exists() and json.loads(record.read_text())['exit'] == 0 and not args.force:
                print(f'{args.system} {dataset} {step}: done, skipped', flush=True)
                continue
            command, out = (leap_prep_command if args.system == 'leap' else otif_prep_command)(dataset, step)
            # OTIF's steps add to their outputs (30_speed_accuracy.go appends to its curve, 16_extract_frames.py keeps
            # images beyond its count), so a rerun starts without the outputs of an earlier attempt (clone only).
            stale = host_path(out, root)
            if args.system == 'otif' and stale.exists():
                assert stale.is_relative_to(SCRATCH), f'not removing {stale} outside {SCRATCH}'
                sh(['rm', '-rf', str(stale)], args.run)
            # GNU time around the step; its log and report go to the dataset's prep directory.
            work = f'/scratch/prep/{args.system}/{dataset}'
            # LEAP runs from its checkout; OTIF's `go run` builds under GOTMPDIR, which must allow execution (the
            # launcher's /tmp is a noexec tmpfs).
            cwd = f'cd {LEAP_CODE_MOUNT} && ' if args.system == 'leap' else 'mkdir -p /root/go-tmp && export GOTMPDIR=/root/go-tmp && '
            wrapped = (f'mkdir -p {work}/time {work}/stdout && {cwd}{GNU_TIME} -v -o {work}/time/{step}.log '
                       f'bash -c {shlex.quote(command)} > {work}/stdout/{step}.log 2>&1')
            summary = launch(args.gpu, mounts, wrapped, args.run, image=image)
            if summary is None:
                continue
            # Host paths of the step's outputs and logs.
            host = lambda p: host_path(p, root)
            host_work = host(work)
            time_log = host_work / 'time' / f'{step}.log'
            time_report = time_log.read_text() if time_log.exists() else ''
            stdout = (host_work / 'stdout' / f'{step}.log').read_bytes()
            # The samples, the log, the comparison with the scored outputs, and the record last (it marks the step done).
            samples = SCRATCH / 'samples' / f"{summary['run']}.jsonl"
            write_jsonl_gz(raw / f'{step}_samples.jsonl.gz',
                           [json.loads(line) for line in samples.read_text().splitlines() if line.strip()])
            with gzip.open(raw / f'{step}_stdout.log.gz', 'wb') as f:
                f.write(stdout)
            # A failed comparison is recorded, not fatal: the timing stands, and the queue goes on.
            try:
                compare_prep(args.system, dataset, step, host(out), raw, args.run)
                compare_error = None
            except Exception as e:
                compare_error = repr(e)
            write_json(record, dict(summary, system=args.system, dataset=dataset, step=step, command=command,
                                    wrapped=wrapped, output=out, output_size=tree_size(host(out)), gnu_time=time_report,
                                    compare_error=compare_error, stdout_tail=stdout[-4000:].decode(errors='replace')))
            print(f"{args.system} {dataset} {step}: exit {summary['exit']}, contaminated {summary['contaminated']}", flush=True)


def gpu_rows(run: str) -> list[dict]:
    """Every GPU's samples during one launcher run: nvidia-smi's CSV lines (timestamp, index, utilization %, memory
    MiB), skipping lines nvidia-smi could not fill (such as [N/A])."""
    lines = (SCRATCH / 'samples' / f'{run}.gpus.csv').read_text().splitlines()
    # Only complete numeric lines; a partial last line (the sampler was stopped mid-write) is dropped too.
    matches = (re.fullmatch(r'\s*([^,]+),\s*(\d+),\s*(\d+),\s*(\d+)\s*', line) for line in lines)
    return [dict(t=m.group(1), gpu=int(m.group(2)), util=float(m.group(3)), mem_mib=float(m.group(4))) for m in matches if m]


def gpus(args):
    """Write <record>_gpus.jsonl.gz for every timed run on record that lacks it (see the module docstring)."""
    # LEAP reruns (raw/<dataset>/leap_rerun_launch.json) and prep steps (raw/prep/<system>/<dataset>/<step>.json).
    records = [(p, p.with_name('leap_rerun_gpus.jsonl.gz')) for p in sorted(RAW.glob('*/leap_rerun_launch.json'))]
    records += [(p, p.with_name(f'{p.stem}_gpus.jsonl.gz')) for p in sorted(RAW.glob('prep/*/*/*.json'))
                if not p.name.endswith('_compare.json')]
    for record, out in records:
        # A file newer than its record is that run's (a step rerun with --force rewrites its record, so its log follows).
        if out.exists() and out.stat().st_mtime >= record.stat().st_mtime:
            continue
        run = json.loads(record.read_text())['run']
        print(f'{record.relative_to(RAW)}: {run} -> {out.name}', flush=True)
        if args.run:
            write_jsonl_gz(out, gpu_rows(run))


def main():
    # Command-line options.
    args = parse_args()
    print(f"{'RUN' if args.run else 'DRY RUN'}: {args.action} {args.datasets}", flush=True)
    # One action per call.
    {'setup': setup, 'scored': scored, 'leap': leap, 'export': export, 'prep': prep, 'compare': compare,
     'gpus': gpus}[args.action](args)


if __name__ == '__main__':
    sys.exit(main())
