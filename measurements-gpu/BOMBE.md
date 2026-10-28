# Running revision GPU jobs on bombe

bombe has 7 Quadro RTX 6000 GPUs (the same model as ace's), 80 CPUs on two NUMA nodes, and rootless docker with the
`polyis-polyis:latest` image (`efe83173d8fd`, identical to ace's). `/work` is the same NFS export on both hosts, so
`/work/cwkt/projects/tetris` is this checkout on both. Set up 2026-10-27.

## Layout (`/data/cwkt/data/bombe-revision/`)

| Path | Contents | Size |
| --- | --- | --- |
| `mirror/polyis-data/` | ace's `/data/cwkt/data/polyis-data` (videos, training sets, weights; the 6 relative dataset symlinks such as `caldot1-y05 -> caldot1` kept) | 66.8 GiB, 52,111 files |
| `mirror/polyis-cache/<ds>/{indexing,evaluation}/` | from ace's cache for the 7 EXEC datasets and the root `caldot1` (the roots `caldot2` and `ams` have only `execution/` on ace) | 21.3 GiB, 15,802,950 files (ams-y05 6.8 M) |
| `mirror/otif-dataset/` | `/work/cwkt/data/otif-dataset` (bombe's own `/data/cwkt/data/otif-dataset` is empty) | 65.9 GiB, 140,997 files |
| `mirror/sota-results/` | `/work/alenachao/tunnel/hota_outputs` | 0.4 GiB, 3,977 files |
| `pylib/` | `duckdb==2.0.0`, which ace's container has pip-installed and the image lacks | |
| `scratch/<track>/` | one directory per track: `polyis-cache/`, `global.yaml`, `samples/` | |
| `logs/`, `bin/` | mirror logs, checksum samples, and the two copy scripts | |

The mirror is a read-only snapshot taken on 2026-10-27 and verified against its sources: equal file, symlink, and
byte counts per tree (per dataset for the cache), an `rsync -n` dry run with no differences (polyis-data, OTIF, SOTA),
and checksums of samples (84 polyis-data and cache files including 1.3 GB of videos; OTIF's `yolov3/`,
`dataset/caldot1/`, and one 4.6 GB video; all of SOTA). ace's trees were pushed from ace; the two `/work` trees were
copied on bombe, which mounts the same NFS export. Nothing writes to the mirror, and the launcher mounts it
read-only. bombe's older `/data/cwkt/data/{polyis-data,polyis-cache,otif-dataset,tetris-annotate,
r1o4-jpeg-control,r9o1-matched-timing}` belong to other work; leave them alone.

`execution/` is not mirrored. A track copies the execution inputs it needs from ace into its own scratch cache,
pushing from ace (ace's key is authorized on bombe, so this works inside tmux without agent forwarding):

```sh
ssh ace 'cd /data/cwkt/data/polyis-cache && rsync -a --relative ./caldot1/execution/te00.mp4/003_groundtruth \
  bombe.millennium.berkeley.edu:/data/cwkt/data/bombe-revision/scratch/<track>/polyis-cache/'
```

Mirrored inputs go into a scratch cache as copy-on-write clones, which take no space and under a second per GB (ZFS
block cloning works on `/data`). Do not hard-link them: a pipeline write with `open(path, 'w')` would truncate the
mirror's file too.

```sh
cp -a --reflink=always /data/cwkt/data/bombe-revision/mirror/polyis-cache/jnc0 \
  /data/cwkt/data/bombe-revision/scratch/<track>/polyis-cache/
```

## Launcher

```sh
# On bombe, in a detached tmux session for anything long.
cd /work/cwkt/projects/tetris
measurements-gpu/bombe_container.sh <GPU 0-6> /data/cwkt/data/bombe-revision/scratch/<track> '<command>' >> <log> 2>&1
# Smoke test: GPU, polyis, all 17 Cython modules, duckdb, mounts, Gurobi.
measurements-gpu/bombe_container.sh 4 /data/cwkt/data/bombe-revision/scratch/<track> 'python measurements-gpu/bombe_check.py'
```

- It refuses to start unless the GPU shows 0% utilization, under 100 MiB, and no compute process in three samples 1 s
  apart (exit 3). The scratch directory must be under `scratch/`, and the GPU index must exist (exit 2).
- Mounts: the checkout at `/polyis` (read-write, so `measurements-results/` lands in the shared checkout),
  `<scratch>` at `/scratch`, `<scratch>/polyis-cache` at `/polyis-cache`, the mirror at `/polyis-data`,
  `/otif-dataset`, and `/sota-results` (read-only), `pylib/` on `PYTHONPATH`, and `~/.gurobi/gurobi.lic` at
  `/opt/gurobi/gurobi.lic`. The environment matches ace's `docker-compose.base.yml` (`POLYIS_DATA`, `POLYIS_CACHE`,
  `PYTHONWARNINGS`, host network, 8 GB `/dev/shm`, tmpfs `/tmp`).
- `CONFIG=/scratch/global.yaml`, copied from `configs/global.yaml` on first use and then kept, so a track can narrow
  `EXEC.DATASETS` without touching the checkout's config (which ace's runs read).
- Only the chosen GPU is visible (`--gpus device=G`), as `cuda:0`. A script that takes a GPU index or sets
  `CUDA_VISIBLE_DEVICES` itself (for example `measurements-gpu/r9o2_preparation_cost.py --gpu`) must be given `0`.
  `BOMBE_GPU`, `BOMBE_CPUS`, and `BOMBE_RUN` carry the physical index, CPU set, and run id.
- Each run writes `<scratch>/samples/<run>.json`, a summary with the GPU, NUMA node, CPU set, image id, start, end,
  exit code, and `contaminated`. It is true when any sample saw a live compute process on the GPU whose cgroup is
  not this container's (other containers, other users). `<run>.jsonl` holds those samples (utilization, memory, pids,
  load average), taken about every 2 s (the sampler sleeps 1 s after each `nvidia-smi` query, which itself takes about
  1 s), and `<run>.gpus.csv` every GPU's utilization and memory each second. Tested with a second container
  started on the same GPU: its 7 samples were flagged and the job's own process never was.
- Killing the launcher (for example `tmux kill-session`) removes the container and stops both samplers; such a run has
  no `<run>.json`. Tested: no container, compute process, or `nvidia-smi` loop was left behind.

## NUMA and GPU mapping

| GPUs | NUMA node | CPUs |
| --- | --- | --- |
| 0, 1, 2, 3 | 0 | 0-19,40-59 |
| 4, 5, 6 | 1 | 20-39,60-79 |

GPU pairs 0-1, 2-3, and 5-6 share a PCIe switch. Rootless docker on bombe ignores `--cpuset-cpus` (the user cgroup
delegates only cpu, memory, and pids), so the launcher pins the command with `taskset -c <node CPUs>` inside the
container (checked: `Cpus_allowed_list` is the node's CPUs). Memory is not bound: `numactl` is not in the image, and
pinned threads allocate on their own node under the default first-touch policy. `r9o1_matched_timing.sh` bound memory
with an apt-installed `numactl`; add that only if a measurement must match it exactly.

## Gurobi: the license has expired

`~/.gurobi/gurobi.lic` (one file, since the home directory is shared NFS) is a Web License Service license, ID 2792523,
and Gurobi rejects it on both hosts: `GurobiError: License 2792523 has expired` (also from ace's running `polyis`
container). Without the file, gurobipy falls back to its size-limited license (2,000 variables), too small for p022
(up to 9,455 variables per clip), and p022's C solver (`polyis.sample.ilp.c.gurobi`) fails at `GRBstartenv`. Nothing
that runs p022 or `solve_ilp` can run on either host until the author renews the academic WLS license at
portal.gurobi.com and replaces `~/.gurobi/gurobi.lic`. Everything else in `bombe_check.py` passes.

## Notes for tracks

- nvidia-smi persistence mode is off, so a cold `nvidia-smi` call takes about 5 s; the idle check takes 15-40 s, and
  the launcher's own `nvidia-smi -lms 1000` loop keeps the driver warm during the run. A cold call sometimes reports
  3-4% on an idle GPU with 0 MiB; if the launcher refuses a GPU with 0 MiB and no process, run it again.
- The rootless docker daemon has no linger: it runs while any of the user's sessions (a tmux session counts) exists
  and restarts on an SSH login. The launcher waits up to 30 s for it.
- Other users can start jobs on bombe at any time; check `nvidia-smi` before choosing a GPU, and read `contaminated`
  in each run's summary.
- `/work` (NFS) reads reach about 75 MB/s for one stream and are latency-bound on small files; read inputs from the
  mirror, not `/work`, during timed runs.
- `measurements-gpu/r9o2_preparation_cost.py` names ace host paths (`OTIF_HOST_DATASET=/work/cwkt/data/otif-dataset`,
  OTIF and LEAP code under `/work`); the same paths exist on bombe through NFS, and the mirror holds the OTIF dataset.
- ultralytics warns that `/root/.config` is not writable and uses `/tmp/Ultralytics`; this is harmless.
