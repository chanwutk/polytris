"""Build LEAP motion patterns without merging clip-local tracker identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Namespace LEAP training-label IDs before its original clustering')
    parser.add_argument('--video-name', required=True)
    parser.add_argument('--label-dir', type=Path, required=True)
    parser.add_argument('--normalized-label-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def normalize_training_labels(label_dir: Path, video_name: str, normalized_label_dir: Path) -> dict:
    # BYTETracker restarts its counter for each clip; clustering needs (clip, ID), not ID alone.
    if not video_name or Path(video_name).name != video_name or video_name in ('.', '..'):
        raise ValueError(f'Not a dataset name: {video_name}')
    source = (label_dir / video_name).resolve()
    target = (normalized_label_dir / video_name).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('Normalized labels must be separate from the source labels')
    files = sorted(source.glob('*/train/label_*.txt'))
    if not files:
        raise FileNotFoundError(f'No clip-local training labels under {source}')
    expected = {p.relative_to(source) for p in files}
    if {p.relative_to(target) for p in target.glob('*/train/label_*.txt')} - expected:
        raise ValueError(f'Unexpected existing normalized labels under {target}')
    identities, records = {}, []
    for path in files:
        relative = path.relative_to(source)
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(f'{output.name}.{os.getpid()}.tmp')
        digest, rows = hashlib.sha256(), 0
        with path.open('rb') as inp, tmp.open('wb') as out:
            for row in inp:
                digest.update(row)
                parts = row.split(b',', 2)
                if len(parts) != 3 or len(row.split(b',')) < 8:
                    raise ValueError(f'Malformed label at {path}:{rows + 1}')
                key = (relative.parts[0], int(parts[1]))
                normalized_id = identities.setdefault(key, len(identities) + 1)
                # Keep every frame, coordinate, confidence, class, and line-ending byte unchanged.
                out.write(parts[0] + b',' + str(normalized_id).encode('ascii') + b',' + parts[2])
                rows += 1
        os.replace(tmp, output)
        records.append(dict(file=str(relative), rows=rows, source_sha256=digest.hexdigest()))
    return dict(dataset=video_name, source=str(source), normalized=str(target),
                files=records, rows=sum(r['rows'] for r in records), identities=len(identities))


def main():
    args = parse_args()
    metadata = normalize_training_labels(args.label_dir, args.video_name, args.normalized_label_dir)
    # Preserve the input mapping and provenance alongside this isolated, derived label copy.
    record = args.normalized_label_dir / 'normalization.json'
    record.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'files'}), flush=True)
    # Keep LEAP's original algorithm, k search, 96-track cap, and filtering thresholds unchanged.
    subprocess.run([sys.executable, 'track_cluster_updated.py', '--video_name', args.video_name,
                    '--label_dir', str(args.normalized_label_dir), '--output_dir', str(args.output_dir),
                    '--k', '0', '--cluster_min_num', '3'], check=True)
    import numpy as np
    patterns = args.output_dir / args.video_name / f'{args.video_name}_0_0_tracks_clustered.npy'
    if len(np.load(patterns, allow_pickle=True)) == 0:
        raise RuntimeError(f'LEAP produced an empty motion-pattern library: {patterns}')


if __name__ == '__main__':
    main()
