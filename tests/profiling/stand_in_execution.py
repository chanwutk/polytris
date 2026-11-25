"""A stand-in for `python -m execution.main` with the same command line, for profiling's subprocess test.

Writes <out>/<stem>.jsonl per video and <out>/summary.json, and <out>/argv.json with its arguments.  Exits 5 without
writing a summary when the config's relevance_threshold is 0.75 (the test's failing configuration).
"""

import argparse
import json
import os
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--videos', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--save', default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    print('stand-in stderr line', file=sys.stderr)
    if config['params']['relevance_threshold'] == 0.75:
        return 5
    os.makedirs(args.out, exist_ok=True)
    stems = sorted(os.path.splitext(v)[0] for v in os.listdir(args.videos))
    for stem in stems:
        open(os.path.join(args.out, f'{stem}.jsonl'), 'w').close()
    with open(os.path.join(args.out, 'argv.json'), 'w') as f:
        json.dump(vars(args), f)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'videos': len(stems), 'runtime_s': 3.5, 'frames': 7 * len(stems), 'saved': args.save is not None,
                   'video_frames': {s: 7 for s in stems}}, f)
    return 0


if __name__ == '__main__':
    sys.exit(main())
