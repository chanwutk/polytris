"""R1.O2: rebuild LEAP's per-clip track exports from a parsed_results directory with the exporter the scored runs used.

The scored exports in /sota-results/<dataset>/leap/<clip>.json were written by reconstruct_tracks_json in
export-leap-outputs.py (ace:/work/alenachao/tunnel/work/scripts/). That script runs its own export loop at import
time over hard-coded paths, so this script parses its source and executes only its import lines and the
reconstruct_tracks_json definition, then calls that function on every <clip>_results.json and
<clip>_tracks.json of --parsed-dir. The default preserves the original function; --clip-negative-starts skips
pre-video frame writes without changing pattern interpolation or LEAP's predicted intervals. The frame lists are written as one gzipped JSON line per clip
({"video_name": ..., "frames": [...]}), so measurements/r1o2_leap_otif_rerun.py can compare them with the scored
exports and score them. Runs in the LEAP image (numpy 1.24.3, as the scored container had).
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description="R1.O2: rebuild LEAP's track exports with export-leap-outputs.py's function")
    parser.add_argument('--exporter', required=True)
    parser.add_argument('--parsed-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--clip-negative-starts', action='store_true')
    return parser.parse_args()


def load_exporter(path: str, clip_negative_starts: bool = False):
    """reconstruct_tracks_json from the exporter's source, without running the script's module-level loop."""
    # The script's syntax tree.
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    # Its imports and the one function; everything else (the hard-coded export loop) is left out.
    keep = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
            or (isinstance(node, ast.FunctionDef) and node.name == 'reconstruct_tracks_json')]
    assert sum(isinstance(n, ast.FunctionDef) for n in keep) == 1, f'{path}: reconstruct_tracks_json not found once'
    if clip_negative_starts:
        # LEAP can predict starts before zero, including indices smaller than the export's negative-index range.
        inserted = 0
        function = next(node for node in keep if isinstance(node, ast.FunctionDef))
        for loop in (node for node in ast.walk(function) if isinstance(node, ast.For)):
            for index, statement in enumerate(loop.body):
                if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'frame_idx' for target in statement.targets):
                    loop.body.insert(index + 1, ast.parse('if frame_idx < 0: continue').body[0])
                    inserted += 1
                    break
        assert inserted == 1, f'{path}: expected exactly one frame assignment, found {inserted}'
    # Execute them in a fresh namespace.
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[])), path, 'exec'), namespace)
    return namespace['reconstruct_tracks_json']


def main():
    # Command-line options.
    args = parse_args()
    # The scored runs' exporter function and the digest of its source, for the record.
    reconstruct = load_exporter(args.exporter, clip_negative_starts=args.clip_negative_starts)
    with open(args.exporter, 'rb') as f:
        exporter_sha256 = hashlib.sha256(f.read()).hexdigest()
    # Clips of the parsed results directory, by their results file.
    names = sorted(f[:-len('_results.json')] for f in os.listdir(args.parsed_dir) if f.endswith('_results.json'))
    assert names, f'no <clip>_results.json in {args.parsed_dir}'
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            # The exporter writes <tmp>/<clip>.json and prints a line; read the file back.
            reconstruct(os.path.join(args.parsed_dir, f'{name}_results.json'),
                        os.path.join(args.parsed_dir, f'{name}_tracks.json'), tmp, name)
            with open(os.path.join(tmp, f'{name}.json')) as f:
                row = dict(video_name=name, exporter_sha256=exporter_sha256, frames=json.load(f))
                if args.clip_negative_starts:
                    row['negative_frame_policy'] = 'drop_before_zero'
                rows.append(row)
    # One gzipped line per clip, written to a temporary name first.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.out + '.tmp', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    os.replace(args.out + '.tmp', args.out)
    print(f'wrote {args.out}: {len(rows)} clips')


if __name__ == '__main__':
    main()
