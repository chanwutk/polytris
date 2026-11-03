"""Copy result files or directories to a sibling named <path><suffix> before a rerun overwrites them.

All paths are checked before anything is copied: every path must exist (so nothing is left unbacked by a
typo) and no snapshot may exist yet (so a second run never replaces the original copy).
"""
import argparse
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Snapshot result paths before they are overwritten')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--suffix', default='.auto-gt')
    return parser.parse_args()


def snapshot_path(path: Path, suffix: str) -> Path:
    # The sibling that keeps the copy, e.g. 070_accuracy -> 070_accuracy.auto-gt, a.json -> a.json.auto-gt.
    return path.with_name(path.name + suffix)


def snapshot(paths: list[Path], suffix: str) -> list[Path]:
    """Copy every path to its snapshot sibling; returns the snapshot paths."""
    # Reject an empty suffix, which would copy a path onto itself.
    if not suffix:
        raise ValueError('suffix must be non-empty')
    # Reject repeated paths, which would otherwise fail halfway through.
    if len(set(paths)) != len(paths):
        raise ValueError('paths repeat')
    # Validate everything before copying anything.
    missing = [str(p) for p in paths if not p.exists()]
    taken = [str(snapshot_path(p, suffix)) for p in paths if snapshot_path(p, suffix).exists()]
    if missing or taken:
        raise FileExistsError(f'missing: {missing}; snapshot already exists: {taken}')
    # Copy files and directory trees, keeping timestamps.
    for path in paths:
        if path.is_dir():
            shutil.copytree(path, snapshot_path(path, suffix))
        else:
            shutil.copy2(path, snapshot_path(path, suffix))
    return [snapshot_path(p, suffix) for p in paths]


def main():
    args = parse_args()
    for target in snapshot([Path(p) for p in args.paths], args.suffix):
        print(target)


if __name__ == '__main__':
    main()
