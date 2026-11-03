"""Provenance of installed ground truth.

A 003_groundtruth directory holding MARKER was installed from external labels (p004), not produced by
p003. The automatic labels it replaced live in the sibling directory named by backup_dir().
"""
from pathlib import Path

from polyis.io import cache


# File p004 writes into an installed ground-truth directory; its presence marks the directory as installed.
MARKER = 'SOURCE.json'
# Suffix of the sibling directory holding the automatic ground truth an install replaced.
BACKUP_SUFFIX = '.auto'


def groundtruth_dir(dataset: str, video: str) -> Path:
    # The canonical ground-truth directory of one video.
    return cache.exec(dataset, 'groundtruth', video)


def backup_dir(dataset: str, video: str) -> Path:
    # The sibling directory that keeps the automatic ground truth after an install.
    directory = groundtruth_dir(dataset, video)
    return directory.with_name(directory.name + BACKUP_SUFFIX)


def is_installed(dataset: str, video: str) -> bool:
    # An installed directory is recognized by its provenance marker.
    return (groundtruth_dir(dataset, video) / MARKER).exists()


def assert_not_installed(dataset: str, videos: list[str]) -> None:
    """Refuse to regenerate ground truth over installed labels (p003 would silently replace them)."""
    # Collect every requested video whose ground truth was installed by p004.
    installed = [video for video in videos if is_installed(dataset, video)]
    # Fail before any work starts, naming every affected video.
    if installed:
        raise RuntimeError(
            f'{dataset}: ground truth for {", ".join(sorted(installed))} was installed from external labels '
            f'({MARKER} present); p003 would overwrite it. Move the installed directories aside first.')
