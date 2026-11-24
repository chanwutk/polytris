#!/usr/local/bin/python

"""
Upload (once) or download the detector weights for caldot1/2-y05, ams-y05, and jnc0/2/6/7,
at the container paths that configs/detectors.yaml and configs/retinanet.json load.
Run inside the polyis container.

    ./run scripts/weights.py download
    ./run scripts/weights.py upload      # once; needs `pip install huggingface_hub` and `hf auth login`

The YOLOv5x6 fine-tunes download from public tetris-db repos on the Hugging Face Hub.
B3D's RetinaNet downloads from B3D's official Google Drive release (b3d-project/b3d, download.py),
which is byte-identical to the checkpoint the experiments used; upload backs it up to a private repo.
"""

import argparse
import hashlib
import os
import shutil
import urllib.request

# Container path -> Hub repo, its visibility, download URL, and sha256 of the checkpoint the experiments used.
WEIGHTS = {
    '/polyis-data/training/ultralytics/caldot1/weights/yolov5x6/weights/best.pt': dict(
        repo='tetris-db/caldot1-yolov5', private=False,
        url='https://huggingface.co/tetris-db/caldot1-yolov5/resolve/main/best.pt',
        sha256='451b44c7102e3cb255b8e3193eca40673a85754d4f7c793d481b0af51aac6135'),
    '/polyis-data/training/ultralytics/caldot2/weights/yolov5x6/weights/best.pt': dict(
        repo='tetris-db/caldot2-yolov5', private=False,
        url='https://huggingface.co/tetris-db/caldot2-yolov5/resolve/main/best.pt',
        sha256='ef2b32e315aca947405be755ac32ef497ed4216de56dbc0fea9ac14ca3987c7c'),
    '/polyis-data/training/ultralytics/amsterdam/weights/yolov5x6/weights/best.pt': dict(
        repo='tetris-db/amsterdam-yolov5', private=False,
        url='https://huggingface.co/tetris-db/amsterdam-yolov5/resolve/main/best.pt',
        sha256='be2b26074f713f27b3ebe36c7a3a6e2c38076874ed64c296fae069ead1337ab6'),
    # B3D's RetinaNet (jnc0/2/6/7); configs/retinanet.json loads it relative to /polyis.
    '/polyis/weights/retinanet_b3d.pth': dict(
        repo='tetris-db/b3d', private=True,
        url='https://drive.usercontent.google.com/download?id=17ZiwW_11q5oLldTCXuXjCpd8FQ7MjKaD&export=download&confirm=t',
        sha256='cd48354aed70787596c4783fe2d1df1d90bfd790e41ae82627580dacd8a1a0c3'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['upload', 'download'])
    return parser.parse_args()


def sha256(path: str) -> str:
    """Return the hex sha256 of a file, read in 1 MiB chunks."""
    # Hash the file incrementally so large checkpoints do not load into memory.
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check(path: str, expected: str):
    """Raise if the file's sha256 differs from the expected digest."""
    # Compare against the recorded digest of the checkpoint the experiments used.
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'{path}: sha256 {actual} != expected {expected}')


def upload():
    """Publish every checkpoint in WEIGHTS to its Hub repo after verifying its digest."""
    # Imported here so download needs only the standard library.
    from huggingface_hub import HfApi
    api = HfApi()
    for path, w in WEIGHTS.items():
        # Refuse to publish anything but the checkpoints the experiments used.
        check(path, w['sha256'])
        # Create the repo with its visibility on first run; a no-op when it already exists.
        api.create_repo(w['repo'], private=w['private'], exist_ok=True)
        # Upload under the checkpoint's own file name.
        name = os.path.basename(path)
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=w['repo'], commit_message=f'Add {name}')
        print('uploaded', f"https://huggingface.co/{w['repo']}", '(private)' if w['private'] else '(public)')


def download():
    """Fetch every checkpoint in WEIGHTS to its container path, skipping ones already in place."""
    for path, w in WEIGHTS.items():
        # Skip files that are already the right checkpoint.
        if os.path.exists(path) and sha256(path) == w['sha256']:
            print('ok', path)
            continue
        # Create the destination directory tree.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Download to a side file so a failed or partial download never replaces the target.
        tmp = path + '.part'
        try:
            with urllib.request.urlopen(w['url']) as r, open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f)
            # Verify before moving into place; also catches an HTML error page served instead of the file.
            check(tmp, w['sha256'])
        except BaseException:
            # Remove the partial or corrupt file; leave any existing target untouched.
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        # Atomically replace the target (same directory, same filesystem).
        os.replace(tmp, path)
        print('downloaded', path)


if __name__ == '__main__':
    # Dispatch to the requested action.
    {'upload': upload, 'download': download}[parse_args().action]()
