import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


class ArtifactDownloadTest(unittest.TestCase):
    def test_stages_both_bundles_and_preserves_existing_files(self):
        script = Path(__file__).resolve().parents[1] / 'reproduce/download.sh'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hf = root / 'hf'
            hf.write_text('''#!/usr/bin/env bash
set -euo pipefail
case "$2" in
    tetris-db/b3d-processed) datasets=(jnc0) ;;
    tetris-db/otif-processed) datasets=(caldot1 caldot2 ams) ;;
    *) exit 1 ;;
esac
for dataset in "${datasets[@]}"; do
    clips="$8/datasets/$dataset/test"
    labels="$8/groundtruth/$dataset/execution/te00.mp4/003_groundtruth"
    mkdir -p "$clips" "$labels"
    printf 'video\n' > "$clips/te00.mp4"
    printf 'labels\n' > "$labels/tracking.jsonl"
done
''')
            hf.chmod(0o755)
            data = root / 'data with spaces'
            cmd = ['bash', str(script), str(data)]
            env = dict(os.environ, PATH=f'{root}{os.pathsep}{os.environ["PATH"]}')
            subprocess.run([*cmd, 'b3d'], env=env, check=True, capture_output=True, text=True)
            datasets = data / 'polyis-data/datasets'
            self.assertTrue((datasets / 'jnc0/test/te00.mp4').is_file())
            self.assertFalse((datasets / 'caldot1').exists())

            for _ in range(2):
                subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

            for name in ('caldot1', 'caldot2', 'ams'):
                self.assertEqual((datasets / f'{name}-y05').readlink(), Path(name))
                self.assertTrue((datasets / f'{name}-y05/test/te00.mp4').is_file())
            tracking = data / 'polyis-cache/caldot1/execution/te00.mp4/003_groundtruth/tracking.jsonl'
            self.assertEqual(tracking.read_text(), 'labels\n')

            tracking.write_text('local labels\n')
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Existing file differs from the release', result.stderr)
            self.assertEqual(tracking.read_text(), 'local labels\n')
