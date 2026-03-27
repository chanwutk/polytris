import pandas as pd

from evaluation import manifests


def test_build_split_video_manifest_filters_supported_extensions(monkeypatch):
    # Redirect the dataset store lookup to a deterministic fake directory.
    monkeypatch.setattr(manifests.store, 'dataset', lambda dataset, videoset: f"/datasets/{dataset}/{videoset}")
    # Pretend the configured split directory always exists.
    monkeypatch.setattr(manifests.os.path, 'exists', lambda path: True)
    # Return both supported and unsupported filenames so the filter is exercised.
    monkeypatch.setattr(
        manifests.os,
        'listdir',
        lambda path: ['te02.avi', 'notes.txt', 'te01.mp4', 'preview.png'],
    )

    # Build the manifest for one dataset/test split pair.
    manifest_df = manifests.build_split_video_manifest(datasets=['demo'], videosets=['test'])

    # Keep only the supported video extensions in sorted order.
    assert manifest_df.to_dict('records') == [
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4'},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.avi'},
    ]


def test_build_sota_video_param_manifest_cross_joins_test_videos(monkeypatch, tmp_path):
    # Create a minimal transformed stat.csv with two param IDs.
    stat_path = tmp_path / 'stat.csv'
    pd.DataFrame({'param_id': [0, 2], 'runtime': [1.0, 2.0]}).to_csv(stat_path, index=False)

    # Redirect the SOTA cache lookup to the temporary stat.csv path.
    monkeypatch.setattr(
        manifests.cache,
        'sota',
        lambda system, dataset, *args: str(stat_path) if args == ('stat.csv',) else str(tmp_path),
    )
    # Redirect the dataset store lookup to a deterministic fake directory.
    monkeypatch.setattr(manifests.store, 'dataset', lambda dataset, videoset: f"/datasets/{dataset}/{videoset}")
    # Pretend the configured split directory always exists.
    monkeypatch.setattr(manifests.os.path, 'exists', lambda path: True)
    # Return the configured test videos for the manifest expansion.
    monkeypatch.setattr(manifests.os, 'listdir', lambda path: ['te01.mp4', 'te02.mp4'])

    # Build the expected test-video/param_id manifest.
    manifest_df = manifests.build_sota_video_param_manifest('otif', 'demo')

    # Cross join the two videos with the two param IDs.
    assert manifest_df.to_dict('records') == [
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'param_id': 0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'param_id': 2},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4', 'param_id': 0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4', 'param_id': 2},
    ]
