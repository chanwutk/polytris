import pandas as pd

from evaluation import p140_otif_transform


def test_normalize_otif_stat_csv_renames_param_id_and_runtime_total(tmp_path):
    # Write a minimal OTIF stat CSV that uses the legacy unnamed param-id column.
    stat_path = tmp_path / 'otif_demo.csv'
    pd.DataFrame({
        'Unnamed: 0': [0, 3],
        'detector_cfg': ['det-a', 'det-b'],
        'segmentation_cfg': ['seg-a', 'seg-b'],
        'tracker_cfg': ['trk-a', 'trk-b'],
        'runtime_total': [1.25, 2.50],
    }).to_csv(stat_path, index=False)

    # Normalize the OTIF CSV to the shared stat schema.
    stat_df = p140_otif_transform.normalize_otif_stat_csv(str(stat_path))

    # Keep the expected stat columns with integer param ids and canonical runtime.
    assert stat_df.to_dict('records') == [
        {
            'param_id': 0,
            'detector_cfg': 'det-a',
            'segmentation_cfg': 'seg-a',
            'tracker_cfg': 'trk-a',
            'runtime': 1.25,
        },
        {
            'param_id': 3,
            'detector_cfg': 'det-b',
            'segmentation_cfg': 'seg-b',
            'tracker_cfg': 'trk-b',
            'runtime': 2.50,
        },
    ]


def test_build_tracking_transform_manifest_matches_configured_test_split(monkeypatch, tmp_path):
    # Create a minimal configured test split with two videos.
    monkeypatch.setattr(
        p140_otif_transform,
        'build_split_video_manifest',
        lambda datasets, videosets: pd.DataFrame([
            {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4'},
            {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4'},
        ]),
    )

    # Redirect transformed SOTA outputs to the temporary directory.
    monkeypatch.setattr(
        p140_otif_transform.cache,
        'sota',
        lambda system, dataset, *args: str(tmp_path / 'out' / system / dataset / '/'.join(args)),
    )

    # Create the raw tracking tree with two param ids and one extra unexpected file.
    tracks_dir = tmp_path / 'tracks'
    for param_id in [0, 3]:
        for video_id in [1, 2]:
            track_path = tracks_dir / str(param_id) / f'{video_id}.json'
            track_path.parent.mkdir(parents=True, exist_ok=True)
            track_path.write_text('[]')
    extra_track_path = tracks_dir / '9' / '99.json'
    extra_track_path.parent.mkdir(parents=True, exist_ok=True)
    extra_track_path.write_text('[]')

    # Build the transform manifest from the configured stat rows and raw tracking tree.
    manifest_df = p140_otif_transform.build_tracking_transform_manifest(
        'otif',
        'demo',
        pd.DataFrame({'param_id': [0, 3]}),
        str(tracks_dir),
    )

    # Keep only the configured test videos crossed with the configured param ids.
    assert manifest_df[['dataset', 'videoset', 'video', 'video_id', 'param_id']].to_dict('records') == [
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'video_id': 1, 'param_id': 0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'video_id': 1, 'param_id': 3},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4', 'video_id': 2, 'param_id': 0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4', 'video_id': 2, 'param_id': 3},
    ]

    # Resolve output paths under the transformed SOTA cache layout.
    assert manifest_df['output_jsonl_path'].tolist() == [
        str(tmp_path / 'out' / 'otif' / 'demo' / 'te01.mp4' / 'tracking_results' / '000' / 'tracking.jsonl'),
        str(tmp_path / 'out' / 'otif' / 'demo' / 'te01.mp4' / 'tracking_results' / '003' / 'tracking.jsonl'),
        str(tmp_path / 'out' / 'otif' / 'demo' / 'te02.mp4' / 'tracking_results' / '000' / 'tracking.jsonl'),
        str(tmp_path / 'out' / 'otif' / 'demo' / 'te02.mp4' / 'tracking_results' / '003' / 'tracking.jsonl'),
    ]
