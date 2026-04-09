import pandas as pd

from evaluation.p130_tradeoff_compute import build_tradeoff_table


def test_build_tradeoff_table_aggregates_split_level_variants(monkeypatch):
    # Build a split-level accuracy table with one Polytris row and one naive row.
    accuracy_df = pd.DataFrame([
        {
            'dataset': 'demo',
            'videoset': 'test',
            'variant': 'polytris',
            'variant_id': 'cfg_a',
            'classifier': 'ShuffleNet05',
            'tilesize': 60,
            'sample_rate': 2,
            'tracking_accuracy_threshold': pd.NA,
            'tilepadding': 'none',
            'canvas_scale': 1.0,
            'tracker': 'bytetrackcython',
            'HOTA_HOTA': 0.8,
        },
        {
            'dataset': 'demo',
            'videoset': 'test',
            'variant': 'naive',
            'variant_id': 'naive',
            'classifier': pd.NA,
            'tilesize': pd.NA,
            'sample_rate': pd.NA,
            'tracking_accuracy_threshold': pd.NA,
            'tilepadding': pd.NA,
            'canvas_scale': pd.NA,
            'tracker': pd.NA,
            'HOTA_HOTA': 0.7,
        },
    ])

    # Build the processed query runtime table with one Polytris config and one naive baseline.
    query_df = pd.DataFrame([
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'variant': 'polytris', 'variant_id': 'cfg_a', 'time': 1.0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'variant': 'polytris', 'variant_id': 'cfg_a', 'time': 2.0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4', 'variant': 'polytris', 'variant_id': 'cfg_a', 'time': 4.0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'variant': 'naive', 'variant_id': 'naive', 'time': 0.5},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te01.mp4', 'variant': 'naive', 'variant_id': 'naive', 'time': 1.0},
        {'dataset': 'demo', 'videoset': 'test', 'video': 'te02.mp4', 'variant': 'naive', 'variant_id': 'naive', 'time': 3.0},
    ])

    # Return deterministic frame counts for the two test videos.
    monkeypatch.setattr(
        'evaluation.p130_tradeoff_compute.get_video_frame_count',
        lambda dataset, video: {'te01.mp4': 100, 'te02.mp4': 50}[video],
    )

    # Build the canonical split-level tradeoff table.
    tradeoff_df = build_tradeoff_table(accuracy_df, query_df)

    # Keep one split-level row for the Polytris variant and one for the naive baseline.
    assert sorted(tradeoff_df['variant_id'].tolist()) == ['cfg_a', 'naive']

    # Validate the Polytris split-level aggregation.
    polytris_row = tradeoff_df[tradeoff_df['variant_id'] == 'cfg_a'].iloc[0]
    assert polytris_row['time'] == 7.0
    assert polytris_row['frame_count'] == 150
    assert polytris_row['throughput_fps'] == 150 / 7.0
    assert polytris_row['classifier'] == 'ShuffleNet05'

    # Validate the naive split-level aggregation and null parameter contract.
    naive_row = tradeoff_df[tradeoff_df['variant_id'] == 'naive'].iloc[0]
    assert naive_row['time'] == 4.5
    assert naive_row['frame_count'] == 150
    assert naive_row['throughput_fps'] == 150 / 4.5
    assert pd.isna(naive_row['classifier'])
    assert pd.isna(naive_row['sample_rate'])
