import pandas as pd

from evaluation.p121_throughput_compute import filter_excluded_runtime_ops


def test_filter_excluded_runtime_ops_drops_configured_stage_ops():
    # Classification excludes frame reading and result retrieval; inference stays.
    classify = pd.DataFrame([
        {'stage': '020_exec_classify', 'time': 1.0, 'op': 'read'},
        {'stage': '020_exec_classify', 'time': 2.0, 'op': 'inference'},
        {'stage': '020_exec_classify', 'time': 3.0, 'op': 'retrieve'},
    ])
    assert filter_excluded_runtime_ops(classify, '020_exec_classify').to_dict('records') == [
        {'stage': '020_exec_classify', 'time': 2.0, 'op': 'inference'},
    ]

    # The reference detector's video decoding is excluded, matching Tetris's extraction boundary.
    reference = pd.DataFrame([
        {'stage': '001_preprocess_groundtruth_detection', 'time': 4.0, 'op': 'read'},
        {'stage': '001_preprocess_groundtruth_detection', 'time': 5.0, 'op': 'detect'},
    ])
    assert filter_excluded_runtime_ops(reference, '001_preprocess_groundtruth_detection').to_dict('records') == [
        {'stage': '001_preprocess_groundtruth_detection', 'time': 5.0, 'op': 'detect'},
    ]


def test_filter_excluded_runtime_ops_keeps_ilp_solve_time():
    # Extraction time includes the ILP (R1.O3), so pruning keeps every op, solve_ilp included.
    prune = pd.DataFrame([
        {'stage': '022_exec_prune_polyominoes', 'time': 1.0, 'op': 'group_tiles'},
        {'stage': '022_exec_prune_polyominoes', 'time': 2.0, 'op': 'solve_ilp'},
        {'stage': '022_exec_prune_polyominoes', 'time': 3.0, 'op': 'write_output'},
    ])
    assert filter_excluded_runtime_ops(prune, '022_exec_prune_polyominoes').to_dict('records') == prune.to_dict('records')


def test_filter_excluded_runtime_ops_keeps_other_stages_unchanged():
    # Build a representative runtime table for a stage without exclusions.
    file_timings = pd.DataFrame([
        {'stage': '020_exec_classify', 'time': 1.0, 'op': 'classify'},
        {'stage': '020_exec_classify', 'time': 2.0, 'op': 'transform'},
    ])

    # Apply the stage-local exclusion policy.
    filtered_df = filter_excluded_runtime_ops(file_timings, '020_exec_classify')

    # Keep every row because the stage has no configured exclusions.
    assert filtered_df.to_dict('records') == file_timings.to_dict('records')
