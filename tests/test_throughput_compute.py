import pandas as pd

from evaluation.p121_throughput_compute import filter_excluded_runtime_ops


def test_filter_excluded_runtime_ops_drops_configured_stage_ops():
    # Build a representative runtime table for the pruning stage.
    file_timings = pd.DataFrame([
        {'stage': '022_exec_prune_polyominoes', 'time': 1.0, 'op': 'group_tiles'},
        {'stage': '022_exec_prune_polyominoes', 'time': 2.0, 'op': 'solve_ilp'},
        {'stage': '022_exec_prune_polyominoes', 'time': 3.0, 'op': 'write_output'},
    ])

    # Apply the stage-local exclusion policy.
    filtered_df = filter_excluded_runtime_ops(file_timings, '022_exec_prune_polyominoes')

    # Keep only the non-excluded operations.
    assert filtered_df.to_dict('records') == [
        {'stage': '022_exec_prune_polyominoes', 'time': 1.0, 'op': 'group_tiles'},
        {'stage': '022_exec_prune_polyominoes', 'time': 3.0, 'op': 'write_output'},
    ]


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
