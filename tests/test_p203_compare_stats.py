from pathlib import Path

import pandas as pd

from evaluation.p203_compare_stats import (
    DEFAULT_THRESHOLDS,
    add_loss_pct,
    build_threshold_reports,
    filter_pareto_by_dataset,
    save_tex_macros,
    select_best_prior_row,
)


def test_add_loss_pct_clamps_negative_loss_to_zero():
    # Build a tiny system table with one row above the oracle and one row below it.
    df = pd.DataFrame([
        {'dataset': 'demo', 'variant_id': 'cfg_hi', 'HOTA_HOTA': 0.82, 'throughput_fps': 50.0},
        {'dataset': 'demo', 'variant_id': 'cfg_lo', 'HOTA_HOTA': 0.76, 'throughput_fps': 75.0},
    ])

    # Compute the clamped relative loss percentages.
    result = add_loss_pct(df, oracle_hota=0.80)

    # Clamp the above-oracle row to zero loss.
    assert result.loc[result['variant_id'] == 'cfg_hi', 'loss_pct'].iloc[0] == 0.0
    # Keep the below-oracle row as an ordinary relative loss.
    assert result.loc[result['variant_id'] == 'cfg_lo', 'loss_pct'].iloc[0] == 0.05


def test_filter_pareto_by_dataset_drops_dominated_rows():
    # Build a throughput-vs-accuracy tradeoff table with one dominated row.
    df = pd.DataFrame([
        {'dataset': 'demo', 'variant_id': 'cfg_a', 'throughput_fps': 100.0, 'HOTA_HOTA': 0.90},
        {'dataset': 'demo', 'variant_id': 'cfg_b', 'throughput_fps': 150.0, 'HOTA_HOTA': 0.85},
        {'dataset': 'demo', 'variant_id': 'cfg_c', 'throughput_fps': 120.0, 'HOTA_HOTA': 0.80},
    ])

    # Pareto-filter the dataset-local rows.
    result = filter_pareto_by_dataset(df)

    # Keep only the two non-dominated tradeoff points.
    assert set(result['variant_id']) == {'cfg_a', 'cfg_b'}


def test_select_best_prior_row_handles_presence_cases():
    # Build four prior-system availability cases to cover OTIF/LEAP selection behavior.
    prior_cases = {
        'otif_only': (
            {'OTIF': pd.DataFrame([{'loss_pct': 0.03, 'throughput_fps': 120.0}])},
            0.05,
            'OTIF',
        ),
        'leap_only': (
            {'LEAP': pd.DataFrame([{'loss_pct': 0.02, 'throughput_fps': 90.0}])},
            0.05,
            'LEAP',
        ),
        'both_present': (
            {
                'OTIF': pd.DataFrame([{'loss_pct': 0.03, 'throughput_fps': 120.0}]),
                'LEAP': pd.DataFrame([{'loss_pct': 0.01, 'throughput_fps': 80.0}]),
            },
            0.05,
            'OTIF',
        ),
        'neither_feasible': (
            {
                'OTIF': pd.DataFrame([{'loss_pct': 0.08, 'throughput_fps': 120.0}]),
                'LEAP': pd.DataFrame([{'loss_pct': 0.09, 'throughput_fps': 80.0}]),
            },
            0.05,
            None,
        ),
    }

    # Validate the selection outcome for each presence case.
    for prior_dfs, threshold, expected_system in prior_cases.values():
        selected_row = select_best_prior_row(prior_dfs, threshold)

        # Expect no row when neither prior system is feasible.
        if expected_system is None:
            assert selected_row is None
            continue

        # Expect the fastest feasible prior system otherwise.
        assert selected_row is not None
        assert selected_row['system'] == expected_system


def test_build_threshold_reports_uses_fixed_thresholds_and_aggregates_counts():
    # Build one Pareto-filtered Polytris table for a single synthetic dataset.
    polytris_df = pd.DataFrame([
        {
            'dataset': 'demo',
            'videoset': 'test',
            'variant_id': 'poly_strict',
            'HOTA_HOTA': 0.99,
            'throughput_fps': 120.0,
        },
        {
            'dataset': 'demo',
            'videoset': 'test',
            'variant_id': 'poly_fast',
            'HOTA_HOTA': 0.95,
            'throughput_fps': 300.0,
        },
    ])

    # Build the dedicated naive oracle row.
    naive_df = pd.DataFrame([
        {
            'dataset': 'demo',
            'videoset': 'test',
            'variant_id': 'naive',
            'HOTA_HOTA': 1.00,
            'throughput_fps': 20.0,
        },
    ])

    # Build two prior-system Pareto tables that trade accuracy for speed differently.
    prior_dfs = {
        'OTIF': pd.DataFrame([
            {'dataset': 'demo', 'videoset': 'test', 'HOTA_HOTA': 0.98, 'throughput_fps': 100.0},
        ]),
        'LEAP': pd.DataFrame([
            {'dataset': 'demo', 'videoset': 'test', 'HOTA_HOTA': 0.995, 'throughput_fps': 70.0},
        ]),
    }

    # Build the default threshold reports from the synthetic tradeoff data.
    summary_df, detail_tables = build_threshold_reports(
        ['demo'],
        polytris_df,
        naive_df,
        prior_dfs,
    )

    # Emit one summary row per default threshold from 1% through 10%.
    assert summary_df['threshold'].tolist() == DEFAULT_THRESHOLDS

    # At 1%, Polytris should pick the strict row and LEAP should be the only feasible prior.
    one_pct_detail = detail_tables[0.01]
    assert one_pct_detail.loc[0, 'polytris_variant_id'] == 'poly_strict'
    assert one_pct_detail.loc[0, 'prior_system'] == 'LEAP'
    assert one_pct_detail.loc[0, 'speedup_x'] == 120.0 / 70.0

    # At 5%, Polytris should pick the faster row and OTIF should be the best feasible prior.
    five_pct_detail = detail_tables[0.05]
    assert five_pct_detail.loc[0, 'polytris_variant_id'] == 'poly_fast'
    assert five_pct_detail.loc[0, 'prior_system'] == 'OTIF'
    assert five_pct_detail.loc[0, 'speedup_x'] == 3.0

    # Keep the threshold-level counts and speedup range consistent with the detail rows.
    five_pct_summary = summary_df.loc[summary_df['threshold'] == 0.05].iloc[0]
    assert five_pct_summary['polytris_meet_count'] == 1
    assert five_pct_summary['prior_meet_count'] == 1
    assert five_pct_summary['prior_fail_count'] == 0
    assert five_pct_summary['speedup_min_x'] == 3.0
    assert five_pct_summary['speedup_max_x'] == 3.0


def test_save_tex_macros_writes_abstract_ready_values(tmp_path: Path):
    # Build a tiny threshold summary table with the required 5% and 10% rows.
    summary_df = pd.DataFrame([
        {
            'threshold': 0.05,
            'polytris_meet_count': 7,
            'prior_meet_count': 4,
            'prior_fail_count': 3,
            'speedup_min_x': 1.3,
            'speedup_max_x': 55.8,
        },
        {
            'threshold': 0.10,
            'polytris_meet_count': 7,
            'prior_meet_count': 7,
            'prior_fail_count': 0,
            'speedup_min_x': 1.3248,
            'speedup_max_x': 15.8395,
        },
    ])

    # Save the macro file into a temporary output path.
    output_path = tmp_path / 'p203_compare_stats.tex'
    save_tex_macros(summary_df, str(output_path))

    # Read the generated macro file back for exact assertions.
    contents = output_path.read_text()

    # Persist the 5% prior-failure count and the rounded 10% speedup bounds.
    assert '\\newcommand{\\comparePriorFailDatasetsFivePct}{\\autogen{3}}' in contents
    assert '\\newcommand{\\compareSpeedupMinTenPct}{\\autogen{1.3}}' in contents
    assert '\\newcommand{\\compareSpeedupMaxTenPct}{\\autogen{15.8}}' in contents
