from pathlib import Path

import pandas as pd

from evaluation.p203_compare_stats import (
    DEFAULT_THRESHOLDS,
    DETAIL_THRESHOLDS,
    add_loss_pct,
    build_threshold_reports,
    filter_pareto_by_dataset,
    save_tex_macros,
    select_accuracy_matched_prior_row,
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


def test_select_accuracy_matched_prior_row_handles_matching_rules():
    # Build a selected Polytris row that anchors the accuracy-matched comparison.
    polytris_row = pd.Series({'HOTA_HOTA': 0.80, 'throughput_fps': 300.0})

    # Reuse the threshold-feasible prior row when it is already no more accurate than Polytris.
    threshold_feasible_prior_dfs = {
        'OTIF': pd.DataFrame([
            {'HOTA_HOTA': 0.79, 'loss_pct': 0.04, 'throughput_fps': 50.0},
        ]),
        'LEAP': pd.DataFrame([
            {'HOTA_HOTA': 0.75, 'loss_pct': 0.03, 'throughput_fps': 60.0},
        ]),
    }
    threshold_feasible_row = select_accuracy_matched_prior_row(
        threshold_feasible_prior_dfs,
        0.05,
        polytris_row,
    )
    assert threshold_feasible_row is not None
    assert threshold_feasible_row['system'] == 'LEAP'
    assert threshold_feasible_row['accuracy_match_rule'] == 'threshold_feasible_not_more_accurate'

    # Fall back to the closest lower-accuracy prior row when threshold-feasible rows are too accurate.
    fallback_prior_dfs = {
        'OTIF': pd.DataFrame([
            {'HOTA_HOTA': 0.90, 'loss_pct': 0.01, 'throughput_fps': 100.0},
            {'HOTA_HOTA': 0.78, 'loss_pct': 0.10, 'throughput_fps': 80.0},
            {'HOTA_HOTA': 0.70, 'loss_pct': 0.20, 'throughput_fps': 200.0},
        ]),
        'LEAP': pd.DataFrame([
            {'HOTA_HOTA': 0.77, 'loss_pct': 0.09, 'throughput_fps': 1000.0},
        ]),
    }
    fallback_row = select_accuracy_matched_prior_row(
        fallback_prior_dfs,
        0.05,
        polytris_row,
    )
    assert fallback_row is not None
    assert fallback_row['system'] == 'OTIF'
    assert fallback_row['HOTA_HOTA'] == 0.78
    assert fallback_row['accuracy_match_rule'] == 'nearest_not_more_accurate'

    # Return no match when every prior row is more accurate than Polytris.
    no_match_prior_dfs = {
        'OTIF': pd.DataFrame([
            {'HOTA_HOTA': 0.90, 'loss_pct': 0.01, 'throughput_fps': 100.0},
        ]),
    }
    assert select_accuracy_matched_prior_row(no_match_prior_dfs, 0.05, polytris_row) is None


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
            {'dataset': 'demo', 'videoset': 'test', 'HOTA_HOTA': 0.97, 'throughput_fps': 90.0},
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
    # Print detail tables for the same thresholds covered by the summary output.
    assert DETAIL_THRESHOLDS == DEFAULT_THRESHOLDS

    # At 1%, Polytris should pick the strict row and LEAP should be the only feasible prior.
    one_pct_detail = detail_tables[0.01]
    assert one_pct_detail.loc[0, 'polytris_variant_id'] == 'poly_strict'
    assert one_pct_detail.loc[0, 'prior_system'] == 'LEAP'
    assert one_pct_detail.loc[0, 'speedup_x'] == 120.0 / 70.0
    assert one_pct_detail.loc[0, 'accuracy_matched_prior_system'] == 'OTIF'
    assert one_pct_detail.loc[0, 'accuracy_matched_prior_rule'] == 'nearest_not_more_accurate'
    assert one_pct_detail.loc[0, 'accuracy_matched_speedup_x'] == 120.0 / 100.0
    # Ignore the faster-looking but Pareto-dominated OTIF row during accuracy matching.
    assert one_pct_detail.loc[0, 'accuracy_matched_prior_hota'] == 0.98
    assert one_pct_detail.loc[0, 'naive_speedup_x'] == 120.0 / 20.0

    # At 5%, Polytris should pick the faster row and OTIF should be the best feasible prior.
    five_pct_detail = detail_tables[0.05]
    assert five_pct_detail.loc[0, 'polytris_variant_id'] == 'poly_fast'
    assert five_pct_detail.loc[0, 'prior_system'] == 'OTIF'
    assert five_pct_detail.loc[0, 'speedup_x'] == 3.0
    assert pd.isna(five_pct_detail.loc[0, 'accuracy_matched_prior_system'])
    assert pd.isna(five_pct_detail.loc[0, 'accuracy_matched_speedup_x'])
    assert five_pct_detail.loc[0, 'naive_speedup_x'] == 15.0

    # Keep the accuracy-matched threshold summary consistent with the detail rows.
    one_pct_summary = summary_df.loc[summary_df['threshold'] == 0.01].iloc[0]
    assert one_pct_summary['accuracy_matched_prior_count'] == 1
    assert one_pct_summary['accuracy_matched_speedup_min_x'] == 120.0 / 100.0
    assert one_pct_summary['accuracy_matched_speedup_max_x'] == 120.0 / 100.0

    # Keep the threshold-level counts and speedup range consistent with the detail rows.
    five_pct_summary = summary_df.loc[summary_df['threshold'] == 0.05].iloc[0]
    assert five_pct_summary['polytris_meet_count'] == 1
    assert five_pct_summary['prior_meet_count'] == 1
    assert five_pct_summary['prior_fail_count'] == 0
    assert five_pct_summary['speedup_min_x'] == 3.0
    assert five_pct_summary['speedup_max_x'] == 3.0
    assert five_pct_summary['accuracy_matched_prior_count'] == 0
    assert pd.isna(five_pct_summary['accuracy_matched_speedup_min_x'])
    assert pd.isna(five_pct_summary['accuracy_matched_speedup_max_x'])
    assert five_pct_summary['naive_speedup_min_x'] == 15.0
    assert five_pct_summary['naive_speedup_max_x'] == 15.0


def test_save_tex_macros_writes_abstract_ready_values(tmp_path: Path):
    # Build a tiny threshold summary table with one row for every reported threshold.
    rows: list[dict[str, float | int]] = []
    for threshold in DEFAULT_THRESHOLDS:
        # Convert each threshold into an integer percent for deterministic test values.
        threshold_percent = int(round(threshold * 100))

        # Add one synthetic summary row for the current threshold.
        rows.append({
            'threshold': threshold,
            'polytris_meet_count': threshold_percent,
            'prior_meet_count': threshold_percent - 1,
            'prior_fail_count': 11 - threshold_percent,
            'speedup_min_x': threshold_percent + 0.31,
            'speedup_max_x': threshold_percent + 0.82,
            'accuracy_matched_prior_count': threshold_percent + 2,
            'accuracy_matched_speedup_min_x': threshold_percent + 3.06,
            'accuracy_matched_speedup_max_x': threshold_percent + 4.07,
            'naive_speedup_min_x': threshold_percent + 1.04,
            'naive_speedup_max_x': threshold_percent + 2.05,
        })

    # Materialize the synthetic summary table.
    summary_df = pd.DataFrame(rows)

    # Build a tiny dominance table so the dominance macros can be emitted.
    dominance_detail_df = pd.DataFrame([
        {'hota_delta': 0.42, 'prior_throughput_fps': 1234.5},
    ])

    # Save the macro file into a temporary output path.
    output_path = tmp_path / 'p203_compare_stats.tex'
    save_tex_macros(summary_df, dominance_detail_df, str(output_path))

    # Read the generated macro file back for exact assertions.
    contents = output_path.read_text()

    # Persist one macro block for every reported threshold.
    for suffix in [
        'OnePct', 'TwoPct', 'ThreePct', 'FourPct', 'FivePct',
        'SixPct', 'SevenPct', 'EightPct', 'NinePct', 'TenPct',
    ]:
        assert f'\\newcommand{{\\comparePolytrisMeet{suffix}}}' in contents
        assert f'\\newcommand{{\\comparePriorMeet{suffix}}}' in contents
        assert f'\\newcommand{{\\comparePriorFailDatasets{suffix}}}' in contents
        assert f'\\newcommand{{\\compareSpeedupMin{suffix}}}' in contents
        assert f'\\newcommand{{\\compareSpeedupMax{suffix}}}' in contents
        assert f'\\newcommand{{\\compareAccuracyMatchedPriorCount{suffix}}}' in contents
        assert f'\\newcommand{{\\compareAccuracyMatchedSpeedupMin{suffix}}}' in contents
        assert f'\\newcommand{{\\compareAccuracyMatchedSpeedupMax{suffix}}}' in contents
        assert f'\\newcommand{{\\compareNaiveSpeedupMin{suffix}}}' in contents
        assert f'\\newcommand{{\\compareNaiveSpeedupMax{suffix}}}' in contents

    # Preserve the full 5% threshold macro block with the existing macro names.
    assert '\\newcommand{\\comparePolytrisMeetFivePct}{\\autogen{5}}' in contents
    assert '\\newcommand{\\comparePriorMeetFivePct}{\\autogen{4}}' in contents
    assert '\\newcommand{\\comparePriorFailDatasetsFivePct}{\\autogen{6}}' in contents
    assert '\\newcommand{\\compareSpeedupMinFivePct}{\\autogen{5.3}}' in contents
    assert '\\newcommand{\\compareSpeedupMaxFivePct}{\\autogen{5.8}}' in contents
    assert '\\newcommand{\\compareAccuracyMatchedPriorCountFivePct}{\\autogen{7}}' in contents
    assert '\\newcommand{\\compareAccuracyMatchedSpeedupMinFivePct}{\\autogen{8.1}}' in contents
    assert '\\newcommand{\\compareAccuracyMatchedSpeedupMaxFivePct}{\\autogen{9.1}}' in contents
    assert '\\newcommand{\\compareNaiveSpeedupMinFivePct}{\\autogen{6.0}}' in contents
    assert '\\newcommand{\\compareNaiveSpeedupMaxFivePct}{\\autogen{7.0}}' in contents

    # Preserve the full 10% threshold macro block with the existing macro names.
    assert '\\newcommand{\\comparePolytrisMeetTenPct}{\\autogen{10}}' in contents
    assert '\\newcommand{\\comparePriorMeetTenPct}{\\autogen{9}}' in contents
    assert '\\newcommand{\\comparePriorFailDatasetsTenPct}{\\autogen{1}}' in contents
    assert '\\newcommand{\\compareSpeedupMinTenPct}{\\autogen{10.3}}' in contents
    assert '\\newcommand{\\compareSpeedupMaxTenPct}{\\autogen{10.8}}' in contents
    assert '\\newcommand{\\compareAccuracyMatchedPriorCountTenPct}{\\autogen{12}}' in contents
    assert '\\newcommand{\\compareAccuracyMatchedSpeedupMinTenPct}{\\autogen{13.1}}' in contents
    assert '\\newcommand{\\compareAccuracyMatchedSpeedupMaxTenPct}{\\autogen{14.1}}' in contents
    assert '\\newcommand{\\compareNaiveSpeedupMinTenPct}{\\autogen{11.0}}' in contents
    assert '\\newcommand{\\compareNaiveSpeedupMaxTenPct}{\\autogen{12.1}}' in contents

    # Continue emitting the dominance macros after the threshold blocks.
    assert '\\newcommand{\\compareMaxHotaImprovement}{\\autogen{0.42}}' in contents
    assert '\\newcommand{\\compareMaxHotaImprovementFps}{\\autogen{1200}}' in contents
