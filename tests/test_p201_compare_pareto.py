import pandas as pd
import pytest

from evaluation.p201_compare_pareto import (
    FACET_COLUMNS,
    FACET_SUBPLOT_HEIGHT,
    FACET_SUBPLOT_WIDTH,
    _filter_pareto_per_dataset,
    compute_accuracy_gain_at_naive_speedup_levels,
    compute_speedup_at_accuracy_levels,
    create_accuracy_gain_chart,
    create_hota_presentation_summary_chart,
    create_pareto_comparison_chart,
    create_speedup_chart,
    PRESENTATION_DROP_BAND_COLOR,
    PRESENTATION_MUTED_COLOR,
)


@pytest.mark.parametrize(
    ('chart', 'expected_title'),
    [
        (
            create_speedup_chart(
                pd.DataFrame([
                    {
                        'dataset': 'demo',
                        'accuracy_level': 0.70,
                        'system': 'OTIF',
                        'comparison_system': 'OTIF',
                        'speedup_ratio': 1.2,
                        'polytris_time': 10.0,
                        'other_time': 12.0,
                    },
                    {
                        'dataset': 'demo',
                        'accuracy_level': 0.80,
                        'system': 'OTIF',
                        'comparison_system': 'OTIF',
                        'speedup_ratio': 1.1,
                        'polytris_time': 11.0,
                        'other_time': 12.1,
                    },
                ]),
                'HOTA',
            ),
            'Speedup Ratio at HOTA Scores (>1 = Tetris faster)',
        ),
        (
            create_accuracy_gain_chart(
                pd.DataFrame([
                    {
                        'dataset': 'demo',
                        'throughput_fps': 2.0,
                        'naive_time': 100.0,
                        'system': 'OTIF',
                        'comparison_system': 'OTIF',
                        'accuracy_gain': 0.05,
                        'polytris_accuracy': 0.80,
                        'other_accuracy': 0.75,
                    },
                    {
                        'dataset': 'demo',
                        'throughput_fps': 4.0,
                        'naive_time': 100.0,
                        'system': 'OTIF',
                        'comparison_system': 'OTIF',
                        'accuracy_gain': 0.04,
                        'polytris_accuracy': 0.82,
                        'other_accuracy': 0.78,
                    },
                ]),
                'HOTA',
            ),
            'HOTA Gain at Throughput (>0 = Tetris more accurate)',
        ),
        (
            create_pareto_comparison_chart(
                pd.DataFrame([
                    {
                        'system': 'Tetris',
                        'dataset': 'demo',
                        'classifier': 'ShuffleNet05',
                        'sample_rate': 1,
                        'tracking_accuracy_threshold': pd.NA,
                        'tilepadding': 'none',
                        'canvas_scale': 1.0,
                        'tracker': 'bytetrackcython',
                        'time': 10.0,
                        'HOTA_HOTA': 0.80,
                    },
                    {
                        'system': 'Tetris',
                        'dataset': 'demo',
                        'classifier': 'ShuffleNet05',
                        'sample_rate': 1,
                        'tracking_accuracy_threshold': pd.NA,
                        'tilepadding': 'none',
                        'canvas_scale': 1.0,
                        'tracker': 'bytetrackcython',
                        'time': 20.0,
                        'HOTA_HOTA': 0.84,
                    },
                ]),
                'HOTA_HOTA',
                'HOTA',
            ),
            'HOTA vs Runtime Pareto Fronts',
        ),
        (
            create_pareto_comparison_chart(
                pd.DataFrame([
                    {
                        'system': 'Tetris',
                        'dataset': 'demo',
                        'classifier': 'ShuffleNet05',
                        'sample_rate': 1,
                        'tracking_accuracy_threshold': pd.NA,
                        'tilepadding': 'none',
                        'canvas_scale': 1.0,
                        'tracker': 'bytetrackcython',
                        'throughput_fps': 100.0,
                        'HOTA_HOTA': 0.80,
                    },
                    {
                        'system': 'Tetris',
                        'dataset': 'demo',
                        'classifier': 'ShuffleNet05',
                        'sample_rate': 1,
                        'tracking_accuracy_threshold': pd.NA,
                        'tilepadding': 'none',
                        'canvas_scale': 1.0,
                        'tracker': 'bytetrackcython',
                        'throughput_fps': 50.0,
                        'HOTA_HOTA': 0.84,
                    },
                ]),
                'HOTA_HOTA',
                'HOTA',
                time_col='throughput_fps',
                x_title='Throughput (frames/sec)',
            ),
            'HOTA vs Runtime Pareto Fronts',
        ),
    ],
)
def test_facet_charts_use_shared_four_column_layout(chart, expected_title):
    # Inspect the serialized Altair spec so layout regressions are caught directly.
    chart_spec = chart.to_dict()

    assert chart_spec['columns'] == FACET_COLUMNS
    assert chart_spec['spec']['width'] == FACET_SUBPLOT_WIDTH
    assert chart_spec['spec']['height'] == FACET_SUBPLOT_HEIGHT
    assert chart_spec['title']['text'] == expected_title


@pytest.mark.parametrize(
    'chart_factory',
    [
        lambda: create_speedup_chart(
            pd.DataFrame(columns=[
                'dataset',
                'accuracy_level',
                'system',
                'comparison_system',
                'speedup_ratio',
                'polytris_time',
                'other_time',
            ]),
            'HOTA',
        ),
        lambda: create_accuracy_gain_chart(
            pd.DataFrame(columns=[
                'dataset',
                'throughput_fps',
                'naive_time',
                'system',
                'comparison_system',
                'accuracy_gain',
                'polytris_accuracy',
                'other_accuracy',
            ]),
            'HOTA',
        ),
        lambda: create_pareto_comparison_chart(
            pd.DataFrame(columns=[
                'system',
                'dataset',
                'classifier',
                'sample_rate',
                'tracking_accuracy_threshold',
                'tilepadding',
                'canvas_scale',
                'tracker',
                'time',
                'HOTA_HOTA',
            ]),
            'HOTA_HOTA',
            'HOTA',
        ),
    ],
)
def test_chart_builders_return_no_data_chart_for_empty_inputs(chart_factory):
    # Empty inputs should still serialize to a valid placeholder chart.
    chart_spec = chart_factory().to_dict()

    assert chart_spec['mark']['type'] == 'text'
    assert chart_spec['encoding']['text']['value'] == 'No data available'


def test_throughput_chart_uses_custom_x_title():
    # Verify that passing x_title propagates to the x-axis encoding.
    df = pd.DataFrame([
        {
            'system': 'Tetris',
            'dataset': 'demo',
            'classifier': 'ShuffleNet05',
            'sample_rate': 1,
            'tracking_accuracy_threshold': pd.NA,
            'tilepadding': 'none',
            'canvas_scale': 1.0,
            'tracker': 'bytetrackcython',
            'throughput_fps': 100.0,
            'HOTA_HOTA': 0.80,
        },
        {
            'system': 'Tetris',
            'dataset': 'demo',
            'classifier': 'ShuffleNet05',
            'sample_rate': 1,
            'tracking_accuracy_threshold': pd.NA,
            'tilepadding': 'none',
            'canvas_scale': 1.0,
            'tracker': 'bytetrackcython',
            'throughput_fps': 50.0,
            'HOTA_HOTA': 0.84,
        },
    ])
    chart = create_pareto_comparison_chart(
        df, 'HOTA_HOTA', 'HOTA',
        time_col='throughput_fps',
        x_title='Throughput (frames/sec)',
    )
    spec = chart.to_dict()
    # The x-axis title lives inside the spec's layer encoding.
    layer_x = spec['spec']['layer'][0]['encoding']['x']
    assert layer_x['title'] == 'Throughput (frames/sec)'
    assert layer_x['field'] == 'throughput_fps'


def test_accuracy_gain_chart_uses_throughput_x_title():
    # Verify that the throughput x-axis is wired through the chart encoding.
    df = pd.DataFrame([
        {
            'dataset': 'demo',
            'throughput_fps': 2.0,
            'naive_time': 100.0,
            'system': 'OTIF',
            'comparison_system': 'OTIF',
            'accuracy_gain': 0.05,
            'polytris_accuracy': 0.80,
            'other_accuracy': 0.75,
        },
        {
            'dataset': 'demo',
            'throughput_fps': 4.0,
            'naive_time': 100.0,
            'system': 'OTIF',
            'comparison_system': 'OTIF',
            'accuracy_gain': 0.03,
            'polytris_accuracy': 0.76,
            'other_accuracy': 0.73,
        },
    ])
    chart = create_accuracy_gain_chart(df, 'HOTA')
    spec = chart.to_dict()

    layer_x = spec['spec']['layer'][0]['encoding']['x']
    assert layer_x['title'] == 'Throughput (FPS)'
    assert layer_x['field'] == 'throughput_fps'


def test_compute_accuracy_gain_at_naive_speedup_levels_discrete_anchors():
    # One result row per SOTA Pareto point; Polytris chosen by strict faster runtime
    # then max accuracy gain (tie-break: higher acc, then lower time).
    naive_df = pd.DataFrame([
        {'dataset': 'demo', 'time': 100.0, 'HOTA_HOTA': 0.60, 'frame_count': 100.0},
    ])
    polytris_df = pd.DataFrame([
        {'dataset': 'demo', 'time': 50.0, 'HOTA_HOTA': 0.80, 'frame_count': 100.0},
        {'dataset': 'demo', 'time': 25.0, 'HOTA_HOTA': 0.70, 'frame_count': 100.0},
    ])
    sota_df = pd.DataFrame([
        {'dataset': 'demo', 'time': 100.0, 'HOTA_HOTA': 0.75},
        {'dataset': 'demo', 'time': 50.0, 'HOTA_HOTA': 0.65},
    ])

    result = compute_accuracy_gain_at_naive_speedup_levels(
        polytris_df,
        {'otif': sota_df},
        naive_df,
        'HOTA_HOTA',
    )

    non_null = result.dropna(subset=['accuracy_gain']).sort_values(
        'throughput_fps',
    ).reset_index(drop=True)
    assert len(non_null) == 2
    assert non_null['system'].tolist() == ['OTIF', 'OTIF']
    # Anchor (100, 0.75): 100 frames / 100 s = 1 FPS; Polytris faster than 100 -> (50, 0.80); gain 0.05
    assert non_null.loc[0, 'throughput_fps'] == pytest.approx(1.0)
    assert non_null.loc[0, 'accuracy_gain'] == pytest.approx(0.05)
    # Anchor (50, 0.65): 100 frames / 50 s = 2 FPS; Polytris faster than 50 -> (25, 0.70); gain 0.05
    assert non_null.loc[1, 'throughput_fps'] == pytest.approx(2.0)
    assert non_null.loc[1, 'accuracy_gain'] == pytest.approx(0.05)
    assert (non_null['naive_time'] == 100.0).all()


def test_compute_speedup_at_accuracy_levels_discrete_anchors():
    polytris_df = pd.DataFrame([
        {'dataset': 'demo', 'time': 5.0, 'HOTA_HOTA': 0.82},
        {'dataset': 'demo', 'time': 10.0, 'HOTA_HOTA': 0.90},
        {'dataset': 'demo', 'time': 20.0, 'HOTA_HOTA': 0.85},
    ])
    sota_df = pd.DataFrame([
        {'dataset': 'demo', 'time': 15.0, 'HOTA_HOTA': 0.80},
        {'dataset': 'demo', 'time': 30.0, 'HOTA_HOTA': 0.70},
        {'dataset': 'demo', 'time': 40.0, 'HOTA_HOTA': 0.95},
    ])

    result = compute_speedup_at_accuracy_levels(
        polytris_df,
        {'otif': sota_df},
        'HOTA_HOTA',
        'time',
    )

    by_acc = result.set_index('accuracy_level').sort_index()
    # acc 0.80: feasible Polytris acc > 0.8 includes (5,0.82); min time 5 -> 15/5
    assert by_acc.loc[0.80, 'speedup_ratio'] == pytest.approx(3.0)
    assert by_acc.loc[0.80, 'polytris_time'] == pytest.approx(5.0)
    # acc 0.70: all three Polytris rows qualify; min time 5 -> 30/5
    assert by_acc.loc[0.70, 'speedup_ratio'] == pytest.approx(6.0)
    assert by_acc.loc[0.70, 'polytris_time'] == pytest.approx(5.0)
    # acc 0.95: no Polytris strictly above 0.95
    assert pd.isna(by_acc.loc[0.95, 'speedup_ratio'])


def test_filter_pareto_per_dataset_maximize_throughput():
    # Maximize both throughput and accuracy (minx=False, miny=False).
    # Tradeoff: higher throughput comes with lower accuracy.
    df = pd.DataFrame([
        {'dataset': 'd1', 'throughput_fps': 50.0, 'acc': 0.90},
        {'dataset': 'd1', 'throughput_fps': 100.0, 'acc': 0.80},
        {'dataset': 'd1', 'throughput_fps': 60.0, 'acc': 0.70},  # dominated by (100, 0.80)
    ])
    result = _filter_pareto_per_dataset(df, 'throughput_fps', 'acc', minx=False, miny=False)
    # Point (60, 0.70) is dominated: (100, 0.80) has higher throughput AND higher accuracy.
    assert len(result) == 2
    assert set(result['throughput_fps']) == {50.0, 100.0}


def test_filter_pareto_per_dataset_minimize_time():
    # Minimize time, maximize accuracy (minx=True, miny=False).
    # Tradeoff: less time comes with lower accuracy.
    df = pd.DataFrame([
        {'dataset': 'd1', 'time': 10.0, 'acc': 0.70},
        {'dataset': 'd1', 'time': 20.0, 'acc': 0.90},
        {'dataset': 'd1', 'time': 15.0, 'acc': 0.60},  # dominated by (10, 0.70)
    ])
    result = _filter_pareto_per_dataset(df, 'time', 'acc', minx=True, miny=False)
    # Point (15, 0.60) is dominated: (10, 0.70) has lower time AND higher accuracy.
    assert len(result) == 2
    assert set(result['time']) == {10.0, 20.0}


def test_presentation_summary_is_single_pareto_chart_with_focus_and_drop_band():
    df_throughput = pd.DataFrame([
        {'system': 'Tetris', 'dataset': 'demo', 'classifier': 'ShuffleNet05',
         'sample_rate': 1, 'tracking_accuracy_threshold': pd.NA,
         'relevance_threshold': 0.5, 'tilepadding': 'none',
         'canvas_scale': 1.0, 'tracker': 'bytetrackcython',
         'throughput_fps': 100.0, 'time': 1.0, 'HOTA_HOTA': 0.80},
        {'system': 'Tetris (-Sampling)', 'dataset': 'demo', 'classifier': 'ShuffleNet05',
         'sample_rate': 1, 'tracking_accuracy_threshold': pd.NA,
         'relevance_threshold': 0.5, 'tilepadding': 'none',
         'canvas_scale': 1.0, 'tracker': 'bytetrackcython',
         'throughput_fps': 80.0, 'time': 1.2, 'HOTA_HOTA': 0.78},
        {'system': 'Reference', 'dataset': 'demo', 'classifier': pd.NA,
         'sample_rate': pd.NA, 'tracking_accuracy_threshold': pd.NA,
         'relevance_threshold': pd.NA, 'tilepadding': pd.NA,
         'canvas_scale': pd.NA, 'tracker': 'bytetrackcython',
         'throughput_fps': 20.0, 'time': 5.0, 'HOTA_HOTA': 0.82},
        {'system': 'OTIF', 'dataset': 'demo', 'classifier': pd.NA,
         'sample_rate': pd.NA, 'tracking_accuracy_threshold': pd.NA,
         'relevance_threshold': pd.NA, 'tilepadding': pd.NA,
         'canvas_scale': pd.NA, 'tracker': pd.NA,
         'throughput_fps': 60.0, 'time': 1.7, 'HOTA_HOTA': 0.70},
        {'system': 'LEAP', 'dataset': 'demo', 'classifier': pd.NA,
         'sample_rate': pd.NA, 'tracking_accuracy_threshold': pd.NA,
         'relevance_threshold': pd.NA, 'tilepadding': pd.NA,
         'canvas_scale': pd.NA, 'tracker': pd.NA,
         'throughput_fps': 40.0, 'time': 2.5, 'HOTA_HOTA': 0.68},
    ])
    # The slide chart is a single faceted HOTA vs Throughput Pareto chart titled below the plots.
    chart = create_hota_presentation_summary_chart(df_throughput, focus_systems={'Tetris'})
    spec = chart.to_dict()
    assert spec['title']['text'] == 'HOTA vs Throughput'
    assert spec['title']['orient'] == 'bottom'
    assert 'vconcat' not in spec and 'facet' in spec

    # Ablation curves are dropped; the focus system keeps its color and the rest are muted.
    layers = spec['spec']['layer']
    assert [layer['mark']['type'] for layer in layers] == ['line', 'point']
    color_scale = layers[-1]['encoding']['color']['scale']
    assert color_scale['domain'] == ['Tetris', 'Reference', 'OTIF', 'LEAP']
    assert color_scale['range'][0] != PRESENTATION_MUTED_COLOR
    assert color_scale['range'][1:] == [PRESENTATION_MUTED_COLOR] * 3

    # The reference-drop variant adds a shaded band and a rule at the dropped HOTA, behind the curves.
    band_chart = create_hota_presentation_summary_chart(df_throughput, show_reference_drop_band=True)
    band_layers = band_chart.to_dict()['spec']['layer']
    assert [layer['mark']['type'] for layer in band_layers] == ['rect', 'rule', 'line', 'point']
    assert band_layers[0]['mark']['color'] == PRESENTATION_DROP_BAND_COLOR
    assert band_layers[0]['mark']['opacity'] == 0.3
    assert band_layers[1]['mark']['color'] == PRESENTATION_DROP_BAND_COLOR
    assert band_layers[1]['encoding']['y']['field'] == 'drop_hota'
    # Without a focus set every system keeps its own color.
    assert PRESENTATION_MUTED_COLOR not in band_layers[-1]['encoding']['color']['scale']['range']
