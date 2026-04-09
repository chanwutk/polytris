import pandas as pd
import pytest

from evaluation.p201_compare_pareto import (
    FACET_COLUMNS,
    FACET_SUBPLOT_HEIGHT,
    FACET_SUBPLOT_WIDTH,
    _filter_pareto_per_dataset,
    create_accuracy_gain_chart,
    create_pareto_comparison_chart,
    create_speedup_chart,
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
                        'comparison_system': 'OTIF',
                        'speedup_ratio': 1.2,
                        'polytris_time': 10.0,
                        'other_time': 12.0,
                    },
                    {
                        'dataset': 'demo',
                        'accuracy_level': 0.80,
                        'comparison_system': 'OTIF',
                        'speedup_ratio': 1.1,
                        'polytris_time': 11.0,
                        'other_time': 12.1,
                    },
                ]),
                'HOTA',
            ),
            'Speedup Ratio at HOTA Levels (>1 = Polytris faster)',
        ),
        (
            create_accuracy_gain_chart(
                pd.DataFrame([
                    {
                        'dataset': 'demo',
                        'runtime_level': 10.0,
                        'comparison_system': 'OTIF',
                        'accuracy_gain': 0.05,
                        'polytris_accuracy': 0.80,
                        'other_accuracy': 0.75,
                    },
                    {
                        'dataset': 'demo',
                        'runtime_level': 20.0,
                        'comparison_system': 'OTIF',
                        'accuracy_gain': 0.04,
                        'polytris_accuracy': 0.82,
                        'other_accuracy': 0.78,
                    },
                ]),
                'HOTA',
            ),
            'HOTA Gain at Runtime Levels (>0 = Polytris more accurate)',
        ),
        (
            create_pareto_comparison_chart(
                pd.DataFrame([
                    {
                        'system': 'Polytris',
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
                        'system': 'Polytris',
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
                        'system': 'Polytris',
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
                        'system': 'Polytris',
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
    assert chart_spec['title'] == expected_title


@pytest.mark.parametrize(
    'chart_factory',
    [
        lambda: create_speedup_chart(
            pd.DataFrame(columns=[
                'dataset',
                'accuracy_level',
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
                'runtime_level',
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
            'system': 'Polytris',
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
            'system': 'Polytris',
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


def test_filter_pareto_per_dataset_minimize_x_for_throughput():
    # minimize_x=True: lower throughput is "better" on x-axis.
    # Tradeoff: (50, 0.90) has best x, (100, 0.95) has best y.
    df = pd.DataFrame([
        {'dataset': 'd1', 'throughput_fps': 50.0, 'acc': 0.90},
        {'dataset': 'd1', 'throughput_fps': 100.0, 'acc': 0.95},
        {'dataset': 'd1', 'throughput_fps': 60.0, 'acc': 0.70},  # dominated by (50, 0.90)
    ])
    result = _filter_pareto_per_dataset(df, 'throughput_fps', 'acc', minimize_x=True)
    # Point (60, 0.70) is dominated: (50, 0.90) has lower throughput AND higher accuracy.
    assert len(result) == 2
    assert set(result['throughput_fps']) == {50.0, 100.0}


def test_filter_pareto_per_dataset_default_minimizes_time():
    # Default (minimize_x=False) means higher time is "better" on x-axis.
    # Tradeoff: (10, 0.90) has best y, (20, 0.80) has best x.
    df = pd.DataFrame([
        {'dataset': 'd1', 'time': 10.0, 'acc': 0.90},
        {'dataset': 'd1', 'time': 20.0, 'acc': 0.80},
        {'dataset': 'd1', 'time': 15.0, 'acc': 0.70},  # dominated by (20, 0.80)
    ])
    result = _filter_pareto_per_dataset(df, 'time', 'acc')
    # Point (15, 0.70) is dominated: (20, 0.80) has higher time AND higher accuracy.
    assert len(result) == 2
    assert set(result['time']) == {10.0, 20.0}
