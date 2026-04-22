from polyis.utilities import (
    build_param_str,
    dataset_name_for_videoset,
    parse_execution_param_str,
    split_dataset_name,
)


def test_parse_execution_param_str_without_threshold():
    param_str = 'ShuffleNet05_60_2_plus_s100_bytetrackcython'
    parsed = parse_execution_param_str(param_str)

    assert parsed['classifier'] == 'ShuffleNet05'
    assert parsed['tilesize'] == 60
    assert parsed['sample_rate'] == 2
    assert parsed['tracking_accuracy_threshold'] is None
    assert parsed['relevance_threshold'] is None
    assert parsed['tilepadding'] == 'plus'
    assert parsed['canvas_scale'] == 1.0
    assert parsed['tracker'] == 'bytetrackcython'


def test_parse_execution_param_str_with_threshold_and_tracker():
    param_str = 'ShuffleNet05_60_4_070_square_s100_ocsortcython'
    parsed = parse_execution_param_str(param_str)

    assert parsed['classifier'] == 'ShuffleNet05'
    assert parsed['tilesize'] == 60
    assert parsed['sample_rate'] == 4
    assert parsed['tracking_accuracy_threshold'] == 0.7
    assert parsed['relevance_threshold'] is None
    assert parsed['tilepadding'] == 'square'
    assert parsed['canvas_scale'] == 1.0
    assert parsed['tracker'] == 'ocsortcython'


def test_parse_execution_param_str_for_pruning_stage_format():
    param_str = 'ShuffleNet05_60_2_070_ocsortcython'
    parsed = parse_execution_param_str(param_str)

    assert parsed['classifier'] == 'ShuffleNet05'
    assert parsed['tilesize'] == 60
    assert parsed['sample_rate'] == 2
    assert parsed['tracking_accuracy_threshold'] == 0.7
    assert parsed['relevance_threshold'] is None
    assert parsed['tilepadding'] is None
    assert parsed['canvas_scale'] is None
    assert parsed['tracker'] == 'ocsortcython'


def test_split_dataset_name_for_validation_alias():
    assert dataset_name_for_videoset('caldot1-y05', 'valid') == 'caldot1-y05-val'
    assert dataset_name_for_videoset('caldot1-y05', 'test') == 'caldot1-y05'

    assert split_dataset_name('caldot1-y05-val') == ('caldot1-y05', 'valid')
    assert split_dataset_name('caldot1-y05') == ('caldot1-y05', 'test')


def test_parse_execution_param_str_with_relevance_only():
    param_str = 'ShuffleNet05_60_4_r050_plus_s100_bytetrackcython'
    parsed = parse_execution_param_str(param_str)
    assert parsed['sample_rate'] == 4
    assert parsed['tracking_accuracy_threshold'] is None
    assert parsed['relevance_threshold'] == 0.5
    assert parsed['tilepadding'] == 'plus'
    assert parsed['canvas_scale'] == 1.0
    assert parsed['tracker'] == 'bytetrackcython'


def test_parse_execution_param_str_with_tracking_and_relevance():
    param_str = 'ShuffleNet05_60_4_070_r025_square_s100_sortcython'
    parsed = parse_execution_param_str(param_str)
    assert parsed['tracking_accuracy_threshold'] == 0.7
    assert parsed['relevance_threshold'] == 0.25
    assert parsed['tilepadding'] == 'square'


def test_build_parse_roundtrip_full_grid_tokens():
    built = build_param_str(
        classifier='ShuffleNet05',
        tilesize=60,
        sample_rate=4,
        tracking_accuracy_threshold=0.7,
        relevance_threshold=0.5,
        tilepadding='square',
        canvas_scale=1.0,
        tracker='sortcython',
    )
    parsed = parse_execution_param_str(built)
    assert parsed['classifier'] == 'ShuffleNet05'
    assert parsed['tilesize'] == 60
    assert parsed['sample_rate'] == 4
    assert parsed['tracking_accuracy_threshold'] == 0.7
    assert parsed['relevance_threshold'] == 0.5
    assert parsed['tilepadding'] == 'square'
    assert parsed['canvas_scale'] == 1.0
    assert parsed['tracker'] == 'sortcython'
    assert parsed['param_str'] == built
