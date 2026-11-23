"""Branch: ordered, immutable parameters that build the scripts' directory names (profiling/PLAN.md 10A)."""

import pickle

import pytest

from profiling.messages import Branch


def base():
    return Branch().with_(classifier='ShuffleNet05', tile_size=60)


def test_branch_builds_the_scripts_directory_names():
    b = base().with_(sample_rate=4).with_(relevance_threshold=0.5, mistrack_tolerance=0.8, tracker='sortcython')
    assert b.param_str('sample_rate') == 'ShuffleNet05_60_4'
    assert b.param_str('sample_rate', 'tracker', 'mistrack_tolerance', 'relevance_threshold') == \
        'ShuffleNet05_60_4_080_r050_sortcython'
    full = b.with_(tile_padding='bl', canvas_scale=1.0)
    assert full.param_str('sample_rate', 'tracker', 'mistrack_tolerance', 'relevance_threshold', 'tile_padding',
                          'canvas_scale') == 'ShuffleNet05_60_4_080_r050_bl_s100_sortcython'


def test_unpruned_tolerance_is_left_out_like_the_scripts():
    b = base().with_(sample_rate=8, relevance_threshold=0.25, mistrack_tolerance=None, tile_padding='none',
                     canvas_scale=1.0, tracker='bytetrackcython')
    assert b.param_str('sample_rate', 'mistrack_tolerance', 'relevance_threshold', 'tile_padding', 'canvas_scale',
                       'tracker') == 'ShuffleNet05_60_8_r025_none_s100_bytetrackcython'


def test_branch_is_immutable_ordered_and_picklable():
    b = base().with_(sample_rate=2)
    assert b.with_(relevance_threshold=0.5) != b and b.as_dict() == {'classifier': 'ShuffleNet05', 'tile_size': 60,
                                                                     'sample_rate': 2}
    assert pickle.loads(pickle.dumps(b)) == b and hash(b) == hash(pickle.loads(pickle.dumps(b)))


def test_branch_rejects_unknown_repeated_and_missing_parameters():
    with pytest.raises(KeyError, match='unknown'):
        base().with_(speed=1)
    with pytest.raises(KeyError, match='already set'):
        base().with_(sample_rate=1).with_(sample_rate=2)
    with pytest.raises(KeyError, match='has no'):
        base().param_str('tracker')
