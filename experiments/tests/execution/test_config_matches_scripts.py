"""The execution config's constants that must stay equal to the original p022 script's."""

from execution.config import MISTRACK_TOLERANCES


def test_mistrack_tolerances_match_p022():
    # The config's list must stay the max_rate_table axis p022 indexes into.
    from experiments.scripts.p022_exec_prune_polyominoes import _ALL_ACCURACY_THRESHOLDS
    assert list(MISTRACK_TOLERANCES) == _ALL_ACCURACY_THRESHOLDS
