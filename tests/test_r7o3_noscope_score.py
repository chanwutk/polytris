import pytest

from measurements.r7o3_noscope_score import recost


def run(**record):
    # A replay with one record charged 30 ms per detector call over 900 frames.
    base = dict(detector_frames=100, difference_seconds=0.5, cnn_seconds=1.5, detector_seconds=3.0, fps=900 / 5.0)
    return dict(frames=900, detector_ms_per_frame=30.0, records=[dict(base, **record)])


def test_recost_charges_detector_frames_at_the_new_rate():
    result = recost(run(), 20.0)
    record = result['records'][0]
    assert result['detector_ms_per_frame'] == 20.0
    assert record['detector_seconds'] == pytest.approx(2.0)
    assert record['fps'] == pytest.approx(900 / (0.5 + 1.5 + 2.0))


def test_recost_keeps_the_replay_charge():
    result = recost(run(), 20.0)
    record = result['records'][0]
    assert result['detector_ms_per_frame_unbatched'] == 30.0
    assert record['detector_seconds_unbatched'] == 3.0
    assert record['fps_unbatched'] == pytest.approx(180.0)


def test_recost_with_the_replay_rate_reproduces_the_replay():
    record = recost(run(), 30.0)['records'][0]
    assert record['fps'] == pytest.approx(record['fps_unbatched'])


def test_recost_without_detector_calls_charges_only_the_cascade():
    record = recost(run(detector_frames=0), 20.0)['records'][0]
    assert record['detector_seconds'] == 0
    assert record['fps'] == pytest.approx(900 / 2.0)
