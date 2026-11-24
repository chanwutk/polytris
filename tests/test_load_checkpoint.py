"""tetris.classify.load_checkpoint: whole-model checkpoints pickled under the old package name still load."""

import io
import zipfile

import torch

from tetris.classify import load_checkpoint
from tetris.models.classifier.shufflenet import ShuffleNet05

# A classifier trained before the polyis -> tetris rename (its pickle names polyis.models.classifier.*).
OLD = 'tests/data/equivalence/caldot1-y05/indexing/training/results/ShuffleNet05_60/model_best.pth'


def test_old_checkpoint_names_the_old_package():
    # Guards the test itself: the committed checkpoint must still carry the old class paths.
    with zipfile.ZipFile(OLD) as z:
        data = z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))
    assert b'polyis.models.classifier' in data and b'tetris.models' not in data


def test_old_checkpoint_loads_as_tetris_classes():
    # The old checkpoint loads, and its classes are the renamed package's (not a second copy of them).
    model = load_checkpoint(OLD, map_location='cpu')
    assert type(model).__module__.startswith('tetris.models.classifier')
    assert any(isinstance(m, ShuffleNet05) for m in model.modules())


def test_old_checkpoint_matches_after_resave():
    # A checkpoint saved after the rename loads to the same weights as the old one.
    old = load_checkpoint(OLD, map_location='cpu')
    buf = io.BytesIO()
    torch.save(old, buf)
    assert b'polyis' not in buf.getvalue()
    buf.seek(0)
    new = load_checkpoint(buf, map_location='cpu')
    a, b = old.state_dict(), new.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_plain_torch_load_fails_on_the_old_checkpoint():
    # Documents why the helper exists: torch.load alone cannot resolve polyis.*.
    try:
        torch.load(OLD, map_location='cpu', weights_only=False)
    except ModuleNotFoundError as e:
        assert 'polyis' in str(e)
    else:
        raise AssertionError('torch.load resolved polyis.*; the old package must still be importable somewhere')
