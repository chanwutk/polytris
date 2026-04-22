"""Tests for folding ImageNet multi-class heads to a single logit."""

import torch
import torch.nn as nn

from polyis.models.classifier.utils import collapse_classifier


def test_collapse_classifier_weight_fold():
    class W:
        meta = {
            'categories': (
                'a',
                'school bus',
                'b',
                'minivan',
            )
        }

    lin = nn.Linear(3, 4, bias=True)
    with torch.no_grad():
        lin.weight.fill_(0.0)
        lin.weight[1] = torch.tensor([1.0, 0.0, 0.0])
        lin.weight[3] = torch.tensor([0.0, 1.0, 0.0])
        lin.weight[0] = torch.tensor([0.0, 0.0, 2.0])
        lin.weight[2] = torch.tensor([0.0, 0.0, 4.0])
        lin.bias.fill_(0.0)
        lin.bias[1] = 3.0
        lin.bias[3] = 5.0
        lin.bias[0] = 10.0
        lin.bias[2] = 20.0

    out = collapse_classifier(W(), lin)
    assert out.out_features == 1 and out.in_features == 3
    expected_w = torch.tensor([[1.0, 1.0, -6.0]])
    assert torch.allclose(out.weight.data, expected_w)
    expected_b = (3.0 + 5.0) - (10.0 + 20.0)
    assert out.bias is not None
    assert abs(float(out.bias.data.item()) - expected_b) < 1e-5


def test_carpet_not_treated_as_vehicle():
    class W:
        meta = {'categories': ('carpet', 'school bus', 'dog')}

    lin = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(6, dtype=torch.float32).view(3, 2))

    out = collapse_classifier(W(), lin)
    # Only class 1 is vehicle: new row = w[1] - (w[0] + w[2])
    expected = lin.weight[1] - (lin.weight[0] + lin.weight[2])
    assert torch.allclose(out.weight.data, expected.unsqueeze(0))


def test_raises_without_categories_key():
    class W:
        meta = {}

    lin = nn.Linear(2, 3, bias=False)
    try:
        collapse_classifier(W(), lin)
    except ValueError as e:
        assert 'categories' in str(e).lower()
    else:
        raise AssertionError('expected ValueError')


def test_raises_on_out_features_mismatch():
    class W:
        meta = {'categories': ('a', 'school bus')}

    lin = nn.Linear(2, 5, bias=False)
    try:
        collapse_classifier(W(), lin)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError')
