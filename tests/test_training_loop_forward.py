"""Smoke tests for training_loop multi-arg vs single-arg forward dispatch."""

import torch
import torch.nn as nn

from polyis.train.training_loop import _forward


class TwoArgModel(nn.Module):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(1, 2, 3), keepdim=True) + p.sum(dim=1, keepdim=True).unsqueeze(1)


class OneArgModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(1, 2, 3), keepdim=True)


def test_forward_with_position_branch():
    model = TwoArgModel()
    x = torch.randn(2, 3, 4, 4)
    p = torch.randn(2, 2)
    out = _forward(model, x, p, pos_in_batch=True)
    assert out.shape == (2, 1)


def test_forward_image_only_branch():
    model = OneArgModel()
    x = torch.randn(2, 3, 4, 4)
    p = torch.randn(2, 2)
    out = _forward(model, x, p, pos_in_batch=False)
    assert out.shape == (2, 1)
