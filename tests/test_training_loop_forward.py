"""Smoke tests for training_loop multi-arg vs single-arg forward dispatch."""

import torch
import torch.nn as nn

from tetris.train.training_loop import _forward


class TwoArgModel(nn.Module):
    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # One logit per sample: image sum plus position sum, shape (N, 1).
        return x.sum(dim=(1, 2, 3)).unsqueeze(1) + p.sum(dim=1, keepdim=True)


class OneArgModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One logit per sample from the image alone, shape (N, 1).
        return x.sum(dim=(1, 2, 3)).unsqueeze(1)


def test_forward_with_position_branch():
    model = TwoArgModel()
    x = torch.randn(2, 3, 4, 4)
    p = torch.randn(2, 2)
    out = _forward(model, x, p, pos_in_batch=True)
    assert out.shape == (2, 1)
    # Positions reach the model on this branch.
    assert torch.allclose(out, x.sum(dim=(1, 2, 3)).unsqueeze(1) + p.sum(dim=1, keepdim=True))


def test_forward_image_only_branch():
    model = OneArgModel()
    x = torch.randn(2, 3, 4, 4)
    p = torch.randn(2, 2)
    out = _forward(model, x, p, pos_in_batch=False)
    assert out.shape == (2, 1)
    # Only the image reaches the model on this branch.
    assert torch.allclose(out, x.sum(dim=(1, 2, 3)).unsqueeze(1))
