"""Anti-transfer loss sanity: bounds, identity case, gradient isolation; GRL sign."""

import pytest
import torch

from turnsense.research.anti_transfer import (
    anti_transfer_loss,
    grad_reverse,
    gram_matrix,
    gram_similarity,
)
from turnsense.research.model import ModelConfig, SpeakerNet, TurnNet


def test_gram_matrix_masks_padding():
    f = torch.randn(1, 4, 10)
    padded = torch.cat([f, torch.randn(1, 4, 5) * 100], dim=2)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 5)], dim=1)
    torch.testing.assert_close(gram_matrix(padded, mask), gram_matrix(f))


@pytest.mark.parametrize("similarity", ["paper", "repo_rowwise"])
def test_identical_maps_give_maximal_similarity(similarity):
    f = torch.relu(torch.randn(3, 8, 20))
    g = gram_matrix(f)
    sim = gram_similarity(g, g, similarity)
    torch.testing.assert_close(sim, torch.ones(3))
    loss, per = anti_transfer_loss({"l": f}, {"l": f}, ["l"], beta=2.5, similarity=similarity)
    assert abs(float(loss) - 2.5) < 1e-4  # beta * cos^2 with cos = 1
    assert abs(per["l"] - 1.0) < 1e-5


def test_similarity_bounded_and_layers_summed():
    a = {"x": torch.randn(2, 6, 30), "y": torch.randn(2, 6, 30)}
    b = {"x": torch.randn(2, 6, 30), "y": torch.randn(2, 6, 30)}
    loss_x, _ = anti_transfer_loss(a, b, ["x"], beta=1.0)
    loss_y, _ = anti_transfer_loss(a, b, ["y"], beta=1.0)
    loss_xy, _ = anti_transfer_loss(a, b, ["x", "y"], beta=1.0)
    assert 0.0 <= float(loss_x) <= 1.0
    torch.testing.assert_close(loss_xy, loss_x + loss_y)  # paper: per-layer losses are summed


def test_gradient_flows_only_into_turn_model():
    torch.manual_seed(0)
    cfg = ModelConfig()
    turn, spk = TurnNet(cfg), SpeakerNet(cfg, 3)
    spk.eval()
    spk.requires_grad_(False)
    x = torch.randn(2, 40, 42)
    _, maps = turn(x)
    with torch.no_grad():
        _, pre = spk.encoder(x)
    loss, _ = anti_transfer_loss(maps, pre, ["conv3"], beta=1.0)
    loss.backward()
    assert all(p.grad is None for p in spk.parameters())
    conv3 = turn.encoder.convs[2].conv.weight.grad
    assert conv3 is not None and conv3.abs().sum() > 0
    assert turn.head.weight.grad is None  # AT loss alone does not touch the task head


def test_gradient_reversal_flips_sign():
    x = torch.randn(5, requires_grad=True)
    (grad_reverse(x, 0.5) * 2.0).sum().backward()
    torch.testing.assert_close(x.grad, torch.full((5,), -1.0))
