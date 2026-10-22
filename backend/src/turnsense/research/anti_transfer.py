"""Anti-transfer (AT) loss after Guizzo, Weyde & Tarroni (2021).

Reference: E. Guizzo, T. Weyde, G. Tarroni, "Anti-transfer learning for task invariance in
convolutional neural networks for speech processing", Neural Networks 142:238–251, 2021
(arXiv:2006.06494; reference code github.com/ericguizzo/anti_transfer).

What the paper specifies (checked against the arXiv text):

* A network with the same architecture (up to the AT layer) is pre-trained on an
  *orthogonal* task and frozen. Here: speaker identification on training speakers only.
* The target network is randomly initialised (not initialised from the orthogonal network).
* At a chosen layer, both networks' feature maps for the same input are summarised as
  channel Gram matrices, ``G_ij = F_i · F_j`` (Eq. 1).
* ``L_AT = β · cos²(vec(G_p), vec(G_t))`` (Eq. 2): squared cosine similarity, bounded in [0, 1],
  added to the task cross-entropy; per-layer losses are *summed* when several layers are used.
* β = 1 was the paper's default; single-layer AT was used in its experiments.

Implementation choices (documented in docs/research_protocol.md):

* ``similarity="paper"`` (default): one Gram matrix per example over its valid (unpadded)
  time steps, normalised by C·T (normalisation does not change a cosine), squared cosine per
  example, averaged over the batch.
* ``similarity="repo_rowwise"``: mirrors the reference code's ``cos_squared`` path —
  cosine between corresponding Gram *rows*, absolute value, mean, then squared. Provided as
  an ablation; it is not the paper's Eq. 2.
* Our feature maps are 1-D in time ([B, C, T]) rather than VGG16's 2-D maps.

Gradient reversal (``SpeakerAdversary``) is a *different* invariance method (domain-
adversarial training). It is offered only as an extra ablation and must never be reported
as anti-transfer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from turnsense.research.model import masked_mean

Similarity = Literal["paper", "repo_rowwise"]


def gram_matrix(fmap: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """fmap [B, C, T], mask [B, T] → per-example Gram matrices [B, C, C], normalised by C·T."""
    b, c, t = fmap.shape
    if mask is not None:
        m = mask.to(fmap.dtype)
        fmap = fmap * m[:, None, :]
        n = m.sum(dim=1).clamp(min=1.0)
    else:
        n = torch.full((b,), float(t), device=fmap.device, dtype=fmap.dtype)
    g = torch.bmm(fmap, fmap.transpose(1, 2))
    return g / (c * n)[:, None, None]


def gram_similarity(
    g_pre: torch.Tensor, g_tgt: torch.Tensor, similarity: Similarity = "paper"
) -> torch.Tensor:
    """Squared cosine similarity between Gram matrices; returns a per-example tensor [B]."""
    if similarity == "paper":
        cos = F.cosine_similarity(g_pre.flatten(1), g_tgt.flatten(1), dim=1, eps=1e-8)
        return cos**2
    if similarity == "repo_rowwise":
        cos = F.cosine_similarity(g_pre, g_tgt, dim=2, eps=1e-8)  # [B, C]
        return cos.abs().mean(dim=1) ** 2
    raise ValueError(f"unknown similarity {similarity!r}")


def anti_transfer_loss(
    target_maps: dict[str, torch.Tensor],
    pretrained_maps: dict[str, torch.Tensor],
    layers: Sequence[str],
    beta: float,
    mask: torch.Tensor | None = None,
    similarity: Similarity = "paper",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Σ_layers β · mean_batch cos²(G_pre, G_tgt). Pretrained maps must carry no gradient."""
    total = torch.zeros((), device=next(iter(target_maps.values())).device)
    per_layer: dict[str, float] = {}
    for name in layers:
        g_t = gram_matrix(target_maps[name], mask)
        g_p = gram_matrix(pretrained_maps[name].detach(), mask)
        sim = gram_similarity(g_p, g_t, similarity).mean()
        per_layer[name] = float(sim.detach())
        total = total + beta * sim
    return total, per_layer


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class SpeakerAdversary(nn.Module):
    """Gradient-reversal speaker classifier on pooled encoder states (NOT anti-transfer)."""

    def __init__(self, hidden: int, n_speakers: int, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_speakers)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.head(grad_reverse(masked_mean(h, mask), self.lam))
