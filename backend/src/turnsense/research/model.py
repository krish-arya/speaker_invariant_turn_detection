"""Causal CNN-GRU encoder, turn-state head and speaker-identification head.

``TurnNet`` and ``SpeakerNet`` share the exact same ``Encoder`` architecture, which
anti-transfer requires (the orthogonal-task network must match the target network up to the
anti-transfer layer; Guizzo et al. 2021).

Encoder (feature frames at 100 Hz, input already in the raw feature domain):

    normalise (stored train mean/std)
    → causal Conv1d k=5 d=1 → ReLU   ("conv1")
    → causal Conv1d k=5 d=2 → ReLU   ("conv2")
    → causal Conv1d k=5 d=4 → ReLU   ("conv3")   receptive field 29 frames ≈ 290 ms
    → GRU                            ("gru")      unbounded causal context

Turn head: per-frame logits over {SPEECH, HOLD, SHIFT}. The detector uses
``P(turn end) = p(SHIFT) / (p(HOLD) + p(SHIFT))`` — the probability that the current
silence is a turn end rather than a within-turn pause.

``step()`` processes newly arrived frames with carried state; its output equals the
full-sequence ``forward()`` output (checked in tests), so offline evaluation with the
streaming detector and batch training see the same function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

SPEECH, HOLD, SHIFT = 0, 1, 2
IGNORE = -100
CLASS_NAMES = ("speech", "hold", "shift")


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 42
    conv_channels: int = 64
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)
    gru_hidden: int = 96
    n_classes: int = 3
    version: str = "causal-cnn-gru-0.1"
    layer_names: tuple[str, ...] = field(default=("conv1", "conv2", "conv3", "gru"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dilations"] = list(self.dilations)
        d["layer_names"] = list(self.layer_names)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "dilations" in d:
            d["dilations"] = tuple(d["dilations"])
        if "layer_names" in d:
            d["layer_names"] = tuple(d["layer_names"])
        return cls(**d)


class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.left = (kernel - 1) * dilation
        self.c_in = c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x [B, C, T]
        return self.conv(F.pad(x, (self.left, 0)))

    def step(self, x: torch.Tensor, buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, C, n] new frames; buf [B, C, left] previous inputs (zeros at start).

        Uses a gather + ``F.linear`` instead of ``conv1d``: identical maths, but avoids the
        per-call conv setup cost that dominates for 2-frame streaming steps on CPU.
        """
        full = torch.cat([buf, x], dim=2)
        n = x.shape[2]
        k, d = self.conv.kernel_size[0], self.conv.dilation[0]
        idx = (torch.arange(n)[:, None] + torch.arange(k)[None, :] * d).reshape(-1)
        taps = full[:, :, idx].reshape(full.shape[0], self.c_in, n, k)  # [B, C, n, k]
        taps = taps.permute(0, 2, 1, 3).reshape(full.shape[0], n, self.c_in * k)
        w = self.conv.weight.reshape(self.conv.out_channels, self.c_in * k)
        y = F.linear(taps, w, self.conv.bias).transpose(1, 2)  # [B, C_out, n]
        return y, full[:, :, full.shape[2] - self.left :]


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        convs = []
        c_in = cfg.input_dim
        for d in cfg.dilations:
            convs.append(CausalConv1d(c_in, cfg.conv_channels, cfg.kernel_size, d))
            c_in = cfg.conv_channels
        self.convs = nn.ModuleList(convs)
        self.gru = nn.GRU(cfg.conv_channels, cfg.gru_hidden, batch_first=True)
        self.register_buffer("feat_mean", torch.zeros(cfg.input_dim))
        self.register_buffer("feat_std", torch.ones(cfg.input_dim))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(torch.clamp(std, min=1e-3))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / self.feat_std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x [B, T, D] → h [B, T, H], maps {layer: [B, C, T]}."""
        z = self._norm(x).transpose(1, 2)
        maps: dict[str, torch.Tensor] = {}
        for i, conv in enumerate(self.convs):
            z = F.relu(conv(z))
            maps[f"conv{i + 1}"] = z
        h, _ = self.gru(z.transpose(1, 2))
        maps["gru"] = h.transpose(1, 2)
        return h, maps

    def init_state(self, batch: int = 1) -> dict[str, object]:
        device = self.feat_mean.device
        bufs = [torch.zeros(batch, c.c_in, c.left, device=device) for c in self.convs]
        h = torch.zeros(1, batch, self.cfg.gru_hidden, device=device)
        return {"bufs": bufs, "h": h}

    def step(
        self, x: torch.Tensor, state: dict[str, object]
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """x [B, n, D] newly arrived frames → h [B, n, H] and the updated state."""
        z = self._norm(x).transpose(1, 2)
        new_bufs = []
        for conv, buf in zip(self.convs, state["bufs"], strict=True):  # type: ignore[arg-type]
            y, nb = conv.step(z, buf)
            z = F.relu(y)
            new_bufs.append(nb)
        h, hn = self.gru(z.transpose(1, 2), state["h"])
        return h, {"bufs": new_bufs, "h": hn}


class TurnNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.head = nn.Linear(cfg.gru_hidden, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h, maps = self.encoder(x)
        return self.head(h), maps

    def init_state(self, batch: int = 1) -> dict[str, object]:
        return self.encoder.init_state(batch)

    def step(
        self, x: torch.Tensor, state: dict[str, object]
    ) -> tuple[torch.Tensor, dict[str, object]]:
        h, state = self.encoder.step(x, state)
        return self.head(h), state

    @staticmethod
    def p_turn_end(logits: torch.Tensor) -> torch.Tensor:
        """P(SHIFT | silence) = p_shift / (p_hold + p_shift), per frame."""
        p = torch.softmax(logits, dim=-1)
        return p[..., SHIFT] / (p[..., HOLD] + p[..., SHIFT] + 1e-8)


class SpeakerNet(nn.Module):
    """Orthogonal-task network: speaker identification with the same encoder architecture."""

    def __init__(self, cfg: ModelConfig, n_speakers: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_speakers = n_speakers
        self.encoder = Encoder(cfg)
        self.head = nn.Linear(cfg.gru_hidden, n_speakers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x [B, T, D], mask [B, T] (1 = pool this frame) → logits [B, n_speakers]."""
        h, maps = self.encoder(x)
        return self.head(masked_mean(h, mask)), maps


def masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """h [B, T, H], mask [B, T] → [B, H]; falls back to all frames where a row is empty."""
    if mask is None:
        return h.mean(dim=1)
    m = mask.to(h.dtype)
    empty = m.sum(dim=1, keepdim=True) == 0
    m = torch.where(empty, torch.ones_like(m), m)
    return (h * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True)
