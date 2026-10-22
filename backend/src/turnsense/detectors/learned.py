"""E1/E2 runtime: streaming learned turn detector loaded from a research checkpoint.

``learned_control`` and ``anti_transfer`` are the same class with different checkpoints;
both use the shared barge-in logic and ``map_turn_end_probability`` from detectors/base.py,
so the only difference between them at runtime is the trained representation.

Per 20 ms frame: numpy features (2 windows) → TurnNet.step (carried conv buffers + GRU
state) → P(turn end) of the newest window. The model state runs continuously across turns
and agent speech so that context is never lost; ``new_turn`` clears only bookkeeping.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from turnsense.audio.frames import resample_linear
from turnsense.detectors.base import (
    DetectorInput,
    Prediction,
    TurnDetector,
    map_turn_end_probability,
)
from turnsense.events import DetectorMode, PolicyConfig
from turnsense.research.features import LogMelPitch, StreamingFeatures
from turnsense.research.model import TurnNet
from turnsense.research.train import load_turn_checkpoint

_THREADS_SET = False


def _limit_threads() -> None:
    """Small per-frame models run fastest single-threaded; override with TURNSENSE_TORCH_THREADS."""
    global _THREADS_SET
    if not _THREADS_SET:
        torch.set_num_threads(int(os.environ.get("TURNSENSE_TORCH_THREADS", "1")))
        _THREADS_SET = True


def read_checkpoint_metadata(path: Path) -> dict:
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = dict(ck.get("metadata", {}))
    if "model_version" not in meta:
        raise ValueError("checkpoint has no model_version")
    return meta


class LearnedTurnDetector(TurnDetector):
    probabilistic = True

    def __init__(
        self,
        mode: DetectorMode,
        checkpoint_path: Path,
        policy: PolicyConfig,
        *,
        allow_experimental: bool = False,
    ) -> None:
        super().__init__(policy)
        _limit_threads()
        model, meta, feature_cfg = load_turn_checkpoint(Path(checkpoint_path))
        ck_mode = str(meta.get("mode"))
        valid_modes = {m.value for m in DetectorMode}
        if ck_mode in valid_modes and ck_mode != mode.value:
            raise ValueError(f"checkpoint was trained as {ck_mode!r}, requested {mode.value!r}")
        if ck_mode not in valid_modes and not allow_experimental:
            raise ValueError(f"checkpoint mode {ck_mode!r} is experimental; offline use only")
        self.mode = mode
        self.model_version = str(meta["model_version"])
        self.metadata = meta
        self.checkpoint_path = Path(checkpoint_path)
        self._model: TurnNet = model
        self._feature_cfg = feature_cfg
        self._features = StreamingFeatures(LogMelPitch(feature_cfg))
        self._state = model.init_state(1)
        self._p_end: float = 0.0
        self._user_spoke = False

    def reset(self) -> None:
        super().reset()
        self._features.reset()
        self._state = self._model.init_state(1)
        self._p_end = 0.0
        self._user_spoke = False

    def new_turn(self) -> None:
        super().new_turn()
        self._user_spoke = False

    def info(self) -> dict[str, object]:
        return {
            **super().info(),
            "dataset_id": self.metadata.get("dataset_id"),
            "split_id": self.metadata.get("split_id"),
            "synthetic": self.metadata.get("synthetic"),
            "beta": self.metadata.get("beta"),
        }

    @torch.inference_mode()
    def _advance(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate != self._feature_cfg.sample_rate:
            samples = resample_linear(samples, sample_rate, self._feature_cfg.sample_rate)
        feats = self._features.push(samples)
        if feats.shape[0]:
            logits, self._state = self._model.step(torch.from_numpy(feats[None]), self._state)
            self._p_end = float(TurnNet.p_turn_end(logits[0, -1]))
        return self._p_end

    def predict(self, inp: DetectorInput) -> Prediction:
        p_end = self._advance(inp.frame.samples, inp.frame.sample_rate)
        overlap = self.speech_overlap_event(inp)
        if overlap is not None:
            return overlap
        if inp.vad.is_speech:
            self._user_spoke = True
        return map_turn_end_probability(p_end, inp.vad, self.policy, self._user_spoke)
