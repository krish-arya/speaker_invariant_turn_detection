"""Detector registry: one factory, so the event consumer never knows which variant runs."""

from __future__ import annotations

from pathlib import Path

from turnsense.detectors.base import (
    DetectorAvailability,
    DetectorInput,
    DetectorOutput,
    Prediction,
    TranscriptHint,
    TurnDetector,
    map_turn_end_probability,
)
from turnsense.detectors.silence import SILENCE_BASELINE_VERSION, SilenceBaselineDetector
from turnsense.events import DetectorMode, PolicyConfig

LABELS = {
    DetectorMode.SILENCE_BASELINE: "Silence/VAD baseline",
    DetectorMode.LEARNED_CONTROL: "Learned control (no anti-transfer)",
    DetectorMode.ANTI_TRANSFER: "Anti-transfer",
}


def checkpoint_path(mode: DetectorMode, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{mode.value}.pt"


def detector_availability(checkpoint_dir: Path) -> list[DetectorAvailability]:
    out = [
        DetectorAvailability(
            mode=DetectorMode.SILENCE_BASELINE,
            label=LABELS[DetectorMode.SILENCE_BASELINE],
            model_version=SILENCE_BASELINE_VERSION,
            available=True,
            unavailable_reason=None,
            probabilistic=False,
        )
    ]
    for mode in (DetectorMode.LEARNED_CONTROL, DetectorMode.ANTI_TRANSFER):
        path = checkpoint_path(mode, checkpoint_dir)
        version, reason, extra = "unavailable", None, {}
        try:
            from turnsense.detectors.learned import read_checkpoint_metadata
        except ImportError as exc:  # torch (research extra) not installed
            reason = f"Learned detector runtime unavailable: {exc}"
        else:
            if not path.is_file():
                reason = f"No checkpoint at {path.relative_to(path.parents[1]).as_posix()}"
            else:
                try:
                    meta = read_checkpoint_metadata(path)
                    version = str(meta.get("model_version", "unknown"))
                    extra = {
                        k: meta[k] for k in ("trained_at", "dataset_id", "split_id") if k in meta
                    }
                except Exception as exc:  # corrupt/incompatible checkpoint
                    reason = f"Checkpoint unreadable: {exc}"
        out.append(
            DetectorAvailability(
                mode=mode,
                label=LABELS[mode],
                model_version=version,
                available=reason is None,
                unavailable_reason=reason,
                probabilistic=True,
                extra=extra,
            )
        )
    return out


def create_detector(mode: DetectorMode, policy: PolicyConfig, checkpoint_dir: Path) -> TurnDetector:
    if mode == DetectorMode.SILENCE_BASELINE:
        return SilenceBaselineDetector(policy)
    from turnsense.detectors.learned import LearnedTurnDetector

    return LearnedTurnDetector(mode, checkpoint_path(mode, checkpoint_dir), policy)


__all__ = [
    "LABELS",
    "DetectorAvailability",
    "DetectorInput",
    "DetectorOutput",
    "Prediction",
    "SilenceBaselineDetector",
    "TranscriptHint",
    "TurnDetector",
    "checkpoint_path",
    "create_detector",
    "detector_availability",
    "map_turn_end_probability",
]
