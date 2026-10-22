from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from turnsense.audio.frames import AudioFrame, FrameBuffer
from turnsense.audio.vad import VadResult
from turnsense.detectors import create_detector
from turnsense.events import DetectorMode, SessionInfo, Transport, TrialType
from turnsense.runtime import SIM_PROVIDERS, build_effective_config, build_policy


def make_info(
    scenario_id: str = "short_answer",
    seed: int = 1,
    mode: DetectorMode = DetectorMode.SILENCE_BASELINE,
    transcript_enabled: bool = True,
    transcript_export_enabled: bool = False,
    policy_overrides: dict | None = None,
) -> SessionInfo:
    policy = build_policy(policy_overrides)
    cfg = build_effective_config(
        detector_mode=mode,
        model_version="silence-baseline-0.1.0",
        policy=policy,
        transport=Transport.SIMULATION,
        scenario_id=scenario_id,
        runtime_profile="simulation",
        providers=SIM_PROVIDERS,
        vad_kind="energy",
        transcript_enabled=transcript_enabled,
        transcript_export_enabled=transcript_export_enabled,
        seed=seed,
    )
    return SessionInfo(
        session_id="ses_0123456789abcdef",
        trial_id="tri_0123456789abcdef",
        transport=Transport.SIMULATION,
        trial_type=TrialType.SIMULATED,
        scenario_id=scenario_id,
        effective_config=cfg,
        started_at="2026-10-22T00:00:00.000Z",
    )


def baseline_detector(info: SessionInfo, tmp_path: Path):
    cfg = info.effective_config
    return create_detector(cfg.detector_mode, cfg.policy, tmp_path)


def frame(seq: int, samples: np.ndarray | None = None) -> AudioFrame:
    s = np.zeros(320, dtype=np.float32) if samples is None else samples
    return AudioFrame(seq, (seq - 1) * 20.0, seq * 20.0, s, 16_000)


def vad_result(
    seq: int,
    speech: bool,
    speech_run_ms: float = 0.0,
    silence_run_ms: float = 0.0,
    changed: bool = False,
) -> VadResult:
    return VadResult(
        frame_seq=seq,
        frame_end_ms=seq * 20.0,
        is_speech=speech,
        raw_speech=speech,
        probability=None,
        speech_run_ms=speech_run_ms,
        silence_run_ms=silence_run_ms,
        last_speech_end_ms=None,
        changed=changed,
    )


def frames_from(audio: np.ndarray) -> list[AudioFrame]:
    return FrameBuffer().push(audio)


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
    return tmp_path
