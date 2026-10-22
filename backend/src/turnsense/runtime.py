"""Builds the frozen, server-accepted configuration for a trial (frontend.md §6.2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from turnsense import __version__
from turnsense.events import (
    DetectorMode,
    EffectiveConfig,
    PolicyConfig,
    ProviderInfo,
    Transport,
    VadConfig,
)

SIM_PROVIDERS = ProviderInfo(stt="mock_stt", llm="mock_llm", tts="mock_tts")


def policy_limits() -> dict[str, list[float]]:
    """[min, max] for every bounded numeric policy field, read from the model's constraints."""
    out: dict[str, list[float]] = {}
    for name, info in PolicyConfig.model_fields.items():
        lo = hi = None
        for meta in info.metadata:
            lo = getattr(meta, "ge", lo)
            hi = getattr(meta, "le", hi)
        if lo is not None and hi is not None:
            out[name] = [lo, hi]
    return out


def build_policy(overrides: dict[str, Any] | None) -> PolicyConfig:
    """Defaults + overrides, validated. Unknown keys and the version field are rejected."""
    overrides = dict(overrides or {})
    if "version" in overrides:
        raise ValueError("policy version is set by the server")
    base = PolicyConfig().model_dump()
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError(f"unknown policy fields: {sorted(unknown)}")
    return PolicyConfig(**{**base, **overrides})


def config_hash(cfg: EffectiveConfig) -> str:
    body = cfg.model_dump(mode="json", exclude={"config_hash"})
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def build_effective_config(
    *,
    detector_mode: DetectorMode,
    model_version: str,
    policy: PolicyConfig,
    transport: Transport,
    scenario_id: str,
    runtime_profile: Literal["simulation", "hosted", "local"],
    providers: ProviderInfo,
    vad_kind: Literal["silero", "energy"],
    transcript_enabled: bool,
    transcript_export_enabled: bool,
    seed: int | None,
    speculative_prefill: bool = False,
) -> EffectiveConfig:
    cfg = EffectiveConfig(
        detector_mode=detector_mode,
        model_version=model_version,
        decision_policy_version=policy.version,
        policy=policy,
        vad=VadConfig(kind=vad_kind),
        providers=providers,
        runtime_profile=runtime_profile,
        transport=transport,
        scenario_id=scenario_id,
        seed=seed,
        transcript_enabled=transcript_enabled,
        transcript_export_enabled=transcript_export_enabled,
        raw_audio_retention=False,
        speculative_prefill=speculative_prefill,
        server_version=__version__,
        config_hash="",
    )
    cfg.config_hash = config_hash(cfg)
    return cfg
