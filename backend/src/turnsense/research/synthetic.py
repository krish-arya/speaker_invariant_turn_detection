"""SYNTHETIC dialogue corpus with an engineered speaker ↔ turn-structure confound.

Purpose: a controlled smoke test of the anti-transfer *mechanism* and of the whole
data → train → streaming-evaluate pipeline. It is **not** evidence about real speakers.

Each recording is one user's channel of a two-party exchange: user turns made of 1–3
phrases separated by within-turn pauses (HOLD), followed by a gap while the (silent, absent)
partner talks (SHIFT). Cues available to a model:

* **genuine turn cue** — phrase-final prosody from ``sim.voice``: a "final" contour (falling
  f0, final lengthening) ends a turn, a "hold" contour (level/rising) precedes a
  within-turn pause. With probability ``1 - prosody_reliability`` a phrase gets the other
  contour, so prosody alone is informative but imperfect.
* **speaker shortcut (the confound)** — voices fall into a low-f0 and a high-f0 group.
  For *training* speakers, the high group produces mostly multi-phrase turns (pauses are
  mostly HOLDs) and the low group mostly single-phrase turns (pauses are mostly SHIFTs).
  For *held-out* speakers the association is reversed. A model that leans on voice identity
  should therefore lose more accuracy on unseen speakers than one that relies on prosody.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from turnsense.research.data import Recording, Turn, write_manifest, write_wav
from turnsense.sim.voice import VoiceProfile, noise_bed, synth_phrase

HIGH_F0_HZ = 155.0


@dataclass(frozen=True)
class SyntheticConfig:
    n_train_speakers: int = 24
    n_unseen_speakers: int = 12
    convs_per_train_speaker: int = 5
    convs_per_unseen_speaker: int = 3
    turns_per_conv: tuple[int, int] = (5, 8)
    syllables_per_phrase: tuple[int, int] = (3, 10)
    hold_pause_ms: tuple[float, float] = (300.0, 1100.0)
    shift_gap_ms: tuple[float, float] = (1600.0, 3600.0)
    prosody_reliability: float = 0.8
    # P(turn has >1 phrase) by voice group; reversed for unseen speakers
    p_multi_phrase_high: float = 0.85
    p_multi_phrase_low: float = 0.15
    noise_dbfs: tuple[float, float] = (-66.0, -52.0)
    train_voice_offset: int = 0
    unseen_voice_offset: int = 50_000
    sample_rate: int = 16_000
    seed: int = 0
    version: str = "synthetic-dialogue-0.1"

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @property
    def dataset_id(self) -> str:
        h = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.version}-{h}"


QUICK = SyntheticConfig(
    n_train_speakers=16,
    n_unseen_speakers=8,
    convs_per_train_speaker=4,
    convs_per_unseen_speaker=2,
    turns_per_conv=(4, 6),
)


def voice_group(v: VoiceProfile) -> str:
    return "high" if v.f0_hz >= HIGH_F0_HZ else "low"


def _pick_voices(n: int, offset: int) -> list[VoiceProfile]:
    """n voices, half per f0 group, deterministic in ``offset``."""
    want = {"high": n - n // 2, "low": n // 2}
    got: list[VoiceProfile] = []
    vid = offset
    while sum(want.values()) > 0:
        v = VoiceProfile.from_id(vid)
        g = voice_group(v)
        if want[g] > 0:
            got.append(v)
            want[g] -= 1
        vid += 1
    return got


def _conversation(
    voice: VoiceProfile, p_multi: float, cfg: SyntheticConfig, rng: np.random.Generator
) -> tuple[np.ndarray, list[Turn], dict]:
    sr = cfg.sample_rate
    pieces: list[tuple[float, np.ndarray]] = []
    turns: list[Turn] = []
    t = float(rng.uniform(300, 900))
    counts = {"phrases": 0, "flipped_contours": 0, "holds": 0, "shifts": 0}
    n_turns = int(rng.integers(cfg.turns_per_conv[0], cfg.turns_per_conv[1] + 1))
    for _ in range(n_turns):
        n_phr = 1 if rng.random() >= p_multi else int(rng.integers(2, 4))
        start = t
        ipus: list[tuple[float, float]] = []
        for j in range(n_phr):
            role = "final" if j == n_phr - 1 else "hold"
            contour = role
            if rng.random() > cfg.prosody_reliability:
                contour = "hold" if role == "final" else "final"
                counts["flipped_contours"] += 1
            elif role == "final" and rng.random() < 0.2:
                contour = "question"
            n_syl = int(rng.integers(cfg.syllables_per_phrase[0], cfg.syllables_per_phrase[1] + 1))
            audio = synth_phrase(voice, n_syl, contour, rng, sr)  # type: ignore[arg-type]
            dur = 1000.0 * audio.size / sr
            pieces.append((t, audio))
            ipus.append((round(t, 3), round(t + dur, 3)))
            counts["phrases"] += 1
            t += dur
            if j < n_phr - 1:
                t += float(rng.uniform(*cfg.hold_pause_ms))
                counts["holds"] += 1
        turns.append(Turn(start_ms=round(start, 3), end_ms=round(t, 3), ipus=ipus))
        counts["shifts"] += 1
        t += float(rng.uniform(*cfg.shift_gap_ms))
    total = int((t + 500.0) * sr / 1000)
    out = noise_bed(total, float(rng.uniform(*cfg.noise_dbfs)), rng)
    for start_ms, audio in pieces:
        i = int(start_ms * sr / 1000)
        out[i : i + audio.size] += audio[: max(0, total - i)]
    return out, turns, counts


def generate_corpus(cfg: SyntheticConfig, out_dir: Path) -> tuple[Path, list[str], dict]:
    """Write WAVs + manifest.jsonl. Returns (manifest path, unseen speaker ids, summary)."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(cfg.seed)
    records: list[Recording] = []
    unseen_ids: list[str] = []
    summary: dict[str, dict] = {"train_pool": {}, "unseen_pool": {}}
    pools = [
        ("train_pool", _pick_voices(cfg.n_train_speakers, cfg.train_voice_offset),
         cfg.convs_per_train_speaker, False),
        ("unseen_pool", _pick_voices(cfg.n_unseen_speakers, cfg.unseen_voice_offset),
         cfg.convs_per_unseen_speaker, True),
    ]  # fmt: skip
    for pool, voices, n_convs, reversed_confound in pools:
        agg = {"high": {"holds": 0, "shifts": 0}, "low": {"holds": 0, "shifts": 0}}
        for v in voices:
            spk = f"syn{v.voice_id:05d}"
            group = voice_group(v)
            high_like = (group == "high") != reversed_confound
            p_multi = cfg.p_multi_phrase_high if high_like else cfg.p_multi_phrase_low
            if reversed_confound:
                unseen_ids.append(spk)
            for c in range(n_convs):
                audio, turns, counts = _conversation(v, p_multi, cfg, rng)
                rec_id = f"{spk}_c{c:02d}"
                wav = out_dir / "audio" / f"{rec_id}.wav"
                write_wav(wav, audio, cfg.sample_rate)
                agg[group]["holds"] += counts["holds"]
                agg[group]["shifts"] += counts["shifts"]
                records.append(
                    Recording(
                        recording_id=rec_id,
                        conversation_id=rec_id,
                        speaker_id=spk,
                        audio_path=wav,
                        sample_rate=cfg.sample_rate,
                        duration_ms=round(1000.0 * audio.size / cfg.sample_rate, 1),
                        turns=turns,
                        condition={"noise": "synthetic_pink", "device": "synthetic"},
                        language="none",
                        synthetic=True,
                        meta={
                            "dataset_id": cfg.dataset_id,
                            "license": "generated; no real speech",
                            "annotation": {"schema": "turn-boundary-1", "annotator": "generator"},
                            "voice_group": group,
                            "pool": pool,
                            "f0_hz": round(v.f0_hz, 1),
                        },
                    )
                )
        summary[pool] = {
            g: {
                **a,
                "p_shift_given_pause": round(a["shifts"] / max(1, a["holds"] + a["shifts"]), 3),
            }
            for g, a in agg.items()
        }
    manifest = out_dir / "manifest.jsonl"
    write_manifest(records, manifest)
    info = {
        "SYNTHETIC": True,
        "dataset_id": cfg.dataset_id,
        "config": cfg.to_dict(),
        "confound_summary": summary,
        "n_recordings": len(records),
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return manifest, unseen_ids, info
