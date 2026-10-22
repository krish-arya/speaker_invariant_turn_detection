"""Corpus manifest, frame labels and speaker-disjoint splits (goal.md §5).

Manifest: JSONL, one record per single-speaker audio stream (one channel of a
conversation). Paths are relative to the manifest file. Example record::

    {"schema_version": "turnsense-manifest-1",
     "recording_id": "conv0007_B", "conversation_id": "conv0007",
     "audio_path": "audio/conv0007.wav", "channel": 1, "sample_rate": 8000,
     "speaker_id": "spk_0042",                      # pseudonymous, never a real identity
     "duration_ms": 312000,
     "condition": {"noise": "clean", "device": "telephone", "mic_distance": null},
     "language": "en",
     "turns": [{"start_ms": 1200, "end_ms": 4810, "type": "turn",
                "ipus": [[1200, 2950], [3400, 4810]], "interrupted": false}],
     "backchannels": [[9050, 9400]], "overlaps": [[15020, 15500]],
     "annotation": {"schema": "turn-boundary-1", "annotator": "A1", "adjudicated": false},
     "dataset_id": "...", "license": "...", "synthetic": false}

``turns[].end_ms`` is the annotated *conversational* turn end (docs/annotation_guide.md),
not merely the last voiced frame of every pause. ``ipus`` (inter-pausal units) are the
speech stretches inside the turn; when omitted they are derived with the energy VAD.

Frame labels (100 Hz feature frames, labelled by the time at the end of each window):
SPEECH inside IPUs (and backchannels); HOLD in pauses between IPUs of one turn; SHIFT from a
turn end until the speaker talks again (capped by ``shift_horizon_ms``); IGNORE elsewhere
(before the first turn, after backchannels, beyond the horizon, overlap regions).
"""

from __future__ import annotations

import hashlib
import json
import wave
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from turnsense.audio.frames import resample_linear
from turnsense.research.features import FeatureConfig, LogMelPitch
from turnsense.research.model import HOLD, IGNORE, SHIFT, SPEECH

MANIFEST_SCHEMA = "turnsense-manifest-1"
SPLIT_SCHEMA = "turnsense-split-1"
SPLIT_NAMES = ("train", "val_seen", "test_unseen")


@dataclass
class Turn:
    start_ms: float
    end_ms: float
    ipus: list[tuple[float, float]] = field(default_factory=list)
    type: str = "turn"  # "turn" | "backchannel"
    interrupted: bool = False


@dataclass
class Recording:
    recording_id: str
    conversation_id: str
    speaker_id: str
    audio_path: Path
    channel: int = 0
    sample_rate: int | None = None
    duration_ms: float | None = None
    turns: list[Turn] = field(default_factory=list)
    backchannels: list[tuple[float, float]] = field(default_factory=list)
    overlaps: list[tuple[float, float]] = field(default_factory=list)
    condition: dict = field(default_factory=dict)
    language: str | None = None
    synthetic: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def conversation_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.type == "turn"]


def load_manifest(path: Path) -> list[Recording]:
    path = Path(path)
    out: list[Recording] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("schema_version") != MANIFEST_SCHEMA:
                raise ValueError(
                    f"{path}:{line_no}: unsupported schema {d.get('schema_version')!r}"
                )
            turns = [
                Turn(
                    start_ms=float(t["start_ms"]),
                    end_ms=float(t["end_ms"]),
                    ipus=[(float(a), float(b)) for a, b in t.get("ipus", [])],
                    type=t.get("type", "turn"),
                    interrupted=bool(t.get("interrupted", False)),
                )
                for t in d.get("turns", [])
            ]
            known = {
                "schema_version", "recording_id", "conversation_id", "speaker_id", "audio_path",
                "channel", "sample_rate", "duration_ms", "turns", "backchannels", "overlaps",
                "condition", "language", "synthetic",
            }  # fmt: skip
            out.append(
                Recording(
                    recording_id=d["recording_id"],
                    conversation_id=d["conversation_id"],
                    speaker_id=d["speaker_id"],
                    audio_path=(path.parent / d["audio_path"]).resolve(),
                    channel=int(d.get("channel", 0)),
                    sample_rate=d.get("sample_rate"),
                    duration_ms=d.get("duration_ms"),
                    turns=sorted(turns, key=lambda t: t.start_ms),
                    backchannels=[(float(a), float(b)) for a, b in d.get("backchannels", [])],
                    overlaps=[(float(a), float(b)) for a, b in d.get("overlaps", [])],
                    condition=d.get("condition", {}) or {},
                    language=d.get("language"),
                    synthetic=bool(d.get("synthetic", False)),
                    meta={k: v for k, v in d.items() if k not in known},
                )
            )
    ids = [r.recording_id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate recording_id in manifest")
    return out


def recording_to_json(rec: Recording, manifest_dir: Path) -> dict:
    d = {
        "schema_version": MANIFEST_SCHEMA,
        "recording_id": rec.recording_id,
        "conversation_id": rec.conversation_id,
        "audio_path": Path(rec.audio_path).resolve().relative_to(manifest_dir.resolve()).as_posix(),
        "channel": rec.channel,
        "sample_rate": rec.sample_rate,
        "speaker_id": rec.speaker_id,
        "duration_ms": rec.duration_ms,
        "condition": rec.condition,
        "language": rec.language,
        "turns": [
            {
                "start_ms": t.start_ms,
                "end_ms": t.end_ms,
                "type": t.type,
                "ipus": [[a, b] for a, b in t.ipus],
                "interrupted": t.interrupted,
            }
            for t in rec.turns
        ],
        "backchannels": [[a, b] for a, b in rec.backchannels],
        "overlaps": [[a, b] for a, b in rec.overlaps],
        "synthetic": rec.synthetic,
    }
    d.update(rec.meta)
    return d


def write_manifest(records: Iterable[Recording], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(recording_to_json(rec, path.parent)) + "\n")


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def load_audio(rec: Recording, target_rate: int = 16_000) -> np.ndarray:
    """Mono float32 at ``target_rate`` for the recording's channel."""
    path = Path(rec.audio_path)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except ImportError:
        with wave.open(str(path), "rb") as w:
            sr, n_ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV supported without soundfile") from None
        data = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).reshape(-1, n_ch)
    ch = min(rec.channel, data.shape[1] - 1)
    audio = data[:, ch]
    if sr != target_rate:
        audio = resample_linear(audio, sr, target_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def derive_ipus(
    audio: np.ndarray, turn: Turn, sample_rate: int = 16_000
) -> list[tuple[float, float]]:
    """IPUs inside a turn from the energy VAD (used when the annotation has none)."""
    from turnsense.audio.frames import FrameBuffer
    from turnsense.audio.vad import EnergyVAD

    vad = EnergyVAD()
    ipus: list[tuple[float, float]] = []
    start: float | None = None
    for fr in FrameBuffer(sample_rate=sample_rate).push(audio):
        r = vad.process(fr)
        inside = turn.start_ms <= fr.end_ms <= turn.end_ms
        if r.is_speech and inside and start is None:
            start = max(fr.start_ms, turn.start_ms)
        elif (not r.is_speech or not inside) and start is not None:
            ipus.append((start, min(fr.start_ms, turn.end_ms)))
            start = None
    if start is not None:
        ipus.append((start, turn.end_ms))
    return ipus or [(turn.start_ms, turn.end_ms)]


def frame_labels(
    rec: Recording,
    n_frames: int,
    hop_ms: float = 10.0,
    shift_horizon_ms: float = 2500.0,
    audio: np.ndarray | None = None,
) -> np.ndarray:
    """Per-feature-frame labels (SPEECH/HOLD/SHIFT/IGNORE); frame k ends at (k+1)·hop_ms."""
    t = (np.arange(n_frames) + 1) * hop_ms
    labels = np.full(n_frames, IGNORE, dtype=np.int64)

    def span(a: float, b: float) -> np.ndarray:
        return (t > a) & (t <= b)

    turns = rec.conversation_turns
    speech_starts = sorted([t_.start_ms for t_ in rec.turns] + [a for a, _ in rec.backchannels])
    for turn in turns:
        ipus = turn.ipus
        if not ipus:
            ipus = [(turn.start_ms, turn.end_ms)] if audio is None else derive_ipus(audio, turn)
        ipus = sorted(ipus)
        for (_, a_end), (b_start, _) in zip(ipus[:-1], ipus[1:], strict=False):
            labels[span(a_end, b_start)] = HOLD
        for a, b in ipus:
            labels[span(a, b)] = SPEECH
        nxt = [s for s in speech_starts if s >= turn.end_ms]
        stop = min([turn.end_ms + shift_horizon_ms, *nxt])
        if not turn.interrupted:
            labels[span(turn.end_ms, stop)] = SHIFT
    for a, b in rec.backchannels:
        labels[span(a, b)] = SPEECH
    for a, b in rec.overlaps:
        labels[span(a, b)] = IGNORE
    return labels


# ---------------------------------------------------------------------------
# Speaker-disjoint splits
# ---------------------------------------------------------------------------


@dataclass
class Split:
    split_id: str
    assignments: dict[str, str]  # recording_id -> split name
    excluded: dict[str, str]  # recording_id -> reason
    seed: int
    params: dict
    stats: dict = field(default_factory=dict)

    def recordings(self, records: Sequence[Recording], name: str) -> list[Recording]:
        return [r for r in records if self.assignments.get(r.recording_id) == name]

    def speakers(self, records: Sequence[Recording], name: str) -> set[str]:
        return {r.speaker_id for r in self.recordings(records, name)}

    def to_json(self) -> dict:
        return {
            "schema_version": SPLIT_SCHEMA,
            "split_id": self.split_id,
            "seed": self.seed,
            "params": self.params,
            "assignments": self.assignments,
            "excluded": self.excluded,
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, d: dict) -> Split:
        if d.get("schema_version") != SPLIT_SCHEMA:
            raise ValueError(f"unsupported split schema {d.get('schema_version')!r}")
        return cls(
            split_id=d["split_id"],
            assignments=dict(d["assignments"]),
            excluded=dict(d.get("excluded", {})),
            seed=int(d["seed"]),
            params=dict(d.get("params", {})),
            stats=dict(d.get("stats", {})),
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Split:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _stable_key(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def build_splits(
    records: Sequence[Recording],
    unseen_frac: float = 0.3,
    val_frac: float = 0.2,
    seed: int = 0,
    unseen_speakers: Iterable[str] | None = None,
) -> Split:
    """Speaker-disjoint split.

    * ``test_unseen``: every stream of speakers that never appear in training. Conversations
      containing an unseen speaker are kept out of training entirely; the *seen* partner's
      stream in such a conversation is excluded (it is neither unseen nor safely trainable).
    * ``val_seen``: held-out conversations of training speakers (same speakers, new audio),
      used for model selection and threshold tuning — and as the "seen" side of the
      generalisation gap.
    * ``train``: the rest.
    """
    speakers = sorted({r.speaker_id for r in records})
    if unseen_speakers is None:
        ordered = sorted(speakers, key=lambda s: _stable_key(s, seed))
        n_unseen = max(1, int(round(unseen_frac * len(speakers)))) if len(speakers) > 1 else 0
        unseen = set(ordered[:n_unseen])
    else:
        unseen = set(unseen_speakers)
        missing = unseen - set(speakers)
        if missing:
            raise ValueError(f"unseen speakers not in manifest: {sorted(missing)[:5]}")

    by_conv: dict[str, list[Recording]] = defaultdict(list)
    for r in records:
        by_conv[r.conversation_id].append(r)

    assignments: dict[str, str] = {}
    excluded: dict[str, str] = {}
    seen_convs: list[str] = []
    for conv_id, recs in by_conv.items():
        if any(r.speaker_id in unseen for r in recs):
            for r in recs:
                if r.speaker_id in unseen:
                    assignments[r.recording_id] = "test_unseen"
                else:
                    excluded[r.recording_id] = "seen speaker in a test conversation"
        else:
            seen_convs.append(conv_id)

    # hold out conversations per training speaker so every val_seen speaker is also in train
    conv_order = sorted(seen_convs, key=lambda c: _stable_key(c, seed + 1))
    n_val = int(round(val_frac * len(conv_order)))
    val_convs: set[str] = set()
    convs_per_speaker: dict[str, int] = defaultdict(int)
    for c in conv_order:
        for r in by_conv[c]:
            convs_per_speaker[r.speaker_id] += 1
    remaining = dict(convs_per_speaker)
    for c in conv_order:
        if len(val_convs) >= n_val:
            break
        spk = [r.speaker_id for r in by_conv[c]]
        if all(remaining[s] > 1 for s in spk):  # keep ≥1 training conversation per speaker
            val_convs.add(c)
            for s in spk:
                remaining[s] -= 1
    for c in seen_convs:
        for r in by_conv[c]:
            assignments[r.recording_id] = "val_seen" if c in val_convs else "train"

    train_spk = {r.speaker_id for r in records if assignments.get(r.recording_id) == "train"}
    test_spk = {r.speaker_id for r in records if assignments.get(r.recording_id) == "test_unseen"}
    val_spk = {r.speaker_id for r in records if assignments.get(r.recording_id) == "val_seen"}
    assert not (train_spk & test_spk), "speaker leak: test speaker in training"
    assert val_spk <= train_spk, "val_seen speaker missing from training"

    params = {
        "unseen_frac": unseen_frac,
        "val_frac": val_frac,
        "explicit_unseen": unseen_speakers is not None,
    }
    digest = hashlib.sha256(
        json.dumps({"a": sorted(assignments.items()), "p": params}, sort_keys=True).encode()
    ).hexdigest()[:12]
    split = Split(
        split_id=f"split-{digest}",
        assignments=assignments,
        excluded=excluded,
        seed=seed,
        params=params,
    )
    split.stats = split_stats(records, split)
    return split


def assert_speaker_disjoint(records: Sequence[Recording], split: Split) -> None:
    train = split.speakers(records, "train")
    test = split.speakers(records, "test_unseen")
    leak = train & test
    if leak:
        raise AssertionError(f"speaker leak between train and test_unseen: {sorted(leak)[:5]}")


def split_stats(records: Sequence[Recording], split: Split) -> dict:
    stats: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        recs = split.recordings(records, name)
        n_turns = sum(len(r.conversation_turns) for r in recs)
        n_holds = sum(max(len(t.ipus) - 1, 0) for r in recs for t in r.conversation_turns)
        stats[name] = {
            "speakers": len({r.speaker_id for r in recs}),
            "conversations": len({r.conversation_id for r in recs}),
            "recordings": len(recs),
            "turns": n_turns,
            "turn_ends": sum(1 for r in recs for t in r.conversation_turns if not t.interrupted),
            "within_turn_pauses": n_holds,
            "backchannels": sum(len(r.backchannels) for r in recs),
            "duration_s": round(sum((r.duration_ms or 0) for r in recs) / 1000.0, 1),
        }
    stats["excluded_recordings"] = len(split.excluded)
    return stats


# ---------------------------------------------------------------------------
# Feature/label preparation with an on-disk cache
# ---------------------------------------------------------------------------


@dataclass
class PreparedRecording:
    rec: Recording
    features: np.ndarray  # [T, D] float32 at 100 Hz
    labels: np.ndarray  # [T] int64


def prepare(
    records: Sequence[Recording],
    feature_cfg: FeatureConfig,
    cache_dir: Path | None = None,
    shift_horizon_ms: float = 2500.0,
) -> list[PreparedRecording]:
    fe = LogMelPitch(feature_cfg)
    cfg_hash = hashlib.sha256(
        json.dumps(feature_cfg.to_dict(), sort_keys=True).encode()
    ).hexdigest()[:10]
    out: list[PreparedRecording] = []
    for rec in records:
        feats = None
        cache = None
        if cache_dir is not None:
            stamp = Path(rec.audio_path).stat().st_mtime_ns
            cache = Path(cache_dir) / f"feat-{cfg_hash}" / f"{rec.recording_id}-{stamp}.npy"
            if cache.is_file():
                feats = np.load(cache)
        audio = None
        if feats is None:
            audio = load_audio(rec, feature_cfg.sample_rate)
            feats = fe.batch(audio)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, feats)
        needs_audio = any(not t.ipus for t in rec.conversation_turns)
        if needs_audio and audio is None:
            audio = load_audio(rec, feature_cfg.sample_rate)
        labels = frame_labels(rec, feats.shape[0], feature_cfg.hop_ms, shift_horizon_ms, audio)
        out.append(PreparedRecording(rec=rec, features=feats, labels=labels))
    return out
