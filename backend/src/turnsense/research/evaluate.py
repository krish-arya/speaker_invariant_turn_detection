"""Streaming evaluation of E0/E1/E2 on speaker-disjoint splits (goal.md §6–7).

Every detector is run exactly as the orchestrator runs it: 20 ms frames from
``FrameBuffer`` → ``EnergyVAD`` → ``detector.process`` → ``DecisionPolicy.update``, with
``detector.new_turn()`` after each committed TURN_COMPLETE. Offline there is no agent, so
``agent_speaking``/``agent_active`` are False and barge-in is not exercised here (it is
covered by the scripted scenarios in the live pipeline).

Scoring uses ``turnsense.metrics.score_turn_ends`` with a fixed ``Tolerance``. Confidence
intervals are percentile bootstraps over *speakers* (the unit of generalisation).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from turnsense.audio.frames import FrameBuffer
from turnsense.audio.vad import EnergyVAD
from turnsense.detectors.base import DetectorInput, TurnDetector
from turnsense.detectors.silence import SilenceBaselineDetector
from turnsense.events import DecisionType, DetectorMode, PolicyConfig, VadConfig
from turnsense.metrics import ReferenceTurn, Tolerance, percentiles, score_turn_ends
from turnsense.metrics.turn import DEFAULT_TOLERANCE
from turnsense.research.data import Recording, Split, load_audio, load_manifest

Log = Callable[[str], None]
SYNTHETIC_BANNER = (
    "SYNTHETIC DATA — pipeline/mechanism smoke test only. These numbers are not evidence "
    "about real speakers, real conversations, or the anti-transfer hypothesis."
)


@dataclass
class RecordingResult:
    recording_id: str
    speaker_id: str
    commits_ms: list[float]
    n_reference: int
    matched: int
    n_commits: int
    premature: int
    delays_ms: list[float]
    n_frames: int
    inference_ms: list[float] = field(repr=False, default_factory=list)


def stream_recording(
    audio: np.ndarray,
    detector: TurnDetector,
    policy_cfg: PolicyConfig,
    vad_cfg: VadConfig | None = None,
    sample_rate: int = 16_000,
    frame_ms: int = 20,
) -> tuple[list[float], list[float], int]:
    """Run one stream through VAD → detector → policy. Returns (commits, inference_ms, frames)."""
    from turnsense.policy import DecisionPolicy

    detector.reset()
    vad = EnergyVAD(vad_cfg)
    policy = DecisionPolicy(policy_cfg, frame_ms)
    commits: list[float] = []
    inference: list[float] = []
    frames = FrameBuffer(sample_rate=sample_rate, frame_ms=frame_ms).push(audio)
    for frame in frames:
        v = vad.process(frame)
        out = detector.process(DetectorInput(frame=frame, vad=v, agent_speaking=False))
        inference.append(out.inference_ms)
        dec = policy.update(out, v, agent_active=False)
        if dec is not None and dec.decision == DecisionType.TURN_COMPLETE:
            commits.append(dec.last_user_frame_ms)
            detector.new_turn()
    return commits, inference, len(frames)


def references(rec: Recording) -> list[ReferenceTurn]:
    return [
        ReferenceTurn(t.start_ms, t.end_ms) for t in rec.conversation_turns if not t.interrupted
    ]


def _f1(m: float, n_ref: float, n_com: float) -> float | None:
    if n_ref == 0 or n_com == 0:
        return None if n_ref == 0 else 0.0
    p, r = m / n_com, m / n_ref
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def summarise(results: Sequence[RecordingResult], n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled metrics + per-speaker table + speaker-bootstrap CIs."""
    by_spk: dict[str, list[RecordingResult]] = defaultdict(list)
    for r in results:
        by_spk[r.speaker_id].append(r)
    spk_ids = sorted(by_spk)
    rows = []
    counts = np.zeros((len(spk_ids), 4))  # matched, n_ref, n_commits, premature
    for i, s in enumerate(spk_ids):
        rs = by_spk[s]
        c = np.array([sum(r.matched for r in rs), sum(r.n_reference for r in rs),
                      sum(r.n_commits for r in rs), sum(r.premature for r in rs)], dtype=float)  # fmt: skip
        counts[i] = c
        delays = [d for r in rs for d in r.delays_ms]
        rows.append({
            "speaker_id": s,
            "recordings": len(rs),
            "reference_turns": int(c[1]),
            "commits": int(c[2]),
            "precision": c[0] / c[2] if c[2] else None,
            "recall": c[0] / c[1] if c[1] else None,
            "f1": _f1(*c[:3]),
            "premature_end_rate": c[3] / c[1] if c[1] else None,
            "delay_p50_ms": percentiles(delays, (50,))[50],
        })  # fmt: skip

    def pooled(c: np.ndarray) -> dict[str, float | None]:
        m, n_ref, n_com, prem = c.sum(axis=0)
        return {
            "precision": m / n_com if n_com else None,
            "recall": m / n_ref if n_ref else None,
            "f1": _f1(m, n_ref, n_com),
            "premature_end_rate": prem / n_ref if n_ref else None,
            "missed_end_rate": (n_ref - m) / n_ref if n_ref else None,
        }

    point = pooled(counts)
    ci: dict[str, list[float | None]] = {}
    if len(spk_ids) >= 2:
        rng = np.random.default_rng(seed)
        boots = defaultdict(list)
        for _ in range(n_boot):
            sample = counts[rng.integers(0, len(spk_ids), size=len(spk_ids))]
            for k, v in pooled(sample).items():
                if v is not None:
                    boots[k].append(v)
        for k in point:
            vals = boots.get(k, [])
            ci[k] = (
                [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
                if vals
                else [None, None]
            )
    delays = [d for r in results for d in r.delays_ms]
    inf = [x for r in results for x in r.inference_ms]
    q = percentiles(delays, (50, 95))
    iq = percentiles(inf, (50, 95, 99))
    return {
        **point,
        "ci95": ci,
        "delay_p50_ms": q[50],
        "delay_p95_ms": q[95],
        "n_speakers": len(spk_ids),
        "n_recordings": len(results),
        "n_reference_turns": int(counts[:, 1].sum()),
        "n_commits": int(counts[:, 2].sum()),
        "n_frames": int(sum(r.n_frames for r in results)),
        "inference_ms": {"p50": iq[50], "p95": iq[95], "p99": iq[99], "n": len(inf)},
        "per_speaker": rows,
        "_counts": counts.tolist(),
    }


def generalisation_gap(seen: dict, unseen: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """seen − unseen pooled F1 with an independent speaker bootstrap of each side."""
    a, b = np.array(seen["_counts"]), np.array(unseen["_counts"])
    point = None
    if seen["f1"] is not None and unseen["f1"] is not None:
        point = seen["f1"] - unseen["f1"]
    if len(a) < 2 or len(b) < 2 or point is None:
        return {"f1_gap": point, "ci95": [None, None]}
    rng = np.random.default_rng(seed + 1)
    diffs = []
    for _ in range(n_boot):
        sa = a[rng.integers(0, len(a), len(a))].sum(0)
        sb = b[rng.integers(0, len(b), len(b))].sum(0)
        fa, fb = _f1(*sa[:3]), _f1(*sb[:3])
        if fa is not None and fb is not None:
            diffs.append(fa - fb)
    return {
        "f1_gap": point,
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


DetectorFactory = Callable[[PolicyConfig], TurnDetector]


def default_factories(checkpoints: dict[str, Path]) -> dict[str, DetectorFactory]:
    from turnsense.detectors.learned import LearnedTurnDetector, read_checkpoint_metadata

    out: dict[str, DetectorFactory] = {"silence_baseline": SilenceBaselineDetector}
    for name, path in checkpoints.items():
        meta_mode = str(read_checkpoint_metadata(path).get("mode"))
        mode = (
            DetectorMode(meta_mode)
            if meta_mode in {m.value for m in DetectorMode}
            else DetectorMode.LEARNED_CONTROL
        )

        def make(policy: PolicyConfig, _p: Path = path, _m: DetectorMode = mode) -> TurnDetector:
            return LearnedTurnDetector(_m, _p, policy, allow_experimental=True)

        out[name] = make
    return out


def evaluate(
    records: Sequence[Recording],
    split: Split,
    factories: dict[str, DetectorFactory],
    policy_cfg: PolicyConfig | None = None,
    tolerance: Tolerance | None = None,
    split_names: Sequence[str] = ("val_seen", "test_unseen"),
    log: Log = print,
) -> dict:
    policy_cfg = policy_cfg or PolicyConfig()
    tol = tolerance or DEFAULT_TOLERANCE
    detectors = {name: f(policy_cfg) for name, f in factories.items()}
    raw: dict[str, dict[str, list[RecordingResult]]] = {n: defaultdict(list) for n in detectors}
    t0 = time.perf_counter()
    for split_name in split_names:
        recs = split.recordings(records, split_name)
        log(f"[eval] {split_name}: {len(recs)} recordings × {len(detectors)} detectors")
        for rec in recs:
            audio = load_audio(rec)
            refs = references(rec)
            for name, det in detectors.items():
                commits, inf, n_frames = stream_recording(audio, det, policy_cfg)
                s = score_turn_ends(refs, commits, tol)
                raw[name][split_name].append(RecordingResult(
                    recording_id=rec.recording_id, speaker_id=rec.speaker_id, commits_ms=commits,
                    n_reference=s.n_reference, matched=s.matched, n_commits=s.n_commits,
                    premature=s.premature_turns, delays_ms=s.delays_ms, n_frames=n_frames,
                    inference_ms=inf))  # fmt: skip
    results: dict[str, dict] = {}
    for name, det in detectors.items():
        per_split = {sn: summarise(raw[name][sn]) for sn in split_names}
        gap = None
        if "val_seen" in per_split and "test_unseen" in per_split:
            gap = generalisation_gap(per_split["val_seen"], per_split["test_unseen"])
        results[name] = {
            "detector": det.info(),
            "splits": per_split,
            "generalisation_gap": gap,
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split_id": split.split_id,
        "split_stats": split.stats,
        "policy": policy_cfg.model_dump(),
        "tolerance_ms": tol.as_dict(),
        "vad": "energy",
        "results": results,
        "eval_seconds": round(time.perf_counter() - t0, 1),
    }


def _fmt(v: float | None, pct: bool = False, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{100 * v:.1f}%" if pct else f"{v:.{digits}f}"


def _ci(ci: Sequence[float | None] | None, pct: bool = False) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" [{_fmt(ci[0], pct)}, {_fmt(ci[1], pct)}]"


def markdown_report(report: dict, synthetic: bool, extra_sections: Sequence[str] = ()) -> str:
    lines = ["# Turn-detection evaluation", ""]
    if synthetic:
        lines += [f"> **{SYNTHETIC_BANNER}**", ""]
    lines += [
        f"- Generated: {report['generated_at']}",
        f"- Split: `{report['split_id']}` (speaker-disjoint; test speakers never seen in training)",
        f"- Policy: `{report['policy']['version']}` — complete_threshold={report['policy']['complete_threshold']}, "
        f"consecutive_frames={report['policy']['consecutive_frames']}, silence_timeout_ms={report['policy']['silence_timeout_ms']}, "
        f"max_silence_ms={report['policy']['max_silence_ms']}",
        f"- Tolerance (ms): early={report['tolerance_ms']['early']}, late={report['tolerance_ms']['late']}",
        "- VAD: energy (synthetic signals do not trigger Silero)",
        "",
        "## Split statistics",
        "",
        "| split | speakers | conversations | turns | within-turn pauses | duration (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sn in ("train", "val_seen", "test_unseen"):
        s = report["split_stats"].get(sn, {})
        lines.append(f"| {sn} | {s.get('speakers', 0)} | {s.get('conversations', 0)} | {s.get('turns', 0)} | "
                     f"{s.get('within_turn_pauses', 0)} | {s.get('duration_s', 0)} |")  # fmt: skip
    lines += [
        "",
        "## Turn-completion quality (pooled; 95% CI = speaker bootstrap)",
        "",
        "| detector | model version | split | precision | recall | F1 | premature-end rate | missed-end rate | delay p50 / p95 (ms) | refs / commits |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in report["results"].items():
        for sn, s in r["splits"].items():
            ci = s.get("ci95", {})
            lines.append(
                f"| {name} | `{r['detector'].get('model_version')}` | {sn} | {_fmt(s['precision'])} | {_fmt(s['recall'])} | "
                f"{_fmt(s['f1'])}{_ci(ci.get('f1'))} | {_fmt(s['premature_end_rate'], True)}{_ci(ci.get('premature_end_rate'), True)} | "
                f"{_fmt(s['missed_end_rate'], True)} | {_fmt(s['delay_p50_ms'], digits=0)} / {_fmt(s['delay_p95_ms'], digits=0)} | "
                f"{s['n_reference_turns']} / {s['n_commits']} |"
            )
    lines += [
        "",
        "## Generalisation gap (F1 on val_seen − F1 on test_unseen; smaller is better)",
        "",
        "| detector | F1 gap | 95% CI |",
        "|---|---:|---|",
    ]
    for name, r in report["results"].items():
        g = r.get("generalisation_gap") or {}
        lines.append(f"| {name} | {_fmt(g.get('f1_gap'))} | {_ci(g.get('ci95')).strip() or '—'} |")
    lines += [
        "",
        "## Detector inference time per 20 ms frame (ms, includes feature extraction)",
        "",
        "| detector | p50 | p95 | p99 | frames |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in report["results"].items():
        inf = r["splits"].get("test_unseen", next(iter(r["splits"].values())))["inference_ms"]
        lines.append(
            f"| {name} | {_fmt(inf['p50'])} | {_fmt(inf['p95'])} | {_fmt(inf['p99'])} | {inf['n']} |"
        )
    lines += ["", *extra_sections]
    return "\n".join(lines) + "\n"


def write_report(
    report: dict, out_dir: Path, synthetic: bool, extra_sections: Sequence[str] = ()
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(report, default=str))
    for r in clean.get("results", {}).values():
        for s in r["splits"].values():
            s.pop("_counts", None)
    clean["SYNTHETIC"] = synthetic
    (out_dir / "evaluation.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    (out_dir / "evaluation.md").write_text(
        markdown_report(report, synthetic, extra_sections), encoding="utf-8"
    )


def load_and_evaluate(
    manifest: Path,
    split_path: Path,
    checkpoints: dict[str, Path],
    policy_cfg: PolicyConfig | None = None,
    log: Log = print,
) -> dict:
    records = load_manifest(manifest)
    split = Split.load(split_path)
    return evaluate(records, split, default_factories(checkpoints), policy_cfg, log=log)
