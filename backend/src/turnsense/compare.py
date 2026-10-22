"""Trial comparison and exports (frontend.md §6.8, goal.md §7).

Two aggregation rules, stated in every result:

* ``trial_mean`` — per-trial scores/rates averaged over trials, 95% t-interval across trials.
* ``pooled_samples`` — latency percentiles over all samples pooled across the group's trials,
  95% percentile-bootstrap interval; withheld below the minimum sample count.

Groups are never ranked. Each metric carries the direction in which it is better.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from turnsense import __version__
from turnsense.metrics.stats import bootstrap_ci95, mean_ci95, percentile
from turnsense.sim.scenarios import SCENARIO_SET_ID
from turnsense.trial import TrialRecord, now_iso


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    better: Literal["higher", "lower"]
    unit: Literal["ratio", "ms", "count"]
    definition: str
    aggregation: Literal["trial_mean", "pooled_samples"] = "trial_mean"
    sample_key: str | None = None  # latency_samples key for pooled metrics
    q: float | None = None


METRICS: list[MetricDef] = [
    MetricDef("turn_f1", "Turn F1", "higher", "ratio",
              "Harmonic mean of turn-completion precision and recall (tolerance window [end−early, end+late])."),
    MetricDef("turn_precision", "Turn precision", "higher", "ratio",
              "Committed turn ends that match an annotated end within the tolerance window."),
    MetricDef("turn_recall", "Turn recall", "higher", "ratio",
              "Annotated turn ends matched by a committed turn end within the tolerance window."),
    MetricDef("premature_end_rate", "Premature-end rate", "lower", "ratio",
              "Annotated turns during which the system committed completion earlier than end − early tolerance."),
    MetricDef("missed_end_rate", "Missed-end rate", "lower", "ratio",
              "Annotated turn ends with no committed completion inside the evaluation window."),
    MetricDef("turn_end_delay_p50_ms", "Turn-end delay p50", "lower", "ms",
              "Committed TURN_COMPLETE minus annotated turn end, matched pairs. Lower is better only if "
              "premature ends do not rise.", "pooled_samples", "turn_end_delay", 50),
    MetricDef("turn_end_delay_p95_ms", "Turn-end delay p95", "lower", "ms",
              "95th percentile of turn-end delay.", "pooled_samples", "turn_end_delay", 95),
    MetricDef("response_start_p50_ms", "Response start p50", "lower", "ms",
              "Last user frame used for the completion decision → first response audio delivered.",
              "pooled_samples", "response_start", 50),
    MetricDef("response_start_p95_ms", "Response start p95", "lower", "ms",
              "95th percentile of response-start latency.", "pooled_samples", "response_start", 95),
    MetricDef("response_start_p99_ms", "Response start p99", "lower", "ms",
              "99th percentile of response-start latency.", "pooled_samples", "response_start", 99),
    MetricDef("detector_inference_p50_ms", "Detector inference p50", "lower", "ms",
              "Wall time of one detector call per 20 ms frame.", "pooled_samples", "detector_inference", 50),
    MetricDef("detector_inference_p95_ms", "Detector inference p95", "lower", "ms",
              "95th percentile of detector inference time.", "pooled_samples", "detector_inference", 95),
    MetricDef("detector_inference_p99_ms", "Detector inference p99", "lower", "ms",
              "99th percentile of detector inference time.", "pooled_samples", "detector_inference", 99),
    MetricDef("false_barge_in_rate", "False barge-in rate", "lower", "ratio",
              "Agent output intervals cancelled without a genuine user interruption (e.g. by a backchannel)."),
    MetricDef("missed_barge_in_rate", "Missed barge-in rate", "lower", "ratio",
              "Genuine interruptions not acted on within the barge-in window."),
    MetricDef("barge_in_cancel_p50_ms", "Barge-in cancellation p50", "lower", "ms",
              "Committed BARGE_IN → cancellation acknowledged.", "pooled_samples", "barge_in_cancel", 50),
    MetricDef("stale_chunks_played", "Stale chunks played", "lower", "count",
              "Audio chunks of a cancelled response played after cancellation (target 0)."),
]  # fmt: skip

METRIC_BY_KEY = {m.key: m for m in METRICS}
GROUPABLE = (
    "detector_mode",
    "scenario_id",
    "model_version",
    "trial_type",
    "decision_policy_version",
)


def _summary(m: MetricDef, trials: list[TrialRecord], min_samples: int) -> dict[str, Any]:
    if m.aggregation == "pooled_samples":
        pooled = [v for t in trials for v in t.latency_samples.get(m.sample_key or "", [])]
        need = 1 if m.q == 50 else min_samples
        if len(pooled) < need:
            return {
                "mean": None,
                "ci95_low": None,
                "ci95_high": None,
                "n": len(pooled),
                "aggregation": m.aggregation,
            }
        q = m.q or 50
        value = percentile(pooled, q)
        lo, hi = (None, None)
        if len(pooled) >= min_samples:
            lo, hi = bootstrap_ci95(pooled, stat=lambda a: np.percentile(a, q), n_boot=1000)
        return {
            "mean": value,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": len(pooled),
            "aggregation": m.aggregation,
        }
    values = [getattr(t.metrics, m.key) for t in trials]
    mean, lo, hi, n = mean_ci95(values)
    # keep intervals inside the metric's range (rates are proportions, counts are non-negative)
    if lo is not None and hi is not None:
        lo = max(lo, 0.0)
        hi = min(hi, 1.0) if m.unit == "ratio" else hi
    return {"mean": mean, "ci95_low": lo, "ci95_high": hi, "n": n, "aggregation": m.aggregation}


def compare_trials(
    trials: list[TrialRecord], group_by: Iterable[str] = ("detector_mode",), min_samples: int = 20
) -> dict[str, Any]:
    group_by = [g for g in group_by if g in GROUPABLE] or ["detector_mode"]
    groups: dict[tuple[str, ...], list[TrialRecord]] = {}
    for t in trials:
        key = tuple(str(getattr(t, g)) for g in group_by)
        groups.setdefault(key, []).append(t)

    out_groups = []
    # like-for-like check: groups that differ only in detector/model must share (scenario, seed)
    inputs_by_bucket: dict[tuple[str, ...], set[frozenset[tuple[str, str]]]] = {}
    for key, members in sorted(groups.items()):
        scenario_ids = sorted({t.scenario_id for t in members})
        bucket = tuple(
            v
            for g, v in zip(group_by, key, strict=True)
            if g not in ("detector_mode", "model_version")
        )
        inputs = frozenset((t.scenario_id, str(t.effective_config.seed)) for t in members)
        inputs_by_bucket.setdefault(bucket, set()).add(inputs)
        out_groups.append(
            {
                "key": dict(zip(group_by, key, strict=True)),
                "n_trials": len(members),
                "scenario_ids": scenario_ids,
                "trial_types": sorted({t.trial_type.value for t in members}),
                "model_versions": sorted({t.model_version for t in members}),
                "metrics": {m.key: _summary(m, members, min_samples) for m in METRICS},
            }
        )

    warnings: list[str] = []
    if any(len(variants) > 1 for variants in inputs_by_bucket.values()):
        warnings.append(
            "Detector groups were run on different scenario/seed sets; the comparison is not like-for-like."
        )
    if any(g["n_trials"] < 3 for g in out_groups):
        warnings.append(
            "Some groups have fewer than 3 trials; confidence intervals are wide or absent."
        )
    types = {t.trial_type.value for t in trials}
    if len(types) > 1:
        warnings.append("Simulated and live trials are mixed; report them separately.")
    if any(t.effective_config.runtime_profile == "simulation" for t in trials):
        warnings.append(
            "Simulated trials use synthetic voices and modelled provider latency; they test the "
            "pipeline, not real-speaker generalisation."
        )
    if any(not t.labelled for t in trials):
        warnings.append(
            "Unlabelled trials contribute latency only; task metrics use labelled trials."
        )

    if types == {"simulated"}:
        evaluation_set = SCENARIO_SET_ID
    elif types == {"replay"}:  # "manifest:<file>/<split_id>", without the recording id
        sources = {
            "/".join((t.annotations.source if t.annotations else "").split("/")[:2]) for t in trials
        }
        evaluation_set = "replay:" + ",".join(sorted(sources))
    elif types == {"live"}:
        evaluation_set = "live-unlabelled"
    else:
        evaluation_set = "mixed"
    return {
        "group_by": group_by,
        "groups": out_groups,
        "metric_definitions": {
            m.key: {
                "label": m.label,
                "better": m.better,
                "unit": m.unit,
                "definition": m.definition,
                "aggregation": m.aggregation,
            }
            for m in METRICS
        },
        "evaluation_set": evaluation_set,
        "min_percentile_samples": min_samples,
        "warnings": warnings,
        "generated_at": now_iso(),
        "server_version": __version__,
    }


# --------------------------------------------------------------------------- exports


def export_meta(trials: list[TrialRecord]) -> dict[str, Any]:
    return {
        "server_version": __version__,
        "exported_at": now_iso(),
        "transcripts_included": any(t.effective_config.transcript_export_enabled for t in trials),
        "raw_audio_included": False,
        "retention_note": "Raw audio is never stored. Transcripts are included only for trials run "
        "with transcript export enabled.",
    }


def trial_json(record: TrialRecord) -> dict[str, Any]:
    return {"export_meta": export_meta([record]), **record.model_dump(mode="json")}


_EVENT_COLUMNS = [
    "trial_id", "session_id", "detector_mode", "model_version", "decision_policy_version",
    "config_hash", "seq", "kind", "t_ms", "wall_time", "generation_id", "state", "event_type",
    "decision", "confidence", "stage", "duration_ms", "metric", "value_ms", "status", "reason",
    "payload_json",
]  # fmt: skip


def trial_events_csv(record: TrialRecord) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_EVENT_COLUMNS, extrasaction="ignore")
    w.writeheader()
    cfg = record.effective_config
    for env in record.events:
        p = env.payload
        reason = p.get("reason")
        if reason is None and p.get("reason_codes"):
            reason = ";".join(p["reason_codes"])
        w.writerow(
            {
                "trial_id": record.trial_id,
                "session_id": record.session_id,
                "detector_mode": record.detector_mode,
                "model_version": record.model_version,
                "decision_policy_version": record.decision_policy_version,
                "config_hash": cfg.config_hash,
                "seq": env.seq,
                "kind": env.kind.value,
                "t_ms": env.t_ms,
                "wall_time": env.wall_time,
                "generation_id": env.generation_id,
                "state": p.get("state"),
                "event_type": p.get("event_type"),
                "decision": p.get("decision"),
                "confidence": p.get("confidence"),
                "stage": p.get("stage"),
                "duration_ms": p.get("duration_ms"),
                "metric": p.get("metric"),
                "value_ms": p.get("value_ms"),
                "status": p.get("status"),
                "reason": reason,
                "payload_json": json.dumps(p, separators=(",", ":")),
            }
        )
    return buf.getvalue()


def _summary_row(t: TrialRecord) -> dict[str, Any]:
    cfg = t.effective_config
    row: dict[str, Any] = {k: v for k, v in t.list_item().items() if k not in ("metrics", "counts")}
    row.update({f"metric_{k}": v for k, v in t.metrics.model_dump().items()})
    row.update({f"count_{k}": v for k, v in t.counts.model_dump().items()})
    row.update({f"policy_{k}": v for k, v in cfg.policy.model_dump().items()})
    row.update(
        {
            "config_hash": cfg.config_hash,
            "seed": cfg.seed,
            "runtime_profile": cfg.runtime_profile,
            "vad": cfg.vad.kind,
            "provider_stt": cfg.providers.stt,
            "provider_llm": cfg.providers.llm,
            "provider_tts": cfg.providers.tts,
            "server_version": cfg.server_version,
            "tolerance_ms": json.dumps(t.tolerance_ms) if t.tolerance_ms else None,
            "notes": " | ".join(t.notes),
        }
    )
    return row


def trials_summary_json(trials: list[TrialRecord]) -> dict[str, Any]:
    return {"export_meta": export_meta(trials), "trials": [_summary_row(t) for t in trials]}


def trials_summary_csv(trials: list[TrialRecord]) -> str:
    rows = [_summary_row(t) for t in trials]
    buf = io.StringIO()
    if rows:
        fields = list(rows[0].keys())
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


def comparison_csv(result: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            *result["group_by"],
            "n_trials",
            "metric",
            "better",
            "aggregation",
            "value",
            "ci95_low",
            "ci95_high",
            "n",
        ]
    )
    for g in result["groups"]:
        key = [g["key"][k] for k in result["group_by"]]
        for metric, s in g["metrics"].items():
            d = result["metric_definitions"][metric]
            w.writerow(
                [
                    *key,
                    g["n_trials"],
                    metric,
                    d["better"],
                    s["aggregation"],
                    s["mean"],
                    s["ci95_low"],
                    s["ci95_high"],
                    s["n"],
                ]
            )
    return buf.getvalue()


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "ms":
        return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}" if abs(v) >= 1 else f"{v:.4f}"
    return f"{v:.3f}" if unit == "ratio" else f"{v:g}"


def comparison_markdown(result: dict[str, Any]) -> str:
    gb = result["group_by"]
    names = [" / ".join(str(g["key"][k]) for k in gb) for g in result["groups"]]
    lines = [
        "# Turn-detection comparison",
        "",
        f"- Evaluation set: `{result['evaluation_set']}`",
        f"- Grouped by: {', '.join(gb)}",
        f"- Generated: {result['generated_at']} (server {result['server_version']})",
        f"- Percentiles need ≥ {result['min_percentile_samples']} pooled samples (p50 needs ≥ 1).",
        "",
    ]
    if result["warnings"]:
        lines += ["**Warnings**", "", *[f"- {w}" for w in result["warnings"]], ""]
    lines += [
        "| Metric | Better | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for key, d in result["metric_definitions"].items():
        cells = []
        for g in result["groups"]:
            s = g["metrics"][key]
            v = _fmt(s["mean"], d["unit"])
            if s["ci95_low"] is not None and s["ci95_high"] is not None:
                v += f" [{_fmt(s['ci95_low'], d['unit'])}, {_fmt(s['ci95_high'], d['unit'])}]"
            cells.append(f"{v} (n={s['n']})")
        arrow = "↑" if d["better"] == "higher" else "↓"
        lines.append(f"| {d['label']} | {arrow} {d['better']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Trial counts: "
        + ", ".join(f"{n}: {g['n_trials']}" for n, g in zip(names, result["groups"], strict=True)),
        "",
        "Values in brackets are 95% intervals (t-interval across trials for rates; bootstrap over pooled "
        "samples for latency percentiles). Groups are not ranked; read each metric with its direction "
        "and alongside premature-end rate.",
        "",
    ]
    return "\n".join(lines)
