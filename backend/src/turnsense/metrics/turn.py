"""Turn-taking task metrics (goal.md §7). One implementation for live trials and offline eval.

Definitions (times in ms on one clock):

* A reference turn has ``start_ms`` and ``end_ms``; ``end_ms`` is the annotated
  conversational turn end (not every pause).
* A *commit* is the time the system committed TURN_COMPLETE.
* The evaluation window for a reference end ``r`` is ``[r - early_ms, r + late_ms]``.
* Commits are matched one-to-one to reference ends, earliest first.
* **premature end**: a commit inside the turn, earlier than ``r - early_ms``
  (counted per reference turn: did *any* premature commit happen during it?).
* **missed end**: a reference end with no matched commit.
* **endpoint delay**: ``commit - r`` for matched pairs (may be slightly negative).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from turnsense.metrics.stats import percentile


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Protocol parameters — fix before the final test run (goal.md §7)."""

    early_ms: float = 100.0
    late_ms: float = 2000.0
    barge_in_ms: float = 800.0

    def as_dict(self) -> dict[str, float]:
        return {"early": self.early_ms, "late": self.late_ms, "barge_in": self.barge_in_ms}


DEFAULT_TOLERANCE = Tolerance()


@dataclass(frozen=True, slots=True)
class ReferenceTurn:
    start_ms: float
    end_ms: float


@dataclass
class TurnScores:
    n_reference: int
    n_commits: int
    matched: int
    precision: float | None
    recall: float | None
    f1: float | None
    premature_turns: int
    premature_end_rate: float | None
    missed_end_rate: float | None
    delays_ms: list[float] = field(default_factory=list)
    delay_p50_ms: float | None = None
    delay_p95_ms: float | None = None


def score_turn_ends(
    references: Sequence[ReferenceTurn],
    commits_ms: Sequence[float],
    tol: Tolerance = DEFAULT_TOLERANCE,
) -> TurnScores:
    refs = sorted(references, key=lambda r: r.end_ms)
    commits = sorted(commits_ms)
    used = [False] * len(commits)
    delays: list[float] = []
    matched = 0
    for ref in refs:
        lo, hi = ref.end_ms - tol.early_ms, ref.end_ms + tol.late_ms
        for i, c in enumerate(commits):
            if not used[i] and lo <= c <= hi:
                used[i] = True
                matched += 1
                delays.append(c - ref.end_ms)
                break

    premature = 0
    for ref in refs:
        if any(ref.start_ms < c < ref.end_ms - tol.early_ms for c in commits):
            premature += 1

    n_ref, n_com = len(refs), len(commits)
    precision = matched / n_com if n_com else None
    recall = matched / n_ref if n_ref else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return TurnScores(
        n_reference=n_ref,
        n_commits=n_com,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        premature_turns=premature,
        premature_end_rate=premature / n_ref if n_ref else None,
        missed_end_rate=(n_ref - matched) / n_ref if n_ref else None,
        delays_ms=delays,
        delay_p50_ms=percentile(delays, 50),
        delay_p95_ms=percentile(delays, 95),
    )


@dataclass
class BargeInScores:
    agent_intervals: int
    false_barge_ins: int
    false_barge_in_rate: float | None
    reference_barge_ins: int
    missed_barge_ins: int
    missed_barge_in_rate: float | None
    reaction_ms: list[float] = field(default_factory=list)


def score_barge_ins(
    agent_intervals: Sequence[tuple[float, float]],
    cancel_commits_ms: Sequence[float],
    reference_onsets_ms: Sequence[float],
    tol: Tolerance = DEFAULT_TOLERANCE,
) -> BargeInScores:
    """Score interruption handling.

    ``agent_intervals``: (start, end) of each agent output interval (generation→playback end
    or cancellation). ``cancel_commits_ms``: committed BARGE_IN times. ``reference_onsets_ms``:
    annotated onsets of genuine user interruptions that fell inside an agent interval.

    A cancel is *false* if no reference onset lies in ``[cancel - barge_in_ms, cancel]``.
    A reference onset is *missed* if no cancel lies in ``[onset, onset + barge_in_ms]``.
    """
    cancels = sorted(cancel_commits_ms)
    onsets = sorted(reference_onsets_ms)
    false = 0
    for c in cancels:
        if not any(c - tol.barge_in_ms <= o <= c for o in onsets):
            false += 1
    missed = 0
    reactions: list[float] = []
    for o in onsets:
        hits = [c for c in cancels if o <= c <= o + tol.barge_in_ms]
        if hits:
            reactions.append(hits[0] - o)
        else:
            missed += 1
    n_int = len(agent_intervals)
    return BargeInScores(
        agent_intervals=n_int,
        false_barge_ins=false,
        false_barge_in_rate=false / n_int if n_int else None,
        reference_barge_ins=len(onsets),
        missed_barge_ins=missed,
        missed_barge_in_rate=missed / len(onsets) if onsets else None,
        reaction_ms=reactions,
    )
