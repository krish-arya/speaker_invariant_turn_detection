import pytest

from turnsense.metrics import (
    ReferenceTurn,
    Tolerance,
    mean_ci95,
    percentiles,
    score_barge_ins,
    score_turn_ends,
)

TOL = Tolerance(early_ms=100, late_ms=2000, barge_in_ms=800)


def test_perfect_matching_and_delays():
    refs = [ReferenceTurn(0, 1000), ReferenceTurn(3000, 4000)]
    s = score_turn_ends(refs, [1700, 4300], TOL)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)
    assert s.delays_ms == [700, 300] and s.delay_p50_ms == 500
    assert s.premature_end_rate == 0 and s.missed_end_rate == 0


def test_premature_commit_counts_against_precision_and_premature_rate():
    refs = [ReferenceTurn(0, 2000)]
    s = score_turn_ends(refs, [900, 2600], TOL)  # committed during a within-turn pause
    assert s.matched == 1 and s.precision == 0.5 and s.recall == 1.0
    assert s.premature_turns == 1 and s.premature_end_rate == 1.0


def test_missed_and_late_commits():
    refs = [ReferenceTurn(0, 1000)]
    s = score_turn_ends(refs, [3500], TOL)  # later than end + late tolerance
    assert s.matched == 0 and s.missed_end_rate == 1.0 and s.f1 == 0.0
    s = score_turn_ends(refs, [], TOL)
    assert s.precision is None and s.recall == 0.0 and s.f1 is None


def test_commit_within_early_tolerance_matches_with_negative_delay():
    s = score_turn_ends([ReferenceTurn(0, 1000)], [950], TOL)
    assert s.matched == 1 and s.delays_ms == [-50] and s.premature_turns == 0


def test_one_commit_cannot_match_two_references():
    refs = [ReferenceTurn(0, 1000), ReferenceTurn(1100, 1500)]
    s = score_turn_ends(refs, [1600], TOL)
    assert s.matched == 1


def test_barge_in_scoring():
    intervals = [(0, 5000), (6000, 9000)]
    b = score_barge_ins(intervals, cancel_commits_ms=[2300, 7000], reference_onsets_ms=[2000], tol=TOL)
    assert b.false_barge_ins == 1 and b.false_barge_in_rate == 0.5
    assert b.missed_barge_ins == 0 and b.reaction_ms == [300]
    b = score_barge_ins(intervals, cancel_commits_ms=[], reference_onsets_ms=[2000], tol=TOL)
    assert b.missed_barge_in_rate == 1.0


def test_percentiles_respect_minimum_samples_and_ci():
    assert percentiles([1, 2, 3], min_samples=20) == {50: None, 95: None, 99: None}
    assert percentiles(list(range(1, 101)), min_samples=20)[50] == pytest.approx(50.5)
    mean, lo, hi, n = mean_ci95([0.5, 0.6, None, 0.7])
    assert n == 3 and mean == pytest.approx(0.6) and lo < 0.6 < hi
    assert mean_ci95([0.5])[1:3] == (None, None)
    assert mean_ci95([None]) == (None, None, None, 0)


def test_interruption_as_agent_finishes_is_not_a_missed_barge_in():
    # the user starts at 4850; the agent's answer ends by itself at 5000 (< 800 ms window)
    b = score_barge_ins([(0, 5000)], cancel_commits_ms=[], reference_onsets_ms=[4850], tol=TOL)
    assert b.yielded_naturally == 1 and b.missed_barge_ins == 0 and b.missed_barge_in_rate is None
    # still talking when the window closes → genuinely missed
    b = score_barge_ins([(0, 9000)], cancel_commits_ms=[], reference_onsets_ms=[4850], tol=TOL)
    assert b.missed_barge_ins == 1 and b.missed_barge_in_rate == 1.0
    # an interval still open at session end counts as still talking
    b = score_barge_ins([(0, None)], cancel_commits_ms=[], reference_onsets_ms=[4850], tol=TOL)
    assert b.missed_barge_ins == 1
