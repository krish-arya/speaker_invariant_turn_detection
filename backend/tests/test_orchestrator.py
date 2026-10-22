"""End-to-end orchestrator behaviour on scripted scenarios, run in virtual time."""

import asyncio

import pytest

from turnsense.adapters.base import LLMAdapter, ProviderError
from turnsense.events import DecisionType, EnvelopeKind, ErrorSource, SessionState
from turnsense.sim.runner import SimulatedSession
from turnsense.sim.scenarios import build_scenario
from turnsense.sim.vtime import run_in_virtual_time

from .conftest import baseline_detector, make_info


def run(scenario_id, seed=1, tmp_path=None, mutate=None, during=None, **info_kw):
    info = make_info(scenario_id, seed, **info_kw)
    sim = SimulatedSession(info, build_scenario(scenario_id, seed), baseline_detector(info, tmp_path), None)

    async def go():
        sim.setup()
        if mutate:
            mutate(sim)
        bg = asyncio.create_task(during(sim)) if during else None
        rec = await sim.run()
        if bg:
            bg.cancel()
        return rec

    return sim, run_in_virtual_time(go)


def kinds(rec, kind):
    return [e for e in rec.events if e.kind == kind]


def states(rec):
    return [e.payload["state"] for e in kinds(rec, EnvelopeKind.SESSION_STATUS)]


def test_same_seed_gives_identical_timeline(tmp_path):
    def timeline(rec):
        return [
            (e.kind.value, e.t_ms, e.payload.get("state") or e.payload.get("decision"))
            for e in rec.events
            if e.kind in (EnvelopeKind.SESSION_STATUS, EnvelopeKind.TURN_DECISION, EnvelopeKind.VAD)
        ]

    _, a = run("pause_then_continue", 4, tmp_path)
    _, b = run("pause_then_continue", 4, tmp_path)
    assert timeline(a) == timeline(b)


def test_envelope_sequence_is_strictly_increasing_and_starts_with_hello(tmp_path):
    _, rec = run("short_answer", 2, tmp_path)
    seqs = [e.seq for e in rec.events]
    assert rec.events[0].kind == EnvelopeKind.HELLO
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert rec.events[-1].kind == EnvelopeKind.SESSION_STATUS and rec.events[-1].payload["state"] == "ended"
    assert kinds(rec, EnvelopeKind.TRIAL_SUMMARY)


def test_short_answers_are_answered_and_scored(tmp_path):
    _, rec = run("short_answer", 2, tmp_path)
    assert rec.status == "completed" and rec.labelled
    assert rec.metrics.turn_f1 == 1.0 and rec.metrics.premature_end_rate == 0.0
    assert rec.counts.responses_started == 3 == rec.counts.agent_speaking_intervals
    assert rec.metrics.response_start_p50_ms is not None and rec.metrics.response_start_p95_ms is None
    assert rec.metrics.frame_trace_coverage == 1.0
    assert "turn_end_delay" in rec.latency_samples and len(rec.latency_samples["turn_end_delay"]) == 3
    st = states(rec)
    assert st[:4] == ["connecting", "listening", "user_speaking", "possible_end"]
    assert "generating" in st and "agent_speaking" in st


def test_predictions_are_distinct_from_committed_decisions(tmp_path):
    _, rec = run("short_answer", 2, tmp_path)
    preds = kinds(rec, EnvelopeKind.TURN_PREDICTION)
    decisions = kinds(rec, EnvelopeKind.TURN_DECISION)
    assert {p.payload["event_type"] for p in preds} >= {"CONTINUE", "POSSIBLE_END", "TURN_COMPLETE"}
    assert all(p.payload["confidence"] is None for p in preds)  # deterministic baseline
    assert all(d.payload["decision"] == "TURN_COMPLETE" for d in decisions)
    for d in decisions:  # every decision is traceable to a prediction and an input frame
        assert d.payload["source_event_sequence"] in {p.payload["sequence"] for p in preds}
        assert d.payload["last_user_frame_ms"] <= d.payload["committed_ms"] + 1e-6


def test_barge_in_cancels_playback_and_truncates_history(tmp_path):
    sim, rec = run("barge_in", 3, tmp_path)
    decisions = [d.payload["decision"] for d in kinds(rec, EnvelopeKind.TURN_DECISION)]
    assert "BARGE_IN" in decisions
    flushes = [e for e in kinds(rec, EnvelopeKind.PLAYBACK) if e.payload["action"] == "flush"]
    assert flushes and flushes[0].generation_id == 1
    st = states(rec)
    i = st.index("interrupted")
    assert st[i - 1] == "agent_speaking" and st[i + 1] == "user_speaking"
    assert rec.metrics.stale_chunks_played == 0
    assert rec.metrics.missed_barge_in_rate == 0.0 and rec.metrics.false_barge_in_rate == 0.0
    history = sim.orchestrator.history
    assert history[1].role == "assistant" and history[1].content.endswith("…")
    # nothing from generation 1 starts playing after its flush
    t_flush = flushes[0].t_ms
    late = [e for e in rec.events if e.generation_id == 1 and e.t_ms > t_flush and e.kind == EnvelopeKind.PLAYBACK and e.payload["action"] == "start"]
    assert not late
    assert rec.latency_samples["barge_in_cancel"]


def test_late_chunk_from_cancelled_generation_is_rejected(tmp_path):
    async def inject(sim):
        orch = sim.orchestrator
        while not any(d.decision == DecisionType.BARGE_IN for d in orch.decisions):
            await asyncio.sleep(0.02)
        import numpy as np

        assert not orch.playback.put(1, np.zeros(320, np.float32), 16_000)

    _, rec = run("barge_in", 3, tmp_path, during=inject)
    assert rec.metrics.stale_chunks_rejected >= 1 and rec.metrics.stale_chunks_played == 0
    assert any(e.payload.get("stage") == "stale_chunk_rejected" for e in kinds(rec, EnvelopeKind.TRACE))


def test_baseline_false_barge_in_on_backchannel_is_reported(tmp_path):
    _, rec = run("backchannel", 0, tmp_path)
    assert rec.annotations.backchannels_ms
    assert rec.metrics.false_barge_in_rate is not None and rec.metrics.false_barge_in_rate > 0


def test_transcripts_hidden_when_disabled_and_not_exported_by_default(tmp_path):
    sim, rec = run("short_answer", 1, tmp_path, transcript_enabled=False)
    assert not kinds(rec, EnvelopeKind.TRANSCRIPT)
    sim, rec = run("short_answer", 1, tmp_path, transcript_enabled=True, transcript_export_enabled=False)
    assert not kinds(rec, EnvelopeKind.TRANSCRIPT)  # shown live, not retained
    sim, rec = run("short_answer", 1, tmp_path, transcript_enabled=True, transcript_export_enabled=True)
    tr = kinds(rec, EnvelopeKind.TRANSCRIPT)
    assert {e.payload["role"] for e in tr} == {"user", "assistant"}
    assert not kinds(rec, EnvelopeKind.AUDIO_LEVEL)  # never stored


class FailingLLM(LLMAdapter):
    name = "failing_llm"

    async def stream(self, messages):
        await asyncio.sleep(0.1)
        raise ProviderError(ErrorSource.LLM, "rate_limited", "LLM provider rate limit reached")
        yield ""  # pragma: no cover


class HangingLLM(LLMAdapter):
    name = "hanging_llm"

    async def stream(self, messages):
        await asyncio.sleep(3600)
        yield ""  # pragma: no cover


def _swap_llm(llm):
    def mutate(sim):
        sim.orchestrator.llm = llm

    return mutate


def test_provider_failure_is_a_recoverable_error_not_a_turn_event(tmp_path):
    _, rec = run("short_answer", 1, tmp_path, mutate=_swap_llm(FailingLLM()))
    errors = kinds(rec, EnvelopeKind.ERROR)
    assert errors and all(e.payload["source"] == "llm" and e.payload["recoverable"] for e in errors)
    assert errors[0].payload["code"] == "rate_limited"
    st = states(rec)
    assert "error" in st and st[st.index("error") + 1] == "listening"
    # the number of committed turn ends equals the user's turns: errors created none
    assert rec.counts.committed_turn_ends == 3 and rec.metrics.turn_f1 == 1.0
    assert rec.counts.errors == 3 and rec.counts.agent_speaking_intervals == 0
    assert rec.metrics.response_start_p50_ms is None


def test_hanging_provider_times_out_and_session_stays_usable(tmp_path):
    from turnsense.orchestrator.session import OrchestratorOptions

    def mutate(sim):
        sim.orchestrator.llm = HangingLLM()
        sim.orchestrator.opts = OrchestratorOptions(llm_timeout_s=2.0)

    _, rec = run("short_answer", 1, tmp_path, mutate=mutate)
    codes = [e.payload["code"] for e in kinds(rec, EnvelopeKind.ERROR)]
    assert codes and set(codes) == {"timeout"}
    assert rec.status == "completed" and states(rec)[-1] == "ended"


def test_manual_interrupt_uses_barge_in_path(tmp_path):
    async def press_interrupt(sim):
        orch = sim.orchestrator
        while orch.state != SessionState.AGENT_SPEAKING:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.5)
        assert await sim.interrupt()

    _, rec = run("short_answer", 1, tmp_path, during=press_interrupt)
    manual = [d for d in kinds(rec, EnvelopeKind.TURN_DECISION) if "manual_interrupt" in d.payload["reason_codes"]]
    assert len(manual) == 1 and manual[0].payload["decision"] == "BARGE_IN"
    assert "Manual interrupts are excluded from barge-in scoring." in rec.notes
    assert rec.metrics.false_barge_in_rate == 0.0


@pytest.mark.parametrize("scenario_id", ["pause_then_continue", "multi_turn", "barge_in", "backchannel"])
def test_all_scenarios_complete_with_no_stale_playback(scenario_id, tmp_path):
    _, rec = run(scenario_id, 5, tmp_path)
    assert rec.status == "completed" and rec.metrics.stale_chunks_played == 0
    assert rec.metrics.turn_recall is not None and rec.counts.reference_turn_ends >= 2


def _with_prefill(sim):
    from dataclasses import replace

    sim.orchestrator.opts = replace(sim.orchestrator.opts, speculative_prefill=True)


def _stages(rec):
    return [e.payload["stage"] for e in kinds(rec, EnvelopeKind.TRACE)]


def test_speculative_prefill_is_reused_and_lowers_response_latency(tmp_path):
    _, base = run("short_answer", 2, tmp_path)
    _, spec = run("short_answer", 2, tmp_path, mutate=_with_prefill)
    assert "speculation_used" not in _stages(base)
    assert _stages(spec).count("speculation_used") == 3
    assert spec.metrics.turn_f1 == base.metrics.turn_f1  # turn decisions are unaffected
    assert spec.metrics.response_start_p50_ms < base.metrics.response_start_p50_ms


def test_speculation_is_discarded_when_the_user_keeps_talking(tmp_path):
    # a long silence timeout guarantees a POSSIBLE_END during each within-turn pause
    _, rec = run(
        "pause_then_continue", 2, tmp_path, mutate=_with_prefill, policy_overrides={"silence_timeout_ms": 1500}
    )
    discarded = [e.payload for e in kinds(rec, EnvelopeKind.TRACE) if e.payload["stage"] == "speculation_discarded"]
    assert discarded and {d["detail"]["reason"] for d in discarded} <= {"speech_resumed", "transcript_changed"}
    assert rec.metrics.premature_end_rate == 0.0 and rec.metrics.stale_chunks_played == 0
