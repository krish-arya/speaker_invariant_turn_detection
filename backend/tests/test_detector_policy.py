from turnsense.detectors import DetectorInput, SilenceBaselineDetector, map_turn_end_probability
from turnsense.detectors.base import DetectorOutput
from turnsense.events import DecisionType, PolicyConfig, TurnEventType
from turnsense.policy import DecisionPolicy

from .conftest import frame, vad_result

POLICY = PolicyConfig()


def run_baseline(vads, agent_speaking=False):
    det = SilenceBaselineDetector(POLICY)
    return [det.process(DetectorInput(frame(i + 1), v, agent_speaking)) for i, v in enumerate(vads)]


def test_baseline_continue_possible_end_complete_and_no_confidence():
    vads = [vad_result(1, True, 20)] + [vad_result(i, False, silence_run_ms=(i - 1) * 20) for i in range(2, 40)]
    outs = run_baseline(vads)
    kinds = [o.event_type for o in outs]
    assert kinds[0] == TurnEventType.CONTINUE
    first_possible = kinds.index(TurnEventType.POSSIBLE_END)
    first_complete = kinds.index(TurnEventType.TURN_COMPLETE)
    assert vads[first_possible].silence_run_ms >= POLICY.possible_end_ms
    assert vads[first_complete].silence_run_ms >= POLICY.silence_timeout_ms
    assert vads[first_complete - 1].silence_run_ms < POLICY.silence_timeout_ms
    assert all(o.confidence is None and o.p_turn_end is None for o in outs)
    assert all(o.inference_ms >= 0 for o in outs)


def test_baseline_never_completes_without_user_speech():
    outs = run_baseline([vad_result(i, False, silence_run_ms=5000) for i in range(1, 10)])
    assert {o.event_type for o in outs} == {TurnEventType.CONTINUE}


def test_overlap_speech_barge_in_vs_backchannel():
    det = SilenceBaselineDetector(POLICY)
    # 120 ms burst then silence → BACKCHANNEL
    burst = [vad_result(i, True, speech_run_ms=i * 20) for i in range(1, 7)] + [vad_result(7, False)]
    outs = [det.process(DetectorInput(frame(i + 1), v, True)) for i, v in enumerate(burst)]
    assert outs[-1].event_type == TurnEventType.BACKCHANNEL
    assert TurnEventType.BARGE_IN not in {o.event_type for o in outs}
    # sustained speech → BARGE_IN once the run reaches the threshold
    det = SilenceBaselineDetector(POLICY)
    sustained = [vad_result(i, True, speech_run_ms=i * 20) for i in range(1, 15)]
    outs = [det.process(DetectorInput(frame(i + 1), v, True)) for i, v in enumerate(sustained)]
    first = next(i for i, o in enumerate(outs) if o.event_type == TurnEventType.BARGE_IN)
    assert sustained[first].speech_run_ms >= POLICY.barge_in_min_speech_ms


def _out(kind, p=None, end_ms=0.0, reasons=()):
    return DetectorOutput(kind, p, p, reasons, 0.01, 1, end_ms)


def test_policy_commits_deterministic_turn_complete_immediately():
    pol = DecisionPolicy(POLICY)
    for i in range(1, 6):  # 100 ms of speech
        assert pol.update(_out(TurnEventType.CONTINUE), vad_result(i, True), agent_active=False) is None
    d = pol.update(_out(TurnEventType.TURN_COMPLETE, end_ms=900), vad_result(40, False, silence_run_ms=700), agent_active=False)
    assert d is not None and d.decision == DecisionType.TURN_COMPLETE and d.last_user_frame_ms == 900
    assert pol.turn_speech_ms == 0  # new turn after commit


def test_policy_requires_consecutive_frames_for_probabilistic_detectors():
    pol = DecisionPolicy(POLICY)
    for i in range(1, 6):
        pol.update(_out(TurnEventType.CONTINUE, 0.1), vad_result(i, True), agent_active=False)
    silent = vad_result(20, False, silence_run_ms=200)
    results = [pol.update(_out(TurnEventType.TURN_COMPLETE, 0.9), silent, agent_active=False) for _ in range(3)]
    assert results[:2] == [None, None] and results[2] is not None
    # an interruption of the streak resets it
    pol = DecisionPolicy(POLICY)
    for i in range(1, 6):
        pol.update(_out(TurnEventType.CONTINUE, 0.1), vad_result(i, True), agent_active=False)
    pol.update(_out(TurnEventType.TURN_COMPLETE, 0.9), silent, agent_active=False)
    pol.update(_out(TurnEventType.POSSIBLE_END, 0.5), silent, agent_active=False)
    assert pol.update(_out(TurnEventType.TURN_COMPLETE, 0.9), silent, agent_active=False) is None


def test_policy_guards_short_noise_and_min_silence():
    pol = DecisionPolicy(POLICY)
    pol.update(_out(TurnEventType.CONTINUE), vad_result(1, True), agent_active=False)  # 20 ms < 60 ms
    assert pol.update(_out(TurnEventType.TURN_COMPLETE), vad_result(50, False, silence_run_ms=900), agent_active=False) is None
    pol = DecisionPolicy(POLICY.model_copy(update={"min_silence_guard_ms": 300}))
    for i in range(1, 6):
        pol.update(_out(TurnEventType.CONTINUE), vad_result(i, True), agent_active=False)
    assert pol.update(_out(TurnEventType.TURN_COMPLETE), vad_result(9, False, silence_run_ms=200), agent_active=False) is None


def test_policy_barge_in_only_while_agent_active():
    pol = DecisionPolicy(POLICY)
    assert pol.update(_out(TurnEventType.BARGE_IN), vad_result(1, True), agent_active=False) is None
    d = pol.update(_out(TurnEventType.BARGE_IN), vad_result(2, True), agent_active=True)
    assert d is not None and d.decision == DecisionType.BARGE_IN
    d = pol.update(_out(TurnEventType.BACKCHANNEL), vad_result(3, False), agent_active=True)
    assert d is not None and d.decision == DecisionType.BACKCHANNEL_IGNORED


def test_shared_probability_mapping():
    silent = vad_result(30, False, silence_run_ms=300)
    assert map_turn_end_probability(0.9, silent, POLICY, True).event_type == TurnEventType.TURN_COMPLETE
    assert map_turn_end_probability(0.4, silent, POLICY, True).event_type == TurnEventType.POSSIBLE_END
    assert map_turn_end_probability(0.9, vad_result(1, True), POLICY, True).event_type == TurnEventType.CONTINUE
    assert map_turn_end_probability(0.9, silent, POLICY, False).event_type == TurnEventType.CONTINUE
    long_silence = vad_result(200, False, silence_run_ms=POLICY.max_silence_ms)
    p = map_turn_end_probability(0.01, long_silence, POLICY, True)
    assert p.event_type == TurnEventType.TURN_COMPLETE and "max_silence_reached" in p.reason_codes
