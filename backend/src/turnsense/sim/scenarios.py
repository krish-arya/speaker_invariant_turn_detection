"""Scripted, labelled scenarios rendered as synthetic user audio (frontend.md §6.2, goal.md §8).

A scenario is a fixed *user track*: phrases, within-turn pauses (holds), turn ends and
backchannels at scripted times, plus the word timings the simulated STT reports. The agent's
behaviour is not scripted — it is whatever the pipeline under test does — so interruption
references are resolved after the run (trial.resolve_annotations).

The same ``seed`` always yields the same audio, so detector variants are compared on
identical input. Voices come from an id range reserved for simulation (never used for
training), so scenario speakers are unseen by any learned detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from turnsense.adapters.mock import ScriptWord
from turnsense.events import TrialType
from turnsense.sim.voice import Contour, VoiceProfile, noise_bed, synth_phrase
from turnsense.trial import ScriptAnnotations, UserSegment

SAMPLE_RATE = 16_000
SIM_VOICE_ID_BASE = 700_000  # reserved: simulation scenarios only, never training data
SCENARIO_SET_ID = "sim-scenarios-v1"

_UTTERANCES = [
    "could you check the train times to manchester for tomorrow morning",
    "actually make that the afternoon if there is anything after two",
    "what is the weather going to be like there this weekend",
    "and can you remind me to pack an umbrella before i leave",
    "i was thinking about booking a table for four people tonight",
    "somewhere quiet near the station would be perfect for us",
    "how long does the journey usually take on a weekday",
    "okay that sounds good please go ahead and book it",
]
_SHORT = ["yes please", "no thanks", "sure", "okay", "that works", "not really"]
_BACKCHANNEL = ["mm hm", "yeah", "right", "uh huh"]


@dataclass
class BuiltScenario:
    scenario_id: str
    seed: int
    voice: VoiceProfile
    audio: np.ndarray
    words: list[ScriptWord]
    annotations: ScriptAnnotations
    llm_sentences: tuple[int, int]

    @property
    def duration_ms(self) -> float:
        return self.audio.size * 1000.0 / SAMPLE_RATE


@dataclass
class _Builder:
    voice: VoiceProfile
    rng: np.random.Generator
    noise_dbfs: float
    parts: list[np.ndarray] = field(default_factory=list)
    t_ms: float = 0.0
    words: list[ScriptWord] = field(default_factory=list)
    segments: list[UserSegment] = field(default_factory=list)
    turns: list[tuple[float, float]] = field(default_factory=list)
    _phrase_id: int = 0
    _utterance: int = 0

    def silence(self, ms: float) -> None:
        n = int(ms * SAMPLE_RATE / 1000)
        self.parts.append(np.zeros(n, dtype=np.float32))
        self.t_ms += n * 1000.0 / SAMPLE_RATE

    def at(self, t_ms: float) -> None:
        """Advance (with silence) to an absolute time on the user track."""
        if t_ms > self.t_ms:
            self.silence(t_ms - self.t_ms)

    def phrase(
        self,
        text: str,
        contour: Contour,
        kind: str = "speech",
        turn_index: int | None = None,
    ) -> tuple[float, float]:
        tokens = text.split()
        n_syll = max(1, round(len(tokens) * 1.35))
        audio = synth_phrase(self.voice, n_syll, contour, self.rng, SAMPLE_RATE)
        start = self.t_ms
        dur = audio.size * 1000.0 / SAMPLE_RATE
        self.parts.append(audio)
        self.t_ms += dur
        self._phrase_id += 1
        step = dur / len(tokens)
        for i, tok in enumerate(tokens):
            self.words.append(
                ScriptWord(tok, start + i * step, start + (i + 1) * step, self._phrase_id)
            )
        self.segments.append(
            UserSegment(start_ms=start, end_ms=self.t_ms, kind=kind, turn_index=turn_index)
        )  # type: ignore[arg-type]
        return start, self.t_ms

    def turn(self, pieces: list[str], pauses_ms: list[float], final: Contour = "final") -> None:
        """One conversational turn: phrases separated by within-turn pauses (holds)."""
        idx = len(self.turns)
        start = self.t_ms
        for i, text in enumerate(pieces):
            last = i == len(pieces) - 1
            self.phrase(text, final if last else "hold", turn_index=idx)
            if not last:
                self.silence(pauses_ms[i])
        self.turns.append((start, self.t_ms))

    def next_utterance(self) -> str:
        text = _UTTERANCES[
            (self._utterance + int(self.rng.integers(len(_UTTERANCES)))) % len(_UTTERANCES)
        ]
        self._utterance += 1
        return text

    def split(self, text: str, parts: int) -> list[str]:
        tokens = text.split()
        parts = max(1, min(parts, len(tokens)))
        cuts = (
            sorted(self.rng.choice(np.arange(1, len(tokens)), size=parts - 1, replace=False))
            if parts > 1
            else []
        )
        bounds = [0, *cuts, len(tokens)]
        return [" ".join(tokens[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]

    def build(self, scenario_id: str, seed: int, llm_sentences: tuple[int, int]) -> BuiltScenario:
        audio = np.concatenate(self.parts).astype(np.float32)
        audio += noise_bed(audio.size, self.noise_dbfs, self.rng)
        return BuiltScenario(
            scenario_id=scenario_id,
            seed=seed,
            voice=self.voice,
            audio=audio,
            words=self.words,
            annotations=ScriptAnnotations(
                source="scenario_script", turns=self.turns, segments=self.segments
            ),
            llm_sentences=llm_sentences,
        )


def _builder(seed: int) -> _Builder:
    rng = np.random.default_rng(seed)
    voice = VoiceProfile.from_id(SIM_VOICE_ID_BASE + seed)
    return _Builder(voice=voice, rng=rng, noise_dbfs=float(rng.uniform(-66, -56)))


def _pause_then_continue(seed: int) -> BuiltScenario:
    b = _builder(seed)
    b.silence(600)
    b.turn(b.split(b.next_utterance(), 2), [float(b.rng.uniform(550, 1000))])
    b.silence(8000)
    b.turn(b.split(b.next_utterance(), 2), [float(b.rng.uniform(450, 900))])
    b.silence(8000)
    return b.build("pause_then_continue", seed, (1, 1))


def _short_answer(seed: int) -> BuiltScenario:
    b = _builder(seed)
    b.silence(600)
    for _ in range(3):
        b.turn([_SHORT[int(b.rng.integers(len(_SHORT)))]], [])
        b.silence(7000)
    return b.build("short_answer", seed, (1, 1))


def _multi_turn(seed: int) -> BuiltScenario:
    b = _builder(seed)
    b.silence(600)
    for _ in range(3):
        n = int(b.rng.integers(1, 4))
        b.turn(
            b.split(b.next_utterance(), n), [float(b.rng.uniform(250, 650)) for _ in range(n - 1)]
        )
        b.silence(8000)
    return b.build("multi_turn", seed, (1, 1))


def _barge_in(seed: int) -> BuiltScenario:
    b = _builder(seed)
    b.silence(600)
    b.turn([b.next_utterance()], [], final="question")
    first_end = b.t_ms
    b.at(first_end + float(b.rng.uniform(3200, 4200)))  # lands inside the agent's long answer
    b.turn([b.next_utterance()], [])
    b.silence(9000)
    return b.build("barge_in", seed, (3, 3))


def _backchannel(seed: int) -> BuiltScenario:
    b = _builder(seed)
    b.silence(600)
    b.turn([b.next_utterance()], [], final="question")
    first_end = b.t_ms
    b.at(first_end + float(b.rng.uniform(3000, 3600)))
    b.phrase(
        _BACKCHANNEL[int(b.rng.integers(len(_BACKCHANNEL)))], "backchannel", kind="backchannel"
    )
    b.at(first_end + float(b.rng.uniform(4800, 5600)))
    b.phrase(
        _BACKCHANNEL[int(b.rng.integers(len(_BACKCHANNEL)))], "backchannel", kind="backchannel"
    )
    b.at(first_end + 15000)
    b.turn([_SHORT[int(b.rng.integers(len(_SHORT)))]], [])
    b.silence(7000)
    return b.build("backchannel", seed, (3, 3))


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    label: str
    description: str
    trial_type: TrialType
    labelled: bool
    transports: tuple[str, ...]
    build: Callable[[int], BuiltScenario] | None = None


SCENARIOS: dict[str, ScenarioSpec] = {
    s.id: s
    for s in [
        ScenarioSpec(
            "pause_then_continue",
            "Pause then continue",
            "Two turns, each with a 0.45–1.0 s thinking pause before the speaker finishes. "
            "Tests premature endpointing.",
            TrialType.SIMULATED, True, ("simulation",), _pause_then_continue,
        ),
        ScenarioSpec(
            "short_answer",
            "Short answer",
            "Three one- or two-word turns. Tests responsiveness to brief, complete answers.",
            TrialType.SIMULATED, True, ("simulation",), _short_answer,
        ),
        ScenarioSpec(
            "multi_turn",
            "Multi-turn",
            "Three turns with zero to two short within-turn pauses each.",
            TrialType.SIMULATED, True, ("simulation",), _multi_turn,
        ),
        ScenarioSpec(
            "barge_in",
            "Barge-in",
            "A question, then the user interrupts the agent's long answer with a new request.",
            TrialType.SIMULATED, True, ("simulation",), _barge_in,
        ),
        ScenarioSpec(
            "backchannel",
            "Backchannel",
            "Short listener responses (\"mm hm\", \"yeah\") while the agent speaks; these should not "
            "cancel the answer.",
            TrialType.SIMULATED, True, ("simulation",), _backchannel,
        ),
        ScenarioSpec(
            "free_conversation",
            "Free conversation",
            "Talk to the agent through your microphone. Unlabelled: latency and events only.",
            TrialType.LIVE, False, ("livekit",), None,
        ),
    ]
}  # fmt: skip


def build_scenario(scenario_id: str, seed: int) -> BuiltScenario:
    spec = SCENARIOS[scenario_id]
    if spec.build is None:
        raise ValueError(f"scenario {scenario_id!r} has no simulated user track")
    return spec.build(seed)
