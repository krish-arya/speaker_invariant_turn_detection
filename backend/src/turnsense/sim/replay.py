"""Replay labelled recordings through the live pipeline (the "Evaluation script" scenario).

Each trial replays one single-speaker recording from a research manifest as the user track,
with its annotated turns as ground truth. When a split file is given, only its
``test_unseen`` recordings are used, so learned detectors are scored on speakers they never
saw. The seed selects the recording, so every detector variant hears the same audio.

Recordings carry no transcript, so the simulated STT reports a placeholder per IPU; the
response content is irrelevant to turn-taking metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from turnsense.adapters.mock import ScriptWord
from turnsense.events import TrialType
from turnsense.sim.scenarios import BuiltScenario, ScenarioSpec
from turnsense.sim.voice import VoiceProfile
from turnsense.trial import ScriptAnnotations, UserSegment

log = logging.getLogger(__name__)

REPLAY_SCENARIO_ID = "replay_eval_set"


def replay_spec(manifest: Path, split_path: Path | None = None) -> ScenarioSpec | None:
    """A scenario over the manifest's held-out recordings, or None if it cannot be loaded."""
    try:
        from turnsense.research.data import Split, load_audio, load_manifest
    except ImportError as exc:  # research extra (torch) not installed
        log.warning("replay scenario unavailable: %s", exc)
        return None
    try:
        records = load_manifest(manifest)
        split_id = "unsplit"
        if split_path is not None:
            split = Split.load(split_path)
            records = split.recordings(records, "test_unseen")
            split_id = split.split_id
    except (OSError, ValueError, KeyError) as exc:
        log.warning("replay scenario unavailable: %s", exc)
        return None
    records = sorted(records, key=lambda r: r.recording_id)
    if not records:
        return None
    synthetic = any(r.synthetic for r in records)
    source = f"manifest:{manifest.name}/{split_id}"

    def build(seed: int) -> BuiltScenario:
        rec = records[seed % len(records)]
        audio = load_audio(rec, 16_000)
        words: list[ScriptWord] = []
        segments: list[UserSegment] = []
        turns = rec.conversation_turns
        for idx, turn in enumerate(turns):
            for a, b in turn.ipus or [(turn.start_ms, turn.end_ms)]:
                pid = len(segments) + 1
                mid = (a + b) / 2
                words += [ScriptWord("(replayed", a, mid, pid), ScriptWord("speech)", mid, b, pid)]
                segments.append(UserSegment(start_ms=a, end_ms=b, kind="speech", turn_index=idx))
        for a, b in rec.backchannels:
            segments.append(UserSegment(start_ms=a, end_ms=b, kind="backchannel"))
        return BuiltScenario(
            scenario_id=REPLAY_SCENARIO_ID,
            seed=seed,
            voice=VoiceProfile.from_id(0),  # unused for replay; kept for the dataclass shape
            audio=audio,
            words=words,
            annotations=ScriptAnnotations(
                source=f"{source}/{rec.recording_id}",
                turns=[(t.start_ms, t.end_ms) for t in turns],
                segments=sorted(segments, key=lambda s: s.start_ms),
            ),
            llm_sentences=(1, 1),
        )

    kind = "SYNTHETIC recordings" if synthetic else "recordings"
    return ScenarioSpec(
        REPLAY_SCENARIO_ID,
        "Evaluation set replay",
        f"Replays {len(records)} labelled {kind} from the held-out speaker set "
        f"({source}); the seed picks the recording.",
        TrialType.REPLAY,
        True,
        ("simulation",),
        build,
    )


def scenario_catalog(manifest: Path | None, split_path: Path | None) -> dict[str, ScenarioSpec]:
    from turnsense.sim.scenarios import SCENARIOS

    catalog = dict(SCENARIOS)
    if manifest is not None:
        spec = replay_spec(manifest, split_path)
        if spec is not None:
            catalog[spec.id] = spec
    return catalog
