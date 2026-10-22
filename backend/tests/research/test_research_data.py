"""Manifest round-trip, frame labels, speaker-disjoint splits."""

from pathlib import Path

import numpy as np
import pytest

from turnsense.research.data import (
    Recording,
    Split,
    Turn,
    assert_speaker_disjoint,
    build_splits,
    frame_labels,
    load_manifest,
    write_manifest,
)
from turnsense.research.model import HOLD, IGNORE, SHIFT, SPEECH


def _rec(rid: str, spk: str, conv: str | None = None, tmp: Path | None = None) -> Recording:
    return Recording(
        recording_id=rid,
        conversation_id=conv or rid,
        speaker_id=spk,
        audio_path=(tmp or Path(".")) / f"{rid}.wav",
        duration_ms=10_000,
        turns=[Turn(1000, 3000, [(1000, 1800), (2300, 3000)]), Turn(6000, 7000, [(6000, 7000)])],
    )


def test_frame_labels_hold_shift_speech():
    rec = _rec("r", "s")
    lab = frame_labels(rec, n_frames=1000, hop_ms=10.0, shift_horizon_ms=2500.0)

    def at(ms: float) -> int:
        return int(lab[int(ms / 10) - 1])  # frame k ends at (k+1)*10 ms

    assert at(500) == IGNORE  # before the first turn
    assert at(1500) == SPEECH
    assert at(2000) == HOLD  # pause inside turn 1
    assert at(2500) == SPEECH
    assert at(3500) == SHIFT  # after turn 1 end
    assert at(5600) == IGNORE  # beyond the 2.5 s shift horizon
    assert at(6500) == SPEECH
    assert at(7100) == SHIFT


def test_shift_stops_at_next_speech():
    rec = _rec("r", "s")
    rec.turns[1] = Turn(4000, 5000, [(4000, 5000)])
    lab = frame_labels(rec, n_frames=1000)
    assert lab[int(3900 / 10) - 1] == SHIFT
    assert lab[int(4100 / 10) - 1] == SPEECH


def test_manifest_round_trip(tmp_path):
    recs = [_rec("a", "s1", tmp=tmp_path), _rec("b", "s2", tmp=tmp_path)]
    write_manifest(recs, tmp_path / "m.jsonl")
    back = load_manifest(tmp_path / "m.jsonl")
    assert [r.recording_id for r in back] == ["a", "b"]
    assert back[0].turns[0].ipus == [(1000.0, 1800.0), (2300.0, 3000.0)]
    assert back[0].audio_path == (tmp_path / "a.wav").resolve()


def _corpus(n_spk: int = 10, convs: int = 4) -> list[Recording]:
    return [_rec(f"s{s}_c{c}", f"s{s}") for s in range(n_spk) for c in range(convs)]


def test_split_is_speaker_disjoint_and_stable():
    recs = _corpus()
    split = build_splits(recs, unseen_frac=0.3, val_frac=0.2, seed=3)
    train = split.speakers(recs, "train")
    test = split.speakers(recs, "test_unseen")
    val = split.speakers(recs, "val_seen")
    assert train and test and val
    assert not train & test
    assert val <= train
    assert_speaker_disjoint(recs, split)
    again = build_splits(recs, unseen_frac=0.3, val_frac=0.2, seed=3)
    assert again.split_id == split.split_id
    assert build_splits(recs, unseen_frac=0.3, val_frac=0.2, seed=4).split_id != split.split_id
    s = split.stats
    assert s["test_unseen"]["speakers"] == 3
    assert s["train"]["turns"] == 2 * s["train"]["recordings"]
    assert s["train"]["within_turn_pauses"] == s["train"]["recordings"]


def test_two_party_conversation_with_unseen_speaker_never_trains():
    recs = [_rec("c1_A", "alice", "c1"), _rec("c1_B", "bob", "c1"),
            _rec("c2_A", "alice", "c2"), _rec("c2_C", "carol", "c2"),
            _rec("c3_B", "bob", "c3"), _rec("c3_C", "carol", "c3")]  # fmt: skip
    split = build_splits(recs, val_frac=0.0, unseen_speakers={"alice"})
    assert split.assignments["c1_A"] == "test_unseen"
    assert "c1_B" in split.excluded and "c2_C" in split.excluded
    assert split.assignments["c3_B"] == split.assignments["c3_C"] == "train"


def test_assert_catches_leak():
    recs = _corpus(4, 2)
    split = build_splits(recs, unseen_frac=0.25, seed=0)
    leaked = next(r for r in recs if split.assignments.get(r.recording_id) == "test_unseen")
    other = next(r for r in recs if r.speaker_id == leaked.speaker_id and r is not leaked)
    split.assignments[other.recording_id] = "train"
    with pytest.raises(AssertionError):
        assert_speaker_disjoint(recs, split)


def test_split_json_round_trip(tmp_path):
    recs = _corpus(6, 3)
    split = build_splits(recs, seed=1)
    split.save(tmp_path / "split.json")
    back = Split.load(tmp_path / "split.json")
    assert back.split_id == split.split_id and back.assignments == split.assignments
    assert np.isclose(back.params["unseen_frac"], 0.3)
