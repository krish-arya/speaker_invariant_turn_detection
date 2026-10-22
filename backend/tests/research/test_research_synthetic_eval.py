"""Synthetic confound design and evaluation summary helpers."""

import numpy as np

from turnsense.research.data import load_manifest
from turnsense.research.evaluate import RecordingResult, generalisation_gap, summarise
from turnsense.research.synthetic import SyntheticConfig, generate_corpus


def test_confound_is_reversed_for_unseen_speakers(tmp_path):
    cfg = SyntheticConfig(n_train_speakers=4, n_unseen_speakers=4, convs_per_train_speaker=2,
                          convs_per_unseen_speaker=2, turns_per_conv=(6, 6), seed=2)  # fmt: skip
    manifest, unseen, info = generate_corpus(cfg, tmp_path)
    recs = load_manifest(manifest)
    assert all(r.synthetic for r in recs)
    assert len(unseen) == 4 and all(
        r.meta["pool"] == "unseen_pool" for r in recs if r.speaker_id in unseen
    )
    s = info["confound_summary"]
    # training: high-f0 voices pause mostly for holds; unseen: the reverse
    assert (
        s["train_pool"]["high"]["p_shift_given_pause"]
        < s["train_pool"]["low"]["p_shift_given_pause"]
    )
    assert (
        s["unseen_pool"]["high"]["p_shift_given_pause"]
        > s["unseen_pool"]["low"]["p_shift_given_pause"]
    )
    # every turn's IPUs lie inside the turn and are ordered
    for r in recs:
        for t in r.turns:
            assert t.ipus[0][0] == t.start_ms and t.ipus[-1][1] == t.end_ms
            assert all(a < b for a, b in t.ipus)


def _res(spk: str, matched: int, n_ref: int, n_com: int) -> RecordingResult:
    return RecordingResult(
        f"{spk}_r", spk, [], n_ref, matched, n_com, 0, [100.0] * matched, 50, [0.5] * 50
    )


def test_summarise_pools_counts_and_bootstraps():
    res = [_res("a", 4, 5, 5), _res("b", 2, 5, 4), _res("c", 5, 5, 6)]
    s = summarise(res, n_boot=200)
    assert s["n_reference_turns"] == 15 and s["n_commits"] == 15
    assert np.isclose(s["precision"], 11 / 15) and np.isclose(s["recall"], 11 / 15)
    lo, hi = s["ci95"]["f1"]
    assert lo <= s["f1"] <= hi
    assert s["inference_ms"]["n"] == 150 and len(s["per_speaker"]) == 3


def test_generalisation_gap_sign():
    seen = summarise([_res("a", 5, 5, 5), _res("b", 5, 5, 5)], n_boot=100)
    unseen = summarise([_res("c", 2, 5, 5), _res("d", 3, 5, 5)], n_boot=100)
    gap = generalisation_gap(seen, unseen, n_boot=200)
    assert gap["f1_gap"] > 0 and gap["ci95"][0] is not None
