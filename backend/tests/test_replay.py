import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from turnsense.api.app import create_app
from turnsense.config import Settings
from turnsense.sim.replay import REPLAY_SCENARIO_ID, replay_spec
from turnsense.sim.voice import VoiceProfile, noise_bed, synth_phrase

data = pytest.importorskip("turnsense.research.data")  # needs the research extra


def _manifest(tmp_path, n_recordings=2):
    """Tiny labelled manifest: each recording has one turn with a within-turn pause."""
    rng = np.random.default_rng(0)
    lines = []
    for i in range(n_recordings):
        voice = VoiceProfile.from_id(900 + i)
        a = synth_phrase(voice, 6, "hold", rng)
        b = synth_phrase(voice, 5, "final", rng)
        parts = [np.zeros(8000, np.float32), a, np.zeros(9600, np.float32), b, np.zeros(48_000, np.float32)]
        audio = np.concatenate(parts)
        audio += noise_bed(audio.size, -62, rng)
        rel = f"audio/r{i}.wav"
        data.write_wav(tmp_path / rel, audio)
        s1, e1 = 500.0, 500.0 + a.size / 16
        s2, e2 = e1 + 600.0, e1 + 600.0 + b.size / 16
        lines.append(
            {
                "schema_version": data.MANIFEST_SCHEMA,
                "recording_id": f"r{i}",
                "conversation_id": f"c{i}",
                "speaker_id": f"spk{i}",
                "audio_path": rel,
                "turns": [{"start_ms": s1, "end_ms": e2, "type": "turn", "ipus": [[s1, e1], [s2, e2]]}],
                "synthetic": True,
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return path, lines


def test_replay_spec_builds_annotated_user_track(tmp_path):
    path, lines = _manifest(tmp_path)
    spec = replay_spec(path)
    assert spec is not None and spec.trial_type.value == "replay" and spec.labelled
    built = spec.build(3)  # seed picks recording 3 % 2 == 1
    turn = lines[1]["turns"][0]
    assert built.annotations.turns == [(turn["start_ms"], turn["end_ms"])]
    assert [s.kind for s in built.annotations.segments] == ["speech", "speech"]
    assert built.annotations.source.endswith("/r1")
    assert abs(built.duration_ms - (len(built.audio) / 16)) < 1e-6


def test_replay_scenario_is_offered_only_when_configured_and_scores_trials(tmp_path):
    path, _ = _manifest(tmp_path)
    plain = Settings(_env_file=None, data_dir=tmp_path / "d0", checkpoint_dir=tmp_path / "ck")
    with TestClient(create_app(plain)) as c:
        ids = {s["id"] for s in c.get("/api/capabilities").json()["scenarios"]}
        assert REPLAY_SCENARIO_ID not in ids

    settings = Settings(
        _env_file=None, data_dir=tmp_path / "d1", checkpoint_dir=tmp_path / "ck", replay_manifest=path
    )
    with TestClient(create_app(settings)) as c:
        scn = {s["id"]: s for s in c.get("/api/capabilities").json()["scenarios"]}
        assert scn[REPLAY_SCENARIO_ID]["trial_type"] == "replay"
        ev = c.post(
            "/api/evaluations",
            json={"detector_modes": ["silence_baseline"], "scenario_ids": [REPLAY_SCENARIO_ID], "repeats": 2},
        ).json()
        for _ in range(200):
            st = c.get(f"/api/evaluations/{ev['evaluation_id']}").json()
            if st["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert st["status"] == "completed", st
        detail = c.get(f"/api/trials/{st['trial_ids'][0]}").json()
        assert detail["trial_type"] == "replay" and detail["labelled"]
        assert detail["metrics"]["turn_recall"] == 1.0
        assert any("Recorded audio replayed" in n for n in detail["notes"])
        cmp = c.post("/api/compare", json={"evaluation_id": ev["evaluation_id"]}).json()
        assert cmp["evaluation_set"] == "replay:manifest:manifest.jsonl/unsplit"
