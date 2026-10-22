import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from turnsense.api.app import create_app
from turnsense.config import Settings


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "ck", _env_file=None)
    with TestClient(create_app(settings)) as c:
        yield c


def test_capabilities_only_offers_what_exists(client):
    cap = client.get("/api/capabilities").json()
    modes = {d["mode"]: d for d in cap["detectors"]}
    assert modes["silence_baseline"]["available"] and not modes["silence_baseline"]["probabilistic"]
    assert not modes["anti_transfer"]["available"] and modes["anti_transfer"]["unavailable_reason"]
    assert cap["transports"]["livekit"]["available"] is False
    assert cap["privacy"]["raw_audio_retention"] is False
    assert cap["policy_limits"]["silence_timeout_ms"] == [50, 5000]
    text = json.dumps(cap).lower()
    assert "secret" not in text.replace("livekit_api_secret", "") and "token" not in text


def test_session_validation_errors(client):
    assert client.post("/api/sessions", json={"scenario_id": "nope"}).status_code == 400
    r = client.post("/api/sessions", json={"transport": "simulation", "scenario_id": "free_conversation"})
    assert r.status_code == 400
    assert client.post("/api/sessions", json={"transport": "livekit", "scenario_id": "free_conversation"}).status_code == 503
    assert client.post("/api/sessions", json={"detector_mode": "anti_transfer"}).status_code == 409
    assert client.post("/api/sessions", json={"policy_overrides": {"silence_timeout_ms": 1}}).status_code == 422
    assert client.post("/api/sessions", json={"policy_overrides": {"version": "x"}}).status_code == 422
    assert client.post("/api/sessions/ses_missing/end").status_code == 404


def test_simulated_session_streams_events_and_records_trial(client):
    r = client.post(
        "/api/sessions",
        json={"scenario_id": "short_answer", "seed": 5, "policy_overrides": {"silence_timeout_ms": 500}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["livekit"] is None and body["effective_config"]["policy"]["silence_timeout_ms"] == 500
    sid = body["session_id"]
    assert client.post(f"/api/sessions/{sid}/interrupt").status_code == 409  # not started yet
    seen = []
    with client.websocket_connect(body["events_url"]) as ws:
        while True:
            try:
                msg = json.loads(ws.receive_text())
            except Exception:
                break
            seen.append(msg)
            if msg["kind"] == "session_status" and msg["payload"]["state"] == "listening" and len(seen) < 6:
                client.post(f"/api/sessions/{sid}/end")  # end early: record marked aborted
    assert seen[0]["kind"] == "hello"
    assert seen[0]["payload"]["session"]["effective_config"]["config_hash"] == body["effective_config"]["config_hash"]
    assert [m["seq"] for m in seen] == sorted(m["seq"] for m in seen)
    assert seen[-1]["payload"]["state"] == "ended"
    for _ in range(50):
        trial = client.get(f"/api/trials/{body['trial_id']}")
        if trial.status_code == 200:
            break
        time.sleep(0.05)
    detail = trial.json()
    assert detail["status"] == "aborted" and detail["metrics"]["turn_f1"] is None
    assert "latency_samples" not in detail


def test_unknown_session_websocket_is_rejected(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/sessions/ses_nope") as ws:
        ws.receive_text()


def test_evaluation_compare_and_exports(client):
    r = client.post(
        "/api/evaluations",
        json={"detector_modes": ["silence_baseline"], "scenario_ids": ["short_answer", "barge_in"], "repeats": 2, "seed": 3},
    )
    assert r.status_code == 202
    eid = r.json()["evaluation_id"]
    for _ in range(300):
        st = client.get(f"/api/evaluations/{eid}").json()
        if st["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert st["status"] == "completed" and st["completed"] == st["total"] == 4
    listing = client.get(f"/api/trials?evaluation_id={eid}").json()
    assert listing["total"] == 4 and all(i["labelled"] for i in listing["items"])

    cmp = client.post("/api/compare", json={"evaluation_id": eid, "group_by": ["detector_mode", "scenario_id"]}).json()
    assert len(cmp["groups"]) == 2 and cmp["evaluation_set"] == "sim-scenarios-v1"
    f1 = cmp["groups"][0]["metrics"]["turn_f1"]
    assert f1["n"] == 2 and f1["aggregation"] == "trial_mean"
    assert cmp["metric_definitions"]["premature_end_rate"]["better"] == "lower"
    md = client.post("/api/compare?format=md", json={"evaluation_id": eid})
    assert md.headers["content-disposition"].endswith('comparison.md"') and "| Turn F1 | ↑ higher |" in md.text
    csv_cmp = client.post("/api/compare?format=csv", json={"evaluation_id": eid}).text
    assert csv_cmp.splitlines()[0].startswith("detector_mode,n_trials,metric")

    tid = st["trial_ids"][0]
    j = client.get(f"/api/trials/{tid}/export?format=json").json()
    assert j["export_meta"]["raw_audio_included"] is False and j["effective_config"]["config_hash"]
    ev_csv = client.get(f"/api/trials/{tid}/export?format=csv").text.splitlines()
    assert ev_csv[0].startswith("trial_id,session_id,detector_mode") and len(ev_csv) > 10
    summary_csv = client.post("/api/trials/export", json={"trial_ids": st["trial_ids"], "format": "csv"}).text.splitlines()
    assert len(summary_csv) == 5 and "metric_turn_f1" in summary_csv[0] and "policy_silence_timeout_ms" in summary_csv[0]

    bad = client.post("/api/evaluations", json={"detector_modes": ["silence_baseline"], "scenario_ids": ["free_conversation"]})
    assert bad.status_code == 400
