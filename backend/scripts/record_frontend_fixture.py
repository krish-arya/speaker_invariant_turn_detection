"""Record real backend output for the frontend's contract regression test.

Writes frontend/src/test/fixtures/backend-recorded.json containing two complete simulated
sessions (every envelope as sent over the WebSocket) and each REST response the console
consumes. Re-run after changing the wire contract:

    backend/.venv/Scripts/python backend/scripts/record_frontend_fixture.py
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from turnsense.api.app import create_app
from turnsense.config import Settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "src" / "test" / "fixtures" / "backend-recorded.json"


def record_session(client: TestClient, body: dict) -> dict:
    created = client.post("/api/sessions", json=body)
    created.raise_for_status()
    resp = created.json()
    messages = []
    with client.websocket_connect(resp["events_url"]) as ws:
        while True:
            try:
                messages.append(json.loads(ws.receive_text()))
            except Exception:
                break
    return {"request": body, "create_response": resp, "messages": messages}


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    settings = Settings(data_dir=tmp / "data", checkpoint_dir=ROOT / "backend" / "checkpoints", _env_file=None)
    with TestClient(create_app(settings)) as c:
        sessions = [
            record_session(c, {"scenario_id": "barge_in", "seed": 3, "transcript_enabled": True, "transcript_export_enabled": True}),
            record_session(
                c,
                {"scenario_id": "backchannel", "seed": 0, "transcript_enabled": False, "speculative_prefill": True},
            ),
        ]
        ev = c.post(
            "/api/evaluations",
            json={"detector_modes": ["silence_baseline"], "scenario_ids": ["short_answer", "barge_in"], "repeats": 2, "seed": 1},
        ).json()
        while (status := c.get(f"/api/evaluations/{ev['evaluation_id']}").json())["status"] in ("queued", "running"):
            time.sleep(0.1)
        trials = c.get("/api/trials", params={"evaluation_id": ev["evaluation_id"]}).json()
        rest = {
            "health": c.get("/api/health").json(),
            "capabilities": c.get("/api/capabilities").json(),
            "evaluation_started": ev,
            "evaluation_status": status,
            "trial_list": trials,
            "trial_detail": c.get(f"/api/trials/{sessions[0]['create_response']['trial_id']}").json(),
            "comparison": c.post(
                "/api/compare", json={"evaluation_id": ev["evaluation_id"], "group_by": ["detector_mode", "scenario_id"]}
            ).json(),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sessions": sessions, "rest": rest}, indent=None), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB, {sum(len(s['messages']) for s in sessions)} envelopes)")


if __name__ == "__main__":
    main()
