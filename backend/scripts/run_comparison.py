"""Run a matched three-way comparison through the live pipeline and write the Markdown report.

Every detector hears identical audio (same scenario, same seed). Trials are stored in the
normal trial store, so they also appear in the console's Results tab.

    backend/.venv/Scripts/python backend/scripts/run_comparison.py --repeats 5 \
        --replay-manifest backend/runs/synthetic-quick-v1/data/manifest.jsonl \
        --replay-split backend/runs/synthetic-quick-v1/split.json \
        --out backend/runs/synthetic-quick-v1/pipeline_comparison.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fastapi.testclient import TestClient

from turnsense.api.app import create_app
from turnsense.config import Settings

SCRIPTED = ["pause_then_continue", "short_answer", "multi_turn", "barge_in", "backchannel"]
MODES = ["silence_baseline", "learned_control", "anti_transfer"]


def run(client: TestClient, scenarios: list[str], repeats: int, label: str) -> str:
    ev = client.post(
        "/api/evaluations",
        json={"detector_modes": MODES, "scenario_ids": scenarios, "repeats": repeats, "seed": 100, "trial_label": label},
    )
    ev.raise_for_status()
    eid = ev.json()["evaluation_id"]
    while (st := client.get(f"/api/evaluations/{eid}").json())["status"] in ("queued", "running"):
        print(f"  {label}: {st['completed']}/{st['total']}", end="\r")
        time.sleep(1)
    print(f"  {label}: {st['status']} {st['completed']}/{st['total']}    ")
    if st["status"] != "completed":
        raise SystemExit(f"evaluation {eid} ended with {st['status']}: {st.get('error')}")
    return eid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--replay-manifest", type=Path)
    ap.add_argument("--replay-split", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    settings = Settings(replay_manifest=args.replay_manifest, replay_split=args.replay_split)
    sections = []
    with TestClient(create_app(settings)) as c:
        caps = c.get("/api/capabilities").json()
        missing = [d["mode"] for d in caps["detectors"] if d["mode"] in MODES and not d["available"]]
        if missing:
            raise SystemExit(f"detectors unavailable: {missing}")
        jobs = [("Scripted scenarios (synthetic voices)", SCRIPTED, args.repeats)]
        if any(s["id"] == "replay_eval_set" for s in caps["scenarios"]):
            jobs.append(("Held-out-speaker replay", ["replay_eval_set"], 16))
        for title, scenarios, repeats in jobs:
            eid = run(c, scenarios, repeats, title)
            for group_by in (["detector_mode"], ["detector_mode", "scenario_id"]):
                md = c.post("/api/compare?format=md", json={"evaluation_id": eid, "group_by": group_by})
                md.raise_for_status()
                body = md.text.replace("# Turn-detection comparison", f"## {title} — grouped by {', '.join(group_by)}")
                sections.append(f"Evaluation `{eid}`\n\n{body}")
    header = (
        "# Live-pipeline comparison (SYNTHETIC-trained detectors)\n\n"
        "> Synthetic voices and modelled provider latency. These numbers check the pipeline and the "
        "experiment machinery; they are not evidence about real speakers or the anti-transfer hypothesis.\n\n"
    )
    args.out.write_text(header + "\n\n".join(sections), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
