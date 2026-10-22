# Speaker-Invariant Turn Detection for Real-Time Voice-to-Voice Conversation

B.Tech IT minor project: a streaming voice agent whose turn detector is trained with **anti-transfer learning** to depend less on speaker identity. The project compares it with a VAD + silence-timeout baseline on speaker-disjoint data.

The specification lives in [project.md](project.md), [architecture.md](architecture.md), [goal.md](goal.md) and [frontend.md](frontend.md). This README covers what is built, how to run it, and what is still open.

```
browser (React console, LiveKit JS) ──► FastAPI (tokens, sessions, evaluations, exports)
         ▲  events (WS / data channel)        │ session config
         │                                    ▼
         └──────────── LiveKit room ◄──► Python worker: FrameBuffer → VAD → TurnDetector
                                                     → DecisionPolicy → Orchestrator
                                                     → STT / LLM → clause chunker → TTS
                                                     → generation-guarded playback
```

## Repository layout

| Path | What it is |
|---|---|
| `backend/src/turnsense/events.py` | The wire contract (envelopes, turn events, traces, configs); prose in [docs/api-contract.md](docs/api-contract.md) |
| `backend/src/turnsense/audio/` | 20 ms frame buffer with gap detection, energy VAD, Silero VAD (ONNX) |
| `backend/src/turnsense/detectors/` | Detector interface and registry; `silence_baseline` (E0); `learned_control`/`anti_transfer` runtime (E1/E2) |
| `backend/src/turnsense/policy.py` | Decision policy that turns predictions into committed decisions, shared by live and offline evaluation |
| `backend/src/turnsense/orchestrator/` | State machine, orchestrator (generations, barge-in, history truncation, speculative prefill), playback queue |
| `backend/src/turnsense/adapters/` | Deepgram STT/TTS, Groq/Ollama LLM, faster-whisper/Kokoro fallbacks, simulated providers |
| `backend/src/turnsense/metrics/` | Turn precision/recall/F1, premature/missed ends, delay, barge-in metrics, percentiles, CIs |
| `backend/src/turnsense/sim/` | Synthetic voices, scripted labelled scenarios, virtual-time runner |
| `backend/src/turnsense/research/` | Feature extraction, model, anti-transfer loss, data/splits, training, evaluation, leakage probe |
| `backend/src/turnsense/api/` | FastAPI service |
| `backend/src/turnsense/worker/` | LiveKit agent worker |
| `frontend/` | React + TypeScript research console (see [frontend/README.md](frontend/README.md)) |
| `docs/` | API contract, research protocol, annotation guide |

## Quick start (no keys needed)

The **simulation transport** runs the full pipeline: real VAD, detector, policy, orchestrator, playback cancellation and tracing. It uses scripted synthetic user audio and modelled STT/LLM/TTS latencies, so it needs no microphone, LiveKit account or API keys.

```bash
# backend (Python 3.11+; uv recommended)
cd backend
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev,vad,livekit,research]"
.venv/Scripts/turnsense-api            # Windows; on Linux/macOS: .venv/bin/turnsense-api
# → http://127.0.0.1:8000  (OpenAPI docs at /docs)

# frontend
cd frontend
npm install
npm run dev                            # → http://localhost:5173 (proxies /api and /ws to 127.0.0.1:8000)
```

If another program already uses port 8000, start the API with `turnsense-api --port 8010` and the console with `TURNSENSE_BACKEND=http://127.0.0.1:8010 npm run dev`.

In the console, choose a detector and a scenario (for example *Pause then continue*), then select **Start session**. The timeline shows VAD activity, detector predictions and committed decisions as separate signals. The **Evaluation** tab runs batches of scripted trials faster than real time. **Results** compares them and exports CSV, JSON or Markdown.

## Live voice sessions (LiveKit + hosted providers)

1. Copy `backend/.env.example` to `backend/.env`, then fill in `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `DEEPGRAM_API_KEY` and `GROQ_API_KEY`. Keys stay on the server and worker. The browser receives only a short-lived room token for a single room.
2. Start the API (`turnsense-api`) and the worker (`turnsense-worker dev`).
3. In the console, select the **LiveKit** transport and the **Free conversation** scenario.

To use the local fallbacks instead (faster-whisper, Ollama, Kokoro-82M), set `TURNSENSE_RUNTIME_PROFILE=local`, run `pip install -e ".[local]"`, and start Ollama.

> **Not yet verified end-to-end.** The live path is written against the installed SDKs (livekit-agents 1.8.8, livekit rtc 1.1.20). Its credential-free parts have unit tests: provider message parsing, the sink, the frame queue and session-ID validation. It has not been run against LiveKit Cloud, Deepgram or Groq, because no credentials were available. Provider quotas and model availability also need rechecking before an evaluation (project.md §1).

## Research pipeline

See [docs/research_protocol.md](docs/research_protocol.md) for the experiment design and [docs/annotation_guide.md](docs/annotation_guide.md) for labelling. The CLI entry point is `turnsense-research`. `turnsense-research pipeline --synthetic --quick` runs the synthetic smoke test end to end and writes checkpoints to `backend/checkpoints/`. Once a checkpoint exists, its detector shows up as available in the console.

To score detectors on labelled recordings through the full live pipeline (not only offline), set `TURNSENSE_REPLAY_MANIFEST` and, optionally, `TURNSENSE_REPLAY_SPLIT` in `backend/.env`. The console then offers an **Evaluation set replay** scenario. It replays the split's `test_unseen` recordings, using the seed to pick the recording.

`backend/scripts/run_comparison.py` runs all three detectors through the live pipeline on identical inputs: the scripted scenarios plus held-out-speaker replay. It writes a Markdown report, and the trials also appear in the console.

**Synthetic results are pipeline checks, not evidence.** The synthetic corpus contains a speaker↔timing confound built in on purpose, so it can show whether the anti-transfer mechanism behaves as intended. The research claim needs a licensed, annotated corpus of real conversations with a speaker-disjoint test set.

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest tests -q          # core + API + research
cd ../frontend && npm test -- --run && npm run build
```

## Status against the definition of done (goal.md §9)

| Item | Status |
|---|---|
| Browser joins a session, streams audio, receives spoken responses, leaves cleanly | Simulation transport: done. LiveKit path: implemented, not yet verified against a live server |
| Baseline runs and is measured with a fixed configuration | Done (E0; frozen `EffectiveConfig` with `config_hash` in every trial) |
| Documented detector event schema wired into the orchestrator | Done (`events.py`, `docs/api-contract.md`) |
| Barge-in cancels playback; no stale audio afterwards | Done; tested with injected late chunks (`stale_chunks_played` = 0) |
| Anti-transfer variant implemented, or blocker documented | Implemented and trained on synthetic data. Result so far: indistinguishable from the learned control, and the penalty is non-binding ([research protocol §11](docs/research_protocol.md)) |
| Speaker-disjoint split; dataset and annotation policy documented | Split builder and docs done; real dataset still to be chosen and licensed |
| Task metrics, delay distribution, cross-speaker comparison, latency | Implemented and exercised end-to-end on synthetic data (offline and through the live pipeline); real-speaker numbers pending data |
| Frontend shows live state and exports trial metrics and configuration | Done (see frontend/README.md) |
| API keys and raw audio not exposed or persisted by default | Done (keys server-side only; raw audio never stored; transcripts exported only on opt-in) |
| Final documentation: failures, limitations, versions, reproduction | Ongoing |

## Known limitations

- Simulated latencies are **modelled**, not measured. Barge-in cancellation is close to 0 ms in virtual time. Only the live path measures real provider and network latency.
- Synthetic voices are harmonic signals, not speech. Silero VAD does not fire on them, so simulations use the energy VAD. Live sessions use Silero.
- Detector inference time is measured with wall-clock time and does not advance virtual time.
- Task metrics need annotated turn ends. Live free-conversation trials report latency and events only.
