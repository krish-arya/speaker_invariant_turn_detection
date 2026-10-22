# Turn-detection research console (frontend)

A React + TypeScript (Vite) console for the speaker-invariant turn-detection voice agent. It does two jobs (see [`../frontend.md`](../frontend.md)):

1. **Voice-agent experience.** Start a session, see whether the system is listening, predicting a turn end, committing a turn end, generating, speaking or cancelling, and interrupt the agent.
2. **Research console.** Pick a detector variant, run repeatable trials, inspect live predictions versus committed decisions and their latency, compare variants, and export evidence.

The wire contract is in [`../docs/api-contract.md`](../docs/api-contract.md), and its source of truth is `backend/src/turnsense/events.py`.

## Run

```bash
npm install
npm run dev          # http://localhost:5173 — proxies /api and /ws to http://127.0.0.1:8000
```

Start the backend API on port 8000 first. To point the dev proxy elsewhere, set `TURNSENSE_BACKEND=http://host:port npm run dev`.

| Command | Purpose |
|---|---|
| `npm run build` | `tsc -b` (strict, zero errors) then the production bundle in `dist/` |
| `npm test -- --run` | Vitest unit, component and integration tests (jsdom) |
| `npm run preview` | Serve `dist/` |

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE` | *(empty: same origin)* | Absolute backend origin for a deployed build, e.g. `https://api.example.org`. WebSocket URLs are derived from it (`http→ws`, `https→wss`). |
| `TURNSENSE_BACKEND` | `http://127.0.0.1:8000` | Dev-server proxy target only (IPv4 on purpose: `localhost` can resolve to `::1` on Windows and reach a different process). |

The frontend never holds provider keys. The only credential it handles is the short-lived LiveKit room token returned by `POST /api/sessions`. That token is passed straight to `room.connect` and is never stored, logged or displayed.

## Mock mode (UI development without a backend)

Open `http://localhost:5173/?mock=1`. If the backend is unreachable, the boot screen also offers **Use mock data**. Nothing switches to mock mode silently.

Mock mode replays a scripted, realistic event stream generated in the browser (`src/events/mock.ts`). It emits **wire-format** envelopes through the same validation path as the real transports. The stream covers every simulation scenario, a barge-in with a stale late chunk that must be ignored, a backchannel, and manual interrupt. It is labelled **MOCK DATA** in the header, at the top of the page, in the latency card and in the trial summary notes, so it cannot be mistaken for measurements. Evaluation and Results need the server and are disabled in mock mode.

## Structure

```
src/
  api/        schemas.ts (zod validation + snake→camel at the boundary), client.ts (typed REST client), download.ts
  events/     SessionEventSource interface + WebSocket (simulation), LiveKit (live mic, lazy-loaded), Mock sources
  state/      sessionReducer.ts (frontend.md §7 state model), sessionStore.ts (rAF-batched external store), sessionController.ts
  components/ Header, Tabs, ConfigBar (+ PolicyFields), VoiceSessionCard, LevelMeter, Timeline, LatencyOverview,
              TranscriptPanel, TrialSummaryCard, TrialsTable, TrialDetailDrawer, ComparisonView, InfoPopover, Icon
  pages/      LiveConsole, EvaluationPage, ResultsPage, SettingsPage
  lib/        labels, formatting, policy field table, settings (localStorage), devices, stats
  styles/     tokens.css (design tokens), app.css
```

## Behaviour worth knowing

- **One event interface.** All three sources hand raw messages to `SessionStore.ingest`. The store validates each envelope once with zod, drops duplicates and messages from other sessions, inserts a gap marker when `seq` jumps, shows late (out-of-order) messages without rolling state back, and folds batches once per animation frame. Unknown kinds and invalid payloads are counted under Settings → Diagnostics.
- **Predictions are not decisions.** `turn_prediction` envelopes render as dashed "Prediction" rows. `turn_decision` envelopes render as solid "Committed" rows. The voice card shows VAD (speech presence), the latest detector prediction and the last committed decision as three separate signals.
- **Stale audio guard.** After a committed `BARGE_IN` for generation *g*, later `transcript` and `playback:start` messages for *g* are ignored and counted. The assistant's partial text for *g* is marked "interrupted — not fully delivered".
- **Honest numbers.** `null` renders as "—", never 0. Latency percentiles appear only once the server reports `count ≥ min_samples`; until then the card shows "Collecting samples (n/min)". Failed samples are shown separately. Accuracy metrics show "Not available for this session" for unlabelled trials. Comparisons show mean, 95% CI and n with a better-direction marker; nothing is ranked or called "best".
- **Bounded memory.** The timeline keeps the last 500 items (the number dropped is shown) and the transcript the last 200 segments. Audio levels are drawn straight to the DOM by a capped animation loop: about 20 fps, or 4 fps with reduced motion.
- **Privacy.** The microphone is requested only after **Start** for live sessions. Simulation sessions never touch it. End session stops publishing, playback and the room. Starting a new session clears the previous transcript. Transcript display and export preferences are visible on the Live Console and in Settings.
- **Configuration is frozen** while a session is active. The UI always displays the server-accepted `effective_config`, not the values that were selected.

## Assumptions about the backend (beyond the written contract)

- `GET /api/trials` accepts `evaluation_id` as a filter (as the contract lists). The Evaluation page relies on it to show a run's trials.
- Comparison `metric_definitions[*].unit` is one of `ms`, `ratio`, `rate`, `%` or `count`. Unknown units render as plain 3-decimal numbers.
- Comparison group keys use trial field names (`detector_mode`, `scenario_id`, `model_version`, `trial_type`). The distribution chart uses them to place each selected trial's own metric value.
- On the LiveKit path, the worker publishes envelopes as UTF-8 JSON on topic `turnsense.events`. The client may join after the first messages were sent; the resulting leading `seq` gap is shown as such.
