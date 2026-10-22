# API and Event Contract (v1)

Source of truth: [`backend/src/turnsense/events.py`](../backend/src/turnsense/events.py). This page restates it for frontend and tooling authors. If they disagree, the Python models win and this page is a bug.

## Conventions

- JSON is **snake_case** on the wire. The frontend converts to camelCase at its validation boundary.
- `*_ms` values are milliseconds on the **session-monotonic clock** (origin = session start). `wall_time` / `*_at` are ISO-8601 UTC strings for display and export only. Never subtract a wall time from a monotonic time.
- A value the system cannot measure is `null`. A deterministic detector has `confidence: null`. A metric with no labels is `null`. The UI renders `—` for `null`, never `0`.
- All REST paths are under `/api`. The dev server runs at `http://localhost:8000`; the Vite dev server proxies `/api` and `/ws` to it.
- No endpoint ever returns provider keys. `livekit.token` is a short-lived room grant: use it to join, never display or log it.

## Enumerations

| Name | Values |
|---|---|
| `DetectorMode` | `silence_baseline`, `learned_control`, `anti_transfer` |
| `TurnEventType` (prediction) | `CONTINUE`, `POSSIBLE_END`, `TURN_COMPLETE`, `BARGE_IN`, `BACKCHANNEL` |
| `DecisionType` (committed) | `TURN_COMPLETE`, `BARGE_IN`, `BACKCHANNEL_IGNORED` |
| `SessionState` (server) | `connecting`, `listening`, `user_speaking`, `possible_end`, `generating`, `agent_speaking`, `interrupted`, `reconnecting`, `error`, `ending`, `ended` |
| `Transport` | `simulation`, `livekit` |
| `TrialType` | `simulated`, `replay`, `live` |
| `TraceStatus` | `ok`, `error`, `cancelled`, `timeout` |
| `ErrorSource` | `transport`, `stt`, `llm`, `tts`, `detector`, `vad`, `worker` |
| `LatencyMetric` | `detector_inference`, `turn_end_delay`, `response_start`, `barge_in_cancel`, `stt_final`, `llm_first_token`, `tts_first_audio` |

The browser adds two local-only states: `idle` and `requesting_permission`.

## REST endpoints

### `GET /api/health`
```json
{ "status": "ok", "server_version": "0.1.0", "protocol_version": 1, "uptime_s": 12.3 }
```

### `GET /api/capabilities`
Describes what this server can actually do. **Only render detector modes with `available: true` as selectable** (unavailable ones may be shown disabled, with the reason), and only scenarios listed here.
```json
{
  "server_version": "0.1.0",
  "protocol_version": 1,
  "detectors": [
    { "mode": "silence_baseline", "label": "Silence/VAD baseline", "model_version": "silence-baseline-0.1.0",
      "available": true, "unavailable_reason": null, "probabilistic": false },
    { "mode": "anti_transfer", "label": "Anti-transfer", "model_version": "turn-detector-at-0.1.0",
      "available": false, "unavailable_reason": "No checkpoint at backend/checkpoints/anti_transfer.pt", "probabilistic": true }
  ],
  "scenarios": [
    { "id": "pause_then_continue", "label": "Pause then continue", "description": "...",
      "trial_type": "simulated", "labelled": true, "transports": ["simulation"] },
    { "id": "free_conversation", "label": "Free conversation", "description": "...",
      "trial_type": "live", "labelled": false, "transports": ["livekit"] }
  ],
  "transports": {
    "simulation": { "available": true, "reason": null },
    "livekit": { "available": false, "reason": "LIVEKIT_URL / LIVEKIT_API_KEY not configured" }
  },
  "policy_defaults": { "version": "policy-0.1", "silence_timeout_ms": 700, "possible_end_ms": 150,
    "complete_threshold": 0.6, "possible_end_threshold": 0.3, "consecutive_frames": 3,
    "min_silence_guard_ms": 100, "max_silence_ms": 2000, "barge_in_min_speech_ms": 200,
    "backchannel_max_ms": 600, "min_speech_ms": 60 },
  "policy_limits": { "silence_timeout_ms": [50, 5000], "complete_threshold": [0, 1], "...": "..." },
  "privacy": { "raw_audio_retention": false, "transcript_display_default": true, "transcript_export_default": false },
  "runtime_profile": "simulation",
  "providers": { "stt": "mock_stt", "llm": "mock_llm", "tts": "mock_tts" },
  "min_percentile_samples": 20,
  "max_evaluation_repeats": 50
}
```
Scenario ids served by the simulation backend: `pause_then_continue`, `short_answer`, `multi_turn`, `barge_in`, `backchannel`. Live LiveKit sessions use `free_conversation`. A replay scenario `replay_eval_set` appears only when a labelled manifest is configured.

### `POST /api/sessions`
Request (`CreateSessionRequest`):
```json
{ "transport": "simulation", "detector_mode": "silence_baseline", "scenario_id": "pause_then_continue",
  "policy_overrides": { "silence_timeout_ms": 600 }, "transcript_enabled": true,
  "transcript_export_enabled": false, "speculative_prefill": false, "trial_label": "demo-1", "seed": 7,
  "client_build": "web-0.1.0" }
```
Response (`CreateSessionResponse`):
```json
{ "session_id": "ses_8f2c...", "trial_id": "tri_41ab...",
  "effective_config": { "detector_mode": "silence_baseline", "model_version": "silence-baseline-0.1.0",
    "decision_policy_version": "policy-0.1", "policy": { "...": "PolicyConfig" },
    "vad": { "kind": "energy", "threshold": 0.5, "min_speech_frames": 2, "min_silence_frames": 2 },
    "frame_ms": 20, "sample_rate": 16000,
    "providers": { "stt": "mock_stt", "llm": "mock_llm", "tts": "mock_tts" },
    "runtime_profile": "simulation", "transport": "simulation", "scenario_id": "pause_then_continue",
    "seed": 7, "transcript_enabled": true, "transcript_export_enabled": false,
    "raw_audio_retention": false, "speculative_prefill": false, "server_version": "0.1.0", "config_hash": "3f9a1c2e" },
  "events_url": "/ws/sessions/ses_8f2c...",
  "livekit": null }
```
For `transport: "livekit"`, `events_url` is `null` and `livekit` is `{ "url", "room", "token", "expires_at" }`. Events then arrive on the LiveKit data channel, topic **`turnsense.events`**, with the same envelope.

Errors: `400` invalid scenario/transport combination, `409` detector unavailable, `422` validation, `503` LiveKit not configured. Body: `{ "detail": "human-readable message" }`.

Display `effective_config` (not the values the user picked) as the session's configuration.

### `POST /api/sessions/{session_id}/interrupt`
Manual "Interrupt agent" button. Goes through the same barge-in cancellation path. `202 {"accepted": true}`; `409` if nothing to interrupt.

### `POST /api/sessions/{session_id}/end`
Stops the session. `202 {"accepted": true}`. The server emits `session_status: ending → ended` and a `trial_summary`.

### `WS /ws/sessions/{session_id}`
Simulation transport event stream. The server sends envelopes as text frames. The client sends nothing (optionally `{"type":"ping"}`). The first message is always `hello`. The socket closes after `session_status: ended`. Connecting to an unknown or finished session closes with code `4404`.

### `POST /api/evaluations`
Runs batches of scripted trials faster than real time, server-side.
```json
{ "detector_modes": ["silence_baseline", "anti_transfer"], "scenario_ids": ["pause_then_continue", "barge_in"],
  "repeats": 3, "seed": 0, "policy_overrides": null, "speculative_prefill": false, "trial_label": "cmp-1" }
```
→ `202` `EvaluationStatus`:
```json
{ "evaluation_id": "eva_...", "status": "running", "total": 12, "completed": 0, "trial_ids": [], "error": null,
  "created_at": "2026-10-22T08:00:00Z" }
```
### `GET /api/evaluations/{id}` → `EvaluationStatus` (poll about once a second).
### `POST /api/evaluations/{id}/stop` → `EvaluationStatus`.

### `GET /api/trials?detector_mode=&scenario_id=&trial_type=&evaluation_id=&limit=50&offset=0`
```json
{ "total": 120, "items": [ TrialListItem, ... ] }
```
`TrialListItem`:
```json
{ "trial_id": "tri_...", "session_id": "ses_...", "evaluation_id": "eva_..." , "created_at": "...", "ended_at": "...",
  "trial_type": "simulated", "transport": "simulation", "scenario_id": "barge_in", "trial_label": null,
  "detector_mode": "silence_baseline", "model_version": "silence-baseline-0.1.0", "decision_policy_version": "policy-0.1",
  "status": "completed", "labelled": true, "metrics": TrialMetrics, "counts": TrialCounts }
```

### `GET /api/trials/{trial_id}`
`TrialListItem` plus:
```json
{ "effective_config": EffectiveConfig,
  "annotations": { "turn_ends_ms": [2140.0], "barge_ins_ms": [5320.0], "backchannels_ms": [], "source": "scenario_script" },
  "tolerance_ms": { "early": 200, "late": 1000, "barge_in": 600 },
  "events": [ Envelope, ... ] }
```
`events` excludes `audio_level`. Transcript envelopes are included only if `transcript_export_enabled` was true.

### Exports
- `GET /api/trials/{trial_id}/export?format=json|csv`: one trial. JSON is the full record. CSV is one row per envelope, with flattened columns.
- `POST /api/trials/export` with body `{ "trial_ids": [...], "format": "json" | "csv" }`: a summary row per trial, including config and metric columns.
- `POST /api/compare?format=json|csv|md` with body `{ "trial_ids": [...] }` or `{ "evaluation_id": "..." }`, plus optional `"group_by": ["detector_mode","scenario_id"]` (default `["detector_mode"]`).

Export responses set `Content-Disposition: attachment; filename=...`. JSON exports carry `"export_meta": { "server_version", "exported_at", "transcripts_included", "raw_audio_included": false, "retention_note" }`.

`ComparisonResult` (`format=json`):
```json
{
  "group_by": ["detector_mode"],
  "groups": [
    { "key": { "detector_mode": "silence_baseline" }, "n_trials": 9,
      "scenario_ids": ["barge_in", "pause_then_continue"], "trial_types": ["simulated"],
      "metrics": { "turn_f1": { "mean": 0.82, "ci95_low": 0.74, "ci95_high": 0.9, "n": 9 } } }
  ],
  "metric_definitions": {
    "turn_f1": { "label": "Turn F1", "better": "higher", "unit": "ratio", "definition": "..." },
    "response_start_p95_ms": { "label": "Response start p95", "better": "lower", "unit": "ms", "definition": "..." }
  },
  "evaluation_set": "sim-scenarios-v1",
  "warnings": ["Groups were run on different scenario sets; comparison is not like-for-like."]
}
```
`mean`, `ci95_low` and `ci95_high` can be `null` when `n` is too small. A 95% CI needs n ≥ 2 trials with a non-null value. Never rank groups. Show `better` as a direction marker only.

## Event envelope (WebSocket text frame / LiveKit data packet)

```json
{ "v": 1, "kind": "turn_prediction", "session_id": "ses_...", "seq": 57, "t_ms": 4281.4,
  "wall_time": "2026-10-22T08:00:04.281Z", "generation_id": null, "payload": { ... } }
```
- `seq` is strictly increasing from 1 per session. A jump means messages were lost: show a gap marker in the timeline. Drop duplicates (`seq` already seen) and anything with a different `session_id`.
- `generation_id` ties messages to a response generation. Once a `turn_decision` `BARGE_IN` has cancelled generation *g*, ignore later `transcript`/`playback` messages for *g*.
- Ignore unknown `kind` values safely and count them in diagnostics.

| kind | payload | notes |
|---|---|---|
| `hello` | `{ protocol_version, server_version, session: SessionInfo }` | first message; `SessionInfo = { session_id, trial_id, transport, trial_type, scenario_id, trial_label, effective_config, started_at }` |
| `session_status` | `{ state, previous_state, reason }` | the UI state comes from these, not from guesses |
| `vad` | `{ speaking, probability, frame_end_ms }` | speech-presence transitions only ("Speech detected" / "Silence") |
| `turn_prediction` | `TurnEvent` (`session_id, sequence, event_type, detector_mode, model_version, event_end_ms, inference_ms, confidence, decision_policy_version, reason_codes`) | emitted when the predicted label changes. A prediction, **not** a decision |
| `turn_decision` | `{ decision, source_event_sequence, last_user_frame_ms, committed_ms, confidence, reason_codes }` | committed by the orchestrator |
| `transcript` | `{ segment_id, role: "user"\|"assistant", text, is_final, revision, start_ms, end_ms }` | update the line with the same `segment_id` in place; `is_final:false` means provisional |
| `trace` | `TraceEvent` (`session_id, trial_id, trace_id, stage, timestamp_monotonic_ms, duration_ms, detector_mode, provider, status, generation_id, detail`) | pipeline stage spans; `stage` values listed below |
| `latency` | `{ metric, value_ms, status, generation_id }` | one discrete sample ("latest event") |
| `latency_summary` | `{ metrics: { [LatencyMetric]: { count, failed, latest_ms, p50_ms, p95_ms, p99_ms, min_samples } } }` | about once a second; percentiles stay `null` until `count >= min_samples` → show "Collecting samples (n/min)" |
| `audio_level` | `{ source: "user"\|"agent", rms_dbfs }` | ≤ 15 Hz, simulation only (live mode measures the mic locally) |
| `playback` | `{ action: "start"\|"end"\|"flush", reason }` | `flush` = queued agent audio was discarded |
| `trial_summary` | `TrialSummaryPayload` (`trial_id, session_id, trial_type, scenario_id, detector_mode, model_version, decision_policy_version, labelled, status, metrics: TrialMetrics, counts: TrialCounts, tolerance_ms, notes`) | sent once at the end |
| `error` | `{ source, code, message, recoverable }` | provider and transport errors are never presented as detector errors |

Trace `stage` values: `vad_speech_start`, `vad_speech_end`, `detector_inference`, `turn_prediction`, `turn_decision`, `stt_partial`, `stt_final`, `llm_request`, `llm_first_token`, `llm_complete`, `tts_request`, `tts_first_audio`, `playback_start`, `playback_end`, `barge_in_detected`, `cancel_requested`, `cancel_acknowledged`, `stale_chunk_rejected`, `frame_gap`, `provider_error`.

`TrialMetrics` fields (all nullable unless noted): `turn_precision, turn_recall, turn_f1, premature_end_rate, missed_end_rate, turn_end_delay_p50_ms, turn_end_delay_p95_ms, response_start_p50_ms, response_start_p95_ms, response_start_p99_ms, detector_inference_p50_ms, detector_inference_p95_ms, detector_inference_p99_ms, false_barge_in_rate, missed_barge_in_rate, barge_in_cancel_p50_ms, stale_chunks_played (int), stale_chunks_rejected (int), dropped_frames (int), frame_trace_coverage`.

`TrialCounts`: `frames, reference_turn_ends, committed_turn_ends, reference_barge_ins, committed_barge_ins, agent_speaking_intervals, responses_started, responses_cancelled, errors` (all ints).

## Metric definitions (shown in UI tooltips)

| Metric | Definition | Better |
|---|---|---|
| Detector inference | Wall time of one detector `process()` call, per 20 ms frame | lower |
| Turn-end delay | Committed `TURN_COMPLETE` time − annotated turn end. Labelled trials only | lower, but not at the cost of premature ends |
| Response start | `last_user_frame_ms` of the committed decision → first response audio frame delivered to playback | lower |
| Barge-in cancellation | `BARGE_IN` committed → playback flush acknowledged | lower |
| Turn precision / recall / F1 | Committed turn ends matched to annotated ends within the tolerance window `[end − early, end + late]` | higher |
| Premature-end rate | Fraction of annotated turns with a commit earlier than `end − early` | lower |
| Missed-end rate | Fraction of annotated ends with no commit in the window | lower |
| False barge-in rate | Fraction of agent-speaking intervals cancelled without an annotated interruption | lower |
| Missed barge-in rate | Fraction of annotated interruptions not cancelled within the barge-in window | lower |

Stage timings overlap, so they do not sum to end-to-end latency.
