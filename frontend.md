# Frontend Product Specification

## 1. Purpose and priority

The frontend is the most important user-facing deliverable for the project. It must do two jobs well:

1. **Voice-agent experience:** let a participant hold a natural, low-latency voice conversation, see when the system is listening or speaking, and interrupt the agent.
2. **Research/evaluation console:** let the project team switch detector modes, run repeatable trials, inspect live turn decisions and latency, compare results, and export evidence for the final report.

Do not build a generic chat dashboard with a microphone button added as an afterthought. The central interface is a **live audio session and its turn timeline**. The layout must make it obvious what the system believes is happening, what the user can do, and where latency is coming from.

## 2. Recommended frontend stack

- **Framework:** React + TypeScript. Next.js is acceptable if it fits the deployment setup; a lightweight React/Vite app is also sufficient.
- **Real-time transport:** LiveKit JS SDK for joining/leaving rooms, microphone publication, and agent audio playback.
- **API layer:** typed service functions for session creation, configuration, trial history, and exports.
- **Event updates:** LiveKit data messages or an authenticated WebSocket/SSE channel, according to backend choice. The UI should consume a single normalised event interface regardless of transport.
- **Charts:** a lightweight chart library for latency history and detector confidence; do not let charts block live audio updates.
- **Styling:** accessible responsive component system with design tokens. Keep the demo visually calm, technical, and legible rather than overly decorative.

These are recommendations, not dependencies already confirmed in the source presentation. Preserve the backend's existing LiveKit JS / FastAPI plan.

## 3. Design principles

1. **State before decoration:** connection, listening, processing, speaking, and interrupted states must be immediately distinguishable.
2. **One primary action at a time:** the main control changes from Start session to End session; interruption should be directly available during playback.
3. **Live timeline over dense logs:** show a human-readable event timeline by default and put raw JSON in a collapsible developer panel.
4. **Progressive disclosure:** users see the essentials; researchers can expand metrics, event details, and advanced configuration.
5. **Honest status:** distinguish `VAD: speech detected`, `Turn: possibly complete`, and `Turn: committed complete`. Never display a model prediction as ground truth.
6. **Latency is contextual:** display the metric's start/end definition, the selected detector mode, and whether values are live estimates or completed-trial measurements.
7. **Privacy is visible:** microphone-active state and transcript/recording settings must not be hidden in a settings submenu.
8. **Keyboard and screen-reader support:** every action must be reachable without a mouse, with visible focus and useful labels.

## 4. Information architecture

Recommended navigation:

- **Live Console** — default landing page; voice session, turn timeline, live latency, transcript toggle.
- **Evaluation** — select detector mode and scenario, run trials, view trial outcome and compare modes.
- **Sessions / Results** — searchable recent trial records, aggregate metrics, export controls.
- **Settings** — audio device, language if supported, privacy settings, advanced endpointing thresholds, connection diagnostics, about/version.

For an initial demo, Live Console, Evaluation, and Results can be tabs on one page rather than separate routes. Do not delay implementation for complex navigation.

## 5. Main screen wireframe

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Logo / Project name              Connection: Connected   Settings       │
├─────────────────────────────────────────────────────────────────────────┤
│ LIVE CONSOLE                                                             │
│ Detector: [Anti-transfer ▾]   Scenario: [Free conversation ▾] [Start]     │
│                                                                         │
│  ┌──────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ VOICE SESSION            │  │ TURN STATE / LIVE EVENT TIMELINE    │  │
│  │                           │  │  10:21:04 Listening                 │  │
│  │       [Waveform /         │  │  10:21:05 User speaking             │  │
│  │        audio level]       │  │  10:21:06 Possible end (0.72)       │  │
│  │                           │  │  10:21:06 User continued             │  │
│  │  Listening...             │  │  10:21:08 Turn complete (0.91)      │  │
│  │  [End session]            │  │  10:21:08 Agent generating          │  │
│  └──────────────────────────┘  └─────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ LATENCY                                                            │  │
│  │ Detector 7 ms | Turn-end delay 210 ms | First audio 1.12 s         │  │
│  │ [Stage timing mini-chart]  [View detailed trace]                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Transcript (optional)  [Show transcript]                                │
│  [Clear session view]                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

Wireframe values are illustrative only. Do not ship these values as hard-coded metrics; render real events and show an empty/placeholder state before the first measurement.

## 6. Page and component specifications

### 6.1 Application header

**Purpose:** anchor the user and surface connection health.

Components:
- Project title, short subtitle: “Real-time voice agent / turn-detection research console”.
- LiveKit/session connection badge: Connecting, Connected, Reconnecting, Disconnected, or Error.
- Current detector mode badge when a session is active.
- Settings link.

Requirements:
- Connection status must be driven by transport events.
- On reconnect, show a non-blocking banner and avoid implying that audio was continuously processed during the outage.
- Show app/build version in the About/diagnostics panel to support reproducibility.

### 6.2 Detector mode and scenario bar

Controls:
- Detector mode selector: `Silence/VAD baseline`, `Anti-transfer`, and optionally `Learned control` if implemented.
- Scenario selector: `Free conversation`, `Pause then continue`, `Short answer`, `Barge-in`, and `Evaluation script` (only list scenarios the backend actually supports).
- Primary `Start session` / `Run trial` action.
- Advanced configuration drawer for endpoint threshold, policy version, language, and experimental flags.

Behaviour:
- Freeze detector mode and thresholds when a session/trial starts. If the user changes configuration mid-session, require ending the session first or clearly label the change as applying to the next trial.
- Persist the actual server-accepted configuration in the trial record; frontend-selected values alone are not sufficient.
- Do not show model variants that the backend has not registered as available.

### 6.3 Voice session card

Elements:
- Large, accessible session status label: Ready, Connecting, Listening, User speaking, Waiting to confirm end, Generating response, Agent speaking, Interrupted, Recovering, or Ended.
- Microphone state icon plus text—not colour alone.
- Animated waveform or simple audio-level meter while audio is active. Use a lightweight visualisation and cap updates so it does not degrade streaming performance.
- Main action: Start session before connecting; End session when active.
- Optional `Interrupt agent` action while the agent is speaking. In normal conversation, user speech should also trigger the barge-in pathway automatically.
- Audio input selector if supported by the browser and LiveKit SDK.
- A short error area for permission denial, device unavailable, or connection failure.

Behaviour:
- Ask for microphone permission only after the user clicks Start session.
- If permission is denied, explain how to enable it and leave a clear Retry option.
- On End session, stop publishing microphone audio, stop playback, leave the room, and reset ephemeral live view state. Do not silently retain the transcript in a later session.
- Do not disable all controls during LLM/TTS generation; the user must be able to end or interrupt.

### 6.4 Turn state / event timeline

This is the signature frontend component and should receive the most design attention.

Display each committed or informative event as a compact item with:
- relative timestamp (e.g. `+4.28 s`) and optional local time;
- event label;
- source: VAD, turn detector, orchestrator, STT, LLM, TTS, playback, transport;
- confidence only when supplied by a probabilistic detector;
- short reason codes/details when available;
- expandable raw payload in developer mode.

Recommended event labels:
- `Listening`
- `Speech detected`
- `Possible turn end`
- `Turn complete`
- `Agent response started`
- `Agent audio started`
- `Barge-in detected`
- `Response cancelled`
- `Backchannel` (only if implemented)
- `Connection warning`

Interaction:
- New events appear at the top or bottom consistently; default to newest-at-bottom for chronological scanning.
- Provide Pause/Resume auto-scroll for debugging.
- Allow selecting an event to highlight relevant trace spans.
- Separate predictions from decisions: e.g. `Turn detector predicted POSSIBLE_END` vs `Orchestrator committed TURN_COMPLETE`.
- Include filters for turn events, pipeline stages, and errors.
- Limit in-memory event count (e.g. last 500 items) and provide pagination/load-more for trial history; do not allow an unbounded event array to grow through long sessions.

Empty state: “Start a session to see live turn decisions and pipeline events.”

### 6.5 Latency overview

Show a small number of metrics first:

1. **Detector inference** — model invocation to result, p50/p95/p99 after enough samples.
2. **Turn-end delay** — annotated turn end to committed decision; this requires labels and is available only in evaluation runs.
3. **Response start latency** — last user audio frame used for completion decision to first response audio frame delivered for playback, matching the project's declared end-to-end metric.
4. **Barge-in cancellation** — detected interruption to acknowledgement/queue flush where measurable.

UI requirements:
- During a live session, show current/most recent values and label them `live sample` or `latest event`.
- For aggregates, show sample count and p50, p95, p99. Do not show percentiles when there are too few valid samples; show `Collecting samples` instead.
- Show a tooltip or info popover with metric definitions.
- Show `—` for unavailable/undefined metrics rather than `0 ms`.
- Make failed stages visible and do not include failed/timeout values silently in successful latency percentiles.
- A stage breakdown mini-chart can show Audio → VAD → Turn detector → STT → LLM → TTS → Playback, but explain that stages overlap; their durations do not necessarily sum to end-to-end duration.

### 6.6 Transcript panel

Purpose: help debug turn timing without turning the interface into a text-chat product.

Controls:
- Transcript display toggle, default based on privacy setting.
- Partial/final transcript styling.
- Clear transcript for current session.
- Small marker on each transcript segment to show whether it was partial or final and to link it to timestamps.

Requirements:
- Clearly label partial hypotheses as provisional.
- When a partial changes, update the relevant line rather than duplicating every revision into the visible transcript. Raw partials may remain in developer trace if needed and configured.
- If transcript display is disabled, do not render transcript text in the UI. Whether transcript text is retained server-side must follow the configured privacy policy and be disclosed separately.
- Do not infer or show speaker identity from voice. In the initial two-party agent flow, use `You` and `Assistant` labels only.

### 6.7 Evaluation page

Purpose: run comparable test conditions and make the experimental method visible.

Controls:
- Detector variant.
- Scenario or test-set selection.
- Number of repeats, with sane limits.
- Optional fixed seed / trial label where supported.
- Start, Stop, and Retry controls.
- Clear summary of effective model version, policy version, input mode, runtime profile, and config.

Results table columns:
- Trial ID and date/time.
- Detector mode/model version.
- Scenario ID.
- Outcome status.
- Turn F1 or applicable task score when ground truth exists.
- Premature-end / missed-end rate when labels exist.
- Turn-end delay p50/p95.
- Response-start latency p50/p95/p99.
- Barge-in result.
- Export/detail action.

Requirements:
- Do not report accuracy/F1 for unlabelled free conversation; show latency and event logs instead.
- Clearly distinguish simulated/scripted tests from human live sessions.
- Show missing ground truth and small sample counts.
- For comparative tables, ensure both variants use the same scenario/test set and filter conditions.

### 6.8 Results and comparison page

Features:
- Select two or more completed trial groups.
- Compare baseline vs anti-transfer across task quality, endpoint delay, latency, and false interruption.
- Show per-metric sample counts and confidence intervals when computed by the backend.
- Plot latency distributions or percentile lines; avoid charts that hide outliers.
- Filter by runtime profile, model version, and scenario.
- Export aggregate CSV/JSON and a human-readable Markdown summary suitable for the project report.

Interpretation rules:
- Visually mark whether a lower or higher value is desirable per metric; do not imply that the lowest latency is always best if it increases premature endpoints.
- Display the evaluation-set/split identifier so that results cannot be separated from the data they represent.
- Never label the anti-transfer variant “best” solely because it is the proposed method.

### 6.9 Settings and privacy

Sections:

**Audio:** selected microphone, input level test, supported sample rate if available.

**Conversation:** language, transcript visibility, auto-start playback preference where supported.

**Evaluation:** default detector mode, scenario defaults, advanced threshold fields, configuration version.

**Privacy:** transcript display/export toggle, raw-audio recording status (off by default), clear current session, retention note.

**Diagnostics:** LiveKit connection, API health, model/provider name (non-secret), browser/device metadata appropriate for debugging, app build and detector versions.

Never display API keys, JWTs, provider credentials, complete secret-bearing URLs, or hidden tokens.

## 7. Interaction state model

The frontend should represent at least these states:

| State | Main label | Primary available action | Key UI behaviour |
|---|---|---|---|
| `idle` | Ready to connect | Start session | No mic streaming; show setup hint |
| `requesting_permission` | Requesting microphone access | Cancel if possible | Explain browser permission prompt |
| `connecting` | Connecting | Cancel | Show progress; disable duplicate Start |
| `listening` | Listening | End session | Show low-level input activity if available |
| `user_speaking` | You're speaking | End session | Mark speech event and active waveform |
| `possible_end` | Checking whether you're done | End session | Do not imply the agent has already committed to responding |
| `generating` | Preparing a response | Interrupt / End session | Show generation and latency updates |
| `agent_speaking` | Assistant speaking | Interrupt / End session | Show playback state; maintain barge-in detection |
| `interrupted` | Listening after interruption | End session | Show cancelled response and resume input processing |
| `reconnecting` | Reconnecting | End session | Freeze or mark gaps in timeline; note lost intervals |
| `error` | Needs attention | Retry / End session | Explain cause, preserve safe diagnostics |
| `ended` | Session ended | Start new session | Clear live-only data per privacy settings |

The client should derive state from backend/transport events and guard against stale events using session IDs and response generation IDs.

## 8. Frontend data contracts

Use TypeScript types (illustrative):

```ts
export type TurnEventType =
  | "CONTINUE"
  | "POSSIBLE_END"
  | "TURN_COMPLETE"
  | "BARGE_IN"
  | "BACKCHANNEL";

export interface TurnEvent {
  sessionId: string;
  sequence: number;
  eventType: TurnEventType;
  detectorMode: "silence_baseline" | "anti_transfer" | "learned_control";
  modelVersion: string;
  eventEndMs: number;
  inferenceMs?: number;
  confidence?: number | null;
  decisionPolicyVersion: string;
  reasonCodes?: string[];
}

export interface TraceEvent {
  sessionId: string;
  trialId?: string;
  traceId: string;
  stage: string;
  timestampMonotonicMs: number;
  durationMs?: number;
  status: "ok" | "error" | "cancelled" | "timeout";
  provider?: string;
}

export interface SessionStatus {
  sessionId: string;
  state:
    | "connecting" | "listening" | "user_speaking" | "possible_end"
    | "generating" | "agent_speaking" | "interrupted"
    | "reconnecting" | "error" | "ended";
  detectorMode: string;
  modelVersion?: string;
  updatedAt: string;
}
```

Keep runtime validation at the API/event boundary. TypeScript types alone do not validate incoming WebSocket messages. Unknown event types should be ignored safely and logged to diagnostics without crashing the session.

## 9. Accessibility and usability requirements

- All controls must have accessible names and visible keyboard focus.
- Do not use colour as the only indicator of listening/speaking/error state; pair colour with text and iconography.
- Waveform and animation must respect reduced-motion preferences.
- Live event updates should not trigger excessive screen-reader announcements. Provide a polite summary region for state changes and leave the full timeline navigable separately.
- Maintain readable contrast and test at 200% zoom.
- Responsive layout: on narrow screens, stack the voice session, event timeline, and latency cards; do not require horizontal scrolling for primary actions.
- Provide meaningful empty, loading, permission-denied, reconnecting, no-results, and error states.

## 10. Visual direction

Use a clean technical-console visual language:

- Neutral dark or light background with strong contrast; support one mode well before adding theme switching.
- Typography hierarchy: clear page title, section title, concise labels, readable timestamps.
- Distinct but restrained state colours: neutral for idle, blue for active/listening, amber for uncertain/possible end, green for completed decision, red for errors/cancelled interruption. Always pair colour with label/icon.
- Use monospace only for IDs, timestamps, model versions, and raw event payloads.
- Cards should organise information, not obscure it. Avoid oversized decorative charts or unnecessary gradients.
- Keep the live status, main control, and interruption path visible above the fold on desktop.

## 11. Frontend performance requirements

- UI rendering must not block audio transport or processing.
- Batch/coalesce high-frequency level and trace updates; do not re-render the entire page for every 20 ms frame.
- Animate at a capped frame rate and stop animations when the session is inactive.
- Virtualise long event histories if necessary.
- Use bounded buffers for live events and remove old ephemeral data when a session ends.
- Handle duplicated/out-of-order messages with sequence numbers; show a warning if sequence gaps are detected.
- Avoid doing model inference or audio feature extraction on the main UI thread unless explicitly justified and benchmarked.

## 12. Error and edge-case UX

| Failure | User-facing behaviour | Developer detail |
|---|---|---|
| Mic permission denied | Explain permission and offer Retry | Browser permission state |
| No microphone available | Offer device check / refresh | Device API error |
| LiveKit join failed | Show connection failure and Retry | Sanitised service error and session ID |
| STT provider rate limit | Explain transcription service is temporarily unavailable; keep state truthful | Provider code, retry timing if known |
| LLM failure | Show response error; keep user able to retry | Request ID and provider status without secret |
| TTS failure | Explain speech output failed; optional text response only if supported | TTS stage trace |
| Detector timeout | Mark detector status unavailable; use documented fallback only if configured | Fallback reason and mode switch |
| Event gap/reconnect | Mark timeline gap; don't interpolate fake state events | Sequence IDs and reconnect times |
| Barge-in cancellation delay | Show “Stopping response…” until acknowledged | Cancel request/ack trace |
| No labelled ground truth | Hide accuracy/F1 and show a clear “not available for this session” label | Trial type and label flag |

## 13. Frontend acceptance checklist

### Core live session

- [ ] Start requests microphone permission only after user action.
- [ ] User can connect, speak, hear streamed response, interrupt, end, and reconnect.
- [ ] UI state matches backend session state and transport state.
- [ ] End session stops microphone publication and audio playback.
- [ ] Old TTS chunks cannot appear after a confirmed cancellation.

### Turn detection visibility

- [ ] VAD speech activity and turn state are shown as different signals.
- [ ] Possible-end predictions are distinguishable from committed turn-complete decisions.
- [ ] Detector mode, model version, policy version, and confidence (if present) are visible in event details.
- [ ] Timeline can be paused, filtered, and inspected without breaking audio.

### Metrics / evaluation

- [ ] Latency definitions are documented in the UI.
- [ ] p50/p95/p99 are shown only when a sufficient valid sample is available.
- [ ] Accuracy metrics are hidden or labelled unavailable for unlabelled sessions.
- [ ] User can run comparable baseline and anti-transfer trials.
- [ ] Trial results record the actual server-side detector configuration.
- [ ] CSV and JSON export are valid and include experiment metadata.

### Privacy / accessibility

- [ ] Raw audio is not retained by default.
- [ ] Transcript display is optional and clearly labelled.
- [ ] API secrets/tokens never reach client code or visible logs.
- [ ] Primary actions work with keyboard only.
- [ ] State is never communicated by colour alone.
- [ ] Permission, reconnect, and provider errors have understandable messages.

## 14. Suggested implementation order

1. Build the page frame, connection indicator, Start/End session, and LiveKit audio round trip.
2. Add the state model and a mock event stream so the timeline UI can be developed independently from detector training.
3. Connect real orchestrator/turn events and handle reconnection and stale session IDs.
4. Add latency cards and trace details, then validate metrics against server logs.
5. Add detector/scenario selection and persist effective configurations into trial records.
6. Add labelled evaluation, comparison charts, and CSV/JSON/Markdown export.
7. Run keyboard/accessibility checks, privacy checks, and long-session stability tests.

**Frontend release gate:** a demo is not complete just because speech can be heard. The interface must accurately show whether the system is listening, predicting a turn end, committing a turn end, generating audio, and cancelling an interruption—and it must connect those visible decisions to traceable measurements.
