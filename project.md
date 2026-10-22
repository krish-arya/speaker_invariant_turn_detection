# Product Requirements Document — Project Overview

## Speaker-Invariant Turn Detection System for Real-Time Voice-to-Voice Conversation Using Anti-Transfer Learning

**Document type:** Product requirements / project definition  
**Status:** Draft for implementation and evaluation  
**Project context:** B.Tech IT minor project — Low-Latency Voice-to-Voice Agent  
**Source baseline:** Second review presentation dated 30 September 2026  
**Primary companion documents:** [`architecture.md`](./architecture.md), [`goal.md`](./goal.md), [`frontend.md`](./frontend.md)

---

## 1. Executive summary

The project extends a streaming voice-to-voice assistant with a dedicated **speaker-invariant turn detector**. The system must estimate whether a user is still speaking, has paused briefly, has completed a conversational turn, is giving a short backchannel, or is interrupting the agent. Its central research idea is **anti-transfer learning**: reduce the influence of speaker-identity representations that are irrelevant to the turn-boundary task, while preserving acoustic and conversational cues that help estimate when a turn is complete.

The product objective is not simply to detect silence. Voice activity detection (VAD) estimates whether speech is present; turn detection estimates whether the user has finished a turn and whether the agent should begin or stop speaking. A robust turn decision should therefore combine acoustic evidence, partial ASR text where available, recent turn state, and timing. The detector must be evaluated against a simple silence/VAD baseline to establish whether the proposed learning method improves generalisation across unseen speakers, voice characteristics, speaking rates, pauses, and accents.

The existing voice-agent design remains the delivery path:

- Browser client using LiveKit JS for microphone capture and speaker playback.
- FastAPI token service for authenticated room access.
- LiveKit Cloud WebRTC room for real-time transport.
- Python asyncio / LiveKit Agents worker as the session orchestrator.
- Silero VAD, streaming speech-to-text, token-streaming LLM, and streaming text-to-speech adapters.
- A new turn-detection component feeding decisions to the orchestrator.
- Frame-level traces for stage and end-to-end latency, with p50, p95, and p99 reporting.

The presentation's baseline API plan is Deepgram Nova-3 for streaming STT, Groq-hosted Llama 3.1 8B Instant for LLM streaming, and Deepgram Aura-2 for streaming TTS, with faster-whisper, Ollama, and Kokoro-82M as open/local fallbacks. Provider quotas and product availability must be rechecked before evaluation; these are integration choices, not the core research contribution.

## 2. Problem statement

Many voice agents make turn decisions using a fixed silence timeout or VAD state. These heuristics are easy to implement but can respond too early when a speaker pauses to think, or too late when a speaker finishes quickly. They can also behave inconsistently when voice characteristics differ from those seen during development. Both errors harm conversational flow:

- **Premature turn end:** the agent begins speaking while the user is still formulating an utterance.
- **Delayed turn end:** silence after a completed thought makes the conversation feel sluggish.
- **False interruption:** brief sounds or backchannels trigger cancellation of agent speech.
- **Missed barge-in:** the user starts speaking while the agent is talking, but playback is not cancelled promptly.
- **Speaker-dependent performance:** a detector performs differently for voices with different pitch, timbre, accent, speaking rate, or recording conditions.

The system will test whether anti-transfer learning can make turn-boundary representations less dependent on speaker identity without removing cues that are genuinely useful for turn-taking.

## 3. Product vision

Deliver a browser-based, measurable, real-time voice agent in which users can speak naturally, pause, resume, interrupt the assistant, and receive streamed spoken responses. Provide a visible live console that explains the current turn state and exposes latency and evaluation metrics without forcing a user to understand the underlying model.

## 4. Users and stakeholders

### Primary users

- **Demo participant:** speaks with the agent and expects natural turn-taking and immediate interruption support.
- **Researcher / student developer:** configures detector modes, runs repeatable trials, compares baseline and anti-transfer variants, and exports results.
- **Project evaluator / mentor:** needs a clear demonstration, reproducible measurements, and evidence that the proposed learning method was tested rather than merely described.

### Key stakeholder needs

| Stakeholder | Need |
|---|---|
| End user | Natural conversation with minimal awkward silence or agent interruption |
| Developer | Modular interfaces, traceable state transitions, model/provider swapability |
| Research evaluator | Baselines, held-out-speaker tests, metrics, reproducible experiment configuration |
| Privacy-conscious participant | Clear mic status, explicit recording/export controls, minimal retention |

## 5. Product goals

1. Implement a speaker-invariant turn-detection component that can be enabled in the real-time voice-agent pipeline.
2. Compare the anti-transfer approach against a transparent VAD-plus-silence baseline.
3. Reduce premature and delayed turn decisions across a diverse, held-out speaker evaluation set.
4. Preserve low latency and make agent barge-in reliable.
5. Show live turn state and latency traces in a frontend console designed for demos and experiments.
6. Make the result reproducible through documented configurations, trial metadata, and exportable metrics.

## 6. Non-goals

- Building a general-purpose speaker identification or biometric authentication system.
- Identifying a person's real-world identity from voice.
- Replacing the LLM, ASR, or TTS models as the main research contribution.
- Supporting large multi-party meeting diarisation in the initial version.
- Claiming speaker invariance, accuracy improvements, or real-time performance before benchmark results exist.
- Treating model confidence as a guarantee that a turn has ended.

## 7. Core user stories

1. As a participant, I can start a voice session and see when the browser microphone is active.
2. As a participant, I can pause naturally without the agent immediately cutting me off.
3. As a participant, I can interrupt the agent and have ongoing playback cancelled promptly.
4. As a participant, I can finish a thought and receive an answer without an unnecessarily long wait.
5. As a researcher, I can switch between baseline and anti-transfer detector modes for a controlled comparison.
6. As a researcher, I can view predicted turn events, confidence, timestamps, and response latency for each trial.
7. As a researcher, I can export trial metrics and configuration metadata as CSV or JSON.
8. As an evaluator, I can replay a scripted evaluation scenario and inspect whether a decision was early, correct, or late.
9. As a user, I can end a session and clear visible session data.

## 8. Proposed product workflow

1. User opens the web application and grants microphone access.
2. The client obtains a short-lived room token from the FastAPI token service and joins the LiveKit room.
3. The worker streams audio frames through VAD and the turn detector while the STT adapter emits partial and final transcripts.
4. The detector emits a structured turn-state event. The orchestrator combines it with VAD, recent transcript stability, session state, and configured thresholds.
5. When the user turn is judged complete, the LLM response begins. The TTS adapter streams cancellable audio chunks to the client.
6. If the user barges in, playback and speculative work are cancelled, conversation state is reconciled, and the detector returns to listening mode.
7. The tracer records timestamps and model decisions. The frontend presents the live timeline and session summary.

## 9. Functional requirements

| ID | Requirement | Priority | Acceptance evidence |
|---|---|---:|---|
| FR-01 | The client shall start and end a LiveKit voice session. | P0 | Successful session lifecycle test |
| FR-02 | The worker shall ingest audio in timestamped frames, with 20 ms frames as the initial configuration from the existing design. | P0 | Frame continuity and timestamp test |
| FR-03 | The system shall run VAD separately from conversational turn-completion prediction. | P0 | Logs distinguish speech activity from turn state |
| FR-04 | The detector shall emit at least `CONTINUE`, `POSSIBLE_END`, `TURN_COMPLETE`, and `BARGE_IN` events; a `BACKCHANNEL` label is recommended when training data supports it. | P0 | Unit and integration tests for event schema |
| FR-05 | The orchestrator shall use detector output to control response start, continued listening, and playback cancellation. | P0 | End-to-end scripted scenarios |
| FR-06 | The system shall retain a fixed silence/VAD baseline for comparisons. | P0 | Baseline run can be selected in UI/config |
| FR-07 | The system shall support an anti-transfer model variant without changing frontend or LLM/TTS interfaces. | P0 | Same event schema across variants |
| FR-08 | Every trial shall record configuration, event timestamps, per-stage durations, and end-to-end latency. | P0 | Export includes required fields |
| FR-09 | The frontend shall display live turn states, connection state, transcript (if enabled), and latency information. | P0 | UI acceptance checklist in `frontend.md` |
| FR-10 | The frontend shall allow selecting detector mode and running a controlled trial. | P1 | Mode reflected in server-side trial metadata |
| FR-11 | The system shall support local/open fallback adapters for repeatable tests when practical. | P1 | Documented fallback configuration runs |
| FR-12 | The system shall export trial records as CSV and JSON. | P1 | Files open and contain valid headers/schema |
| FR-13 | The system shall provide a privacy-aware option to disable transcript display and avoid retaining raw audio. | P0 | Privacy mode test and UI indication |

## 10. Non-functional requirements

### Latency and responsiveness

- The detector must operate incrementally on streaming input; it must not require a complete recording before making a decision.
- Instrument audio-frame arrival, VAD result, detector result, STT partial/final, LLM first token, TTS first audio chunk, playback start, and interruption cancellation.
- Report p50, p95, and p99 for each relevant stage and end-to-end latency.
- Define the primary response metric as **last user audio frame used for the turn-completion decision to first response audio frame delivered for playback**. Also report **true end of user turn to detector decision** in the labelled evaluation set; these are different metrics.
- Provisional engineering targets, subject to baseline measurements: p95 turn-decision delay at or below 300 ms after annotated turn end; p95 last-user-frame-to-first-response-audio at or below 1.8 seconds in the hosted stack. These are targets, not established results. Any target changes must be recorded in the experiment configuration.

### Reliability

- Network or provider failures must be shown as recoverable errors, not falsely labelled as turn-detection errors.
- A model or adapter timeout must not deadlock the voice session.
- Late events from a cancelled response must not resume obsolete playback.
- The system must handle browser permission denial, dropped connection, provider rate limits, and worker restart gracefully.

### Modularity

- Turn detector, STT, LLM, TTS, transport, and tracing interfaces must be decoupled.
- The detector must be swappable through a configuration option.
- Hosted and local model results must use a common event schema.

### Privacy and security

- Require microphone permission through the browser's normal permission flow.
- Do not store raw audio by default.
- Display whether transcript or event logging is enabled.
- Keep API keys on the server/worker, never in frontend bundles.
- Use short-lived room tokens and validate room/session access on the server.
- Document that evaluation exports may contain transcripts or derived events if those features are enabled.

## 11. Research hypothesis and method

**Hypothesis:** reducing representational similarity to speaker-identity features that are not required for turn-boundary prediction can improve turn-detection generalisation across unseen speakers while maintaining or improving event timing.

The term *anti-transfer learning* is used in prior speech-processing research to describe discouraging transfer of features learned for an orthogonal, potentially confounding task. In this project, the orthogonal task is speaker identity, and the target task is conversational turn state. The specific model/loss must be selected, implemented, and documented during experimentation; the phrase alone does not specify an architecture.

Suggested controlled comparison:

- **Baseline A:** VAD plus fixed silence threshold.
- **Baseline B:** a learned turn detector without anti-transfer objective, where feasible.
- **Proposed:** turn detector trained with an anti-transfer objective intended to reduce speaker-identity leakage.

Evaluate on speaker-disjoint splits. Keep the audio preprocessing, event labels, decision policy, and evaluation scripts consistent across variants. Report results by speaker group and acoustic condition where sample sizes permit. Do not claim that speaker identity has been removed unless a separate probe or equivalent test supports that claim.

## 12. Success metrics

Primary metrics:

- Turn-completion precision, recall, and F1.
- Premature-end rate: detector ends the turn before the annotated conversational turn ends.
- Delayed-end rate and median/p95 delay after the annotated turn end.
- False barge-in rate and missed-barge-in rate.
- Cross-speaker generalisation gap between seen-speaker validation and held-out-speaker test sets.
- End-to-end voice response latency p50/p95/p99.

Secondary metrics:

- Backchannel misclassification rate, if that label is implemented.
- False-start and cancellation frequency.
- Per-stage latency, model inference duration, dropped-frame rate, and provider error rate.
- Subjective conversational quality feedback from a small structured user test.

Minimum success criterion for the research claim: report a reproducible comparison on a speaker-disjoint test set, with confidence intervals or repeated runs where feasible. Whether the anti-transfer variant wins must be decided by results; a negative result is still reportable if the evaluation is sound.

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No appropriately labelled turn-boundary dataset | Model quality cannot be measured reliably | Start with a small manually labelled set and a documented public dataset where licensing permits |
| Anti-transfer objective removes useful prosodic information | Lower turn accuracy despite lower speaker leakage | Track task quality and speaker leakage jointly; tune objective strength |
| Hosted API quota or network variation contaminates benchmark | Noisy latency comparison | Separate local detector latency from network pipeline latency; save provider/mode metadata |
| VAD mistaken for turn detector | Premature endpointing | Keep distinct event types and explicit baseline |
| Transcript partials fluctuate | Unstable speculative LLM calls | Gate speculative prefill on partial stability; cancel and reconcile obsolete work |
| User audio is retained unintentionally | Privacy risk | No raw audio retention by default; explicit opt-in for any recording |
| Demo UI shows false precision | Misleading evaluation | Label confidence and provisional thresholds honestly; expose raw timestamps |

## 14. Release phases

### Phase 1 — Instrumented baseline

Connect the existing LiveKit/STT/LLM/TTS pipeline, keep the silence/VAD baseline, and establish reliable event and latency logs.

### Phase 2 — Turn detector interface

Implement the shared event schema and wire detector output to the orchestrator. Test continue, complete-turn, and interruption scenarios.

### Phase 3 — Anti-transfer experiment

Prepare/label the dataset, create speaker-disjoint splits, train or adapt the proposed model, and benchmark against baselines.

### Phase 4 — Frontend evaluation console

Deliver the live conversation UI, state timeline, latency panels, detector selector, trial history, and export workflows.

### Phase 5 — Final evaluation/demo

Run repeatable trials, report error cases and latency distributions, demonstrate barge-in, and publish the reproducibility notes.

## 15. Open decisions

- Which dataset(s) and licensing terms will be used for training and evaluation?
- Will the first model use acoustic features only, or combine audio embeddings with ASR partials?
- Which anti-transfer implementation will be tested, and how will speaker-information leakage be measured?
- What final latency and accuracy targets are realistic after baseline measurement?
- Will raw audio ever be retained for debugging, and if so, what explicit consent and deletion process applies?
- Is multilingual/code-switched evaluation in the initial scope or a stretch goal?

## 16. Source and design basis

### Existing project baseline

The 30 September 2026 second-review presentation establishes the current integration plan: LiveKit Cloud / LiveKit Agents, Silero VAD, Deepgram Nova-3 streaming STT, Groq Llama 3.1 8B Instant, Deepgram Aura-2 streaming TTS, local fallbacks, a Python asyncio worker, speculative prefill, clause chunking, barge-in, history truncation, and a frame-level tracer reporting p50/p95/p99.

### External research context

- Anti-transfer learning for task invariance in speech processing describes discouraging transfer of features from an orthogonal/confounding task: [PubMed record](https://pubmed.ncbi.nlm.nih.gov/34034071/) and [ScienceDirect article](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002045).
- A recent review provides broader context on turn-taking modelling, acoustic/text inputs, and the difference between silence-based segmentation and turn-taking decisions: [Turn-Taking Modelling in Conversational Systems: A Review of Recent Advances](https://www.mdpi.com/2227-7080/13/12/591).

External papers motivate the experiment; they do not demonstrate that this project has already achieved speaker invariance or better latency.
