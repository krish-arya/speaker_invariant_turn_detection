# Goals, Research Plan, Metrics, and Acceptance Criteria

## 1. Main goal

Build and evaluate a streaming turn-detection system that is less sensitive to speaker identity and more reliable at deciding when the user has completed a conversational turn in a low-latency voice-to-voice agent.

The project combines a product goal—natural, interruptible voice conversation—with a research goal—testing whether anti-transfer learning improves turn detection across unseen speakers. Both must be measured independently.

## 2. Research question

**Does discouraging transfer of speaker-identity representations into a turn-detection model improve turn-boundary generalisation across unseen speakers, without introducing unacceptable endpointing delay or reducing conversational responsiveness?**

### Hypothesis

A model trained to reduce the influence of speaker-identity features on its turn-state representation will show a smaller performance gap between seen and unseen speakers than a comparable model trained without the anti-transfer objective.

### Null / alternative outcomes

- **Positive:** held-out-speaker turn metrics improve or remain stable, speaker-related leakage decreases, and latency remains acceptable.
- **Mixed:** speaker leakage decreases but turn detection quality worsens, suggesting over-regularisation or removal of useful acoustic cues.
- **Negative:** no reliable reduction in cross-speaker performance gap, or endpointing performance becomes worse than baseline.

All outcomes should be reported honestly. The project is an experiment, not a pre-proven performance claim.

## 3. Product outcomes

1. A working live voice session from browser microphone to streamed spoken answer.
2. A turn detector integrated with the orchestrator and a reliable cancellation path for barge-in.
3. A configurable baseline and anti-transfer model variant using the same event interface.
4. A frontend console that exposes current turn state, session health, and timing.
5. A repeatable evaluation workflow and exportable records.
6. A final report that distinguishes measured results, provisional targets, and limitations.

## 4. Scope boundaries

### Must have

- Real-time microphone capture and streamed response playback.
- VAD/turn-detection separation.
- Turn event contract and session state machine.
- Baseline silence/VAD policy.
- Anti-transfer model experiment or a documented experimental implementation.
- Barge-in cancellation.
- Stage-level and end-to-end latency instrumentation.
- Basic frontend live view and trial export.
- Speaker-disjoint test methodology.

### Should have

- A learned control model without the anti-transfer objective for ablation.
- Backchannel label and scenarios.
- Repeatable scenario runner.
- Local/open fallbacks for core pipeline stages.
- Per-condition result tables and confidence intervals.

### Could have

- Multilingual and code-switching evaluation.
- Configurable detector-policy sweep.
- Human rating of perceived interruption quality.
- A model-information probe that predicts speaker identity from learned representations to quantify remaining speaker leakage.

### Out of scope for initial release

- Voice identity verification, emotion profiling, or demographic inference.
- Multi-party diarisation as a production feature.
- Autonomous training on user conversations without explicit consent.

## 5. Evaluation dataset and protocol

### Data requirements

The benchmark needs conversational audio with enough temporal detail to label turn boundaries. Where possible, retain:

- start/end times for primary speaking turns;
- pauses and speech resumption;
- overlapping speech and interruptions;
- short backchannels such as acknowledgements;
- audio condition (clean/noisy/device/mic distance) and language where licensable and available;
- a pseudonymous speaker identifier used only to create disjoint splits.

### Split discipline

- Split by speaker, not only by utterance or clip, so speakers in the held-out test set do not appear in training.
- Keep the evaluation set fixed after model decisions begin.
- Tune thresholds using training/validation data only.
- Report the number of speakers, conversations, turns, and events in each split.
- Document dataset licence and any manual annotation protocol.

### Annotation policy

Write a short annotation guide defining the conversational turn boundary. Do not define every pause as a turn end. Annotators should consider whether the speaker appears to retain the floor, including short within-turn pauses, trailing phrases, self-repair, and interrupted turns. For a subset of the data, use a second annotator or adjudication pass to estimate annotation agreement.

## 6. Experiment matrix

| ID | Variant | Purpose |
|---|---|---|
| E0 | VAD + fixed silence timeout | Transparent deployable baseline |
| E1 | Learned turn model without anti-transfer objective | Estimate contribution of learning alone (if feasible) |
| E2 | Learned turn model with anti-transfer objective | Test proposed research hypothesis |
| E3 | E2 with threshold/policy sweep | Analyse responsiveness/error trade-off; optional |

Keep input features, output labels, evaluation set, and metric implementation matched across variants wherever possible. Include model and policy version in every trial record.

## 7. Metrics and definitions

### Task quality

- **Turn-completion precision:** fraction of predicted turn completions that match a reference completion within the accepted tolerance window.
- **Turn-completion recall:** fraction of reference turn completions detected within the accepted tolerance window.
- **F1:** harmonic mean of turn-completion precision and recall.
- **Premature-end rate:** fraction of annotated turns for which the system commits completion before the reference turn boundary by more than the allowed early tolerance.
- **Endpoint delay:** detector commit time minus annotated end time; report median, p95, and the full distribution where possible.
- **Missed endpoint rate:** fraction of reference turn ends with no completion decision inside the defined evaluation window.
- **False barge-in rate:** fraction of agent-speaking intervals where the system cancels due to a false user-interruption detection.
- **Missed barge-in rate:** fraction of scripted/annotated user interruptions not acted upon within the agreed timing window.

The tolerance window is a protocol parameter and must be fixed before the final test run. Use a sensible range based on frame resolution; document exact values in the experiment record rather than selecting them after looking at results.

### Speaker invariance / generalisation

- **Held-out speaker score:** task quality measured only on speakers absent from training.
- **Generalisation gap:** seen-speaker validation score minus held-out-speaker test score, calculated with the same metric and evaluation policy.
- **Speaker leakage probe (optional but recommended):** train/evaluate a probe to predict speaker labels from the learned representation. Lower probe performance may suggest less speaker information, but should be interpreted alongside turn-detection quality and the probe protocol.

### Latency and runtime

- Detector inference time p50/p95/p99.
- Turn-end delay from annotated end to committed decision p50/p95/p99.
- Last-user-frame-to-first-response-audio latency p50/p95/p99.
- STT, LLM first-token, TTS first-audio and playback stage timings.
- Dropped-frame rate, cancellation completion time, API/network error rate.

### User-perceived quality

For a small exploratory user test, ask participants to rate interruption frequency, response delay, and conversational naturalness on a short consistent scale. These subjective ratings complement—but do not replace—task metrics.

## 8. Provisional performance targets

These are engineering targets for early planning, not existing results. Reassess them after a baseline measurement and record any changed target.

| Metric | Initial target | Notes |
|---|---:|---|
| p95 turn-decision delay after annotated end | ≤ 300 ms | Measure separately from end-to-end agent latency |
| p95 last-user-frame-to-first-response-audio | ≤ 1.8 s | Hosted pipeline target; report provider and network conditions |
| Frame trace coverage | ≥ 99% of processed frames/events have timestamps | Missing trace records invalidate fine-grained latency claims |
| Stale playback after confirmed barge-in | 0 known stale chunks in scripted tests | Requires generation IDs and playback queue invalidation |
| Held-out speaker evaluation | 100% of final test speakers unseen in training | A design requirement for the main generalisation claim |
| Repeatability | 3 or more repeats for key latency scenarios where feasible | Report hardware/network conditions and variability |

A metric can miss its target and still yield a valid project result, provided the failure is visible, explained, and not concealed.

## 9. Definition of done

The project is ready for final review when:

- [ ] The browser can join a session, stream audio, receive spoken responses, and leave cleanly.
- [ ] The existing baseline can run and is measured with a fixed configuration.
- [ ] The detector uses a documented event schema and is wired into the orchestrator.
- [ ] Barge-in cancels current playback and stale generated audio is not played afterward.
- [ ] The anti-transfer variant is implemented or the remaining blocker is explicitly documented, with no unsupported claim of success.
- [ ] The test split is speaker-disjoint and dataset/annotation policy are documented.
- [ ] Results include task-quality metrics, endpoint-delay distribution, cross-speaker comparison, and stage/end-to-end latency.
- [ ] The frontend shows live state and can export trial metrics/configuration.
- [ ] API keys and raw audio are not exposed or persisted by default.
- [ ] Final documentation includes failed cases, limitations, provider/config versions, and steps to reproduce.

## 10. Work plan and dependencies

### Milestone 1 — Instrument and stabilise baseline

**Outputs:** working pipeline, fixed timeout policy, trace schema, first p50/p95/p99 report.  
**Dependencies:** LiveKit room, Python worker, STT/LLM/TTS adapters.

### Milestone 2 — Detector event contract and state integration

**Outputs:** turn detector interface, orchestrator transitions, barge-in cancellation and scripted tests.  
**Dependencies:** stable event timestamps and session state design.

### Milestone 3 — Dataset and model experiment

**Outputs:** annotation guide, speaker-disjoint splits, baseline metrics, anti-transfer experiment and config.  
**Dependencies:** dataset licensing, labels, model training/runtime environment.

### Milestone 4 — Frontend research console

**Outputs:** live conversation view, state timeline, latency cards, detector selector, trial history/export.  
**Dependencies:** stable server event and metrics API.

### Milestone 5 — Controlled evaluation and final demonstration

**Outputs:** frozen test run, results table, error-case analysis, reproducibility instructions, final live demo.  
**Dependencies:** no code or threshold changes after final test set is unblinded, except documented bug fixes followed by a new run.

## 11. Final report outline

1. Problem and motivation.
2. Related work and definition of anti-transfer learning used in this project.
3. System architecture and integration.
4. Data, annotation, speaker split, and experimental controls.
5. Baselines and anti-transfer implementation.
6. Task results and confidence/variability analysis.
7. Latency and barge-in results.
8. Speaker-generalisation and speaker-leakage analysis.
9. Failure cases, trade-offs, and limitations.
10. Reproduction instructions and future work.
