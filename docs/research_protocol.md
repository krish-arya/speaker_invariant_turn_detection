# Research Protocol: Anti-Transfer Learning for Speaker-Invariant Turn Detection

Companion to [`goal.md`](../goal.md) §2–§11. Code: [`backend/src/turnsense/research/`](../backend/src/turnsense/research/) and [`backend/src/turnsense/detectors/learned.py`](../backend/src/turnsense/detectors/learned.py).

## 1. Question and hypothesis

**Question.** If the turn detector is discouraged from reusing speaker-identity representations, does it generalise better to unseen speakers at turn boundaries, without unacceptable endpointing delay?

**Hypothesis.** A turn model trained with an anti-transfer (AT) objective against a frozen speaker-identification network will have a smaller *generalisation gap* than an otherwise identical model trained without it. The gap is F1 on seen-speaker held-out audio minus F1 on unseen speakers. Its endpoint delay and premature-end rate should stay comparable or improve.

**Outcomes and how we report them.** All outcomes are reported:

- *Positive*: the gap narrows and unseen-speaker F1 holds or improves.
- *Mixed*: leakage falls but turn quality drops (over-regularisation).
- *Negative*: no reliable change, or worse endpointing.

## 2. Experiment matrix and where it lives in code

| ID | Variant | Code | Runtime mode |
|---|---|---|---|
| E0 | Energy/Silero VAD + fixed silence timeout | `detectors/silence.py` | `silence_baseline` |
| E1 | Learned turn model, no AT (β = 0) | `research/train.py::train_turn_net` with `TrainConfig(beta=0)` | `learned_control` |
| E2 | Same model + AT objective (β > 0) | the same function, `TrainConfig(beta=1.0)` | `anti_transfer` |
| E3 | Policy/threshold sweep on E1/E2 | `PolicyConfig` fields; tune on `val_seen` only | (offline) |
| ablation | Gradient-reversal speaker adversary | `TrainConfig(adversary="grl")` | offline only, **not** anti-transfer |

E1 and E2 are matched. They share the feature config, architecture, seed, initial weights, batch sequence, epochs, optimiser, class weights and model-selection rule. The runtime mapping from P(turn end) to events is also shared (`map_turn_end_probability`), as is the barge-in logic (`speech_overlap_event`). The only difference is the AT term in the loss.

Every variant is evaluated through the same streaming path the orchestrator uses: `FrameBuffer` 20 ms → VAD → `detector.process` → `DecisionPolicy.update`, with `detector.new_turn()` after each committed `TURN_COMPLETE`. See `research/evaluate.py::stream_recording`.

## 3. Model

- **Features** (`research/features.py`, numpy only): 40 log-mel bands (25 ms Hann window, 10 ms hop, 1024-point FFT, 60–7800 Hz). Two pitch features are added from a window-compensated normalised autocorrelation in the 70–400 Hz range, using first-peak picking to avoid octave errors:
  - periodicity;
  - log2(f0 / 100 Hz) when voiced, else 0.

  The signal is left-padded with 15 ms of zeros, so exactly two windows complete per 20 ms frame. Batch and streaming outputs are identical (tested). Pitch and spectral shape carry speaker identity (f0 level, formants) and turn-final prosody (f0 movement). Both are deliberately available.
- **Encoder** (`research/model.py`): three causal Conv1d layers (k = 5, dilations 1/2/4, 64 channels, ReLU), named `conv1..conv3`, followed by a GRU with 96 units (`gru`). The causal receptive field before the GRU is 290 ms.
- **Turn head**: per-frame logits over {SPEECH, HOLD, SHIFT}. The detector uses `P(turn end) = p(SHIFT) / (p(HOLD) + p(SHIFT))`.
- **Streaming**: `TurnNet.step()` carries conv buffers and the GRU state, and equals the full-sequence forward pass (tested). It runs continuously across turns.
- **Speaker network** (`SpeakerNet`): the *same* encoder, plus masked mean-pooling over speech frames, plus a linear speaker classifier. It is trained on **training speakers only**, then frozen.

## 4. Anti-transfer as implemented

Source: Guizzo, Weyde & Tarroni, *Anti-transfer learning for task invariance in convolutional neural networks for speech processing*, Neural Networks 142 (2021) 238–251, [arXiv:2006.06494](https://arxiv.org/abs/2006.06494), [PubMed 34034071](https://pubmed.ncbi.nlm.nih.gov/34034071/), reference code [ericguizzo/anti_transfer](https://github.com/ericguizzo/anti_transfer).

What the paper specifies, checked against the arXiv text, and how we follow it:

| Paper | This project |
|---|---|
| Pre-train a network with the same architecture (up to the AT layer) on an orthogonal task, then freeze it | `SpeakerNet` with the identical `Encoder`, trained for speaker ID on `train` speakers, `requires_grad_(False)` and `eval()` |
| The target network is randomly initialised, not initialised from the orthogonal network | `torch.manual_seed(seed); TurnNet(cfg)` |
| Gram matrix of a layer's feature maps, `G_ij = F_i · F_j` (Eq. 1) | `anti_transfer.gram_matrix`: per example over valid (unpadded) time steps, divided by C·T. A constant scale does not change a cosine |
| `L_AT = β · cos²(vec(G_p), vec(G_t))` (Eq. 2), squared so the term stays in [0, 1] and high similarity is penalised harder | `gram_similarity(similarity="paper")`, averaged over the batch and multiplied by β |
| Total loss = task cross-entropy + Σ over AT layers of L_AT; multiple layers are summed | `anti_transfer_loss` sums over `at_layers` |
| One AT layer per experiment; the best layer varies by task; β = 1 is the default | default `at_layers=("conv3",)`, `beta=1.0`; both configurable (`--at-layer`, `--beta`) |

Deviations and uncertainties, stated plainly:

1. **1-D feature maps.** The paper used VGG16 with 2-D (frequency × time) maps. Our maps are [channels × time]. The Gram matrix is therefore a channel co-activation statistic over time.
2. **Batch handling.** The paper defines the loss per datapoint. The reference code instead builds one Gram matrix over all batch·channel rows, which equals the per-example version only at batch size 1. It then takes a **row-wise** cosine, absolute value, mean, and squares the result. We implement the paper's per-example Eq. 2 as the default and provide the repository's variant as `similarity="repo_rowwise"` for ablation. The reference code also applies β twice; we apply it once (identical at β = 1).
3. **Layer choice for this task** has not been tuned. The default `conv3` is the deepest convolution. The paper reports that layer choice matters and that there is "no overall unequivocal best choice".
4. **Recurrent layer.** The paper uses convolutional layers only. `at_layers` may include `gru`, but that would be an extension beyond the paper.
5. **Different task.** The paper's tasks were utterance-level classification. Ours is frame-level, streaming, three-class turn-state prediction. Whether the method transfers to this setting is exactly what is being tested.

The **gradient-reversal speaker adversary** (`SpeakerAdversary`, `--grl`) is a different invariance method (domain-adversarial training). It exists as an extra ablation and must never be reported as anti-transfer.

## 5. Data, labels and splits

**Manifest.** JSONL with one record per single-speaker stream (see the docstring of `research/data.py`). Each record holds:

- the audio path and channel;
- a pseudonymous `speaker_id` and the `conversation_id`;
- annotated `turns` (`start_ms`, `end_ms`, optional `ipus`, `type`, `interrupted`);
- optional `backchannels` and `overlaps`;
- `condition`, `language` and annotation-schema metadata.

**Frame labels** (100 Hz):

- SPEECH inside IPUs;
- HOLD in pauses between IPUs of one turn;
- SHIFT from the annotated turn end until the speaker talks again, capped at 2.5 s;
- IGNORE elsewhere: before the first turn, after a backchannel, beyond the cap, during overlaps, and after interrupted turns.

**Splits** (`build_splits`). Speakers are selected by a seeded hash, or passed explicitly.

- **`test_unseen`**: every stream of the selected speakers.
- **Excluded**: in a two-party conversation that contains an unseen speaker, the seen partner's stream. It is excluded, and the conversation never enters training.
- **`val_seen`**: held-out *conversations* of training speakers. It is used for model selection and threshold tuning, and is the "seen" side of the generalisation gap.
- **`train`**: everything else.

An assertion prevents any `test_unseen` speaker from appearing in `train`. `split_id` is a hash of the assignments and parameters. Split statistics (speakers, conversations, turns, within-turn pauses, duration, exclusions) are written into every report.

**Discipline.**

- Thresholds and model selection use `train`/`val_seen` only.
- `test_unseen` is evaluated with a frozen configuration.
- After the final test run, change no code or thresholds except documented bug fixes, each followed by a full re-run (goal.md Milestone 5).

## 6. Tolerances and policy: freeze before the final test run

| Parameter | Value | Where |
|---|---|---|
| Early tolerance | 100 ms | `metrics.Tolerance.early_ms` |
| Late tolerance (evaluation window) | 2000 ms | `metrics.Tolerance.late_ms` |
| Silence timeout (E0) | 700 ms | `PolicyConfig.silence_timeout_ms` |
| Complete / possible-end thresholds (E1/E2) | 0.6 / 0.3 | `PolicyConfig` |
| Consecutive frames to commit (probabilistic detectors) | 3 | `PolicyConfig.consecutive_frames` |
| Minimum silence before commit | 100 ms | `PolicyConfig.min_silence_guard_ms` |
| Max-silence fallback commit | 2000 ms | `PolicyConfig.max_silence_ms` |

These values are recorded in every evaluation JSON. Record any change before unblinding `test_unseen`.

## 7. Metrics and statistics

Metrics come from `turnsense.metrics.score_turn_ends`, the same code the live trials use:

- turn-completion precision, recall and F1;
- premature-end rate;
- missed-end rate;
- endpoint delay p50 and p95.

Additional measurements:

- **Generalisation gap**: pooled F1 on `val_seen` minus pooled F1 on `test_unseen`.
- **Confidence intervals**: percentile bootstrap over **speakers**, the unit of generalisation. Pooled metrics are recomputed from resampled speakers; for the gap, each side is resampled independently.
- **Detector inference time** p50/p95/p99 per 20 ms frame, including feature extraction.
- **Speaker-leakage probe** (`research/probe.py`): a linear logistic probe on utterance-pooled mean ⊕ std of a layer's activations, with a within-speaker train/test split of IPUs. It is reported against two reference points:
  - `random_init`: the same architecture with random weights, which already carries voice information;
  - `speaker_net`: the orthogonal network.

  Lower probe accuracy suggests less *linearly decodable* speaker information at that layer. It does not prove that information is absent.

## 8. Synthetic smoke test

`turnsense-research pipeline --synthetic --quick` generates a **synthetic** corpus (`research/synthetic.py`) from harmonic, syllable-structured signals (`sim/voice.py`). The corpus has an **engineered confound**:

- **Training speakers.** High-f0 voices produce mostly multi-phrase turns, so their pauses are mostly HOLDs. Low-f0 voices produce mostly single-phrase turns, so their pauses are mostly SHIFTs.
- **Held-out speakers.** The association is reversed.
- **Prosody.** It is informative but deliberately imperfect: 20% of phrases get the opposite contour.

The run then trains the speaker net, E1 and E2, evaluates E0/E1/E2, runs the probe, and writes `evaluation.md`, `evaluation.json`, `probe.json`, `pipeline.json` and `pipeline.log` under `backend/runs/<timestamp>-synthetic-quick/`. Unless `--no-install` is given, it copies the E1/E2 checkpoints to `backend/checkpoints/` so the live system can select them. Their metadata carries `"synthetic": true`.

**This verifies the plumbing and lets the mechanism be observed under a known confound. It is not evidence for or against the hypothesis on real speech.** The confound is constructed so that a speaker shortcut exists, and real conversations need not have one.

## 9. Real-data options (licence check required before use)

| Corpus | Why it fits | Licence / access | Notes |
|---|---|---|---|
| AMI Meeting Corpus | Close-talk headset channels, word timings, many speakers with IDs, meeting overlap and backchannels | CC BY 4.0 | Multi-party: derive per-speaker streams from headset channels. Turn ends need a two-party-style definition, or annotation of a subset |
| Switchboard-1 Release 2 (LDC97S62) | Two-party telephone speech, separate channels, ~500 speakers, word alignments (MS-State) | LDC licence (fee / membership) | 8 kHz; resampled with the linear resampler (document it) |
| Fisher English (LDC2004S13 / S14 …) | Very many speakers, two-party telephone | LDC licence | Transcripts are segment-level; IPUs via VAD |
| CANDOR | Naturalistic video-call conversations, many speakers | Research licence (check terms) | Check consent and redistribution terms |
| Own recordings | Exact fit to the demo setting | Needs informed consent, a retention policy and a deletion process (project.md §15) | Small scale; good as an extra test set |

Whatever is used, record:

- the dataset ID and version;
- the licence;
- the annotation protocol ([`annotation_guide.md`](annotation_guide.md));
- annotator agreement on a doubly-annotated subset;
- the split statistics.

## 10. Reporting rules

- Report E0, E1 and E2 on the *same* split, policy, tolerances and metric code, and include the split statistics.
- Report the generalisation gap with its CI, not only unseen-speaker F1. Always report delay and premature-end rate alongside F1.
- Report the probe together with the task metrics and both reference points. Do not claim that speaker identity was "removed".
- Report negative and mixed results in full. Never present synthetic results as evidence about real speakers. Never call a variant "best" because it is the proposed method.
- Include the model versions (`turn-detector-{lc,at}-0.1.0+<hash>`), `split_id`, `dataset_id`, β, the AT layer(s), the seeds, the hardware and the run time.

## 11. Results so far: synthetic smoke test only (2026-10-22)

> **SYNTHETIC.** These results come from synthetic voices with an engineered confound and modelled provider latency. They show that the machinery works and how the mechanism behaves. They are not evidence about real speakers or about the hypothesis.

Run: `backend/runs/synthetic-quick-v1/` (`pipeline --synthetic --quick`, seed 0, CPU, 414 s). Split `split-b16506d4a505`: 16 training speakers; 13 val_seen speakers; 8 test_unseen speakers who never appear in training. Models: `turn-detector-lc-0.1.0+25120ffb` (E1, β = 0) and `turn-detector-at-0.1.0+eeac6e93` (E2, β = 1, AT on `conv3`). A repeat run from scratch reproduced every task metric exactly.

**Offline streaming evaluation (test_unseen).** The pipeline is VAD → detector → `DecisionPolicy`, with the same policy and tolerances for every detector. CIs are a speaker bootstrap.

| detector | F1 | premature-end | delay p50 / p95 (ms) | F1 gap (seen − unseen) |
|---|---:|---:|---:|---:|
| E0 silence baseline | 0.845 [0.785, 0.917] | 30.4% | 706 / 716 | 0.006 [−0.082, 0.097] |
| E1 learned control | 0.888 [0.837, 0.947] | 20.3% | 804 / 1040 | −0.019 [−0.091, 0.055] |
| E2 anti-transfer | 0.883 [0.829, 0.944] | 21.5% | 797 / 986 | −0.004 [−0.081, 0.066] |

**The same checkpoints through the full live pipeline.** These runs go through the orchestrator, generations and barge-in, in virtual time (`backend/scripts/run_comparison.py`; report in `pipeline_comparison.md`). Replaying the 16 test_unseen recordings gives F1 of 0.860 (E0), 0.898 (E1) and 0.893 (E2). Premature-end rates are 30.1%, 20.4% and 21.7%. These agree with the offline evaluation to within noise, which cross-checks the two code paths. On the five scripted scenarios with 5 seeds each, E1 and E2 make identical turn decisions: F1 0.931 for both, against 0.920 for E0. None of the 123 trials played a stale audio chunk. Learned-detector inference is 0.58–0.60 ms p50 and 1.4–1.6 ms p99 per 20 ms frame.

**What this shows**

1. **Learning helps with pauses, at a latency cost.** E1 and E2 cut premature endpointing by about a third compared with E0, using phrase-final prosody, but their p95 turn-end delay is about 300 ms worse. The CIs overlap at this sample size. That trade-off is the main E0-vs-learned result to test on real data.
2. **E2 is indistinguishable from E1** on every task metric and on the generalisation gap.
3. **The anti-transfer penalty is not binding here.** Even without the penalty, E1's Gram similarity to the speaker net is only cos² = 0.040, and the penalty takes it to 0.001. The speaker-leakage probe is unchanged: on test_unseen, linear speaker decoding is 92.7%/95.1% (E1, conv3/gru) against 97.6%/90.2% (E2), with a random-init reference of 85–90%. The penalty compares Gram matrices channel by channel between two independently initialised networks, whose channel orders are arbitrary. It can therefore be satisfied almost trivially without removing speaker information. This is consistent with the paper's formulation (§4). Whether it is a weakness of the method in this setting, or a consequence of our choices, is open.
4. **The synthetic testbed did not create the condition the hypothesis needs.** E1 shows no seen/unseen gap despite the engineered confound. Prosody (80% reliable) apparently dominates the speaker shortcut. A testbed in which the control model does not degrade cannot show an improvement from E2.
5. **Barge-in and backchannels are not differentiated by the learned models.** Barge-in uses the shared overlap rule by design (§2), so all variants inherit the baseline's false barge-ins on two-syllable backchannels (about 9% of agent turns in the scripted set).

**Next steps, in order**

1. Make the testbed discriminative before testing a fix. Lower `prosody_reliability`, or strengthen the confound, until E1 shows a significant seen/unseen gap. Report that calibration.
2. Run the planned AT ablations:
   - a β sweep (0.1, 1, 10);
   - AT on `conv1`/`conv2` or several layers;
   - `similarity="repo_rowwise"`;
   - a stronger speaker net (the quick run reaches 0.336 val_seen accuracy against 0.062 chance);
   - a warm-start variant that initialises E2 from the speaker net, so channels correspond. This is a documented deviation from the paper.

   Run the gradient-reversal adversary (`--grl`) as a separate invariance baseline.
3. Move to licensed, annotated real conversations (§9) with a fixed speaker-disjoint split, and freeze tolerances before unblinding the test set (§6).
4. Consider a BACKCHANNEL head, so a learned model can address the false barge-in failure mode that all variants currently share.
