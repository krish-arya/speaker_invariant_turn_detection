# Turn-Boundary Annotation Guide (schema `turn-boundary-1`)

This guide is for people annotating conversational audio for the turn-detection benchmark (goal.md §5). The output goes into the manifest format described in [`research_protocol.md`](research_protocol.md) §5.

## 1. What you are marking

A **turn** is a stretch in which one speaker holds the conversational floor: they are the person currently "having their say". A **turn end** is the moment that speaker gives up the floor, so it would be appropriate for the other party to respond.

**A pause is not automatically a turn end.** People pause to think, breathe, search for words, or emphasise a point, and still keep the floor. The detector we are building must learn exactly this difference, so your labels must not just copy silence.

For each speaker stream, mark:

| Field | Meaning |
|---|---|
| `turns[].start_ms` | onset of the first speech sound of the turn |
| `turns[].end_ms` | offset of the last speech sound before the speaker yields the floor |
| `turns[].ipus` | the speech stretches inside the turn (inter-pausal units), separated by silences ≥ 200 ms; optional if you use forced alignment |
| `turns[].type` | `turn` (default) or `backchannel` |
| `turns[].interrupted` | `true` if the turn was cut off by the other party before it was complete |
| `backchannels` | `[start, end]` for short listener responses (see §4) |
| `overlaps` | `[start, end]` where both parties speak at once |

Times are in milliseconds from the start of the audio file. Place each boundary at the acoustic onset or offset of the speech, using the waveform and spectrogram, to within about 20 ms. Do not place boundaries at the midpoint of a silence.

## 2. Deciding whether a pause ends the turn

Ask: **"Had the speaker finished what they were saying, so that a response here would feel natural rather than like an interruption?"** Use everything available:

- **Syntax and meaning**: is the sentence or idea complete? "I went to the shop and…" is not complete. "I went to the shop." usually is.
- **Prosody**: falling pitch at the end of a statement, rising pitch on a question, and final lengthening all suggest an end. Level or continuing intonation suggests the speaker will continue.
- **What happens next**: if the *same* speaker resumes after the pause and the other party did not take or attempt the floor, it was usually a within-turn pause (HOLD). If the other party responds, it was usually a turn end (SHIFT).
- **Gaze and gesture** cannot be used for audio-only data; do not guess them.

When the evidence conflicts, prefer what actually happened in the conversation. Add a `note` field to the turn if it was genuinely ambiguous.

## 3. Hard cases

| Case | Rule |
|---|---|
| Thinking pause ("so… um… I think we should go") | One turn. The filler and pause are inside it (HOLD) |
| Trailing phrase ("…and that's it, really.") | The turn ends after the trailing phrase, not before it |
| Self-repair ("we met on Tues– Wednesday") | One turn. A repair is never a turn end |
| Turn completed after a long pause, with no response in between | One turn, with a long HOLD. Mark it even if the pause is longer than 1 s |
| Question answered by the same speaker ("Why? Because…") | One turn, unless the other party took the floor in between |
| Speaker stops because the other party interrupts | End the turn at the speaker's last sound and set `interrupted: true`. Interrupted turns are excluded from turn-end scoring |
| Two attempts to start ("I– I think") | One turn starting at the first attempt |
| Laughter at the end of a turn | Include it in the turn if it is the speaker's own continuation. Otherwise end the turn before it |
| Turn end then immediate restart by the same speaker, with no response | Treat as one turn (HOLD), unless there is a clear topic shift *and* a long pause (> 2 s) inviting response |

## 4. Backchannels

A **backchannel** is a short listener response that does not claim the floor: "mm-hm", "yeah", "right", "okay", a short laugh, or "wow". Mark it in `backchannels` (or as a turn with `type: backchannel`) on the listener's stream, not as a turn.

- A backchannel does not end the *other* speaker's turn.
- If the listener's response goes on to claim the floor ("yeah, but I think…"), it is a turn, not a backchannel.
- When unsure, a response under about 1 s with no propositional content is a backchannel.

## 5. Overlap

Mark regions where both people speak in `overlaps` on both streams. These regions are ignored for frame labels. Turn starts and ends inside overlap are still annotated normally.

## 6. Privacy

Use the pseudonymous `speaker_id` you are given. Never write names, contact details or other identifying information into annotations or notes. Do not transcribe content beyond what the turn decision needs, unless the protocol asks for transcripts.

## 7. Agreement and adjudication

1. A second annotator independently annotates at least **10% of conversations**, drawn at random and stratified across speakers and conditions. Neither annotator sees the other's labels.
2. Compute agreement on:
   - **Turn-end detection**: treat annotator A's ends as reference and B's as predictions, and report precision/recall/F1 with a ±200 ms window (`turnsense.metrics.score_turn_ends` with `Tolerance(early_ms=200, late_ms=200)`). Repeat with B as reference.
   - **Boundary placement**: the median absolute difference of matched ends, in ms.
   - **HOLD vs SHIFT at each pause** present in both annotations: Cohen's κ.
3. Discuss disagreements in an adjudication pass. Update this guide when a recurring disagreement reveals an unclear rule, and bump the schema version (`turn-boundary-2`, …) if labels change meaning.
4. Report the agreement figures in the final report next to the model results. Model scores above human agreement should be treated with suspicion.

## 8. Checklist per file

- [ ] Every speech stretch of the speaker belongs to exactly one turn or backchannel.
- [ ] Turn ends are at the last speech sound, not in the silence.
- [ ] Within-turn pauses are inside turns (IPUs), not separate turns.
- [ ] Interrupted turns are flagged.
- [ ] The overlap regions are marked.
- [ ] `annotation.annotator` and `annotation.schema` are filled in.
