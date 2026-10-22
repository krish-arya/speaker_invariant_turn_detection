import type {
  DecisionType,
  DetectorMode,
  ErrorSource,
  LatencyMetric,
  ServerState,
  TrialType,
  TurnEventType,
} from "../api/schemas";

/** Browser-side session phase: server states plus two local-only states. */
export type UiPhase = "idle" | "requesting_permission" | ServerState;

/** State colours follow frontend.md §10 and are always paired with text and an icon. */
export type Tone = "neutral" | "active" | "uncertain" | "complete" | "agent" | "danger";

export type IconName =
  | "power"
  | "mic"
  | "micOff"
  | "wave"
  | "hourglass"
  | "spark"
  | "speaker"
  | "hand"
  | "refresh"
  | "alert"
  | "check"
  | "plug"
  | "stop"
  | "info"
  | "gap"
  | "chevron"
  | "download"
  | "filter"
  | "pause"
  | "play"
  | "gear"
  | "dot";

export interface PhaseMeta {
  label: string;
  tone: Tone;
  icon: IconName;
  hint: string;
}

export const PHASE_META: Record<UiPhase, PhaseMeta> = {
  idle: { label: "Ready to connect", tone: "neutral", icon: "power", hint: "Choose a detector and scenario, then start a session." },
  requesting_permission: {
    label: "Requesting microphone access",
    tone: "neutral",
    icon: "mic",
    hint: "Your browser is asking for microphone permission. Allow it to continue.",
  },
  connecting: { label: "Connecting", tone: "neutral", icon: "plug", hint: "Joining the session…" },
  listening: { label: "Listening", tone: "active", icon: "mic", hint: "Speak when you are ready." },
  user_speaking: { label: "You're speaking", tone: "active", icon: "wave", hint: "Speech detected. Pause naturally — a pause is not a turn end." },
  possible_end: {
    label: "Checking whether you're done",
    tone: "uncertain",
    icon: "hourglass",
    hint: "A pause was detected. The agent has not committed to responding yet.",
  },
  generating: { label: "Preparing a response", tone: "complete", icon: "spark", hint: "Your turn was committed as complete. You can interrupt at any time." },
  agent_speaking: { label: "Assistant speaking", tone: "agent", icon: "speaker", hint: "Start speaking to interrupt, or use Interrupt agent." },
  interrupted: { label: "Listening after interruption", tone: "danger", icon: "hand", hint: "The response was cancelled. Keep talking." },
  reconnecting: {
    label: "Reconnecting",
    tone: "uncertain",
    icon: "refresh",
    hint: "Connection lost. Audio during the outage was not processed.",
  },
  error: { label: "Needs attention", tone: "danger", icon: "alert", hint: "Something went wrong. See the details below." },
  ending: { label: "Ending session", tone: "neutral", icon: "power", hint: "Closing the session and collecting the trial summary…" },
  ended: { label: "Session ended", tone: "neutral", icon: "power", hint: "Start a new session to continue." },
};

export const DETECTOR_LABELS: Record<DetectorMode, string> = {
  silence_baseline: "Silence/VAD baseline",
  learned_control: "Learned control",
  anti_transfer: "Anti-transfer",
};

export const TURN_EVENT_LABELS: Record<TurnEventType, string> = {
  CONTINUE: "Continue — user holds the floor",
  POSSIBLE_END: "Possible turn end",
  TURN_COMPLETE: "Turn complete",
  BARGE_IN: "Barge-in",
  BACKCHANNEL: "Backchannel",
};

export const DECISION_LABELS: Record<DecisionType, string> = {
  TURN_COMPLETE: "Turn complete",
  BARGE_IN: "Barge-in detected",
  BACKCHANNEL_IGNORED: "Backchannel — agent keeps speaking",
};

export const ERROR_SOURCE_LABELS: Record<ErrorSource, string> = {
  transport: "Connection",
  stt: "Transcription service",
  llm: "Language model",
  tts: "Speech output",
  detector: "Turn detector",
  vad: "Voice activity detector",
  worker: "Agent worker",
};

export const TRIAL_TYPE_LABELS: Record<TrialType, string> = {
  simulated: "Simulated (scripted audio)",
  replay: "Replay (labelled clips)",
  live: "Live (human session)",
};

export const STATE_LABELS: Record<ServerState, string> = {
  connecting: "Connecting",
  listening: "Listening",
  user_speaking: "User speaking",
  possible_end: "Waiting to confirm end",
  generating: "Agent generating",
  agent_speaking: "Agent speaking",
  interrupted: "Interrupted",
  reconnecting: "Reconnecting",
  error: "Error",
  ending: "Ending",
  ended: "Ended",
};

export interface LatencyDefinition {
  label: string;
  short: string;
  definition: string;
  start: string;
  end: string;
  better: "lower";
  note?: string;
}

export const LATENCY_DEFS: Record<LatencyMetric, LatencyDefinition> = {
  detector_inference: {
    label: "Detector inference",
    short: "Detector",
    definition: "Time for one detector process() call on a 20 ms frame.",
    start: "Detector invocation",
    end: "Detector result",
    better: "lower",
  },
  turn_end_delay: {
    label: "Turn-end delay",
    short: "Turn-end delay",
    definition: "Committed TURN_COMPLETE time minus the annotated turn end.",
    start: "Annotated end of the user turn",
    end: "Orchestrator commits TURN_COMPLETE",
    better: "lower",
    note: "Requires labels — available in labelled evaluation runs only. Lower is better, but not at the cost of premature ends.",
  },
  response_start: {
    label: "Response start",
    short: "Response start",
    definition: "Last user audio frame used for the completion decision → first response audio frame delivered for playback.",
    start: "Last user frame used by the committed decision",
    end: "First response audio frame delivered to playback",
    better: "lower",
  },
  barge_in_cancel: {
    label: "Barge-in cancellation",
    short: "Barge-in cancel",
    definition: "BARGE_IN committed → playback flush acknowledged.",
    start: "BARGE_IN committed",
    end: "Cancellation acknowledged / queue flushed",
    better: "lower",
  },
  stt_final: {
    label: "STT final",
    short: "STT",
    definition: "Speech-to-text final hypothesis latency reported by the STT stage.",
    start: "STT request / audio sent",
    end: "Final transcript received",
    better: "lower",
  },
  llm_first_token: {
    label: "LLM first token",
    short: "LLM",
    definition: "LLM request started → first token received.",
    start: "LLM request",
    end: "First token",
    better: "lower",
  },
  tts_first_audio: {
    label: "TTS first audio",
    short: "TTS",
    definition: "TTS request started → first audio chunk received.",
    start: "TTS request",
    end: "First audio chunk",
    better: "lower",
  },
};

export const PRIMARY_LATENCY_METRICS: LatencyMetric[] = [
  "detector_inference",
  "turn_end_delay",
  "response_start",
  "barge_in_cancel",
];

/** Trace stages that are worth a line in the human-readable timeline. */
export const TRACE_TIMELINE_LABELS: Record<string, string> = {
  llm_request: "Agent response started",
  llm_first_token: "LLM first token",
  llm_complete: "LLM response complete",
  tts_first_audio: "TTS first audio",
  stt_final: "Final transcript received",
  cancel_requested: "Cancellation requested",
  cancel_acknowledged: "Response cancelled",
  stale_chunk_rejected: "Stale audio chunk rejected",
  frame_gap: "Audio frame gap",
};

export const TRACE_SOURCE: Record<string, TimelineSource> = {
  llm_request: "LLM",
  llm_first_token: "LLM",
  llm_complete: "LLM",
  tts_first_audio: "TTS",
  stt_final: "STT",
  cancel_requested: "Orchestrator",
  cancel_acknowledged: "Playback",
  stale_chunk_rejected: "Playback",
  frame_gap: "Transport",
};

export type TimelineSource =
  | "Session"
  | "VAD"
  | "Turn detector"
  | "Orchestrator"
  | "STT"
  | "LLM"
  | "TTS"
  | "Playback"
  | "Transport";

export const ERROR_SOURCE_TO_TIMELINE: Record<ErrorSource, TimelineSource> = {
  transport: "Transport",
  stt: "STT",
  llm: "LLM",
  tts: "TTS",
  detector: "Turn detector",
  vad: "VAD",
  worker: "Orchestrator",
};
