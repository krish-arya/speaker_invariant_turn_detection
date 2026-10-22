import type { TransportStatus } from "../state/sessionReducer";

export interface SourceHandlers {
  /** One raw message (JSON string or object). Validation happens in the store. */
  onMessage(raw: unknown): void;
  onStatus(status: TransportStatus, detail?: string | null): void;
}

/**
 * The single normalised interface the UI consumes, whatever the transport
 * (frontend.md §2): WebSocket (simulation), LiveKit data channel, or the mock script.
 */
export interface SessionEventSource {
  readonly kind: "websocket" | "livekit" | "mock";
  /** Resolves once connected; rejects if the initial connection fails. */
  start(handlers: SourceHandlers): Promise<void>;
  /** Idempotent. Stops the microphone and playback for transports that own them. */
  stop(): Promise<void>;
  /** The live microphone track, for local level metering (LiveKit only). */
  getLocalAudioTrack?(): MediaStreamTrack | null;
  switchAudioInput?(deviceId: string): Promise<void>;
}
