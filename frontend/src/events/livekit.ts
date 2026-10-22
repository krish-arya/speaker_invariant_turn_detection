import {
  ConnectionState,
  DisconnectReason,
  Room,
  RoomEvent,
  Track,
  type RemoteTrack,
} from "livekit-client";
import type { LiveKitJoinInfo } from "../api/schemas";
import type { SessionEventSource, SourceHandlers } from "./types";

/** Data-channel topic the worker publishes envelopes on (docs/api-contract.md). */
export const LIVEKIT_EVENTS_TOPIC = "turnsense.events";

export interface LiveKitSourceOptions {
  deviceId: string | null;
}

function describeDisconnect(reason: DisconnectReason | undefined): string {
  switch (reason) {
    case DisconnectReason.CLIENT_INITIATED:
      return "You left the room.";
    case DisconnectReason.SERVER_SHUTDOWN:
      return "The LiveKit server shut down.";
    case DisconnectReason.PARTICIPANT_REMOVED:
      return "You were removed from the room.";
    case DisconnectReason.ROOM_DELETED:
      return "The session room was closed.";
    case DisconnectReason.JOIN_FAILURE:
      return "Joining the room failed.";
    case DisconnectReason.DUPLICATE_IDENTITY:
      return "The same identity joined from another tab or device.";
    default:
      return "The connection to the session was lost.";
  }
}

/**
 * Live transport: joins the LiveKit room with a server-issued short-lived token,
 * publishes the microphone, plays the agent's audio and forwards structured events
 * from the data channel. The token is used only for `connect` and never stored or shown.
 */
export class LiveKitEventSource implements SessionEventSource {
  readonly kind = "livekit" as const;
  private room: Room | null = null;
  private stopped = false;
  private readonly audioEls = new Set<HTMLMediaElement>();
  private container: HTMLElement | null = null;

  constructor(
    private readonly join: LiveKitJoinInfo,
    private readonly opts: LiveKitSourceOptions,
  ) {}

  async start(h: SourceHandlers): Promise<void> {
    this.stopped = false;
    const room = new Room({
      adaptiveStream: false,
      dynacast: false,
      audioCaptureDefaults: {
        deviceId: this.opts.deviceId ?? undefined,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.room = room;
    const decoder = new TextDecoder();

    room.on(RoomEvent.DataReceived, (payload: Uint8Array, _participant, _kind, topic?: string) => {
      if (topic !== LIVEKIT_EVENTS_TOPIC) return;
      h.onMessage(decoder.decode(payload));
    });

    room.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
      if (this.stopped) return;
      if (state === ConnectionState.Connecting) h.onStatus("connecting");
      else if (state === ConnectionState.Connected) h.onStatus("connected");
      else if (state === ConnectionState.Reconnecting || state === ConnectionState.SignalReconnecting) {
        h.onStatus("reconnecting", "LiveKit is reconnecting.");
      }
    });

    room.on(RoomEvent.Disconnected, (reason?: DisconnectReason) => {
      this.detachAll();
      h.onStatus("disconnected", this.stopped ? null : describeDisconnect(reason));
    });

    room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
      if (track.kind !== Track.Kind.Audio) return;
      const el = track.attach();
      el.setAttribute("data-role", "agent-audio");
      this.audioContainer().appendChild(el);
      this.audioEls.add(el);
    });

    room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
      track.detach().forEach((el) => {
        el.remove();
        this.audioEls.delete(el);
      });
    });

    h.onStatus("connecting");
    await room.connect(this.join.url, this.join.token, { autoSubscribe: true });
    await room.localParticipant.setMicrophoneEnabled(
      true,
      this.opts.deviceId ? { deviceId: this.opts.deviceId } : undefined,
    );
    try {
      // Called within the Start click's activation window where possible.
      await room.startAudio();
    } catch {
      // Autoplay may still be blocked; LiveKit retries on the next user gesture.
    }
    h.onStatus("connected");
  }

  private audioContainer(): HTMLElement {
    if (!this.container) {
      const div = document.createElement("div");
      div.setAttribute("aria-hidden", "true");
      div.style.display = "none";
      document.body.appendChild(div);
      this.container = div;
    }
    return this.container;
  }

  private detachAll(): void {
    this.audioEls.forEach((el) => {
      el.pause?.();
      el.remove();
    });
    this.audioEls.clear();
    this.container?.remove();
    this.container = null;
  }

  getLocalAudioTrack(): MediaStreamTrack | null {
    const pub = this.room?.localParticipant.getTrackPublication(Track.Source.Microphone);
    return pub?.track?.mediaStreamTrack ?? null;
  }

  async switchAudioInput(deviceId: string): Promise<void> {
    await this.room?.switchActiveDevice("audioinput", deviceId);
  }

  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    const room = this.room;
    this.room = null;
    if (room) {
      try {
        await room.localParticipant.setMicrophoneEnabled(false);
      } catch {
        // already unpublished
      }
      await room.disconnect(true);
    }
    this.detachAll();
  }
}
