import type { SessionEventSource, SourceHandlers } from "./types";

export const WS_CLOSE_UNKNOWN_SESSION = 4404;

export interface WebSocketSourceOptions {
  maxReconnects?: number;
  pingIntervalMs?: number;
  WebSocketImpl?: typeof WebSocket;
}

/** Event stream for the simulation transport (`WS /ws/sessions/{id}`). */
export class WebSocketEventSource implements SessionEventSource {
  readonly kind = "websocket" as const;
  private ws: WebSocket | null = null;
  private handlers: SourceHandlers | null = null;
  private stopped = false;
  private everConnected = false;
  private attempts = 0;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly decoder = new TextDecoder();

  constructor(
    private readonly url: string,
    private readonly opts: WebSocketSourceOptions = {},
  ) {}

  start(handlers: SourceHandlers): Promise<void> {
    this.handlers = handlers;
    this.stopped = false;
    return new Promise((resolve, reject) => this.open(resolve, reject));
  }

  private open(resolve?: () => void, reject?: (e: Error) => void): void {
    const h = this.handlers;
    if (!h || this.stopped) return;
    h.onStatus(this.attempts === 0 ? "connecting" : "reconnecting");
    const Impl = this.opts.WebSocketImpl ?? WebSocket;
    let ws: WebSocket;
    try {
      ws = new Impl(this.url);
    } catch (err) {
      reject?.(err instanceof Error ? err : new Error(String(err)));
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.attempts = 0;
      this.everConnected = true;
      h.onStatus("connected");
      this.startPing(ws);
      resolve?.();
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this.ws !== ws) return;
      const data: unknown = ev.data;
      if (typeof data === "string") h.onMessage(data);
      else if (data instanceof ArrayBuffer) h.onMessage(this.decoder.decode(data));
      else if (typeof Blob !== "undefined" && data instanceof Blob) void data.text().then((t) => h.onMessage(t));
    };

    ws.onclose = (ev: CloseEvent) => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.stopPing();
      if (this.stopped) {
        h.onStatus("disconnected");
        return;
      }
      if (ev.code === WS_CLOSE_UNKNOWN_SESSION) {
        const msg = "The server does not know this session or it has already finished (close code 4404).";
        if (!this.everConnected) reject?.(new Error(msg));
        else h.onStatus("error", msg);
        return;
      }
      if (ev.code === 1000) {
        h.onStatus("disconnected", "The server closed the event stream.");
        return;
      }
      if (!this.everConnected) {
        reject?.(new Error(`Could not open the event stream (close code ${ev.code}).`));
        return;
      }
      const max = this.opts.maxReconnects ?? 3;
      if (this.attempts < max) {
        this.attempts += 1;
        h.onStatus("reconnecting", `Event stream dropped (code ${ev.code}). Reconnecting, attempt ${this.attempts} of ${max}…`);
        const delay = Math.min(4000, 500 * 2 ** (this.attempts - 1));
        this.reconnectTimer = setTimeout(() => this.open(), delay);
      } else {
        h.onStatus("error", "The event stream was lost and could not be re-established.");
      }
    };
  }

  private startPing(ws: WebSocket): void {
    this.stopPing();
    const every = this.opts.pingIntervalMs ?? 20000;
    this.pingTimer = setInterval(() => {
      if (ws.readyState === 1) ws.send(JSON.stringify({ type: "ping" }));
    }, every);
  }

  private stopPing(): void {
    if (this.pingTimer) clearInterval(this.pingTimer);
    this.pingTimer = null;
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.stopPing();
    const ws = this.ws;
    if (ws && ws.readyState <= 1) {
      ws.close(1000, "client ended session");
    } else {
      this.ws = null;
      this.handlers?.onStatus("disconnected");
    }
  }
}
