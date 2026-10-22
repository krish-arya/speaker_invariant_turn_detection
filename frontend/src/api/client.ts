/**
 * Typed REST client for every endpoint in docs/api-contract.md.
 * Responses are validated with zod; a shape mismatch is reported as a contract error
 * instead of leaking malformed data into the UI.
 */
import type { z } from "zod";
import {
  acceptedSchema,
  capabilitiesSchema,
  comparisonResultSchema,
  createSessionResponseSchema,
  evaluationStatusSchema,
  healthSchema,
  trialDetailSchema,
  trialListSchema,
  type Capabilities,
  type ComparisonResult,
  type CreateSessionResponse,
  type DetectorMode,
  type EvaluationStatus,
  type Health,
  type TrialDetail,
  type TrialList,
  type TrialType,
  type Transport,
} from "./schemas";
import { downloadResponse } from "./download";

const RAW_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
export const API_BASE = RAW_BASE.replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly kind: "http" | "network" | "contract";

  constructor(kind: "http" | "network" | "contract", status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const d = (body as { detail: unknown }).detail;
      if (typeof d === "string") return d;
      if (Array.isArray(d)) {
        return d
          .map((item) =>
            item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : "",
          )
          .filter(Boolean)
          .join("; ");
      }
    }
  } catch {
    // body was not JSON — fall through to status text
  }
  return res.statusText || `HTTP ${res.status}`;
}

async function send(path: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch (err) {
    throw new ApiError("network", 0, err instanceof Error ? err.message : "Network request failed");
  }
  if (!res.ok) throw new ApiError("http", res.status, await errorDetail(res));
  return res;
}

async function request<S extends z.ZodTypeAny>(path: string, schema: S, init?: RequestInit): Promise<z.output<S>> {
  const res = await send(path, init);
  let json: unknown;
  try {
    json = await res.json();
  } catch {
    throw new ApiError("contract", res.status, `Response from ${path} was not JSON`);
  }
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    throw new ApiError(
      "contract",
      res.status,
      `Unexpected response from ${path}: ${issue ? `${issue.path.join(".")} ${issue.message}` : "shape mismatch"}`,
    );
  }
  return parsed.data;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

// ---------------------------------------------------------------------------
// Request bodies (camelCase in the UI, snake_case on the wire)
// ---------------------------------------------------------------------------

export interface CreateSessionInput {
  transport: Transport;
  detectorMode: DetectorMode;
  scenarioId: string;
  /** Wire-keyed overrides (only changed policy fields). */
  policyOverrides: Record<string, number> | null;
  transcriptEnabled: boolean;
  transcriptExportEnabled: boolean;
  speculativePrefill: boolean;
  trialLabel: string | null;
  seed: number | null;
  clientBuild: string;
}

export interface CreateEvaluationInput {
  detectorModes: DetectorMode[];
  scenarioIds: string[];
  repeats: number;
  seed: number;
  policyOverrides: Record<string, number> | null;
  speculativePrefill: boolean;
  trialLabel: string | null;
}

export interface TrialFilters {
  detectorMode?: DetectorMode | "";
  scenarioId?: string;
  trialType?: TrialType | "";
  evaluationId?: string;
  limit?: number;
  offset?: number;
}

export type CompareSelection = { trialIds: string[] } | { evaluationId: string };

function compareBody(sel: CompareSelection, groupBy: string[]): Record<string, unknown> {
  return {
    ...("trialIds" in sel ? { trial_ids: sel.trialIds } : { evaluation_id: sel.evaluationId }),
    group_by: groupBy,
  };
}

const enc = encodeURIComponent;

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  health(): Promise<Health> {
    return request("/api/health", healthSchema);
  },

  capabilities(): Promise<Capabilities> {
    return request("/api/capabilities", capabilitiesSchema);
  },

  createSession(input: CreateSessionInput): Promise<CreateSessionResponse> {
    return request(
      "/api/sessions",
      createSessionResponseSchema,
      json({
        transport: input.transport,
        detector_mode: input.detectorMode,
        scenario_id: input.scenarioId,
        policy_overrides: input.policyOverrides,
        transcript_enabled: input.transcriptEnabled,
        transcript_export_enabled: input.transcriptExportEnabled,
        speculative_prefill: input.speculativePrefill,
        trial_label: input.trialLabel,
        seed: input.seed,
        client_build: input.clientBuild,
      }),
    );
  },

  interruptSession(sessionId: string): Promise<{ accepted: boolean }> {
    return request(`/api/sessions/${enc(sessionId)}/interrupt`, acceptedSchema, { method: "POST" });
  },

  endSession(sessionId: string): Promise<{ accepted: boolean }> {
    return request(`/api/sessions/${enc(sessionId)}/end`, acceptedSchema, { method: "POST" });
  },

  createEvaluation(input: CreateEvaluationInput): Promise<EvaluationStatus> {
    return request(
      "/api/evaluations",
      evaluationStatusSchema,
      json({
        detector_modes: input.detectorModes,
        scenario_ids: input.scenarioIds,
        repeats: input.repeats,
        seed: input.seed,
        policy_overrides: input.policyOverrides,
        speculative_prefill: input.speculativePrefill,
        trial_label: input.trialLabel,
      }),
    );
  },

  getEvaluation(evaluationId: string): Promise<EvaluationStatus> {
    return request(`/api/evaluations/${enc(evaluationId)}`, evaluationStatusSchema);
  },

  stopEvaluation(evaluationId: string): Promise<EvaluationStatus> {
    return request(`/api/evaluations/${enc(evaluationId)}/stop`, evaluationStatusSchema, { method: "POST" });
  },

  listTrials(filters: TrialFilters = {}): Promise<TrialList> {
    const q = new URLSearchParams();
    if (filters.detectorMode) q.set("detector_mode", filters.detectorMode);
    if (filters.scenarioId) q.set("scenario_id", filters.scenarioId);
    if (filters.trialType) q.set("trial_type", filters.trialType);
    if (filters.evaluationId) q.set("evaluation_id", filters.evaluationId);
    q.set("limit", String(filters.limit ?? 50));
    q.set("offset", String(filters.offset ?? 0));
    return request(`/api/trials?${q.toString()}`, trialListSchema);
  },

  getTrial(trialId: string): Promise<TrialDetail> {
    return request(`/api/trials/${enc(trialId)}`, trialDetailSchema);
  },

  async exportTrial(trialId: string, format: "json" | "csv"): Promise<string> {
    const res = await send(`/api/trials/${enc(trialId)}/export?format=${format}`);
    return downloadResponse(res, `trial_${trialId}.${format}`);
  },

  async exportTrials(trialIds: string[], format: "json" | "csv"): Promise<string> {
    const res = await send("/api/trials/export", json({ trial_ids: trialIds, format }));
    return downloadResponse(res, `trials_${trialIds.length}.${format}`);
  },

  compare(selection: CompareSelection, groupBy: string[]): Promise<ComparisonResult> {
    return request("/api/compare?format=json", comparisonResultSchema, json(compareBody(selection, groupBy)));
  },

  async exportComparison(
    selection: CompareSelection,
    groupBy: string[],
    format: "json" | "csv" | "md",
  ): Promise<string> {
    const res = await send(`/api/compare?format=${format}`, json(compareBody(selection, groupBy)));
    return downloadResponse(res, `comparison.${format}`);
  },
};

export type Api = typeof api;

/** Resolve the simulation `events_url` (a path) to an absolute ws:// or wss:// URL. */
export function resolveWebSocketUrl(eventsUrl: string, base: string = API_BASE): string {
  if (/^wss?:\/\//i.test(eventsUrl)) return eventsUrl;
  const origin = base || (typeof window !== "undefined" ? window.location.origin : "http://localhost");
  const url = new URL(eventsUrl, origin.endsWith("/") ? origin : `${origin}/`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export function describeApiError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.kind === "network") return "The server could not be reached. Check that the backend is running.";
    if (err.kind === "contract") return `The server response did not match the expected contract. ${err.detail}`;
    return `${err.detail} (HTTP ${err.status})`;
  }
  return err instanceof Error ? err.message : String(err);
}
