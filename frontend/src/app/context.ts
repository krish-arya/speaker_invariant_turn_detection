import { createContext, useContext } from "react";
import type { Capabilities, Health } from "../api/schemas";
import type { SessionController } from "../state/sessionController";

export type TabId = "live" | "evaluation" | "results" | "settings";

export const TABS: Array<{ id: TabId; label: string }> = [
  { id: "live", label: "Live Console" },
  { id: "evaluation", label: "Evaluation" },
  { id: "results", label: "Results" },
  { id: "settings", label: "Settings" },
];

export interface AppContextValue {
  caps: Capabilities;
  health: Health | null;
  /** True when the UI runs on the in-browser mock stream (no backend). */
  mock: boolean;
  controller: SessionController;
  goTo(tab: TabId): void;
  refreshServer(): void;
}

export const AppContext = createContext<AppContextValue | null>(null);

export function useApp(): AppContextValue {
  const v = useContext(AppContext);
  if (!v) throw new Error("AppContext is missing");
  return v;
}
