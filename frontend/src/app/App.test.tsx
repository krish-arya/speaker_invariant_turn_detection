import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

function setUrl(search: string) {
  window.history.replaceState(null, "", `/${search}`);
}

describe("App", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    vi.restoreAllMocks();
    setUrl("");
  });

  it("offers Retry and an explicit mock opt-in when the backend is unreachable", async () => {
    setUrl("");
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    render(<App />);
    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retry/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Use mock data/ }));
    expect(await screen.findByText("Start a session to see live turn decisions and pipeline events.")).toBeInTheDocument();
    expect(screen.getAllByText("MOCK DATA").length).toBeGreaterThan(0);
  });

  it("runs a mock session end to end and shows live state, predictions and decisions", async () => {
    setUrl("?mock=1");
    render(<App />);
    expect(screen.getByRole("tab", { name: "Live Console" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Ready to connect")).toBeInTheDocument();
    const start = screen.getByRole("button", { name: /Start session/ });
    await act(async () => {
      fireEvent.click(start);
    });
    await waitFor(() => expect(screen.getByText("Listening")).toBeInTheDocument(), { timeout: 3000 });
    // The Detector selector is frozen during the session.
    expect(screen.getByLabelText("Detector")).toBeDisabled();
    await waitFor(() => expect(screen.getAllByText(/Predicted:/).length).toBeGreaterThan(0), { timeout: 5000 });
    await waitFor(() => expect(screen.getAllByText("Committed: Turn complete").length).toBeGreaterThan(0), { timeout: 6000 });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /End session/ }));
    });
    await waitFor(() => expect(screen.getByText("Session ended")).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getByText("Completed trial")).toBeInTheDocument();
    expect(screen.getByLabelText("Detector")).not.toBeDisabled();
  }, 20000);

  it("switches tabs with the keyboard", async () => {
    setUrl("?mock=1");
    render(<App />);
    const live = screen.getByRole("tab", { name: "Live Console" });
    live.focus();
    fireEvent.keyDown(live, { key: "ArrowRight" });
    expect(screen.getByRole("tab", { name: "Evaluation" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/unavailable in mock mode/)).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Evaluation" }), { key: "End" });
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/raw audio is not retained/)).toBeInTheDocument();
  });
});
