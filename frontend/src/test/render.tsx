import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import type { Capabilities } from "../api/schemas";
import { AppContext, type AppContextValue } from "../app/context";
import { MOCK_CAPABILITIES } from "../events/mock";
import { SessionController } from "../state/sessionController";
import { SessionStore, SessionStoreContext } from "../state/sessionStore";

/** Synchronous scheduler so envelopes fold immediately in tests. */
export function testStore(): SessionStore {
  return new SessionStore((cb) => {
    cb();
    return () => {};
  });
}

export function renderWithApp(
  ui: ReactElement,
  opts: { store?: SessionStore; caps?: Capabilities; mock?: boolean } = {},
) {
  const store = opts.store ?? testStore();
  const controller = new SessionController(store);
  const ctx: AppContextValue = {
    caps: opts.caps ?? MOCK_CAPABILITIES,
    health: null,
    mock: opts.mock ?? false,
    controller,
    goTo: () => {},
    refreshServer: () => {},
  };
  const result = render(
    <SessionStoreContext.Provider value={store}>
      <AppContext.Provider value={ctx}>{ui}</AppContext.Provider>
    </SessionStoreContext.Provider>,
  );
  return { ...result, store, controller };
}
