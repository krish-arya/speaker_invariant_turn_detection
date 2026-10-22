import { useCallback, useEffect, useId, useState } from "react";
import { api, describeApiError, type TrialFilters } from "../api/client";
import type { ComparisonResult, DetectorMode, TrialListItem, TrialType } from "../api/schemas";
import { TRIAL_TYPES } from "../api/schemas";
import { useApp } from "../app/context";
import { Card, ErrorNotice, InfoNotice } from "../components/common";
import { ComparisonView } from "../components/ComparisonView";
import { TrialDetailDrawer } from "../components/TrialDetailDrawer";
import { TrialsTable } from "../components/TrialsTable";
import { DETECTOR_LABELS, TRIAL_TYPE_LABELS } from "../lib/labels";

const PAGE_SIZE = 25;

const GROUPINGS: Array<{ id: string; label: string; groupBy: string[] }> = [
  { id: "mode", label: "Detector mode", groupBy: ["detector_mode"] },
  { id: "mode_scenario", label: "Detector mode × scenario", groupBy: ["detector_mode", "scenario_id"] },
  { id: "mode_model", label: "Detector mode × model version", groupBy: ["detector_mode", "model_version"] },
];

export function ResultsPage() {
  const { caps, mock } = useApp();
  const ids = useId();
  const [filters, setFilters] = useState<TrialFilters>({ detectorMode: "", scenarioId: "", trialType: "", evaluationId: "" });
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<{ total: number; items: TrialListItem[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Map<string, TrialListItem>>(new Map());
  const [grouping, setGrouping] = useState(GROUPINGS[0]!.id);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [openTrial, setOpenTrial] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .listTrials({ ...filters, limit: PAGE_SIZE, offset })
      .then((d) => setData(d))
      .catch((e: unknown) => setError(describeApiError(e)))
      .finally(() => setLoading(false));
  }, [filters, offset]);

  useEffect(() => {
    if (!mock) load();
  }, [load, mock]);

  if (mock) {
    return (
      <div className="page">
        <InfoNotice tone="uncertain">
          Trial records are stored by the server and are unavailable in mock mode. Mock sessions are never saved.
        </InfoNotice>
      </div>
    );
  }

  const setFilter = (patch: Partial<TrialFilters>) => {
    setFilters((f) => ({ ...f, ...patch }));
    setOffset(0);
  };

  const toggle = (id: string) => {
    setSelected((cur) => {
      const next = new Map(cur);
      if (next.has(id)) next.delete(id);
      else {
        const item = data?.items.find((t) => t.trialId === id);
        if (item) next.set(id, item);
      }
      return next;
    });
  };

  const toggleAll = (pageIds: string[], on: boolean) => {
    setSelected((cur) => {
      const next = new Map(cur);
      for (const id of pageIds) {
        if (on) {
          const item = data?.items.find((t) => t.trialId === id);
          if (item) next.set(id, item);
        } else next.delete(id);
      }
      return next;
    });
  };

  const groupBy = GROUPINGS.find((g) => g.id === grouping)!.groupBy;
  const selectedIds = [...selected.keys()];
  const selectedItems = [...selected.values()];
  const mixedTypes = new Set(selectedItems.map((t) => t.trialType)).size > 1;

  const compare = async () => {
    setCompareError(null);
    try {
      setComparison(await api.compare({ trialIds: selectedIds }, groupBy));
    } catch (e) {
      setCompareError(describeApiError(e));
    }
  };

  const guardExport = (p: Promise<unknown>) => {
    setExportError(null);
    p.catch((e: unknown) => setExportError(describeApiError(e)));
  };

  const total = data?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="page">
      <Card title="Trial records" labelledBy={`${ids}-title`} actions={<button type="button" className="btn btn-ghost btn-sm" onClick={load}>Refresh</button>}>
        <div className="filters" role="group" aria-label="Filter trials">
          <div className="field">
            <label htmlFor={`${ids}-mode`}>Detector mode</label>
            <select id={`${ids}-mode`} value={filters.detectorMode ?? ""} onChange={(e) => setFilter({ detectorMode: e.target.value as DetectorMode | "" })}>
              <option value="">All</option>
              {caps.detectors.map((d) => (
                <option key={d.mode} value={d.mode}>
                  {d.label || DETECTOR_LABELS[d.mode]}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${ids}-scn`}>Scenario</label>
            <select id={`${ids}-scn`} value={filters.scenarioId ?? ""} onChange={(e) => setFilter({ scenarioId: e.target.value })}>
              <option value="">All</option>
              {caps.scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${ids}-type`}>Trial type</label>
            <select id={`${ids}-type`} value={filters.trialType ?? ""} onChange={(e) => setFilter({ trialType: e.target.value as TrialType | "" })}>
              <option value="">All</option>
              {TRIAL_TYPES.map((t) => (
                <option key={t} value={t}>
                  {TRIAL_TYPE_LABELS[t]}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor={`${ids}-eval`}>Evaluation ID</label>
            <input
              id={`${ids}-eval`}
              type="text"
              value={filters.evaluationId ?? ""}
              placeholder="eva_…"
              onChange={(e) => setFilter({ evaluationId: e.target.value.trim() })}
            />
          </div>
        </div>

        {error && (
          <ErrorNotice title="Could not load trials" actions={<button type="button" className="btn btn-ghost" onClick={load}>Retry</button>}>
            {error}
          </ErrorNotice>
        )}
        {exportError && <p className="field-error">{exportError}</p>}

        <div className="selection-bar" aria-live="polite">
          <span>
            {selected.size} selected
            {selected.size > 0 && mixedTypes && " · mixed trial types (simulated and live) — keep comparisons like-for-like"}
          </span>
          <div className="selection-actions">
            <button type="button" className="btn btn-ghost btn-sm" disabled={selected.size === 0} onClick={() => guardExport(api.exportTrials(selectedIds, "csv"))}>
              Export CSV
            </button>
            <button type="button" className="btn btn-ghost btn-sm" disabled={selected.size === 0} onClick={() => guardExport(api.exportTrials(selectedIds, "json"))}>
              Export JSON
            </button>
            <label className="inline-label" htmlFor={`${ids}-group`}>
              Group by
            </label>
            <select id={`${ids}-group`} value={grouping} onChange={(e) => setGrouping(e.target.value)}>
              {GROUPINGS.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.label}
                </option>
              ))}
            </select>
            <button type="button" className="btn btn-primary btn-sm" disabled={selected.size < 2} onClick={() => void compare()}>
              Compare selected
            </button>
            <button type="button" className="btn btn-ghost btn-sm" disabled={selected.size === 0} onClick={() => setSelected(new Map())}>
              Clear selection
            </button>
          </div>
        </div>
        {compareError && <p className="field-error">{compareError}</p>}

        {loading && !data && <p className="muted">Loading trials…</p>}
        {data && data.items.length === 0 && <p className="empty-state">No trials match these filters.</p>}
        {data && data.items.length > 0 && (
          <TrialsTable
            caption="Trial records table"
            trials={data.items}
            selectable
            selected={new Set(selected.keys())}
            onToggle={toggle}
            onToggleAll={toggleAll}
            onOpen={setOpenTrial}
            onExport={(id, f) => guardExport(api.exportTrial(id, f))}
          />
        )}
        <div className="pager">
          <button type="button" className="btn btn-ghost btn-sm" disabled={offset === 0 || loading} onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}>
            Previous
          </button>
          <span>
            {from}–{to} of {total}
          </span>
          <button type="button" className="btn btn-ghost btn-sm" disabled={to >= total || loading} onClick={() => setOffset((o) => o + PAGE_SIZE)}>
            Next
          </button>
        </div>
      </Card>

      {comparison && (
        <ComparisonView
          result={comparison}
          trials={selectedItems}
          exportError={exportError}
          onExport={(format) => guardExport(api.exportComparison({ trialIds: selectedIds }, groupBy, format))}
        />
      )}

      {openTrial && <TrialDetailDrawer trialId={openTrial} onClose={() => setOpenTrial(null)} />}
    </div>
  );
}
