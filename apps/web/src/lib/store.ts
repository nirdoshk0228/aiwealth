"use client";

/**
 * Client state (Zustand): the portfolio being analysed and the analysis settings.
 * Persisted to localStorage so a page refresh keeps the portfolio. Nothing here is sent
 * anywhere except as a request body to the API.
 */
import { useEffect, useState } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { HoldingRow } from "@/lib/portfolio";

export type Settings = {
  confidence: number; // VaR confidence, e.g. 0.95
  lookbackDays: number; // most recent return rows used
};

type PortfolioState = {
  holdings: HoldingRow[];
  label: string | null; // e.g. the sample's name, or null for a custom portfolio
  settings: Settings;
  setHoldings: (rows: HoldingRow[], label?: string | null) => void;
  upsert: (row: HoldingRow) => void;
  setAmount: (assetId: string, amount: number) => void;
  remove: (assetId: string) => void;
  clear: () => void;
  setSettings: (s: Partial<Settings>) => void;
};

export const DEFAULT_SETTINGS: Settings = { confidence: 0.95, lookbackDays: 756 };

export const usePortfolio = create<PortfolioState>()(
  persist(
    (set) => ({
      holdings: [],
      label: null,
      settings: DEFAULT_SETTINGS,
      setHoldings: (rows, label = null) => set({ holdings: rows, label }),
      upsert: (row) =>
        set((s) => ({
          label: null,
          holdings: s.holdings.some((h) => h.asset_id === row.asset_id)
            ? s.holdings.map((h) => (h.asset_id === row.asset_id ? row : h))
            : [...s.holdings, row],
        })),
      setAmount: (assetId, amount) =>
        set((s) => ({
          label: null,
          holdings: s.holdings.map((h) => (h.asset_id === assetId ? { ...h, amount } : h)),
        })),
      remove: (assetId) =>
        set((s) => ({ label: null, holdings: s.holdings.filter((h) => h.asset_id !== assetId) })),
      clear: () => set({ holdings: [], label: null }),
      setSettings: (p) => set((s) => ({ settings: { ...s.settings, ...p } })),
    }),
    { name: "aiwealth-portfolio", version: 1 },
  ),
);

/**
 * True once the persisted state has been read from localStorage. Before that the server
 * render and the first client render both see the empty default, which avoids a hydration
 * mismatch; components wait for this before deciding "no portfolio yet".
 */
export function useHydrated(): boolean {
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => {
    const done = () => setHydrated(true);
    const unsub = usePortfolio.persist.onFinishHydration(done);
    if (usePortfolio.persist.hasHydrated()) done();
    return unsub;
  }, []);
  return hydrated;
}
