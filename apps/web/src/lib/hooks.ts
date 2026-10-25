"use client";

import { useMemo } from "react";

import { api } from "@/lib/api/endpoints";
import { queryKey, useQuery } from "@/lib/api/use-query";
import { type PortfolioIn, toPortfolioIn } from "@/lib/portfolio";
import { type Settings, useHydrated, usePortfolio } from "@/lib/store";

export function useUniverse() {
  return useQuery("universe", (signal) => api.universe({ signal }));
}

/** asset id -> name, from the universe (empty until it loads). */
export function useAssetNames(): Record<string, string> {
  const u = useUniverse();
  return useMemo(() => {
    if (u.status !== "success") return {};
    return Object.fromEntries(u.data.assets.map((a) => [a.asset_id, a.name]));
  }, [u]);
}

/**
 * The portfolio in API form plus the analysis settings. `portfolio` is null until the stored
 * state has loaded, and stays null if there are no valid holdings.
 */
export function useAnalysisInput(): {
  hydrated: boolean;
  portfolio: PortfolioIn | null;
  settings: Settings;
  label: string | null;
} {
  const hydrated = useHydrated();
  const holdings = usePortfolio((s) => s.holdings);
  const settings = usePortfolio((s) => s.settings);
  const label = usePortfolio((s) => s.label);
  const portfolio = useMemo(
    () => (hydrated ? toPortfolioIn(holdings) : null),
    [hydrated, holdings],
  );
  return { hydrated, portfolio, settings, label };
}

/** useQuery for a POST endpoint whose body depends on the portfolio (skipped while null). */
export function usePortfolioQuery<B, T>(
  path: string,
  body: B | null,
  fetcher: (body: B, signal: AbortSignal) => Promise<T>,
) {
  return useQuery(body === null ? null : queryKey(path, body), (signal) =>
    fetcher(body as B, signal),
  );
}
