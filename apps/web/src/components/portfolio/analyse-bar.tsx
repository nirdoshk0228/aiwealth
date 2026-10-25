"use client";

import { ArrowRight } from "lucide-react";
import Link from "next/link";

import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { portfolioProblems } from "@/lib/portfolio";
import { useHydrated, usePortfolio } from "@/lib/store";
import { cn } from "@/lib/utils";

export const CONFIDENCE_OPTIONS = [
  { value: 0.95, label: "95%" },
  { value: 0.99, label: "99%" },
];

export const LOOKBACK_OPTIONS = [
  { value: 252, label: "1 year" },
  { value: 756, label: "3 years" },
  { value: 1260, label: "5 years" },
  { value: 5000, label: "All (from 2015)" },
];

const selectClass =
  "h-8 rounded-lg border border-input bg-background px-2 text-sm focus-visible:ring-3 focus-visible:ring-ring/50 focus-visible:outline-none";

/** Analysis settings (confidence, window) shared by the dashboard. */
export function SettingsControls() {
  const settings = usePortfolio((s) => s.settings);
  const setSettings = usePortfolio((s) => s.setSettings);
  return (
    <div className="flex flex-wrap items-center gap-4 text-sm">
      <label className="flex items-center gap-2">
        <span className="text-muted-foreground">VaR confidence</span>
        <select
          className={selectClass}
          value={settings.confidence}
          onChange={(e) => setSettings({ confidence: Number(e.target.value) })}
        >
          {CONFIDENCE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-2">
        <span className="text-muted-foreground">History window</span>
        <select
          className={selectClass}
          value={settings.lookbackDays}
          onChange={(e) => setSettings({ lookbackDays: Number(e.target.value) })}
        >
          {LOOKBACK_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

export function AnalyseBar() {
  const hydrated = useHydrated();
  const holdings = usePortfolio((s) => s.holdings);
  const clear = usePortfolio((s) => s.clear);
  const problems = hydrated ? portfolioProblems(holdings) : [];
  const ready = hydrated && problems.length === 0;

  return (
    <Card>
      <CardContent className="space-y-4">
        <SettingsControls />
        {hydrated && holdings.length > 0 && problems.length > 0 && (
          <ul className="list-disc pl-5 text-sm text-destructive" role="alert">
            {problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        )}
        <div className="flex flex-wrap gap-2">
          <Link
            href="/dashboard"
            aria-disabled={!ready}
            className={cn(
              buttonVariants({ size: "lg", className: "h-10 px-4" }),
              !ready && "pointer-events-none opacity-50",
            )}
          >
            Open risk dashboard <ArrowRight />
          </Link>
          <Button variant="ghost" size="lg" className="h-10" onClick={clear} disabled={!holdings.length}>
            Clear holdings
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
