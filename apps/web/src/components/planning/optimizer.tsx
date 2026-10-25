"use client";

import { AlertTriangle, ArrowRight } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { FrontierChart } from "@/components/charts/planning-charts";
import { DataState } from "@/components/data-state";
import { PortfolioGate } from "@/components/portfolio-gate";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Schemas } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { num, pct } from "@/lib/format";
import { usePortfolioQuery, useUniverse } from "@/lib/hooks";
import { shortName } from "@/lib/names";
import type { PortfolioIn } from "@/lib/portfolio";
import { usePortfolio } from "@/lib/store";

type Universe = "portfolio" | "funds" | "stocks" | "all";

const UNIVERSES: { id: Universe; label: string }[] = [
  { id: "portfolio", label: "Current holdings" },
  { id: "funds", label: "All 15 funds" },
  { id: "stocks", label: "All 20 stocks" },
  { id: "all", label: "Funds and stocks" },
];
const CAPS = [1, 0.4, 0.25, 0.2];
const WINDOWS = [
  { days: 756, label: "3 years" },
  { days: 1260, label: "5 years" },
  { days: 5000, label: "All (from 2015)" },
];

const DUST = 0.0005;

const selectClass = "h-8 rounded-lg border border-input bg-background px-2 text-sm";

function PortfolioTable({
  p,
  onAnalyse,
}: {
  p: Schemas["OptimizedPortfolio"];
  onAnalyse?: () => void;
}) {
  // Weights below 0.05% would print as 0.0%; group them into one line.
  const shown = p.weights.map((w, i) => ({ ...w, share: p.risk_shares?.[i] })).filter((w) => w.weight >= DUST);
  const dust = p.weights.filter((w) => w.weight < DUST);
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>{p.label}</CardTitle>
        <CardDescription>
          return {pct(p.expected_return)} · volatility {pct(p.volatility)} · Sharpe {num(p.sharpe)}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Asset</TableHead>
              <TableHead className="text-right">Weight</TableHead>
              {p.risk_shares && <TableHead className="text-right">Risk share</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((w) => (
              <TableRow key={w.asset_id}>
                <TableCell className="max-w-0 truncate" title={w.name}>
                  {shortName(w.name)}
                </TableCell>
                <TableCell className="text-right tabular-nums">{pct(w.weight, 1)}</TableCell>
                {p.risk_shares && <TableCell className="text-right tabular-nums">{pct(w.share, 1)}</TableCell>}
              </TableRow>
            ))}
            {dust.length > 0 && (
              <TableRow>
                <TableCell className="text-muted-foreground">
                  {dust.length} other asset{dust.length === 1 ? "" : "s"} below 0.05% each
                </TableCell>
                <TableCell className="text-right tabular-nums text-muted-foreground">
                  {pct(dust.reduce((s, w) => s + w.weight, 0), 2)}
                </TableCell>
                {p.risk_shares && <TableCell />}
              </TableRow>
            )}
          </TableBody>
        </Table>
        {onAnalyse && (
          <Button variant="outline" size="sm" onClick={onAnalyse}>
            Analyse these weights on the dashboard <ArrowRight />
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

function OptimizerView({ portfolio }: { portfolio: PortfolioIn }) {
  const router = useRouter();
  const setHoldings = usePortfolio((s) => s.setHoldings);
  const universe = useUniverse();
  const [scope, setScope] = useState<Universe>("portfolio");
  const [cap, setCap] = useState(1);
  const [days, setDays] = useState(1260);
  const [includeLiquid, setIncludeLiquid] = useState(false);

  const body = useMemo<Schemas["OptimizeRequest"] | null>(() => {
    const common = { max_weight: cap, lookback_days: days, include_liquid: includeLiquid };
    if (scope === "portfolio") return { portfolio, ...common };
    if (universe.status !== "success") return null;
    const ids = universe.data.assets
      .filter((a) => a.available && a.kind !== "index")
      .filter((a) => scope === "all" || (scope === "funds" ? a.kind === "mutual_fund" : a.kind === "equity"))
      .map((a) => a.asset_id);
    return { asset_ids: ids, portfolio, ...common };
  }, [scope, cap, days, includeLiquid, portfolio, universe]);

  const q = usePortfolioQuery("/optimize", body, (b, signal) => api.optimize(b, { signal }));

  function analyse(p: Schemas["OptimizedPortfolio"]) {
    const value = portfolio.portfolio_value ?? 1_000_000;
    setHoldings(
      p.weights.map((w) => ({ asset_id: w.asset_id, amount: w.weight * value })),
      `${p.label} (from the optimizer)`,
    );
    router.push("/dashboard");
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Assets</span>
          <select className={selectClass} value={scope} onChange={(e) => setScope(e.target.value as Universe)}>
            {UNIVERSES.map((u) => (
              <option key={u.id} value={u.id}>
                {u.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">Max per asset</span>
          <select className={selectClass} value={cap} onChange={(e) => setCap(Number(e.target.value))}>
            {CAPS.map((c) => (
              <option key={c} value={c}>
                {c === 1 ? "No cap" : pct(c, 0)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted-foreground">History</span>
          <select className={selectClass} value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {WINDOWS.map((w) => (
              <option key={w.days} value={w.days}>
                {w.label}
              </option>
            ))}
          </select>
        </label>
        <label
          className="flex items-center gap-2"
          title="Liquid funds behave like cash: their tiny volatility makes them dominate minimum-variance, maximum-Sharpe and risk-parity portfolios, so they are left out unless you include them."
        >
          <input type="checkbox" checked={includeLiquid} onChange={(e) => setIncludeLiquid(e.target.checked)} />
          <span className="text-muted-foreground">Include liquid fund (cash-like)</span>
        </label>
      </div>
      <DataState query={q} loadingLabel="Optimising" height={420}>
        {(r) => (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Efficient frontier</CardTitle>
                <CardDescription>
                  {r.window.start} to {r.window.end} ({r.window.n_obs.toLocaleString("en-IN")} days); Sharpe uses a
                  risk-free rate of {pct(r.risk_free.annual)} ({r.risk_free.source}).
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <FrontierChart result={r} />
                {r.warnings.map((w) => (
                  <p key={w} className="flex gap-2 text-sm text-amber-700">
                    <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {w}
                  </p>
                ))}
                <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                  {r.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>
            <div className="grid gap-4 md:grid-cols-2">
              {r.current && <PortfolioTable p={r.current} />}
              {[r.min_variance, r.max_sharpe, r.risk_parity, r.max_return].map((p) => (
                <PortfolioTable key={p.label} p={p} onAnalyse={() => analyse(p)} />
              ))}
            </div>
          </div>
        )}
      </DataState>
    </div>
  );
}

export function Optimizer() {
  return (
    <PortfolioGate>
      {({ portfolio }) => (
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Optimizer</h1>
            <p className="text-muted-foreground">
              The long-only trade-off between historical return and volatility, and a risk-parity
              allocation that ignores return estimates. For analysis only.
            </p>
          </div>
          <OptimizerView portfolio={portfolio} />
        </div>
      )}
    </PortfolioGate>
  );
}
