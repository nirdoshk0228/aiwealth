"use client";

/**
 * Risk dashboard cards. Each card owns one API query and its own loading and error state, so
 * a slow endpoint (the backtest) never blocks the others. Every number shown comes from the
 * API response; the client only formats it.
 */
import { Info } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import {
  AttributionChart,
  BacktestChart,
  CorrelationHeatmap,
  DrawdownChart,
  FactorChart,
  GrowthChart,
} from "@/components/charts/risk-charts";
import { DataState } from "@/components/data-state";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Schemas } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { num, pct, pValue, rupees } from "@/lib/format";
import { usePortfolioQuery } from "@/lib/hooks";
import { axisLabel, shortName } from "@/lib/names";
import type { PortfolioIn } from "@/lib/portfolio";
import type { Settings } from "@/lib/store";

type Props = { portfolio: PortfolioIn; settings: Settings };

const METHOD_LABEL: Record<string, string> = {
  historical: "Historical",
  parametric: "Parametric (normal)",
  monte_carlo: "Monte Carlo (normal)",
};

function windowText(w: Schemas["AnalysisWindow"]): string {
  return `${w.start} to ${w.end}, ${w.n_obs.toLocaleString("en-IN")} trading days`;
}

function Notes({ notes }: { notes: string[] }) {
  if (notes.length === 0) return null;
  return (
    <details className="text-xs text-muted-foreground">
      <summary className="flex cursor-pointer items-center gap-1 select-none">
        <Info className="size-3.5" aria-hidden /> Assumptions and how to read this
      </summary>
      <ul className="mt-2 list-disc space-y-1 pl-4">
        {notes.map((n) => (
          <li key={n}>{n}</li>
        ))}
      </ul>
    </details>
  );
}

function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="space-y-0.5">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-semibold tabular-nums">{value}</p>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

// --- headline metrics ---

export function SummaryCards({ portfolio, settings }: Props) {
  const metricsBody = useMemo(
    () => ({ portfolio, lookback_days: settings.lookbackDays }),
    [portfolio, settings.lookbackDays],
  );
  const varBody = useMemo(
    () => ({ portfolio, lookback_days: settings.lookbackDays, confidence: settings.confidence }),
    [portfolio, settings.lookbackDays, settings.confidence],
  );
  const metrics = usePortfolioQuery("/risk/metrics", metricsBody, (b, signal) => api.metrics(b, { signal }));
  const vaR = usePortfolioQuery("/risk/var", varBody, (b, signal) => api.var(b, { signal }));

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardContent>
          <DataState query={vaR} loadingLabel="Loading VaR" height={48}>
            {(v) => {
              const h = v.estimates.find((e) => e.method === "historical");
              return (
                <Stat
                  label={`1-day VaR, ${pct(v.confidence, 0)} (historical)`}
                  value={pct(h?.var)}
                  sub={`${rupees(h?.var_rupees)} of ${rupees(v.portfolio_value)}`}
                />
              );
            }}
          </DataState>
        </CardContent>
      </Card>
      <Card>
        <CardContent>
          <DataState query={metrics} loadingLabel="Loading volatility" height={48}>
            {(m) => (
              <Stat
                label="Annualised volatility"
                value={pct(m.annualised_volatility)}
                sub={`Sharpe ${num(m.sharpe)} (risk-free ${pct(m.risk_free.annual)})`}
              />
            )}
          </DataState>
        </CardContent>
      </Card>
      <Card>
        <CardContent>
          <DataState query={metrics} loadingLabel="Loading drawdown" height={48}>
            {(m) => (
              <Stat
                label="Maximum drawdown"
                value={pct(m.max_drawdown)}
                sub={
                  m.max_drawdown_episode
                    ? `trough ${m.max_drawdown_episode.trough_date}${m.max_drawdown_episode.recovered ? "" : ", not yet recovered"}`
                    : "no drawdown in the window"
                }
              />
            )}
          </DataState>
        </CardContent>
      </Card>
      <Card>
        <CardContent>
          <DataState query={metrics} loadingLabel="Loading return" height={48}>
            {(m) => (
              <Stat
                label="Annualised return (CAGR)"
                value={pct(m.annualised_return)}
                sub={`over ${m.window.start} to ${m.window.end}`}
              />
            )}
          </DataState>
        </CardContent>
      </Card>
    </div>
  );
}

// --- VaR / CVaR by method ---

export function VaRCard({ portfolio, settings }: Props) {
  const body = useMemo(
    () => ({ portfolio, lookback_days: settings.lookbackDays, confidence: settings.confidence }),
    [portfolio, settings.lookbackDays, settings.confidence],
  );
  const q = usePortfolioQuery("/risk/var", body, (b, signal) => api.var(b, { signal }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Value-at-Risk and CVaR</CardTitle>
        <CardDescription>
          The loss not exceeded on {pct(settings.confidence, 0)} of days (VaR), and the average
          loss on the remaining worst days (CVaR), over one trading day.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <DataState query={q} loadingLabel="Loading VaR estimates" height={120}>
          {(v) => (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Method</TableHead>
                    <TableHead className="text-right">VaR</TableHead>
                    <TableHead className="text-right">VaR (₹)</TableHead>
                    <TableHead className="text-right">CVaR</TableHead>
                    <TableHead className="text-right">CVaR (₹)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {v.estimates.map((e) => (
                    <TableRow key={e.method}>
                      <TableCell>{METHOD_LABEL[e.method] ?? e.method}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(e.var)}</TableCell>
                      <TableCell className="text-right tabular-nums">{rupees(e.var_rupees)}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(e.cvar)}</TableCell>
                      <TableCell className="text-right tabular-nums">{rupees(e.cvar_rupees)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <p className="text-xs text-muted-foreground">Estimated from {windowText(v.window)}.</p>
              <Notes notes={v.notes} />
            </>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}

// --- growth and drawdown ---

export function PerformanceCard({ portfolio, settings }: Props) {
  const body = useMemo(
    () => ({ portfolio, lookback_days: settings.lookbackDays }),
    [portfolio, settings.lookbackDays],
  );
  const q = usePortfolioQuery("/risk/metrics", body, (b, signal) => api.metrics(b, { signal }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Growth and drawdowns</CardTitle>
        <CardDescription>Value of ₹1 invested at the start of the window, and the fall from each running peak.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <DataState query={q} loadingLabel="Loading performance" height={300}>
          {(m) => (
            <>
              <GrowthChart portfolio={m.growth} benchmark={m.benchmark_growth ?? null} />
              <DrawdownChart drawdown={m.drawdown} />
              <div className="grid gap-4 text-sm sm:grid-cols-3">
                <Stat label="Sortino ratio" value={num(m.sortino)} sub="downside below the risk-free rate" />
                <Stat
                  label="Beta to Nifty 50"
                  value={num(m.benchmark?.beta)}
                  sub={`correlation ${num(m.benchmark?.correlation)}`}
                />
                <Stat
                  label="Tracking error"
                  value={pct(m.benchmark?.tracking_error)}
                  sub={`active return ${pct(m.benchmark?.active_return)} a year`}
                />
              </div>
              {m.longest_drawdown && (
                <p className="text-xs text-muted-foreground">
                  Longest time under water: {m.longest_drawdown.duration_days} trading days from{" "}
                  {m.longest_drawdown.peak_date ?? "the start"}
                  {m.longest_drawdown.recovered ? ` to ${m.longest_drawdown.recovery_date}` : " (not yet recovered)"}.
                  Risk-free rate: {m.risk_free.source}.
                </p>
              )}
              <Notes notes={m.notes} />
            </>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}

// --- attribution ---

export function AttributionCard({ portfolio, settings }: Props) {
  const body = useMemo(
    () => ({ portfolio, lookback_days: settings.lookbackDays, confidence: settings.confidence }),
    [portfolio, settings.lookbackDays, settings.confidence],
  );
  const q = usePortfolioQuery("/risk/attribution", body, (b, signal) => api.attribution(b, { signal }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Where the risk comes from</CardTitle>
        <CardDescription>
          Each holding’s share of capital compared with its share of portfolio volatility.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <DataState query={q} loadingLabel="Loading risk attribution" height={260}>
          {(a) => (
            <>
              <AttributionChart
                labels={a.assets.map((x) => axisLabel(x.asset_id, x.name))}
                weights={a.assets.map((x) => x.weight)}
                shares={a.assets.map((x) => x.volatility_share)}
              />
              <div className="grid gap-4 text-sm sm:grid-cols-3">
                <Stat label="Portfolio volatility" value={pct(a.portfolio_volatility)} sub="annualised" />
                <Stat
                  label={`Parametric VaR, ${pct(a.confidence, 0)}`}
                  value={pct(a.parametric_var)}
                  sub={`CVaR ${pct(a.parametric_cvar)}`}
                />
                <Stat
                  label="Diversification ratio"
                  value={num(a.diversification_ratio)}
                  sub="1 = no diversification benefit"
                />
              </div>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Holding</TableHead>
                    <TableHead className="text-right">Weight</TableHead>
                    <TableHead className="text-right">Volatility share</TableHead>
                    <TableHead className="text-right">VaR share</TableHead>
                    <TableHead className="text-right">Risk / weight</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {a.assets.map((x) => (
                    <TableRow key={x.asset_id}>
                      <TableCell className="max-w-0 truncate" title={x.name}>
                        {shortName(x.name)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{pct(x.weight, 1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(x.volatility_share, 1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{pct(x.var_share, 1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{num(x.risk_to_weight)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <Notes notes={a.notes} />
            </>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}

// --- correlation ---

export function CorrelationCard({ portfolio, settings }: Props) {
  const body = useMemo(
    () => ({ portfolio, lookback_days: settings.lookbackDays }),
    [portfolio, settings.lookbackDays],
  );
  const q = usePortfolioQuery("/risk/correlation", body, (b, signal) => api.correlation(b, { signal }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Correlations</CardTitle>
        <CardDescription>
          How the holdings’ daily returns moved together (−1 opposite, +1 together), with the
          Nifty 50 for reference.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <DataState query={q} loadingLabel="Loading correlations" height={320}>
          {(c) => (
            <>
              <CorrelationHeatmap
                labels={c.asset_ids.map((id, i) => axisLabel(id, c.names[i] ?? id))}
                matrix={c.matrix}
              />
              <p className="text-xs text-muted-foreground">
                {windowText(c.window)}. Correlations are averages over the window and tend to rise
                in sell-offs.
              </p>
            </>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}

// --- factor exposures ---

export function FactorCard({ portfolio }: { portfolio: PortfolioIn }) {
  const body = useMemo(() => ({ portfolio, include_assets: true }), [portfolio]);
  const q = usePortfolioQuery("/factors/exposures", body, (b, signal) => api.factors(b, { signal }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Factor exposures</CardTitle>
        <CardDescription>
          Carhart four-factor betas on IIM Ahmedabad’s Indian factors: market (MKT_RF), size
          (SMB), value (HML) and momentum (WML).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <DataState query={q} loadingLabel="Loading factor exposures" height={240}>
          {(f) => (
            <>
              <FactorChart coefficients={f.portfolio.coefficients} />
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Factor</TableHead>
                    <TableHead className="text-right">Beta</TableHead>
                    <TableHead className="text-right">t-stat</TableHead>
                    <TableHead className="text-right">p-value</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {f.portfolio.coefficients.map((c) => (
                    <TableRow key={c.factor}>
                      <TableCell>{c.factor}</TableCell>
                      <TableCell className="text-right tabular-nums">{num(c.beta, 3)}</TableCell>
                      <TableCell className="text-right tabular-nums">{num(c.t_stat, 1)}</TableCell>
                      <TableCell className="text-right tabular-nums">{pValue(c.p_value)}</TableCell>
                    </TableRow>
                  ))}
                  <TableRow>
                    <TableCell>Alpha (annual)</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(f.portfolio.alpha_annual)}</TableCell>
                    <TableCell className="text-right tabular-nums">{num(f.portfolio.alpha_t_stat, 1)}</TableCell>
                    <TableCell />
                  </TableRow>
                </TableBody>
              </Table>
              <p className="text-xs text-muted-foreground">
                R² {num(f.portfolio.r_squared, 3)}, {windowText(f.window)}, Newey-West lags{" "}
                {f.portfolio.hac_lags}. Source: {f.source}
              </p>
              {f.portfolio.warnings.map((w) => (
                <p key={w} className="text-xs text-amber-700">
                  {w}
                </p>
              ))}
              <Notes notes={f.notes} />
            </>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}

// --- VaR backtest ---

const ZONE_CLASS: Record<string, string> = {
  green: "bg-green-100 text-green-800",
  yellow: "bg-amber-100 text-amber-800",
  red: "bg-red-100 text-red-800",
};

export function BacktestCard({ portfolio }: { portfolio: PortfolioIn }) {
  const [method, setMethod] = useState<"historical" | "parametric">("historical");
  const [confidence, setConfidence] = useState(0.99);
  const body = useMemo(() => ({ portfolio, method, confidence }), [portfolio, method, confidence]);
  const q = usePortfolioQuery("/backtest", body, (b, signal) => api.backtest(b, { signal }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Does the VaR hold up? (backtest)</CardTitle>
        <CardDescription>
          Each day’s VaR is estimated from the previous two years only, then compared with that
          day’s actual loss. A breach is a loss larger than the forecast.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Backtest settings">
          {(["historical", "parametric"] as const).map((m) => (
            <Button key={m} size="sm" variant={method === m ? "default" : "outline"} onClick={() => setMethod(m)}>
              {METHOD_LABEL[m]}
            </Button>
          ))}
          {[0.95, 0.99].map((c) => (
            <Button key={c} size="sm" variant={confidence === c ? "default" : "outline"} onClick={() => setConfidence(c)}>
              {pct(c, 0)}
            </Button>
          ))}
        </div>
        <DataState query={q} loadingLabel="Running the backtest (a few seconds)" height={300}>
          {(b) => (
            <>
              <BacktestChart result={b} />
              <div className="grid gap-4 text-sm sm:grid-cols-4">
                <Stat
                  label="Breaches"
                  value={`${b.n_breaches} of ${b.test_period.n_obs.toLocaleString("en-IN")}`}
                  sub={`expected ${num(b.expected_breaches, 1)} (${pct(b.expected_rate, 0)})`}
                />
                <Stat label="Breach rate" value={pct(b.breach_rate)} sub={`${b.test_period.start} to ${b.test_period.end}`} />
                <Stat
                  label="Kupiec p-value"
                  value={pValue(b.kupiec.p_value)}
                  sub={`independence p ${pValue(b.christoffersen.p_ind)}`}
                />
                <div className="space-y-0.5">
                  <p className="text-xs text-muted-foreground">Basel traffic light (last 250 days)</p>
                  {b.basel_zone_last_250 ? (
                    <Badge className={ZONE_CLASS[b.basel_zone_last_250]}>
                      {b.basel_zone_last_250} · {b.basel_breaches_last_250} breaches
                    </Badge>
                  ) : (
                    <p className="text-sm text-muted-foreground">99% only</p>
                  )}
                </div>
              </div>
              <Notes notes={b.notes} />
            </>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}
