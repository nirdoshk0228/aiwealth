"use client";

import { AlertTriangle, Play, Plus, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { ContributionChart } from "@/components/charts/planning-charts";
import { DataState } from "@/components/data-state";
import { DescribeScenario } from "@/components/planning/describe-scenario";
import { PortfolioGate } from "@/components/portfolio-gate";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Schemas } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { queryKey, useQuery } from "@/lib/api/use-query";
import { rupees, signedPct, pct } from "@/lib/format";
import { useUniverse } from "@/lib/hooks";
import { shortName } from "@/lib/names";
import type { PortfolioIn } from "@/lib/portfolio";
import { cn } from "@/lib/utils";

type Custom = Schemas["HypotheticalScenarioSpec"];
type Target = Schemas["DirectShockSpec"]["target"];

const FACTORS = [
  { id: "MKT_RF", label: "Market (MKT_RF)" },
  { id: "SMB", label: "Size (SMB)" },
  { id: "HML", label: "Value (HML)" },
  { id: "WML", label: "Momentum (WML)" },
] as const;

const SOURCE_LABEL: Record<string, string> = {
  historical: "replayed",
  direct_asset: "direct (asset)",
  direct_sector: "direct (sector)",
  direct_category: "direct (category)",
  factor: "factor betas",
  unshocked: "no shock",
};

function ScenarioPicker({
  selected,
  onSelect,
}: {
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const q = useQuery("scenarios", (signal) => api.scenarios({ signal }));
  return (
    <DataState query={q} loadingLabel="Loading scenarios" height={200}>
      {(list) => (
        <div className="grid gap-2 sm:grid-cols-2">
          {list.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => onSelect(s.id)}
              aria-pressed={selected === s.id}
              className={cn(
                "rounded-lg border p-3 text-left transition-colors hover:bg-muted",
                selected === s.id && "border-blue-600 bg-blue-50",
              )}
            >
              <div className="flex items-center gap-2">
                <Badge variant="secondary">{s.type === "historical" ? "Historical" : "Hypothetical"}</Badge>
                <span className="text-sm font-medium">{s.name}</span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {s.type === "historical" ? `${s.start} to ${s.end}. ` : ""}
                {s.description}
              </p>
            </button>
          ))}
        </div>
      )}
    </DataState>
  );
}

function ResultView({ body }: { body: Schemas["StressRequest"] }) {
  const q = useQuery(queryKey("/stress/run", body), (signal) => api.stress(body, { signal }));
  return (
    <DataState query={q} loadingLabel="Running the scenario" height={240}>
      {(r) => (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-8">
            <div>
              <p className="text-xs text-muted-foreground">Portfolio P&amp;L in the scenario</p>
              <p className={cn("text-3xl font-semibold tabular-nums", r.pnl < 0 ? "text-red-600" : "text-green-700")}>
                {signedPct(r.pnl)}
              </p>
              <p className="text-sm text-muted-foreground">
                {rupees(r.pnl_rupees)} on {rupees(r.portfolio_value)}
              </p>
            </div>
            {r.scenario_type === "historical" && (
              <div className="text-sm text-muted-foreground">
                <p>
                  {r.start} to {r.end}, {r.sessions} sessions
                </p>
                <p>
                  Largest fall inside the window: {pct(r.max_drawdown)}
                  {r.drawdown_trough_date ? ` (trough ${r.drawdown_trough_date})` : ""}
                </p>
              </div>
            )}
          </div>
          {r.warnings.map((w) => (
            <p key={w} className="flex gap-2 text-sm text-amber-700">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {w}
            </p>
          ))}
          <ContributionChart impacts={r.impacts} />
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Holding</TableHead>
                <TableHead className="text-right">Weight</TableHead>
                <TableHead className="text-right">Asset return</TableHead>
                <TableHead className="text-right">Contribution</TableHead>
                <TableHead className="text-right">₹</TableHead>
                <TableHead>Shock source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {r.impacts.map((i) => (
                <TableRow key={i.asset_id}>
                  <TableCell className="max-w-0 truncate" title={i.name}>
                    {shortName(i.name)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{pct(i.weight, 1)}</TableCell>
                  <TableCell className="text-right tabular-nums">{signedPct(i.asset_return)}</TableCell>
                  <TableCell className="text-right tabular-nums">{signedPct(i.contribution)}</TableCell>
                  <TableCell className="text-right tabular-nums">{rupees(i.contribution_rupees)}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {SOURCE_LABEL[i.source] ?? i.source}
                    {i.r_squared !== null && i.r_squared !== undefined ? ` (R² ${i.r_squared.toFixed(2)})` : ""}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
            {r.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      )}
    </DataState>
  );
}

type DirectRow = { kind: "sector" | "category"; value: string; shockPct: string };

function CustomBuilder({ onRun }: { onRun: (s: Custom) => void }) {
  const universe = useUniverse();
  const [factorPct, setFactorPct] = useState<Record<string, string>>({ MKT_RF: "-10" });
  const [rows, setRows] = useState<DirectRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  const options = useMemo(() => {
    if (universe.status !== "success") return { sector: [], category: [] };
    const a = universe.data.assets;
    return {
      sector: [...new Set(a.map((x) => x.industry).filter((x): x is string => !!x))].sort(),
      category: [...new Set(a.map((x) => x.category).filter((x): x is string => !!x))].sort(),
    };
  }, [universe]);

  function build(): Custom | string {
    const factor_shocks: Record<string, number> = {};
    for (const f of FACTORS) {
      const raw = (factorPct[f.id] ?? "").trim();
      if (raw === "") continue;
      const v = Number(raw);
      if (!Number.isFinite(v) || v <= -100 || v > 100) return `${f.label}: enter a percentage between -100 and 100`;
      factor_shocks[f.id] = v / 100;
    }
    const direct_shocks: Schemas["DirectShockSpec"][] = [];
    for (const r of rows) {
      const v = Number(r.shockPct);
      if (!r.value || !Number.isFinite(v) || v <= -100 || v > 100) return "Each direct shock needs a target and a percentage";
      const target: Target =
        r.kind === "sector"
          ? { kind: "sector", sector: r.value }
          : { kind: "category", category: r.value as Schemas["CategoryTarget"]["category"] };
      direct_shocks.push({ target, shock: v / 100 });
    }
    if (Object.keys(factor_shocks).length === 0 && direct_shocks.length === 0) return "Add at least one shock";
    return {
      id: "custom",
      type: "hypothetical",
      name: "Custom scenario",
      description: "Built in the stress lab",
      rationale: "User-defined shocks",
      factor_shocks,
      direct_shocks,
    };
  }

  const selectClass = "h-8 rounded-lg border border-input bg-background px-2 text-sm";
  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-4">
        {FACTORS.map((f) => (
          <label key={f.id} className="space-y-1 text-sm">
            <span className="text-muted-foreground">{f.label} %</span>
            <Input
              inputMode="decimal"
              value={factorPct[f.id] ?? ""}
              placeholder="0"
              onChange={(e) => setFactorPct((s) => ({ ...s, [f.id]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <div className="space-y-2">
        <p className="text-sm text-muted-foreground">
          Direct shocks replace the factor-implied return for matching holdings (sector = stock
          industry, category = fund category).
        </p>
        {rows.map((r, i) => (
          <div key={i} className="flex flex-wrap items-center gap-2">
            <select
              className={selectClass}
              value={r.kind}
              aria-label="Shock target type"
              onChange={(e) =>
                setRows((rs) => rs.map((x, j) => (j === i ? { ...x, kind: e.target.value as DirectRow["kind"], value: "" } : x)))
              }
            >
              <option value="sector">Sector</option>
              <option value="category">Fund category</option>
            </select>
            <select
              className={cn(selectClass, "min-w-48")}
              value={r.value}
              aria-label="Shock target"
              onChange={(e) => setRows((rs) => rs.map((x, j) => (j === i ? { ...x, value: e.target.value } : x)))}
            >
              <option value="">Choose…</option>
              {options[r.kind].map((o) => (
                <option key={o} value={o}>
                  {o.replaceAll("_", " ")}
                </option>
              ))}
            </select>
            <Input
              className="w-24"
              inputMode="decimal"
              placeholder="%"
              aria-label="Shock in percent"
              value={r.shockPct}
              onChange={(e) => setRows((rs) => rs.map((x, j) => (j === i ? { ...x, shockPct: e.target.value } : x)))}
            />
            <Button variant="ghost" size="icon-sm" aria-label="Remove shock" onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}>
              <Trash2 />
            </Button>
          </div>
        ))}
        <Button variant="outline" size="sm" onClick={() => setRows((rs) => [...rs, { kind: "sector", value: "", shockPct: "" }])}>
          <Plus /> Add direct shock
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <Button
        onClick={() => {
          const s = build();
          if (typeof s === "string") setError(s);
          else {
            setError(null);
            onRun(s);
          }
        }}
      >
        <Play /> Run custom scenario
      </Button>
    </div>
  );
}

function Lab({ portfolio }: { portfolio: PortfolioIn }) {
  const [scenarioId, setScenarioId] = useState<string | null>("covid_crash_2020");
  const [custom, setCustom] = useState<Custom | null>(null);
  const body: Schemas["StressRequest"] | null =
    custom !== null ? { portfolio, scenario: custom } : scenarioId !== null ? { portfolio, scenario_id: scenarioId } : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Scenarios</CardTitle>
            <CardDescription>
              Historical replays apply each holding’s actual return over a past window to
              today’s weights; hypothetical shocks go through the holdings’ factor betas.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScenarioPicker
              selected={custom ? null : scenarioId}
              onSelect={(id) => {
                setCustom(null);
                setScenarioId(id);
              }}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Describe a scenario in words</CardTitle>
            <CardDescription>The assistant converts the description into factor and sector shocks.</CardDescription>
          </CardHeader>
          <CardContent>
            <DescribeScenario onRun={setCustom} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Build a hypothetical scenario</CardTitle>
            <CardDescription>Factor shocks in percent (e.g. −20 for a 20% market fall in excess of the risk-free rate).</CardDescription>
          </CardHeader>
          <CardContent>
            <CustomBuilder onRun={setCustom} />
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Result</CardTitle>
          <CardDescription>
            P&amp;L is signed: negative is a loss. A scenario is one possible path under stated
            assumptions, not a probability or a worst case.
          </CardDescription>
        </CardHeader>
        <CardContent>{body ? <ResultView body={body} /> : <p className="text-sm text-muted-foreground">Choose a scenario.</p>}</CardContent>
      </Card>
    </div>
  );
}

export function StressLab() {
  return (
    <PortfolioGate>
      {({ portfolio }) => (
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Stress lab</h1>
            <p className="text-muted-foreground">How the current portfolio would have fared in past episodes and in hypothetical shocks.</p>
          </div>
          <Lab portfolio={portfolio} />
        </div>
      )}
    </PortfolioGate>
  );
}
