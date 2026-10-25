"use client";

import { Play } from "lucide-react";
import { useState } from "react";

import { GoalFanChart } from "@/components/charts/planning-charts";
import { DataState } from "@/components/data-state";
import { PortfolioGate } from "@/components/portfolio-gate";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { Schemas } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { useQuery } from "@/lib/api/use-query";
import { pct, rupees } from "@/lib/format";
import { usePortfolioQuery } from "@/lib/hooks";
import type { PortfolioIn } from "@/lib/portfolio";

type Form = {
  initial: string;
  monthly: string;
  stepUp: string;
  years: string;
  goal: string;
  inflation: string;
};

const DEFAULT_FORM: Form = {
  initial: "100000",
  monthly: "15000",
  stepUp: "0",
  years: "15",
  goal: "5000000",
  inflation: "4",
};

const FIELDS: { key: keyof Form; label: string; hint: string }[] = [
  { key: "initial", label: "Invested today (₹)", hint: "lump sum at the start" },
  { key: "monthly", label: "Monthly SIP (₹)", hint: "in the first year" },
  { key: "stepUp", label: "Yearly SIP increase (%)", hint: "0 for a flat SIP" },
  { key: "years", label: "Years", hint: "1 to 40" },
  { key: "goal", label: "Goal in today’s ₹", hint: "inflated to the horizon" },
  { key: "inflation", label: "Inflation assumption (%)", hint: "default 4%, the RBI’s CPI target" },
];

function toRequest(f: Form, portfolio: PortfolioIn): Schemas["GoalRequest"] | string {
  const n = (s: string) => Number(s.replace(/,/g, ""));
  const v = {
    initial: n(f.initial),
    monthly: n(f.monthly),
    stepUp: n(f.stepUp),
    years: n(f.years),
    goal: n(f.goal),
    inflation: n(f.inflation),
  };
  if (Object.values(v).some((x) => !Number.isFinite(x))) return "Every field needs a number.";
  if (!Number.isInteger(v.years) || v.years < 1 || v.years > 40) return "Years must be a whole number from 1 to 40.";
  if (v.initial < 0 || v.monthly < 0) return "Amounts cannot be negative.";
  if (v.initial === 0 && v.monthly === 0) return "Enter an amount invested today or a monthly SIP.";
  if (v.goal <= 0) return "The goal must be above zero.";
  if (v.stepUp < 0 || v.stepUp > 50) return "The yearly increase must be between 0% and 50%.";
  if (v.inflation <= -5 || v.inflation >= 50) return "Inflation must be between -5% and 50%.";
  return {
    portfolio,
    initial_amount: v.initial,
    monthly_sip: v.monthly,
    step_up_annual: v.stepUp / 100,
    years: v.years,
    goal_amount: v.goal,
    inflation_annual: v.inflation / 100,
  };
}

function Simulation({ portfolio }: { portfolio: PortfolioIn }) {
  const [form, setForm] = useState<Form>(DEFAULT_FORM);
  const [submitted, setSubmitted] = useState<Schemas["GoalRequest"] | null>(() => {
    const r = toRequest(DEFAULT_FORM, portfolio);
    return typeof r === "string" ? null : r;
  });
  const [error, setError] = useState<string | null>(null);
  const q = usePortfolioQuery("/goals/simulate", submitted, (b, signal) => api.goal(b, { signal }));

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
      <Card>
        <CardHeader>
          <CardTitle>Your plan</CardTitle>
          <CardDescription>
            Returns are simulated from the current portfolio’s history; change the holdings on
            the portfolio page to test another mix.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              const r = toRequest(form, portfolio);
              if (typeof r === "string") setError(r);
              else {
                setError(null);
                setSubmitted(r);
              }
            }}
          >
            {FIELDS.map((f) => (
              <label key={f.key} className="block space-y-1 text-sm">
                <span>{f.label}</span>
                <Input
                  inputMode="decimal"
                  value={form[f.key]}
                  onChange={(e) => setForm((s) => ({ ...s, [f.key]: e.target.value }))}
                />
                <span className="text-xs text-muted-foreground">{f.hint}</span>
              </label>
            ))}
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            <Button type="submit">
              <Play /> Run simulation
            </Button>
          </form>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Range of outcomes</CardTitle>
          <CardDescription>10,000 simulated paths; percentiles show how often values were reached, not a prediction.</CardDescription>
        </CardHeader>
        <CardContent>
          <DataState query={q} loadingLabel="Simulating" height={360}>
            {(g) => (
              <div className="space-y-4">
                <div className="flex flex-wrap items-end gap-8">
                  <div>
                    <p className="text-xs text-muted-foreground">Paths that reached the goal</p>
                    <p className="text-3xl font-semibold tabular-nums">{pct(g.probability, 0)}</p>
                    <p className="text-xs text-muted-foreground">± {pct(g.probability_se, 1)} simulation error</p>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    <p>Goal at the horizon: {rupees(g.goal_nominal)} (inflated)</p>
                    <p>Total invested: {rupees(g.total_invested)}</p>
                    <p>
                      Median value: {rupees(g.percentiles.find((p) => p.percentile === 50)?.nominal)}
                    </p>
                  </div>
                </div>
                <GoalFanChart fan={g.fan} goal={g.goal_nominal} />
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Percentile</TableHead>
                      <TableHead className="text-right">Final value</TableHead>
                      <TableHead className="text-right">In today’s ₹</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {g.percentiles.map((p) => (
                      <TableRow key={p.percentile}>
                        <TableCell>{p.percentile}th</TableCell>
                        <TableCell className="text-right tabular-nums">{rupees(p.nominal)}</TableCell>
                        <TableCell className="text-right tabular-nums">{rupees(p.real)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <p className="text-xs text-muted-foreground">
                  Calibrated on {g.calibration.window.start} to {g.calibration.window.end}: median growth{" "}
                  {pct(g.calibration.median_growth)} a year, volatility {pct(g.calibration.volatility)}.
                </p>
                <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                  {g.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </div>
            )}
          </DataState>
        </CardContent>
      </Card>
    </div>
  );
}

function Questionnaire({ portfolio }: { portfolio: PortfolioIn }) {
  const qs = useQuery("questionnaire", (signal) => api.questionnaire({ signal }));
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState<Schemas["ProfileRequest"] | null>(null);
  const result = usePortfolioQuery("/goals/profile", submitted, (b, signal) => api.profile(b, { signal }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Risk profile questionnaire</CardTitle>
        <CardDescription>
          Five questions mapped to an illustrative band of equity-oriented allocation, compared
          with the current portfolio. Not a suitability assessment.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <DataState query={qs} loadingLabel="Loading questions" height={200}>
          {(data) => (
            <div className="grid gap-6 lg:grid-cols-2">
              <form
                className="space-y-4"
                onSubmit={(e) => {
                  e.preventDefault();
                  setSubmitted({ answers, portfolio });
                }}
              >
                {data.questions.map((q) => (
                  <fieldset key={q.id} className="space-y-1">
                    <legend className="text-sm font-medium">{q.text}</legend>
                    {q.options.map((o) => (
                      <label key={o.id} className="flex items-center gap-2 text-sm">
                        <input
                          type="radio"
                          name={q.id}
                          value={o.id}
                          checked={answers[q.id] === o.id}
                          onChange={() => setAnswers((a) => ({ ...a, [q.id]: o.id }))}
                        />
                        {o.label}
                      </label>
                    ))}
                  </fieldset>
                ))}
                <Button type="submit" disabled={Object.keys(answers).length < data.questions.length}>
                  See the band
                </Button>
              </form>
              <div className="space-y-3">
                {submitted === null ? (
                  <p className="text-sm text-muted-foreground">Answer all five questions to see the band.</p>
                ) : (
                  <DataState query={result} loadingLabel="Scoring" height={120}>
                    {(r) => (
                      <div className="space-y-3 text-sm">
                        <p>
                          Score {r.score} of {r.max_score}:{" "}
                          <Badge>{r.band.label}</Badge>, equity-oriented share {pct(r.band.equity_min, 0)} to{" "}
                          {pct(r.band.equity_max, 0)}.
                        </p>
                        <p className="text-muted-foreground">{r.band.description}</p>
                        {r.portfolio_mix && (
                          <p>
                            The current portfolio is {pct(r.portfolio_mix.equity_oriented, 0)} equity-oriented
                            (equity {pct(r.portfolio_mix.equity, 0)}, balanced advantage {pct(r.portfolio_mix.hybrid, 0)},
                            debt {pct(r.portfolio_mix.debt, 0)}, gold {pct(r.portfolio_mix.gold, 0)}), which is{" "}
                            {r.within_band ? "inside" : "outside"} this band.
                          </p>
                        )}
                        <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                          {r.notes.map((n) => (
                            <li key={n}>{n}</li>
                          ))}
                        </ul>
                        <p className="text-xs font-medium text-muted-foreground">{r.disclaimer}</p>
                      </div>
                    )}
                  </DataState>
                )}
              </div>
            </div>
          )}
        </DataState>
      </CardContent>
    </Card>
  );
}

export function GoalPlanner() {
  return (
    <PortfolioGate>
      {({ portfolio }) => (
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Goal planner</h1>
            <p className="text-muted-foreground">
              Simulate a SIP on the current portfolio’s mix and see the range of outcomes and how
              often a goal was reached.
            </p>
          </div>
          <Simulation portfolio={portfolio} />
          <Questionnaire portfolio={portfolio} />
        </div>
      )}
    </PortfolioGate>
  );
}
