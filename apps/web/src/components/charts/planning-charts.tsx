"use client";

import type { Data } from "plotly.js";
import { useMemo } from "react";

import { COLORS, Plot } from "@/components/charts/plot";
import type { Schemas } from "@/lib/api/client";
import { axisLabel } from "@/lib/names";

/** Per-asset contribution to the scenario P&L (signed, % of portfolio value). */
export function ContributionChart({ impacts }: { impacts: Schemas["AssetImpactOut"][] }) {
  const data = useMemo<Data[]>(() => {
    const sorted = [...impacts].sort((a, b) => a.contribution - b.contribution);
    return [
      {
        type: "bar",
        orientation: "h",
        y: sorted.map((i) => axisLabel(i.asset_id, i.name)),
        x: sorted.map((i) => i.contribution * 100),
        marker: { color: sorted.map((i) => (i.contribution < 0 ? COLORS.loss : COLORS.gain)) },
        customdata: sorted.map((i) => [i.asset_return * 100, i.weight * 100]),
        hovertemplate:
          "%{y}<br>contribution %{x:.2f}% of value<br>asset return %{customdata[0]:.2f}%, weight %{customdata[1]:.1f}%<extra></extra>",
      },
    ];
  }, [impacts]);
  return (
    <Plot
      data={data}
      height={Math.max(220, 60 + impacts.length * 28)}
      layout={{
        margin: { l: 150, r: 16, t: 8, b: 40 },
        xaxis: { ticksuffix: "%", title: { text: "Contribution to P&L (% of portfolio value)" } },
        yaxis: { automargin: true },
        showlegend: false,
      }}
      ariaLabel="Each holding's contribution to the scenario profit or loss"
    />
  );
}

const MARKERS: { key: "min_variance" | "max_sharpe" | "max_return" | "risk_parity" | "current"; color: string; symbol: string }[] = [
  { key: "min_variance", color: "#0891b2", symbol: "diamond" },
  { key: "max_sharpe", color: "#16a34a", symbol: "star" },
  { key: "max_return", color: "#9333ea", symbol: "triangle-up" },
  { key: "risk_parity", color: "#ea580c", symbol: "square" },
  { key: "current", color: "#111827", symbol: "circle" },
];

export function FrontierChart({ result }: { result: Schemas["OptimizeResponse"] }) {
  const data = useMemo<Data[]>(() => {
    const traces: Data[] = [
      {
        type: "scatter",
        mode: "markers",
        name: "Individual assets",
        x: result.assets.map((a) => a.volatility * 100),
        y: result.assets.map((a) => a.expected_return * 100),
        text: result.assets.map((a) => axisLabel(a.asset_id, a.name)),
        marker: { color: COLORS.muted, size: 7 },
        hovertemplate: "%{text}<br>vol %{x:.1f}%, return %{y:.1f}%<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "Efficient frontier (long-only)",
        x: result.frontier.map((p) => p.volatility * 100),
        y: result.frontier.map((p) => p.expected_return * 100),
        line: { color: COLORS.portfolio, width: 2.5 },
        hovertemplate: "vol %{x:.2f}%, return %{y:.2f}%<extra>frontier</extra>",
      },
    ];
    for (const m of MARKERS) {
      const p = result[m.key];
      if (!p) continue;
      traces.push({
        type: "scatter",
        mode: "markers",
        name: p.label,
        x: [p.volatility * 100],
        y: [p.expected_return * 100],
        marker: { color: m.color, size: 13, symbol: m.symbol, line: { color: "white", width: 1 } },
        hovertemplate: `${p.label}<br>vol %{x:.2f}%, return %{y:.2f}%<extra></extra>`,
      });
    }
    return traces;
  }, [result]);
  return (
    <Plot
      data={data}
      height={420}
      layout={{
        xaxis: { title: { text: "Annual volatility (%)" }, ticksuffix: "%", rangemode: "tozero" },
        yaxis: { title: { text: "Historical average annual return (%)" }, ticksuffix: "%" },
        legend: { orientation: "h", y: -0.2 },
      }}
      ariaLabel="Efficient frontier with the minimum-variance, maximum-Sharpe, maximum-return, risk-parity and current portfolios"
    />
  );
}

export function GoalFanChart({ fan, goal }: { fan: Schemas["GoalFan"]; goal: number }) {
  const data = useMemo<Data[]>(() => {
    const lakh = (xs: number[]) => xs.map((x) => x / 1e5);
    const band = (lo: number[], hi: number[], name: string, color: string): Data[] => [
      { type: "scatter", mode: "lines", x: fan.years, y: lakh(lo), line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      {
        type: "scatter",
        mode: "lines",
        x: fan.years,
        y: lakh(hi),
        fill: "tonexty",
        fillcolor: color,
        line: { width: 0 },
        name,
        hoverinfo: "skip",
      },
    ];
    return [
      ...band(fan.p5, fan.p95, "5th to 95th percentile", "rgba(37,99,235,0.12)"),
      ...band(fan.p25, fan.p75, "25th to 75th percentile", "rgba(37,99,235,0.25)"),
      {
        type: "scatter",
        mode: "lines",
        name: "Median",
        x: fan.years,
        y: lakh(fan.p50),
        line: { color: COLORS.portfolio, width: 2.5 },
        hovertemplate: "year %{x}: median ₹%{y:.1f} lakh<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "Amount invested",
        x: fan.years,
        y: lakh(fan.invested),
        line: { color: COLORS.benchmark, dash: "dot", width: 1.5 },
        hovertemplate: "year %{x}: invested ₹%{y:.1f} lakh<extra></extra>",
      },
      {
        type: "scatter",
        mode: "markers",
        name: "Goal (inflated)",
        x: [fan.years[fan.years.length - 1]],
        y: [goal / 1e5],
        marker: { color: COLORS.var, size: 12, symbol: "star" },
        hovertemplate: "goal ₹%{y:.1f} lakh<extra></extra>",
      },
    ];
  }, [fan, goal]);
  return (
    <Plot
      data={data}
      height={380}
      layout={{
        xaxis: { title: { text: "Years from today" }, dtick: fan.years.length > 20 ? 5 : 1 },
        yaxis: { title: { text: "Portfolio value (₹ lakh, nominal)" }, rangemode: "tozero" },
      }}
      ariaLabel="Range of simulated portfolio values over time, with the median, the amount invested and the goal"
    />
  );
}
