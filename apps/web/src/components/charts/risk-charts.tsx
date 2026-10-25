"use client";

/** Chart components for the risk dashboard. Inputs are API responses; no numbers are computed
 * here except unit conversions for display (decimal -> percent). */
import type { Data } from "plotly.js";
import { useMemo } from "react";

import { COLORS, Plot } from "@/components/charts/plot";
import type { Schemas } from "@/lib/api/client";

type Series = Schemas["TimeSeries"];

export function GrowthChart({ portfolio, benchmark }: { portfolio: Series; benchmark: Series | null }) {
  const data = useMemo<Data[]>(() => {
    const traces: Data[] = [
      {
        type: "scatter",
        mode: "lines",
        name: "Portfolio",
        x: portfolio.dates,
        y: portfolio.values,
        line: { color: COLORS.portfolio, width: 2 },
        hovertemplate: "%{x}<br>₹1 grew to ₹%{y:.3f}<extra>Portfolio</extra>",
      },
    ];
    if (benchmark) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: "Nifty 50 (price index)",
        x: benchmark.dates,
        y: benchmark.values,
        line: { color: COLORS.benchmark, width: 1.5, dash: "dot" },
        hovertemplate: "%{x}<br>₹1 grew to ₹%{y:.3f}<extra>Nifty 50</extra>",
      });
    }
    return traces;
  }, [portfolio, benchmark]);
  return (
    <Plot
      data={data}
      layout={{ yaxis: { title: { text: "Value of ₹1 invested" } } }}
      ariaLabel="Growth of one rupee invested in the portfolio and in the Nifty 50"
    />
  );
}

export function DrawdownChart({ drawdown }: { drawdown: Series }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "scatter",
        mode: "lines",
        name: "Drawdown",
        x: drawdown.dates,
        y: drawdown.values.map((v) => v * 100),
        fill: "tozeroy",
        line: { color: COLORS.loss, width: 1 },
        fillcolor: "rgba(220,38,38,0.18)",
        hovertemplate: "%{x}<br>%{y:.2f}% below the previous peak<extra></extra>",
      },
    ],
    [drawdown],
  );
  return (
    <Plot
      data={data}
      height={240}
      layout={{ yaxis: { title: { text: "Drawdown (%)" }, ticksuffix: "%" }, showlegend: false }}
      ariaLabel="Drawdown of the portfolio from its running peak"
    />
  );
}

export function CorrelationHeatmap({ labels, matrix }: { labels: string[]; matrix: (number | null)[][] }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "heatmap",
        x: labels,
        y: labels,
        z: matrix,
        zmin: -1,
        zmax: 1,
        colorscale: "RdBu",
        reversescale: true,
        texttemplate: labels.length <= 12 ? "%{z:.2f}" : "",
        hovertemplate: "%{y} vs %{x}<br>correlation %{z:.2f}<extra></extra>",
        colorbar: { thickness: 12, tickvals: [-1, -0.5, 0, 0.5, 1] },
      },
    ],
    [labels, matrix],
  );
  const size = Math.max(320, Math.min(640, 140 + labels.length * 34));
  return (
    <Plot
      data={data}
      height={size}
      layout={{
        margin: { l: 150, r: 16, t: 8, b: 120 },
        xaxis: { tickangle: -40, automargin: true },
        yaxis: { autorange: "reversed", automargin: true },
      }}
      ariaLabel="Correlation matrix of the holdings' daily returns"
    />
  );
}

export function AttributionChart({
  labels,
  weights,
  shares,
}: {
  labels: string[];
  weights: number[];
  shares: number[];
}) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        name: "Share of capital",
        y: labels,
        x: weights.map((w) => w * 100),
        marker: { color: COLORS.muted },
        hovertemplate: "%{y}<br>%{x:.1f}% of capital<extra></extra>",
      },
      {
        type: "bar",
        orientation: "h",
        name: "Share of volatility",
        y: labels,
        x: shares.map((s) => s * 100),
        marker: { color: COLORS.portfolio },
        hovertemplate: "%{y}<br>%{x:.1f}% of portfolio volatility<extra></extra>",
      },
    ],
    [labels, weights, shares],
  );
  return (
    <Plot
      data={data}
      height={Math.max(260, 70 + labels.length * 34)}
      layout={{
        barmode: "group",
        margin: { l: 150, r: 16, t: 8, b: 56 },
        xaxis: { ticksuffix: "%" },
        yaxis: { autorange: "reversed", automargin: true },
      }}
      ariaLabel="Each holding's share of capital compared with its share of portfolio volatility"
    />
  );
}

export function FactorChart({ coefficients }: { coefficients: Schemas["FactorCoefficient"][] }) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        x: coefficients.map((c) => c.factor),
        y: coefficients.map((c) => c.beta),
        error_y: {
          type: "data",
          array: coefficients.map((c) => 1.96 * c.std_error),
          color: "#52525b",
          thickness: 1.2,
        },
        marker: { color: coefficients.map((c) => (c.beta >= 0 ? COLORS.portfolio : COLORS.var)) },
        hovertemplate: "%{x}: beta %{y:.3f}<extra></extra>",
      },
    ],
    [coefficients],
  );
  return (
    <Plot
      data={data}
      height={260}
      layout={{ yaxis: { title: { text: "Beta (±1.96 standard errors)" }, zeroline: true }, showlegend: false }}
      ariaLabel="Factor betas of the portfolio with 95% confidence bars"
    />
  );
}

export function BacktestChart({ result }: { result: Schemas["BacktestResponse"] }) {
  const data = useMemo<Data[]>(() => {
    const loss = result.realised.map((r) => -r * 100);
    const breachSet = new Set(result.breach_dates);
    const bx: string[] = [];
    const by: number[] = [];
    result.dates.forEach((d, i) => {
      if (breachSet.has(d)) {
        bx.push(d);
        by.push(loss[i]);
      }
    });
    return [
      {
        type: "scatter",
        mode: "lines",
        name: "Daily loss (gain below 0)",
        x: result.dates,
        y: loss,
        line: { color: "#94a3b8", width: 0.8 },
        hovertemplate: "%{x}<br>loss %{y:.2f}%<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: `VaR forecast (${(result.confidence * 100).toFixed(0)}%, 1 day)`,
        x: result.dates,
        y: result.var.map((v) => v * 100),
        line: { color: COLORS.var, width: 1.5 },
        hovertemplate: "%{x}<br>VaR %{y:.2f}%<extra></extra>",
      },
      {
        type: "scatter",
        mode: "markers",
        name: `Breach (${result.n_breaches})`,
        x: bx,
        y: by,
        marker: { color: COLORS.loss, size: 6, symbol: "x" },
        hovertemplate: "%{x}<br>loss %{y:.2f}% exceeded VaR<extra></extra>",
      },
    ];
  }, [result]);
  return (
    <Plot
      data={data}
      height={320}
      layout={{ yaxis: { title: { text: "Loss (% of value)" }, ticksuffix: "%" } }}
      ariaLabel="Daily portfolio losses against the VaR forecast made the day before, with breaches marked"
    />
  );
}
