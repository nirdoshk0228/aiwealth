"use client";

/**
 * Minimal Plotly wrapper. Plotly (cartesian bundle, ~1.5 MB) is imported dynamically on first
 * use, so pages without charts never download it; `Plotly.react` updates the chart in place
 * when data changes, and the chart is purged on unmount.
 */
import type { Config, Data, Layout } from "plotly.js";
import { useEffect, useRef, useState } from "react";

import { Skeleton } from "@/components/ui/skeleton";

type PlotlyModule = typeof import("plotly.js-cartesian-dist-min").default;

const BASE_LAYOUT: Partial<Layout> = {
  autosize: true,
  margin: { l: 56, r: 16, t: 16, b: 40 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  // A concrete family: Plotly measures legend text with it and cannot resolve CSS variables.
  font: { family: "Geist, ui-sans-serif, system-ui, Segoe UI, sans-serif", size: 12, color: "#3f3f46" },
  xaxis: { gridcolor: "#e4e4e7", zerolinecolor: "#a1a1aa" },
  yaxis: { gridcolor: "#e4e4e7", zerolinecolor: "#a1a1aa" },
  legend: { orientation: "h", y: -0.18 },
  hoverlabel: { font: { size: 12 } },
};

const BASE_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};

export const COLORS = {
  portfolio: "#2563eb",
  benchmark: "#71717a",
  loss: "#dc2626",
  gain: "#16a34a",
  var: "#ea580c",
  muted: "#a1a1aa",
};

export function Plot({
  data,
  layout,
  height = 320,
  ariaLabel,
}: {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  ariaLabel: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [plotly, setPlotly] = useState<PlotlyModule | null>(null);

  useEffect(() => {
    let active = true;
    import("plotly.js-cartesian-dist-min").then((m) => {
      if (active) setPlotly(() => m.default);
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!plotly || !el) return;
    const merged: Partial<Layout> = {
      ...BASE_LAYOUT,
      ...layout,
      height,
      xaxis: { ...BASE_LAYOUT.xaxis, ...layout?.xaxis },
      yaxis: { ...BASE_LAYOUT.yaxis, ...layout?.yaxis },
    };
    void plotly.react(el, data, merged, BASE_CONFIG);
  }, [plotly, data, layout, height]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (plotly && el) plotly.purge(el);
    };
  }, [plotly]);

  return (
    <div role="img" aria-label={ariaLabel} className="w-full">
      {!plotly && <Skeleton className="w-full" style={{ height }} />}
      <div ref={ref} className={plotly ? "w-full" : "hidden"} />
    </div>
  );
}
