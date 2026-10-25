"use client";

/** Describe a scenario in words; the assistant turns it into shocks (validated by the API),
 * which are shown before the engine runs them. The model never computes the P&L. */
import { Loader2, Sparkles } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { Schemas } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { signedPct } from "@/lib/format";

type Spec = Schemas["HypotheticalScenarioSpec"];

function describeTarget(t: Schemas["DirectShockSpec"]["target"]): string {
  if (t.kind === "sector") return `sector ${t.sector}`;
  if (t.kind === "category") return `category ${t.category.replaceAll("_", " ")}`;
  return t.asset_id;
}

export function DescribeScenario({ onRun }: { onRun: (s: Spec) => void }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [spec, setSpec] = useState<Spec | null>(null);

  async function parse() {
    setBusy(true);
    setError(null);
    setSpec(null);
    try {
      const s = await api.parseScenario({ text });
      setSpec(s);
      onRun(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <Textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. Banks fall 12%, the market falls 6% and gold rises 5%"
        maxLength={1000}
        rows={2}
        aria-label="Scenario in words"
      />
      <Button onClick={() => void parse()} disabled={busy || text.trim().length < 3}>
        {busy ? <Loader2 className="animate-spin" /> : <Sparkles />} Turn into shocks and run
      </Button>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {spec && (
        <div className="rounded-md border p-2 text-xs">
          <p className="font-medium">{spec.name}: shocks applied</p>
          <ul className="mt-1 list-disc pl-4">
            {Object.entries(spec.factor_shocks ?? {}).map(([f, v]) => (
              <li key={f}>
                factor {f}: {signedPct(v)}
              </li>
            ))}
            {(spec.direct_shocks ?? []).map((d, i) => (
              <li key={i}>
                {describeTarget(d.target)}: {signedPct(d.shock)}
              </li>
            ))}
          </ul>
          <p className="mt-1 text-muted-foreground">
            Check these match what you meant; the P&amp;L is computed by the risk engine.
          </p>
        </div>
      )}
    </div>
  );
}
