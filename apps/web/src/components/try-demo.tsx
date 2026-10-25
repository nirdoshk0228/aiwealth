"use client";

import { ArrowRight, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { fromPortfolioIn } from "@/lib/portfolio";
import { usePortfolio } from "@/lib/store";

/** Load a sample portfolio from the API (no login) and open the dashboard. */
export function TryDemoButton({ sampleId = "sample_fund_mix" }: { sampleId?: string }) {
  const router = useRouter();
  const setHoldings = usePortfolio((s) => s.setHoldings);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      const samples = await api.demoPortfolios();
      const sample = samples.find((s) => s.id === sampleId) ?? samples[0];
      if (!sample) throw new ApiError(404, "No sample portfolio is available.");
      setHoldings(fromPortfolioIn(sample.portfolio), sample.name);
      router.push("/dashboard");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <Button size="lg" onClick={start} disabled={busy} className="h-10 px-4">
        {busy ? <Loader2 className="animate-spin" /> : null}
        Try the demo
        <ArrowRight />
      </Button>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
