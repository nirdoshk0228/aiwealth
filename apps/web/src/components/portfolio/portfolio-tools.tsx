"use client";

import { FileUp, Loader2 } from "lucide-react";
import { useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api/endpoints";
import { useQuery } from "@/lib/api/use-query";
import { useUniverse } from "@/lib/hooks";
import { fromPortfolioIn, parseHoldingsCsv } from "@/lib/portfolio";
import { usePortfolio } from "@/lib/store";

export function SampleLoader() {
  const samples = useQuery("demo-portfolios", (signal) => api.demoPortfolios({ signal }));
  const setHoldings = usePortfolio((s) => s.setHoldings);
  if (samples.status === "error") {
    return <p className="text-sm text-destructive">Samples unavailable: {samples.error.message}</p>;
  }
  if (samples.status !== "success") {
    return (
      <Button variant="outline" size="sm" disabled>
        <Loader2 className="animate-spin" /> Loading samples
      </Button>
    );
  }
  return (
    <div className="flex flex-wrap gap-2">
      {samples.data.map((s) => (
        <Button
          key={s.id}
          variant="outline"
          size="sm"
          title={s.description}
          onClick={() => setHoldings(fromPortfolioIn(s.portfolio), s.name)}
        >
          {s.name.replace(/^Sample:\s*/i, "Load sample: ")}
        </Button>
      ))}
    </div>
  );
}

export function CsvUpload() {
  const universe = useUniverse();
  const setHoldings = usePortfolio((s) => s.setHoldings);
  const input = useRef<HTMLInputElement>(null);
  const [report, setReport] = useState<{ ok: number; errors: string[] } | null>(null);

  async function onFile(file: File) {
    if (universe.status !== "success") return;
    const text = await file.text();
    const res = parseHoldingsCsv(text, universe.data.assets);
    if (res.rows.length > 0) setHoldings(res.rows, null);
    setReport({ ok: res.rows.length, errors: res.errors });
  }

  return (
    <div className="space-y-2">
      <input
        ref={input}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void onFile(f);
          e.target.value = "";
        }}
      />
      <Button
        variant="outline"
        size="sm"
        disabled={universe.status !== "success"}
        onClick={() => input.current?.click()}
      >
        <FileUp /> Upload CSV
      </Button>
      <p className="text-xs text-muted-foreground">
        Columns: <code>symbol</code> (or <code>asset_id</code>, scheme <code>code</code>) and{" "}
        <code>amount</code> in rupees, or <code>weight</code> (0.25 or 25%).
      </p>
      {report && (
        <Alert variant={report.errors.length ? "destructive" : "default"}>
          <AlertTitle>
            {report.ok > 0
              ? `Imported ${report.ok} holding${report.ok === 1 ? "" : "s"}`
              : "Nothing imported"}
          </AlertTitle>
          {report.errors.length > 0 && (
            <AlertDescription>
              <ul className="list-disc pl-4">
                {report.errors.slice(0, 8).map((e) => (
                  <li key={e}>{e}</li>
                ))}
                {report.errors.length > 8 && <li>…and {report.errors.length - 8} more</li>}
              </ul>
            </AlertDescription>
          )}
        </Alert>
      )}
    </div>
  );
}
