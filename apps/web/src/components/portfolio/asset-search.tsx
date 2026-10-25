"use client";

import { Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { DataState } from "@/components/data-state";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { Schemas } from "@/lib/api/client";
import { useUniverse } from "@/lib/hooks";
import { shortName } from "@/lib/names";
import { usePortfolio } from "@/lib/store";

const DEFAULT_ADD_AMOUNT = 100_000;

const KIND_LABEL: Record<Schemas["AssetOut"]["kind"], string> = {
  mutual_fund: "Fund",
  equity: "Stock",
  index: "Index",
};

function groupLabel(a: Schemas["AssetOut"]): string {
  if (a.category) return a.category.replaceAll("_", " ");
  return a.industry ?? "benchmark";
}

export function AssetSearch() {
  const universe = useUniverse();
  const holdings = usePortfolio((s) => s.holdings);
  const upsert = usePortfolio((s) => s.upsert);
  const [q, setQ] = useState("");
  const held = useMemo(() => new Set(holdings.map((h) => h.asset_id)), [holdings]);

  return (
    <div className="space-y-3">
      <label className="relative block">
        <span className="sr-only">Search funds and stocks</span>
        <Search className="absolute top-2 left-2.5 size-4 text-muted-foreground" aria-hidden />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search by name, symbol, category or industry"
          className="pl-8"
        />
      </label>
      <DataState query={universe} loadingLabel="Loading the universe" height={240}>
        {(u) => {
          const needle = q.trim().toLowerCase();
          const matches = u.assets.filter(
            (a) =>
              a.available &&
              (needle === "" ||
                [a.name, a.symbol, a.category ?? "", a.industry ?? "", a.amc ?? ""]
                  .join(" ")
                  .toLowerCase()
                  .includes(needle)),
          );
          return (
            <ul className="max-h-[28rem] divide-y overflow-y-auto rounded-lg border">
              {matches.length === 0 && (
                <li className="p-3 text-sm text-muted-foreground">No asset matches “{q}”.</li>
              )}
              {matches.map((a) => (
                <li key={a.asset_id} className="flex items-center gap-3 p-2.5">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium" title={a.name}>
                      {shortName(a.name)}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      <Badge variant="secondary" className="mr-1.5">
                        {KIND_LABEL[a.kind]}
                      </Badge>
                      {a.kind === "mutual_fund" ? `code ${a.symbol}` : a.symbol} ·{" "}
                      <span className="capitalize">{groupLabel(a)}</span>
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant={held.has(a.asset_id) ? "secondary" : "outline"}
                    disabled={held.has(a.asset_id)}
                    onClick={() => upsert({ asset_id: a.asset_id, amount: DEFAULT_ADD_AMOUNT })}
                    aria-label={`Add ${shortName(a.name)}`}
                  >
                    <Plus /> {held.has(a.asset_id) ? "Added" : "Add"}
                  </Button>
                </li>
              ))}
            </ul>
          );
        }}
      </DataState>
    </div>
  );
}
