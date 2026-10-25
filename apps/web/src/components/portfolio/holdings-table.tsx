"use client";

import { Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { pct, rupees } from "@/lib/format";
import { useAssetNames } from "@/lib/hooks";
import { shortName } from "@/lib/names";
import { totalAmount } from "@/lib/portfolio";
import { usePortfolio } from "@/lib/store";

export function HoldingsTable() {
  const holdings = usePortfolio((s) => s.holdings);
  const setAmount = usePortfolio((s) => s.setAmount);
  const remove = usePortfolio((s) => s.remove);
  const names = useAssetNames();
  const total = totalAmount(holdings);

  if (holdings.length === 0) {
    return (
      <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No holdings yet. Add funds or stocks from the list, upload a CSV, or load a sample.
      </p>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Holding</TableHead>
          <TableHead className="w-40 text-right">Amount (₹)</TableHead>
          <TableHead className="w-20 text-right">Weight</TableHead>
          <TableHead className="w-10">
            <span className="sr-only">Remove</span>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {holdings.map((h) => {
          const name = names[h.asset_id] ?? h.asset_id;
          return (
            <TableRow key={h.asset_id}>
              <TableCell className="max-w-0">
                <p className="truncate font-medium" title={name}>
                  {shortName(name)}
                </p>
                <p className="text-xs text-muted-foreground">{h.asset_id}</p>
              </TableCell>
              <TableCell className="text-right">
                <Input
                  type="number"
                  inputMode="decimal"
                  min={0}
                  step={1000}
                  value={Number.isFinite(h.amount) ? Math.round(h.amount) : ""}
                  onChange={(e) => setAmount(h.asset_id, e.target.valueAsNumber)}
                  aria-label={`Amount for ${shortName(name)} in rupees`}
                  className="text-right tabular-nums"
                />
              </TableCell>
              <TableCell className="text-right tabular-nums">
                {total > 0 ? pct(h.amount / total, 1) : "–"}
              </TableCell>
              <TableCell>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => remove(h.asset_id)}
                  aria-label={`Remove ${shortName(name)}`}
                >
                  <Trash2 />
                </Button>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
      <TableFooter>
        <TableRow>
          <TableCell>Total ({holdings.length} holdings)</TableCell>
          <TableCell className="text-right tabular-nums">{rupees(total)}</TableCell>
          <TableCell className="text-right tabular-nums">{total > 0 ? "100%" : "–"}</TableCell>
          <TableCell />
        </TableRow>
      </TableFooter>
    </Table>
  );
}
