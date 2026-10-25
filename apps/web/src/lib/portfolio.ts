/**
 * Portfolio logic on the client: holdings are entered as rupee amounts and converted to the
 * API's `PortfolioIn` (weights + portfolio value). Pure functions, unit-tested.
 *
 * The API re-validates everything (long-only, weights summing to 1, known asset ids), so these
 * checks exist for fast feedback, not as the source of truth.
 */
import type { Schemas } from "@/lib/api/client";

export type HoldingRow = { asset_id: string; amount: number };
export type PortfolioIn = Schemas["PortfolioIn"];
export type AssetInfo = Schemas["AssetOut"];

export const MAX_HOLDINGS = 40;
export const DEFAULT_VALUE = 1_000_000;

export function totalAmount(rows: HoldingRow[]): number {
  return rows.reduce((s, r) => s + (Number.isFinite(r.amount) ? r.amount : 0), 0);
}

/** Problems that would make the API reject the portfolio, in plain words. */
export function portfolioProblems(rows: HoldingRow[]): string[] {
  const problems: string[] = [];
  if (rows.length === 0) problems.push("Add at least one holding.");
  if (rows.length > MAX_HOLDINGS) problems.push(`At most ${MAX_HOLDINGS} holdings.`);
  if (rows.some((r) => !Number.isFinite(r.amount) || r.amount < 0)) {
    problems.push("Amounts must be zero or positive numbers.");
  }
  const ids = rows.map((r) => r.asset_id);
  if (new Set(ids).size !== ids.length) problems.push("Each asset may appear only once.");
  if (rows.length > 0 && totalAmount(rows) <= 0) problems.push("The total amount must be above zero.");
  return problems;
}

/** Rupee amounts -> weights (amount / total) and portfolio value (total). Null if invalid. */
export function toPortfolioIn(rows: HoldingRow[]): PortfolioIn | null {
  if (portfolioProblems(rows).length > 0) return null;
  const total = totalAmount(rows);
  return {
    portfolio_value: total,
    holdings: rows.map((r) => ({ asset_id: r.asset_id, weight: r.amount / total })),
  };
}

/** API portfolio (weights) -> rows of rupee amounts (weight x value). */
export function fromPortfolioIn(p: PortfolioIn): HoldingRow[] {
  const value = p.portfolio_value ?? DEFAULT_VALUE;
  return p.holdings.map((h) => ({ asset_id: h.asset_id, amount: h.weight * value }));
}

// --- CSV import ---

export type CsvResult = { rows: HoldingRow[]; errors: string[] };

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else quoted = !quoted;
    } else if (c === "," && !quoted) {
      out.push(cur.trim());
      cur = "";
    } else cur += c;
  }
  out.push(cur.trim());
  return out;
}

/** Match an asset by full id ("equity:TCS"), symbol ("TCS", "^NSEI") or scheme code ("122639"). */
export function resolveAsset(token: string, universe: AssetInfo[]): AssetInfo | undefined {
  const t = token.trim().toUpperCase();
  return universe.find(
    (a) => a.asset_id.toUpperCase() === t || a.symbol.toUpperCase() === t,
  );
}

const ASSET_COLS = ["asset", "asset_id", "symbol", "code", "scheme_code", "ticker"];
const AMOUNT_COLS = ["amount", "value", "rupees", "inr"];
const WEIGHT_COLS = ["weight", "allocation", "percent", "pct"];

/**
 * Parse holdings from CSV with a header row: one asset column (asset_id, symbol or scheme
 * code) and either an amount column (rupees) or a weight column (decimals or percentages,
 * scaled to `defaultValue`). Rows that cannot be used are reported, not silently dropped.
 */
export function parseHoldingsCsv(
  text: string,
  universe: AssetInfo[],
  defaultValue = DEFAULT_VALUE,
): CsvResult {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith("#"));
  if (lines.length < 2) return { rows: [], errors: ["The file needs a header row and at least one holding."] };

  const header = splitCsvLine(lines[0]).map((h) => h.toLowerCase());
  const assetCol = header.findIndex((h) => ASSET_COLS.includes(h));
  const amountCol = header.findIndex((h) => AMOUNT_COLS.includes(h));
  const weightCol = header.findIndex((h) => WEIGHT_COLS.includes(h));
  if (assetCol < 0) return { rows: [], errors: [`No asset column (expected one of: ${ASSET_COLS.join(", ")}).`] };
  if (amountCol < 0 && weightCol < 0) {
    return { rows: [], errors: ["No amount or weight column (e.g. 'amount' in rupees or 'weight')."] };
  }
  const valueCol = amountCol >= 0 ? amountCol : weightCol;

  const errors: string[] = [];
  const parsed: { asset_id: string; x: number }[] = [];
  const seen = new Set<string>();
  for (let i = 1; i < lines.length; i++) {
    const cells = splitCsvLine(lines[i]);
    const token = cells[assetCol] ?? "";
    const raw = (cells[valueCol] ?? "").replace(/[₹,%\s]/g, "");
    const x = Number(raw);
    const asset = resolveAsset(token, universe);
    if (!asset) {
      errors.push(`Line ${i + 1}: "${token}" is not in the universe.`);
      continue;
    }
    if (raw === "" || !Number.isFinite(x) || x < 0) {
      errors.push(`Line ${i + 1}: "${cells[valueCol] ?? ""}" is not a valid non-negative number.`);
      continue;
    }
    if (seen.has(asset.asset_id)) {
      errors.push(`Line ${i + 1}: ${asset.asset_id} appears more than once.`);
      continue;
    }
    seen.add(asset.asset_id);
    parsed.push({ asset_id: asset.asset_id, x });
  }

  if (amountCol >= 0) return { rows: parsed.map((p) => ({ asset_id: p.asset_id, amount: p.x })), errors };
  const sum = parsed.reduce((s, p) => s + p.x, 0);
  if (parsed.length > 0 && sum <= 0) return { rows: [], errors: [...errors, "Weights sum to zero."] };
  return {
    rows: parsed.map((p) => ({ asset_id: p.asset_id, amount: (p.x / sum) * defaultValue })),
    errors,
  };
}
