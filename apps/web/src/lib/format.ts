/** Number formatting. The API sends decimals (0.0234 = 2.34%) and rupees as plain numbers. */

const inr0 = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

/** 0.0234 -> "2.34%". null/undefined/NaN -> "–". */
export function pct(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return `${(x * 100).toFixed(digits)}%`;
}

/** Signed percentage: 0.05 -> "+5.00%", -0.05 -> "−5.00%". */
export function signedPct(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  const s = (Math.abs(x) * 100).toFixed(digits);
  if (Number(s) === 0) return `${s}%`;
  return x > 0 ? `+${s}%` : `−${s}%`;
}

/** Rupees with Indian digit grouping: 1000000 -> "₹10,00,000". */
export function rupees(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return inr0.format(x);
}

/** Plain number with fixed digits: ratios, betas, t-stats. */
export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return x.toFixed(digits);
}

/** p-values: small values in scientific notation so they never show as 0.000. */
export function pValue(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "–";
  return p < 0.001 ? p.toExponential(1) : p.toFixed(3);
}
