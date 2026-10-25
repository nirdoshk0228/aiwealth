/** Short display names: scheme names carry plan/option suffixes that clutter tables and charts. */

/** "ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund) - Direct Plan - Growth"
 *  -> "ICICI Prudential Large Cap Fund". Stock and index names are returned unchanged. */
export function shortName(name: string): string {
  const head = name.split(" - ")[0] ?? name;
  return head.replace(/\s*\(erstwhile[^)]*\)/i, "").trim();
}

/** Very short label for chart axes: the stock symbol, or the first words of a fund name. */
export function axisLabel(assetId: string, name: string, maxLen = 22): string {
  if (assetId.startsWith("equity:")) return assetId.slice("equity:".length);
  if (assetId.startsWith("index:")) return name;
  const s = shortName(name).replace(/\bFund\b/i, "").replace(/\s+/g, " ").trim();
  return s.length > maxLen ? `${s.slice(0, maxLen - 1)}…` : s;
}
