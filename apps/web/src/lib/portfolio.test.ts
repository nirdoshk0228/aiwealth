import { describe, expect, it } from "vitest";

import {
  type AssetInfo,
  fromPortfolioIn,
  parseHoldingsCsv,
  portfolioProblems,
  resolveAsset,
  toPortfolioIn,
} from "./portfolio";

const universe: AssetInfo[] = [
  { asset_id: "equity:TCS", kind: "equity", symbol: "TCS", name: "TCS", available: true },
  { asset_id: "equity:M&M", kind: "equity", symbol: "M&M", name: "M&M", available: true },
  {
    asset_id: "mutual_fund:122639",
    kind: "mutual_fund",
    symbol: "122639",
    name: "Parag Parikh Flexi Cap",
    available: true,
  },
  { asset_id: "index:^NSEI", kind: "index", symbol: "^NSEI", name: "Nifty 50", available: true },
];

describe("toPortfolioIn", () => {
  it("turns rupee amounts into weights and a portfolio value", () => {
    const p = toPortfolioIn([
      { asset_id: "equity:TCS", amount: 300_000 },
      { asset_id: "mutual_fund:122639", amount: 700_000 },
    ]);
    expect(p).toEqual({
      portfolio_value: 1_000_000,
      holdings: [
        { asset_id: "equity:TCS", weight: 0.3 },
        { asset_id: "mutual_fund:122639", weight: 0.7 },
      ],
    });
  });

  it("weights sum to 1 for awkward amounts", () => {
    const p = toPortfolioIn([
      { asset_id: "a", amount: 1 },
      { asset_id: "b", amount: 1 },
      { asset_id: "c", amount: 1 },
    ]);
    const sum = p!.holdings.reduce((s, h) => s + h.weight, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-12);
  });

  it("returns null and explains why for invalid holdings", () => {
    expect(toPortfolioIn([])).toBeNull();
    expect(portfolioProblems([])).toEqual(["Add at least one holding."]);
    expect(portfolioProblems([{ asset_id: "a", amount: -5 }])).toContain(
      "Amounts must be zero or positive numbers.",
    );
    expect(portfolioProblems([{ asset_id: "a", amount: 0 }])).toContain(
      "The total amount must be above zero.",
    );
    expect(
      portfolioProblems([
        { asset_id: "a", amount: 1 },
        { asset_id: "a", amount: 2 },
      ]),
    ).toContain("Each asset may appear only once.");
  });

  it("round-trips with fromPortfolioIn", () => {
    const rows = fromPortfolioIn({
      portfolio_value: 500_000,
      holdings: [
        { asset_id: "equity:TCS", weight: 0.4 },
        { asset_id: "equity:M&M", weight: 0.6 },
      ],
    });
    expect(rows).toEqual([
      { asset_id: "equity:TCS", amount: 200_000 },
      { asset_id: "equity:M&M", amount: 300_000 },
    ]);
  });
});

describe("resolveAsset", () => {
  it("matches ids, symbols and scheme codes, case-insensitively", () => {
    expect(resolveAsset("tcs", universe)?.asset_id).toBe("equity:TCS");
    expect(resolveAsset("equity:m&m", universe)?.asset_id).toBe("equity:M&M");
    expect(resolveAsset("122639", universe)?.asset_id).toBe("mutual_fund:122639");
    expect(resolveAsset("^nsei", universe)?.asset_id).toBe("index:^NSEI");
    expect(resolveAsset("INFY", universe)).toBeUndefined();
  });
});

describe("parseHoldingsCsv", () => {
  it("reads amounts with rupee signs, commas and quotes", () => {
    const csv = 'symbol,amount\nTCS,"₹2,50,000"\n122639,750000\n';
    expect(parseHoldingsCsv(csv, universe)).toEqual({
      rows: [
        { asset_id: "equity:TCS", amount: 250_000 },
        { asset_id: "mutual_fund:122639", amount: 750_000 },
      ],
      errors: [],
    });
  });

  it("scales weights (decimals or percent) to the default value", () => {
    const pctCsv = "asset_id,weight\nequity:TCS,25%\nequity:M&M,75%";
    expect(parseHoldingsCsv(pctCsv, universe, 1_000_000).rows).toEqual([
      { asset_id: "equity:TCS", amount: 250_000 },
      { asset_id: "equity:M&M", amount: 750_000 },
    ]);
    const decCsv = "code,weight\n122639,0.5\nTCS,0.5";
    expect(parseHoldingsCsv(decCsv, universe, 200_000).rows).toEqual([
      { asset_id: "mutual_fund:122639", amount: 100_000 },
      { asset_id: "equity:TCS", amount: 100_000 },
    ]);
  });

  it("reports bad rows instead of dropping them silently", () => {
    const csv = "symbol,amount\nTCS,100\nINFY,100\nM&M,abc\nTCS,5\n# comment\n";
    const out = parseHoldingsCsv(csv, universe);
    expect(out.rows).toEqual([{ asset_id: "equity:TCS", amount: 100 }]);
    expect(out.errors).toEqual([
      'Line 3: "INFY" is not in the universe.',
      'Line 4: "abc" is not a valid non-negative number.',
      "Line 5: equity:TCS appears more than once.",
    ]);
  });

  it("rejects files without the needed columns", () => {
    expect(parseHoldingsCsv("name,qty\nTCS,1", universe).errors[0]).toMatch(/No asset column/);
    expect(parseHoldingsCsv("symbol,qty\nTCS,1", universe).errors[0]).toMatch(/No amount or weight/);
    expect(parseHoldingsCsv("symbol,amount", universe).errors[0]).toMatch(/header row and at least/);
  });
});
