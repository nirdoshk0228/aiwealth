import { describe, expect, it } from "vitest";

import { num, pct, pValue, rupees, signedPct } from "./format";

describe("format", () => {
  it("formats decimals as percentages", () => {
    expect(pct(0.0234)).toBe("2.34%");
    expect(pct(0.1, 1)).toBe("10.0%");
    expect(pct(null)).toBe("–");
    expect(pct(Number.NaN)).toBe("–");
  });

  it("signs percentages with a real minus sign", () => {
    expect(signedPct(0.05)).toBe("+5.00%");
    expect(signedPct(-0.3416)).toBe("−34.16%");
    expect(signedPct(0)).toBe("0.00%");
    expect(signedPct(-0.00001)).toBe("0.00%");
  });

  it("uses Indian digit grouping for rupees", () => {
    expect(rupees(1_000_000)).toBe("₹10,00,000");
    expect(rupees(12_345_678.9)).toBe("₹1,23,45,679");
    expect(rupees(undefined)).toBe("–");
  });

  it("formats ratios and p-values", () => {
    expect(num(0.81591, 2)).toBe("0.82");
    expect(pValue(0.6932)).toBe("0.693");
    expect(pValue(0.00002)).toBe("2.0e-5");
  });
});
