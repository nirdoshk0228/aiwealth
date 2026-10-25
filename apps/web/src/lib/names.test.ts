import { describe, expect, it } from "vitest";

import { axisLabel, shortName } from "./names";

describe("names", () => {
  it("drops plan/option suffixes and 'erstwhile' notes", () => {
    expect(
      shortName("ICICI Prudential Large Cap Fund (erstwhile Bluechip Fund) - Direct Plan - Growth"),
    ).toBe("ICICI Prudential Large Cap Fund");
    expect(shortName("SBI GILT FUND - Direct Plan - Growth")).toBe("SBI GILT FUND");
    expect(shortName("Tata Consultancy Services Ltd.")).toBe("Tata Consultancy Services Ltd.");
  });

  it("uses symbols for stocks and truncates long fund labels", () => {
    expect(axisLabel("equity:M&M", "Mahindra & Mahindra Ltd.")).toBe("M&M");
    expect(axisLabel("index:^NSEI", "Nifty 50")).toBe("Nifty 50");
    expect(
      axisLabel("mutual_fund:119533", "Aditya Birla Sun Life Corporate Bond Fund - Direct Plan"),
    ).toBe("Aditya Birla Sun Life…");
  });
});
