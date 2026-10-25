import type { Metadata } from "next";

import { MarketStatus } from "@/components/market-status";
import { Optimizer } from "@/components/planning/optimizer";

export const metadata: Metadata = { title: "Optimizer · AIWealth" };

export default function OptimizerPage() {
  return (
    <>
      <MarketStatus />
      <Optimizer />
    </>
  );
}
