import type { Metadata } from "next";

import { MarketStatus } from "@/components/market-status";
import { StressLab } from "@/components/planning/stress-lab";

export const metadata: Metadata = { title: "Stress lab · AIWealth" };

export default function StressPage() {
  return (
    <>
      <MarketStatus />
      <StressLab />
    </>
  );
}
