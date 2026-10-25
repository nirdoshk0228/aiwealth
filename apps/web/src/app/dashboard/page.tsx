import type { Metadata } from "next";

import { Dashboard } from "@/components/dashboard/dashboard";
import { MarketStatus } from "@/components/market-status";

export const metadata: Metadata = { title: "Risk dashboard · AIWealth" };

export default function DashboardPage() {
  return (
    <>
      <MarketStatus />
      <Dashboard />
    </>
  );
}
