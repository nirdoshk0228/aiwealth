import type { Metadata } from "next";

import { MarketStatus } from "@/components/market-status";
import { GoalPlanner } from "@/components/planning/goal-planner";

export const metadata: Metadata = { title: "Goal planner · AIWealth" };

export default function GoalsPage() {
  return (
    <>
      <MarketStatus />
      <GoalPlanner />
    </>
  );
}
