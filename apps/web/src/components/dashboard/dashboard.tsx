"use client";

import Link from "next/link";

import {
  AttributionCard,
  BacktestCard,
  CorrelationCard,
  FactorCard,
  PerformanceCard,
  SummaryCards,
  VaRCard,
} from "@/components/dashboard/cards";
import { SettingsControls } from "@/components/portfolio/analyse-bar";
import { PortfolioGate } from "@/components/portfolio-gate";
import { rupees } from "@/lib/format";

export function Dashboard() {
  return (
    <PortfolioGate>
      {({ portfolio, settings, label }) => (
        <div className="space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Risk dashboard</h1>
              <p className="text-muted-foreground">
                {label ?? "Your portfolio"} · {portfolio.holdings.length} holdings ·{" "}
                {rupees(portfolio.portfolio_value)} ·{" "}
                <Link href="/portfolio" className="underline underline-offset-4">
                  edit
                </Link>
              </p>
            </div>
            <SettingsControls />
          </div>
          <SummaryCards portfolio={portfolio} settings={settings} />
          <div className="grid gap-6 lg:grid-cols-2">
            <VaRCard portfolio={portfolio} settings={settings} />
            <CorrelationCard portfolio={portfolio} settings={settings} />
          </div>
          <PerformanceCard portfolio={portfolio} settings={settings} />
          <AttributionCard portfolio={portfolio} settings={settings} />
          <div className="grid gap-6 lg:grid-cols-2">
            <FactorCard portfolio={portfolio} />
            <BacktestCard portfolio={portfolio} />
          </div>
        </div>
      )}
    </PortfolioGate>
  );
}
