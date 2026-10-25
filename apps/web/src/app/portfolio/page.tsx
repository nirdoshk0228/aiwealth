import type { Metadata } from "next";

import { MarketStatus } from "@/components/market-status";
import { AssetSearch } from "@/components/portfolio/asset-search";
import { HoldingsTable } from "@/components/portfolio/holdings-table";
import { AnalyseBar } from "@/components/portfolio/analyse-bar";
import { CsvUpload, SampleLoader } from "@/components/portfolio/portfolio-tools";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = { title: "Portfolio · AIWealth" };

export default function PortfolioPage() {
  return (
    <div className="space-y-6">
      <MarketStatus />
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Portfolio</h1>
        <p className="text-muted-foreground">
          Enter holdings as rupee amounts; weights are worked out from them. The portfolio stays
          in this browser.
        </p>
      </div>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Universe</CardTitle>
            <CardDescription>15 mutual funds (Direct, Growth), 20 Nifty 50 stocks and the Nifty 50 index.</CardDescription>
          </CardHeader>
          <CardContent>
            <AssetSearch />
          </CardContent>
        </Card>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Holdings</CardTitle>
              <CardDescription>
                Long-only: amounts are zero or positive. Each asset appears once.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap items-start gap-4">
                <SampleLoader />
              </div>
              <HoldingsTable />
              <CsvUpload />
            </CardContent>
          </Card>
          <AnalyseBar />
        </div>
      </div>
    </div>
  );
}
