"use client";

import Link from "next/link";
import type { ReactNode } from "react";

import { LoadingBlock } from "@/components/data-state";
import { TryDemoButton } from "@/components/try-demo";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useAnalysisInput } from "@/lib/hooks";
import type { PortfolioIn } from "@/lib/portfolio";
import type { Settings } from "@/lib/store";

/** Render children only once a valid portfolio is loaded; otherwise explain how to get one. */
export function PortfolioGate({
  children,
}: {
  children: (p: { portfolio: PortfolioIn; settings: Settings; label: string | null }) => ReactNode;
}) {
  const { hydrated, portfolio, settings, label } = useAnalysisInput();
  if (!hydrated) return <LoadingBlock label="Loading your portfolio" height={200} />;
  if (portfolio === null) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No portfolio to analyse yet</CardTitle>
          <CardDescription>Enter holdings on the portfolio page, or load the demo sample.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-start gap-3">
          <TryDemoButton />
          <Link
            href="/portfolio"
            className={buttonVariants({ variant: "outline", size: "lg", className: "h-10 px-4" })}
          >
            Enter holdings
          </Link>
        </CardContent>
      </Card>
    );
  }
  return <>{children({ portfolio, settings, label })}</>;
}
