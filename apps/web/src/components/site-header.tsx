"use client";

import { LineChart } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

export const NAV = [
  { href: "/portfolio", label: "Portfolio" },
  { href: "/dashboard", label: "Risk dashboard" },
  { href: "/stress", label: "Stress lab" },
  { href: "/optimizer", label: "Optimizer" },
  { href: "/goals", label: "Goal planner" },
] as const;

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="border-b bg-background/95 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <LineChart className="size-5 text-blue-600" aria-hidden />
          AIWealth
        </Link>
        <nav aria-label="Main" className="flex items-center gap-1 overflow-x-auto text-sm">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-md px-3 py-1.5 whitespace-nowrap text-muted-foreground hover:bg-muted hover:text-foreground",
                pathname.startsWith(item.href) && "bg-muted font-medium text-foreground",
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-auto border-t">
      <div className="mx-auto max-w-6xl px-4 py-6 text-xs text-muted-foreground">
        <p>
          For education and information only; not investment advice. Figures are statistical
          estimates from historical data, with the assumptions stated next to them, and do not
          predict future returns or losses.
        </p>
        <p className="mt-1">
          Data: AMFI NAVs via mfapi.in, NSE prices via Yahoo Finance, Indian Fama-French-Carhart
          factors from IIM Ahmedabad (Agarwalla, Jacob and Varma, 2013). Independent student
          project, not affiliated with any fund house or with BlackRock.
        </p>
      </div>
    </footer>
  );
}
