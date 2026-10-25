import { BarChart3, FlaskConical, Layers, ShieldCheck } from "lucide-react";
import Link from "next/link";

import { TryDemoButton } from "@/components/try-demo";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const FEATURES = [
  {
    icon: ShieldCheck,
    title: "Value-at-Risk and drawdowns",
    text: "Historical, parametric and Monte Carlo VaR/CVaR in percent and rupees, with the confidence level, horizon and data window always stated.",
  },
  {
    icon: Layers,
    title: "Where the risk comes from",
    text: "Each holding's share of portfolio volatility and VaR (Euler decomposition), correlations, and Carhart factor exposures on IIM Ahmedabad's Indian factors.",
  },
  {
    icon: FlaskConical,
    title: "Stress tests",
    text: "Replays of real episodes such as the 2020 COVID crash, and hypothetical factor and sector shocks, with each scenario's rationale and caveats.",
  },
  {
    icon: BarChart3,
    title: "Checked against history",
    text: "Rolling out-of-sample VaR backtests with Kupiec and Christoffersen tests, so you can see how often the model's losses were exceeded.",
  },
];

export default function Home() {
  return (
    <div className="space-y-12">
      <section className="space-y-5 pt-6">
        <p className="text-sm font-medium text-blue-600">Portfolio risk analytics for India</p>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Understand the risk in a portfolio of Indian mutual funds and stocks.
        </h1>
        <p className="max-w-2xl text-lg text-muted-foreground">
          Enter holdings from 15 popular mutual funds and 20 Nifty 50 stocks and see how the
          portfolio has behaved: how much it could lose on a bad day, how deep its drawdowns
          went, which holdings drive its risk, and how it would have fared in past crises.
        </p>
        <div className="flex flex-wrap items-start gap-3">
          <TryDemoButton />
          <Link href="/portfolio" className={buttonVariants({ variant: "outline", size: "lg", className: "h-10 px-4" })}>
            Enter your own holdings
          </Link>
        </div>
        <p className="text-sm text-muted-foreground">
          The demo loads an illustrative sample portfolio; no account is needed.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        {FEATURES.map(({ icon: Icon, title, text }) => (
          <Card key={title}>
            <CardHeader>
              <Icon className="size-5 text-blue-600" aria-hidden />
              <CardTitle>{title}</CardTitle>
              <CardDescription>{text}</CardDescription>
            </CardHeader>
          </Card>
        ))}
      </section>

      <Card>
        <CardHeader>
          <CardTitle>How the numbers are made</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            Every figure comes from a deterministic risk engine working on cached daily data
            (2015 onwards). Methods, formulas and known data limitations, such as survivorship
            bias in today’s fund and index lists and the price-only Nifty 50 benchmark, are
            documented alongside the code.
          </p>
          <p>
            The analytics describe past behaviour under stated assumptions. They are not
            forecasts and not advice to buy or sell anything.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
