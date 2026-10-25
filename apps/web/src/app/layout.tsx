import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { ChatPanel } from "@/components/chat/chat-panel";
import { SiteFooter, SiteHeader } from "@/components/site-header";
import { TooltipProvider } from "@/components/ui/tooltip";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AIWealth: portfolio risk and goal planning",
  description:
    "Portfolio risk analytics for Indian mutual funds and Nifty stocks: VaR/CVaR, drawdowns, factor exposures, stress tests and backtests.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-background text-foreground">
        <TooltipProvider>
          <SiteHeader />
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">{children}</main>
          <SiteFooter />
          <ChatPanel />
        </TooltipProvider>
      </body>
    </html>
  );
}
