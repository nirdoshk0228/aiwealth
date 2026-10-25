"use client";

/**
 * Cold-start banner. The free API host sleeps when idle and the API then loads market data
 * from the database; this polls GET /status (which never blocks) and explains the wait
 * instead of letting every card spin silently.
 */
import { Loader2, ServerCrash } from "lucide-react";
import { useEffect, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { Schemas } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";

type View =
  | { kind: "checking" }
  | { kind: "ready" }
  | { kind: "loading" }
  | { kind: "unreachable" }
  | { kind: "error"; message: string };

const POLL_MS = 3000;

function view(s: Schemas["MarketDataStatus"]): View {
  if (s.state === "ready") return { kind: "ready" };
  if (s.state === "error") return { kind: "error", message: s.message ?? "unknown error" };
  return { kind: "loading" };
}

export function MarketStatus() {
  const [state, setState] = useState<View>({ kind: "checking" });

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      let next: View;
      try {
        next = view(await api.status());
      } catch {
        next = { kind: "unreachable" };
      }
      if (!active) return;
      setState(next);
      if (next.kind !== "ready" && next.kind !== "error") timer = setTimeout(poll, POLL_MS);
    };
    void poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  if (state.kind === "ready" || state.kind === "checking") return null;
  if (state.kind === "error") {
    return (
      <Alert variant="destructive" className="mb-4">
        <ServerCrash />
        <AlertTitle>Market data is unavailable</AlertTitle>
        <AlertDescription>{state.message}</AlertDescription>
      </Alert>
    );
  }
  return (
    <Alert className="mb-4">
      <Loader2 className="animate-spin" />
      <AlertTitle>
        {state.kind === "unreachable" ? "Waking up the API" : "Loading market data"}
      </AlertTitle>
      <AlertDescription>
        {state.kind === "unreachable"
          ? "The API is not answering yet. On the free hosting tier it sleeps when idle, and the first request can take up to a minute. This page retries automatically."
          : "The API is loading cached prices and factor data. Results appear as soon as it is ready."}
      </AlertDescription>
    </Alert>
  );
}
