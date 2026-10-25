"use client";

/**
 * Assistant panel. Sends the conversation and the portfolio on screen; shows every tool the
 * assistant called (with its output) before the answer, so each figure can be traced back.
 */
import { AlertTriangle, Bot, ChevronDown, Loader2, MessageSquare, Send, Wrench, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, type Schemas } from "@/lib/api/client";
import { type ChatEvent, streamChat } from "@/lib/api/chat";
import { useAnalysisInput } from "@/lib/hooks";
import { cn } from "@/lib/utils";

type ToolStep = { id: string; name: string; args: Record<string, unknown>; ok?: boolean; output?: Record<string, unknown> };
type Turn = {
  question: string;
  tools: ToolStep[];
  answer?: Schemas["AnswerEvent"];
  error?: string;
  pending: boolean;
};

const SUGGESTIONS = [
  "What is my 1-day 99% VaR in rupees?",
  "Which holdings drive most of the risk?",
  "How would my portfolio have done in the COVID crash?",
  "What if IT stocks fell 15% and the market fell 8%?",
];

const TOOL_LABEL: Record<string, string> = {
  get_risk_summary: "Risk summary",
  get_risk_contributions: "Risk contributions",
  get_factor_exposures: "Factor exposures",
  backtest_var: "VaR backtest",
  list_stress_scenarios: "List scenarios",
  run_stress_scenario: "Stress scenario",
  run_custom_stress: "Custom stress",
  optimize_portfolio: "Optimizer",
  simulate_goal: "Goal simulation",
  search_knowledge_base: "Document search",
};

function ToolChip({ step }: { step: ToolStep }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-md border bg-muted/40 text-xs">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 px-2 py-1 text-left"
        aria-expanded={open}
      >
        {step.ok === undefined ? (
          <Loader2 className="size-3 animate-spin" />
        ) : step.ok ? (
          <Wrench className="size-3 text-blue-600" />
        ) : (
          <AlertTriangle className="size-3 text-amber-600" />
        )}
        <span className="font-medium">{TOOL_LABEL[step.name] ?? step.name}</span>
        <span className="truncate text-muted-foreground">{JSON.stringify(step.args)}</span>
        <ChevronDown className={cn("ml-auto size-3 shrink-0", open && "rotate-180")} />
      </button>
      {open && step.output && (
        <pre className="max-h-48 overflow-auto border-t px-2 py-1 whitespace-pre-wrap">
          {JSON.stringify(step.output, null, 1)}
        </pre>
      )}
    </div>
  );
}

function apply(turn: Turn, e: ChatEvent): Turn {
  switch (e.type) {
    case "tool_call":
      return { ...turn, tools: [...turn.tools, { id: e.id, name: e.name, args: e.arguments }] };
    case "tool_result":
      return {
        ...turn,
        tools: turn.tools.map((t) => (t.id === e.id ? { ...t, ok: e.ok, output: e.output } : t)),
      };
    case "answer":
      return { ...turn, answer: e };
    case "error":
      return { ...turn, error: e.message };
    case "done":
      return { ...turn, pending: false };
  }
}

export function ChatPanel() {
  const { portfolio } = useAnalysisInput();
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const busy = turns.some((t) => t.pending);

  // Block bodies: an effect must return nothing or a cleanup function. (Newer browsers return
  // a Promise from scrollIntoView, which React would otherwise call as the cleanup.)
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [turns]);
  useEffect(() => {
    return () => {
      abort.current?.abort();
    };
  }, []);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    const history: Schemas["ChatMessage"][] = [];
    for (const t of turns.slice(-4)) {
      if (t.answer) history.push({ role: "user", content: t.question }, { role: "assistant", content: t.answer.content });
    }
    history.push({ role: "user", content: q });
    const index = turns.length;
    setTurns((ts) => [...ts, { question: q, tools: [], pending: true }]);
    const update = (fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t, i) => (i === index ? fn(t) : t)));
    abort.current = new AbortController();
    try {
      await streamChat({ messages: history, portfolio }, (e) => update((t) => apply(t, e)), abort.current.signal);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      update((t) => ({ ...t, error: msg }));
    } finally {
      update((t) => ({ ...t, pending: false }));
    }
  }

  if (!open) {
    return (
      <Button
        onClick={() => setOpen(true)}
        className="fixed right-4 bottom-4 z-40 h-11 rounded-full px-4 shadow-lg"
        aria-label="Open the assistant"
      >
        <MessageSquare /> Ask the assistant
      </Button>
    );
  }

  return (
    <aside
      aria-label="Assistant"
      className="fixed right-0 bottom-0 z-40 flex h-[min(42rem,100dvh)] w-full flex-col border bg-background shadow-2xl sm:right-4 sm:bottom-4 sm:w-[28rem] sm:rounded-xl"
    >
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <Bot className="size-4 text-blue-600" />
        <p className="text-sm font-semibold">Assistant</p>
        <p className="truncate text-xs text-muted-foreground">
          {portfolio ? `${portfolio.holdings.length} holdings loaded` : "no portfolio loaded"}
        </p>
        <Button variant="ghost" size="icon-sm" className="ml-auto" onClick={() => setOpen(false)} aria-label="Close">
          <X />
        </Button>
      </header>
      <div className="flex-1 space-y-4 overflow-y-auto p-3 text-sm" aria-live="polite">
        {turns.length === 0 && (
          <div className="space-y-2">
            <p className="text-muted-foreground">
              Ask about the portfolio on screen. Every figure comes from the risk engine; the
              tools used are listed above each answer.
            </p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => void ask(s)}
                className="block w-full rounded-md border px-2 py-1.5 text-left text-xs hover:bg-muted"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className="space-y-2">
            <p className="ml-8 rounded-lg bg-blue-600 px-3 py-2 text-white">{t.question}</p>
            {t.tools.map((s) => (
              <ToolChip key={s.id} step={s} />
            ))}
            {t.pending && !t.answer && (
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="size-3 animate-spin" /> Working…
              </p>
            )}
            {t.answer && (
              <div className="space-y-2 rounded-lg border px-3 py-2">
                <div className="prose-chat">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.answer.content}</ReactMarkdown>
                </div>
                {t.answer.citations && t.answer.citations.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {t.answer.citations.map((c) => (
                      <a
                        key={`${c.document}-${c.page}`}
                        href={c.url ?? undefined}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-full border px-2 py-0.5 text-xs hover:bg-muted"
                        title={c.title}
                      >
                        {c.title.length > 32 ? `${c.title.slice(0, 31)}…` : c.title}, p. {c.page}
                      </a>
                    ))}
                  </div>
                )}
                {!t.answer.grounded && (
                  <p className="text-xs text-amber-700">
                    Not verified: {t.answer.ungrounded_numbers?.join(", ")} could not be matched to any
                    tool output or document passage. Treat these figures with caution.
                  </p>
                )}
                {(t.answer.flags ?? []).map((f) => (
                  <p key={f} className="text-xs text-amber-700">
                    {f.startsWith("advice: ") ? "A sentence giving a recommendation was removed." : f}
                  </p>
                ))}
                <p className="text-xs text-muted-foreground">{t.answer.disclaimer}</p>
              </div>
            )}
            {t.error && (
              <p role="alert" className="flex gap-2 rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
                <AlertTriangle className="size-4 shrink-0" /> {t.error}
              </p>
            )}
          </div>
        ))}
        <div ref={bottom} />
      </div>
      <form
        className="flex gap-2 border-t p-2"
        onSubmit={(e) => {
          e.preventDefault();
          void ask(input);
        }}
      >
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void ask(input);
            }
          }}
          placeholder="Ask about this portfolio…"
          maxLength={2000}
          rows={2}
          className="min-h-0 resize-none"
          aria-label="Your question"
        />
        <Button type="submit" size="icon" disabled={busy || !input.trim()} aria-label="Send">
          {busy ? <Loader2 className="animate-spin" /> : <Send />}
        </Button>
      </form>
    </aside>
  );
}
