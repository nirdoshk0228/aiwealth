import { describe, expect, it } from "vitest";

import { parseSse } from "./chat";

const block = (event: object) => `event: x\ndata: ${JSON.stringify({ event })}\n\n`;

describe("parseSse", () => {
  it("returns complete events and keeps the unfinished tail", () => {
    const a = { type: "tool_call", id: "1", name: "get_risk_summary", arguments: {} };
    const b = { type: "error", message: "boom" };
    const text = block(a) + block(b) + 'event: done\ndata: {"event": {"ty';
    const { events, rest } = parseSse(text);
    expect(events).toEqual([a, b]);
    expect(rest).toBe('event: done\ndata: {"event": {"ty');
  });

  it("handles CRLF line endings and chunk boundaries", () => {
    const a = { type: "error", message: "x" };
    const full = block(a).replace(/\n/g, "\r\n");
    const first = parseSse(full.slice(0, 10));
    expect(first.events).toEqual([]);
    expect(parseSse(first.rest + full.slice(10)).events).toEqual([a]);
  });
});
