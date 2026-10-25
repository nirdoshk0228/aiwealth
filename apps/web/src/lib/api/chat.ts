/**
 * Assistant chat over server-sent events. EventSource only supports GET, so the stream is read
 * with fetch: each SSE block's `data:` line is a ChatEventEnvelope (generated type).
 */
import { API_URL, ApiError, errorMessage, type Schemas } from "./client";

export type ChatEvent = Schemas["ChatEventEnvelope"]["event"];

/** Split buffered SSE text into complete events and the unfinished remainder. */
export function parseSse(buffer: string): { events: ChatEvent[]; rest: string } {
  const normalised = buffer.replace(/\r\n/g, "\n");
  const blocks = normalised.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    const data = block
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).trimStart())
      .join("\n");
    if (!data) continue;
    const env = JSON.parse(data) as Schemas["ChatEventEnvelope"];
    events.push(env.event);
  }
  return { events, rest };
}

export async function streamChat(
  body: Schemas["ChatRequest"],
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/v1/ai/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, errorMessage(null, 0));
  }
  if (!res.ok || !res.body) {
    let detail: unknown = null;
    try {
      detail = await res.json();
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, errorMessage(detail, res.status));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  const tail = parseSse(buffer + "\n\n");
  tail.events.forEach(onEvent);
}
