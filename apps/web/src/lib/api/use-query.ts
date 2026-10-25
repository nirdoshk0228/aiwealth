"use client";

/**
 * A small data hook: loading / error / success state for one API call.
 *
 * - `key` identifies the request (path + body as JSON); `null` means "do not fetch yet".
 * - Requests are shared per key: two cards asking for the same data make one request, and a
 *   page revisited later shows the last result at once. A shared request always runs to the
 *   end (it is never aborted by one component unmounting); a component that is gone, or whose
 *   key changed, simply ignores the late result.
 * - Failed requests are not cached; `retry()` drops the cached entry and fetches again.
 */
import { useEffect, useState } from "react";

import { ApiError } from "./client";

export type QueryState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: ApiError }
  | { status: "success"; data: T };

const inflight = new Map<string, Promise<unknown>>();
const results = new Map<string, unknown>();

function toApiError(e: unknown): ApiError {
  return e instanceof ApiError ? e : new ApiError(0, e instanceof Error ? e.message : String(e));
}

function load<T>(key: string, fetcher: (signal: AbortSignal) => Promise<T>): Promise<T> {
  let promise = inflight.get(key) as Promise<T> | undefined;
  if (!promise) {
    promise = fetcher(new AbortController().signal).then(
      (data) => {
        results.set(key, data);
        inflight.delete(key);
        return data;
      },
      (e: unknown) => {
        inflight.delete(key);
        throw e;
      },
    );
    inflight.set(key, promise);
  }
  return promise;
}

function initial<T>(key: string | null): QueryState<T> {
  if (key === null) return { status: "idle" };
  if (results.has(key)) return { status: "success", data: results.get(key) as T };
  return { status: "loading" };
}

export function useQuery<T>(
  key: string | null,
  fetcher: (signal: AbortSignal) => Promise<T>,
): QueryState<T> & { retry: () => void } {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{ key: string | null; value: QueryState<T> }>(() => ({
    key,
    value: initial<T>(key),
  }));

  useEffect(() => {
    if (key === null || results.has(key)) return;
    let active = true;
    load(key, fetcher).then(
      (data) => {
        if (active) setState({ key, value: { status: "success", data } });
      },
      (e: unknown) => {
        if (active) setState({ key, value: { status: "error", error: toApiError(e) } });
      },
    );
    return () => {
      active = false;
    };
    // `fetcher` is recreated on every render; the key identifies the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, attempt]);

  // A key change shows the cached result or "loading" immediately, without an extra render.
  const value = state.key === key ? state.value : initial<T>(key);
  const shown = key !== null && results.has(key) && value.status !== "success" ? initial<T>(key) : value;

  return {
    ...shown,
    retry: () => {
      if (key !== null) results.delete(key);
      setState({ key, value: initial<T>(key) });
      setAttempt((a) => a + 1);
    },
  };
}

/** Build a cache key from a path and a JSON-serialisable body. */
export function queryKey(path: string, body?: unknown): string {
  return body === undefined ? path : `${path} ${JSON.stringify(body)}`;
}
