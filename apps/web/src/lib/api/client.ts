/**
 * Typed API client. Every type comes from `schema.ts`, generated from the FastAPI OpenAPI
 * spec (`python scripts/export_openapi.py && npm run gen:api`); nothing here is hand-typed.
 */
import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const client = createClient<paths>({ baseUrl: API_URL });

/** An API error with the HTTP status and a readable message. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type ValidationItem = { loc?: (string | number)[]; msg?: string };

/** FastAPI errors are `{detail: string}` or, for schema errors, `{detail: [{loc, msg}]}`. */
export function errorMessage(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return (detail as ValidationItem[])
        .map((d) => {
          const where = (d.loc ?? []).filter((p) => p !== "body").join(".");
          return where ? `${where}: ${d.msg ?? "invalid"}` : (d.msg ?? "invalid");
        })
        .join("; ");
    }
  }
  if (status === 0) return "The API could not be reached. Is it running?";
  return `Request failed (HTTP ${status})`;
}

/** Turn an openapi-fetch result into data, or throw an ApiError. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  let result: { data?: T; error?: unknown; response: Response };
  try {
    result = await call;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, errorMessage(null, 0));
  }
  const { data, error, response } = result;
  if (error !== undefined || data === undefined) {
    throw new ApiError(response.status, errorMessage(error, response.status));
  }
  return data;
}
