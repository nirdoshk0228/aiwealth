"use client";

/** Loading, error and empty states shared by every data-driven view. */
import { AlertTriangle, RotateCw } from "lucide-react";
import type { ReactNode } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { QueryState } from "@/lib/api/use-query";

export function LoadingBlock({ height = 160, label }: { height?: number; label: string }) {
  return (
    <div role="status" aria-live="polite" aria-label={label} className="space-y-2">
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="w-full" style={{ height }} />
      <span className="sr-only">{label}</span>
    </div>
  );
}

export function ErrorBlock({
  title = "Could not load this view",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <Alert variant="destructive">
      <AlertTriangle />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        <p>{message}</p>
        {onRetry && (
          <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
            <RotateCw /> Try again
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
}

/** Render `children(data)` on success, a skeleton while loading, an error with retry. */
export function DataState<T>({
  query,
  loadingLabel,
  height,
  children,
}: {
  query: QueryState<T> & { retry: () => void };
  loadingLabel: string;
  height?: number;
  children: (data: T) => ReactNode;
}) {
  if (query.status === "success") return <>{children(query.data)}</>;
  if (query.status === "error") {
    return <ErrorBlock message={query.error.message} onRetry={query.retry} />;
  }
  if (query.status === "idle") return null;
  return <LoadingBlock label={loadingLabel} height={height} />;
}
