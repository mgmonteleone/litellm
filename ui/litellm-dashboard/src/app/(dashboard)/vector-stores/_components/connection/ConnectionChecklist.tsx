"use client";

import React, { useState } from "react";
import { CircleAlert, CircleCheck, CircleX, MinusCircle } from "lucide-react";

import CopyButton from "@/components/shared/CopyButton";
import type { VectorStoreConnectionCheck, VectorStoreTestConnectionResponse } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

type CheckStatus = NonNullable<VectorStoreConnectionCheck["status"]>;

const STATUS_STYLE: Record<CheckStatus, { icon: React.ElementType; className: string; label: string }> = {
  pass: { icon: CircleCheck, className: "text-success", label: "Passed" },
  warn: { icon: CircleAlert, className: "text-warning", label: "Warning" },
  fail: { icon: CircleX, className: "text-destructive", label: "Failed" },
  skip: { icon: MinusCircle, className: "text-muted-foreground", label: "Skipped" },
};

/** Turns "mongodb_sample_document" into "Mongodb sample document" so the row reads like a sentence. */
export const humanizeCheckName = (name: string | undefined): string => {
  const words = (name ?? "").replaceAll("_", " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : "Check";
};

const CheckRow: React.FC<{ check: VectorStoreConnectionCheck }> = ({ check }) => {
  const [showDetails, setShowDetails] = useState(false);
  const status = STATUS_STYLE[check.status ?? "skip"] ?? STATUS_STYLE.skip;
  const Icon = status.icon;
  const details = check.details;
  const hasDetails = details !== null && details !== undefined && Object.keys(details).length > 0;

  return (
    <li className="flex gap-2.5 py-2">
      <Icon className={cn("mt-0.5 size-4 shrink-0", status.className)} aria-label={status.label} />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-foreground">{humanizeCheckName(check.check)}</p>
        <p className="text-sm break-words text-muted-foreground">{check.message}</p>
        {hasDetails && (
          <>
            <Button
              type="button"
              variant="link"
              size="sm"
              className="h-auto p-0 text-xs"
              aria-expanded={showDetails}
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? "Hide details" : "Show details"}
            </Button>
            {showDetails && (
              <pre className="mt-1 max-h-48 overflow-auto rounded-sm border bg-muted/50 p-2 font-mono text-xs">
                {JSON.stringify(details, null, 2)}
              </pre>
            )}
          </>
        )}
      </div>
    </li>
  );
};

interface ConnectionChecklistProps {
  result: VectorStoreTestConnectionResponse;
  /** Shown behind a "Copy as curl" button so an admin can rerun the same check outside the dashboard. */
  curlCommand?: string;
}

/**
 * A passing result with skipped rows means the sidecar itself checked out fine but the admin has not
 * picked a database and collection yet, so "Connection verified" (implying every row ran) would overstate it.
 */
const headline = (result: VectorStoreTestConnectionResponse): string => {
  if (!result.ok) return "Connection failed";
  const hasSkippedChecks = (result.checks ?? []).some((check) => check.status === "skip");
  return hasSkippedChecks ? "Sidecar connected" : "Connection verified";
};

export const ConnectionChecklist: React.FC<ConnectionChecklistProps> = ({ result, curlCommand }) => {
  const checks = result.checks ?? [];
  const summaryStatus = result.ok ? STATUS_STYLE.pass : STATUS_STYLE.fail;
  const SummaryIcon = summaryStatus.icon;

  return (
    <div className="rounded-lg border bg-card" data-testid="connection-checklist">
      <div className="flex items-start gap-2.5 border-b p-3">
        <SummaryIcon className={cn("mt-0.5 size-5 shrink-0", summaryStatus.className)} aria-hidden />
        <div className="min-w-0 flex-1">
          <p className={cn("text-sm font-medium", summaryStatus.className)}>{headline(result)}</p>
          <p className="text-sm break-words text-muted-foreground">{result.summary}</p>
        </div>
        {curlCommand && <CopyButton value={curlCommand} label="Copy as curl" />}
      </div>
      {checks.length > 0 && (
        <ul className="divide-y px-3">
          {checks.map((check, index) => (
            <CheckRow key={`${check.check}-${index}`} check={check} />
          ))}
        </ul>
      )}
    </div>
  );
};

export default ConnectionChecklist;
