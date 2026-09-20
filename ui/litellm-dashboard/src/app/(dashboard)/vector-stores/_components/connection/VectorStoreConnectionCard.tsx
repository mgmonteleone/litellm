"use client";

import React from "react";
import { PlugZap } from "lucide-react";

import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

import ConnectionChecklist from "./ConnectionChecklist";
import { connectionChips } from "./connectionStatus";
import { describeLitellmParams } from "./litellmParamsDisplay";
import type { ConnectionTestState } from "./useVectorStoreConnectionTest";

interface VectorStoreConnectionCardProps {
  provider: string;
  litellmParams: Record<string, unknown> | string | null | undefined;
  connectionTest: ConnectionTestState;
  onRunConnectionTest: () => void;
}

/**
 * What this store is actually pointed at. The proxy redacts the secrets before they leave it,
 * so the card shows the saved connection without ever holding a credential.
 */
export const VectorStoreConnectionCard: React.FC<VectorStoreConnectionCardProps> = ({
  provider,
  litellmParams,
  connectionTest,
  onRunConnectionTest,
}) => {
  const params = describeLitellmParams(provider, litellmParams);

  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-base font-medium">Connection</h3>
            {connectionChips(connectionTest.result).map((chip) => (
              <StatusBadge key={chip.label} tone={chip.tone} label={chip.label} tooltip={chip.tooltip} />
            ))}
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={connectionTest.isRunning}
            onClick={onRunConnectionTest}
          >
            {connectionTest.isRunning ? <UiLoadingSpinner className="size-4" /> : <PlugZap className="size-4" />}
            Test connection
          </Button>
        </div>

        {params.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            This store has no saved connection settings. It is configured through the proxy config or resolves from the
            provider&apos;s own environment variables.
          </p>
        ) : (
          <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[minmax(0,14rem)_1fr]">
            {params.map((param) => (
              <React.Fragment key={param.name}>
                <dt className="text-sm text-muted-foreground" title={param.name}>
                  {param.label}
                </dt>
                <dd className="text-sm break-words">
                  {param.secret ? (
                    <span className="text-muted-foreground italic">{param.value}</span>
                  ) : (
                    <span className="font-mono text-xs">{param.value}</span>
                  )}
                </dd>
              </React.Fragment>
            ))}
          </dl>
        )}

        {connectionTest.result && (
          <ConnectionChecklist result={connectionTest.result} curlCommand={connectionTest.curlCommand ?? undefined} />
        )}
      </CardContent>
    </Card>
  );
};

export default VectorStoreConnectionCard;
