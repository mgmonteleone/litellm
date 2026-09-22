"use client";

import React, { useState } from "react";
import { PlugZap } from "lucide-react";

import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

import ConnectionChecklist from "./ConnectionChecklist";
import { ConnectionEditForm } from "./ConnectionEditForm";
import { connectionChips } from "./connectionStatus";
import { describeLitellmParams, type LitellmParams } from "./litellmParamsDisplay";
import type { ConnectionTestState } from "./useVectorStoreConnectionTest";

interface VectorStoreConnectionCardProps {
  vectorStoreId: string;
  provider: string;
  litellmParams: LitellmParams;
  connectionTest: ConnectionTestState;
  onRunConnectionTest: () => void;
  accessToken: string | null;
  /**
   * Gates the edit form and the Test connection button the same way the rest of the info page gates
   * its own "Edit Vector Store" action: /vector_store/test_connection is admin-only, so a non-admin
   * must not see a control for it that would just 403.
   */
  canEdit: boolean;
  onConnectionUpdated: () => void;
}

/**
 * What this store is actually pointed at. The proxy redacts the secrets before they leave it,
 * so the card shows the saved connection without ever holding a credential.
 */
export const VectorStoreConnectionCard: React.FC<VectorStoreConnectionCardProps> = ({
  vectorStoreId,
  provider,
  litellmParams,
  connectionTest,
  onRunConnectionTest,
  accessToken,
  canEdit,
  onConnectionUpdated,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const params = describeLitellmParams(provider, litellmParams);

  if (isEditing) {
    return (
      <Card>
        <CardContent className="space-y-4">
          <h3 className="text-base font-medium">Edit connection</h3>
          <ConnectionEditForm
            vectorStoreId={vectorStoreId}
            provider={provider}
            litellmParams={litellmParams}
            accessToken={accessToken}
            onCancel={() => setIsEditing(false)}
            onSaved={() => {
              setIsEditing(false);
              onConnectionUpdated();
            }}
          />
        </CardContent>
      </Card>
    );
  }

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
          <div className="flex items-center gap-2">
            {canEdit && (
              <Button type="button" variant="outline" size="sm" onClick={() => setIsEditing(true)}>
                Edit connection
              </Button>
            )}
            {canEdit && (
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
            )}
          </div>
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
