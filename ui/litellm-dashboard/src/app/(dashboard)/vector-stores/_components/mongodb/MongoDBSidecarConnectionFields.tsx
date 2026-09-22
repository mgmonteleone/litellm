"use client";

import React, { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Control, FieldPath, FieldValues, UseFormSetValue } from "react-hook-form";
import { useWatch } from "react-hook-form";

import { getProviderSpecificFields } from "@/components/vector_store_providers";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/cva.config";

import VectorStoreField from "../fields/VectorStoreField";
import { hasDeploymentDefaults, type MongoDBProviderDefaultsState } from "./useMongoDBProviderDefaults";

export interface MongoDBSidecarConnectionFieldsProps<TValues extends FieldValues> {
  control: Control<TValues>;
  setValue: UseFormSetValue<TValues>;
  providerDefaults: MongoDBProviderDefaultsState;
}

/**
 * The sidecar connection block shared by every place a MongoDB store's connection is edited: the
 * add dialog, the create-and-ingest tab, and the saved store's connection edit form. Behind a
 * deployment default it collapses to a one-line summary with an "Override" toggle; without one it
 * shows the URL and API key fields plainly, as the only way to reach the sidecar at all.
 */
export const MongoDBSidecarConnectionFields = <TValues extends FieldValues>({
  control,
  setValue,
  providerDefaults,
}: MongoDBSidecarConnectionFieldsProps<TValues>) => {
  const [apiBase, apiKey] = useWatch({
    control,
    name: ["api_base" as FieldPath<TValues>, "api_key" as FieldPath<TValues>],
  });

  const usingDeploymentDefaults = hasDeploymentDefaults(providerDefaults);
  // Only consulted while usingDeploymentDefaults is true (the fields render unconditionally
  // otherwise), so it only ever needs to start open when a saved override already has a value.
  // Deriving that on mount would race the provider-defaults fetch: text typed into api_base/api_key
  // before the fetch resolves would then get hidden inside a collapsed override while still holding
  // a value that submit still sends. Deriving it instead on the transition into
  // usingDeploymentDefaults picks up whatever the fields hold by the time it actually matters.
  const [overrideOpen, setOverrideOpen] = useState(false);
  const hasInitializedOverrideOpen = useRef(false);

  useEffect(() => {
    if (!usingDeploymentDefaults || hasInitializedOverrideOpen.current) return;
    hasInitializedOverrideOpen.current = true;
    setOverrideOpen(Boolean(apiBase) || Boolean(apiKey));
  }, [usingDeploymentDefaults, apiBase, apiKey]);

  const handleOverrideOpenChange = (open: boolean) => {
    setOverrideOpen(open);
    if (!open) {
      setValue("api_base" as FieldPath<TValues>, "" as never);
      setValue("api_key" as FieldPath<TValues>, "" as never);
    }
  };

  const connectionFields = getProviderSpecificFields("mongodb").filter((field) => field.group === "connection");

  const renderConnectionFields = () =>
    connectionFields.map((field) => (
      <VectorStoreField key={field.name} field={field} control={control} name={field.name as FieldPath<TValues>} />
    ));

  if (!usingDeploymentDefaults) {
    return <>{renderConnectionFields()}</>;
  }

  return (
    <>
      <p className="text-sm text-muted-foreground">
        Using this deployment&apos;s MongoDB sidecar at <code>{providerDefaults.apiBase}</code>.
      </p>
      <Collapsible open={overrideOpen} onOpenChange={handleOverrideOpenChange}>
        <CollapsibleTrigger
          render={
            <Button type="button" variant="ghost" size="sm" className="group/override gap-1.5 px-0">
              <ChevronDown
                className={cn("size-4 transition-transform", "group-data-[panel-open]/override:rotate-180")}
              />
              Override sidecar connection
            </Button>
          }
        />
        <CollapsibleContent className="flex flex-col gap-3 pt-2">{renderConnectionFields()}</CollapsibleContent>
      </Collapsible>
    </>
  );
};

export default MongoDBSidecarConnectionFields;
