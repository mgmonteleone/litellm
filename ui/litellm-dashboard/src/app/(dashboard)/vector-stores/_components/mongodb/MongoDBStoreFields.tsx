"use client";

import React, { useMemo, useState } from "react";
import { ChevronDown, PlugZap } from "lucide-react";
import type { Control, UseFormSetValue } from "react-hook-form";
import { useWatch } from "react-hook-form";

import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { FormField } from "@/components/shared/form/FormField";
import { getProviderSpecificFields, type VectorStoreFieldConfig } from "@/components/vector_store_providers";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";

import ConnectionChecklist from "../connection/ConnectionChecklist";
import type { ConnectionTestState } from "../connection/useVectorStoreConnectionTest";
import VectorStoreField, { type SelectOption } from "../fields/VectorStoreField";
import type { VectorStoreFormValues } from "../vectorStoreFormSchema";
import { dimensionVerdict, dimensionVerdictMessage, supportsFeature } from "./mongodbConnection";
import { hasDeploymentDefaults, useMongoDBProviderDefaults } from "./useMongoDBProviderDefaults";
import { useMongoDiscovery, type DiscoveryState } from "./useMongoDiscovery";

const MONGODB_FIELDS = getProviderSpecificFields("mongodb");

const fieldsInGroup = (group: VectorStoreFieldConfig["group"]): VectorStoreFieldConfig[] =>
  MONGODB_FIELDS.filter((field) => field.group === group);

const Section: React.FC<{ title: string; hint?: string; children: React.ReactNode }> = ({ title, hint, children }) => (
  <section className="flex flex-col gap-3 rounded-lg border p-4">
    <div>
      <h4 className="text-sm font-medium text-foreground">{title}</h4>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
    {children}
  </section>
);

export interface MongoDBStoreFieldsProps {
  control: Control<VectorStoreFormValues>;
  setValue: UseFormSetValue<VectorStoreFormValues>;
  accessToken: string | null;
  embeddingModelOptions: readonly SelectOption[];
  connectionTest: ConnectionTestState;
  onRunConnectionTest: () => void;
}

export const MongoDBStoreFields: React.FC<MongoDBStoreFieldsProps> = ({
  control,
  setValue,
  accessToken,
  embeddingModelOptions,
  connectionTest,
  onRunConnectionTest,
}) => {
  const [createNewIndex, setCreateNewIndex] = useState(false);
  const [apiBase, apiKey, database, collection] = useWatch({
    control,
    name: ["api_base", "api_key", "mongodb_database", "mongodb_collection"],
  });

  const providerDefaults = useMongoDBProviderDefaults(accessToken);
  const usingDeploymentDefaults = hasDeploymentDefaults(providerDefaults);
  // Only consulted while usingDeploymentDefaults is true (the fields render unconditionally
  // otherwise), so it only ever needs to start open when a saved override already has a value.
  const [overrideOpen, setOverrideOpen] = useState(() => Boolean(apiBase) || Boolean(apiKey));

  const handleOverrideOpenChange = (open: boolean) => {
    setOverrideOpen(open);
    if (!open) {
      setValue("api_base", "");
      setValue("api_key", "");
    }
  };

  const connectionReady = usingDeploymentDefaults || Boolean(apiBase && apiKey);
  /**
   * Each discovery kind is scoped to only the fields it actually depends on, so typing in one
   * field doesn't invalidate a sibling's cache and refire its request too: the database list
   * only needs the sidecar credentials, the collection list also needs the database, and the
   * field lists need the collection on top of that.
   *
   * A blank override is left out entirely rather than sent as an empty string, so the backend
   * resolves it from the deployment's own configuration exactly as a real search would.
   */
  const connectionOverrides = useMemo(
    () => ({ ...(apiBase ? { api_base: apiBase } : {}), ...(apiKey ? { api_key: apiKey } : {}) }),
    [apiBase, apiKey],
  );
  const collectionParams = useMemo(
    () => ({ ...connectionOverrides, mongodb_database: database }),
    [connectionOverrides, database],
  );
  const fieldParams = useMemo(
    () => ({ ...connectionOverrides, mongodb_database: database, mongodb_collection: collection }),
    [connectionOverrides, database, collection],
  );

  const databases = useMongoDiscovery(
    accessToken,
    "databases",
    { litellmParams: connectionOverrides },
    connectionReady,
  );
  const collections = useMongoDiscovery(
    accessToken,
    "collections",
    { litellmParams: collectionParams },
    connectionReady && !!database,
  );
  const fieldsEnabled = connectionReady && !!database && !!collection;
  const vectorFields = useMongoDiscovery(accessToken, "vector_fields", { litellmParams: fieldParams }, fieldsEnabled);
  const textFields = useMongoDiscovery(accessToken, "text_fields", { litellmParams: fieldParams }, fieldsEnabled);
  const filterFields = useMongoDiscovery(accessToken, "filter_fields", { litellmParams: fieldParams }, fieldsEnabled);

  const discoveryFor = (field: VectorStoreFieldConfig): DiscoveryState | undefined => {
    switch (field.discovery) {
      case "databases":
        return databases;
      case "collections":
        return collections;
      case "vector_fields":
        return vectorFields;
      case "text_fields":
        return textFields;
      case "filter_fields":
        return filterFields;
      default:
        return undefined;
    }
  };

  const renderField = (field: VectorStoreFieldConfig, description?: React.ReactNode) => (
    <VectorStoreField
      key={field.name}
      field={field}
      control={control}
      name={field.name as keyof VectorStoreFormValues & string}
      fallbackOptions={embeddingModelOptions}
      discovery={discoveryFor(field)}
      description={description}
    />
  );

  const hybridSupported = supportsFeature(connectionTest.result, "hybrid");
  const verdict = dimensionVerdict(connectionTest.result);
  const verdictMessage = dimensionVerdictMessage(verdict);

  return (
    <div className="flex flex-col gap-4">
      <Section title="Connection" hint="Where LiteLLM reaches the MongoDB sidecar.">
        {usingDeploymentDefaults ? (
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
              <CollapsibleContent className="flex flex-col gap-3 pt-2">
                {fieldsInGroup("connection").map((field) => renderField(field))}
              </CollapsibleContent>
            </Collapsible>
          </>
        ) : (
          fieldsInGroup("connection").map((field) => renderField(field))
        )}
        <div className="flex items-center gap-3">
          <Button
            type="button"
            variant="outline"
            disabled={!connectionReady || connectionTest.isRunning}
            onClick={onRunConnectionTest}
          >
            {connectionTest.isRunning ? <UiLoadingSpinner className="size-4" /> : <PlugZap className="size-4" />}
            Test connection
          </Button>
          {!connectionReady && (
            <span className="text-sm text-muted-foreground">Enter the sidecar URL and API key first.</span>
          )}
        </div>
        {connectionTest.result && (
          <ConnectionChecklist result={connectionTest.result} curlCommand={connectionTest.curlCommand ?? undefined} />
        )}
      </Section>

      <Section title="Data" hint="Which collection holds your documents, and which fields carry the text and vectors.">
        {fieldsInGroup("data").map((field) => renderField(field))}
      </Section>

      <Section title="Index" hint="The MongoDB Vector Search index LiteLLM queries.">
        <RadioGroup
          value={createNewIndex ? "create" : "existing"}
          onValueChange={(value: unknown) => setCreateNewIndex(value === "create")}
          className="grid-cols-2"
        >
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <RadioGroupItem value="existing" />
            Use an existing index
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <RadioGroupItem value="create" />
            Create a new index
          </label>
        </RadioGroup>

        <FormField
          control={control}
          name="vector_store_id"
          label={labelWithHint(
            "Index Name",
            createNewIndex
              ? "The Vector Search index LiteLLM creates on this collection. It is also how you refer to this store from the API"
              : "The name of the MongoDB Vector Search index that already exists on this collection",
          )}
          description={
            createNewIndex
              ? "LiteLLM creates the index on first ingest and reports it as Building until MongoDB finishes."
              : undefined
          }
        >
          {({ ref, value, ...controlProps }) => (
            <Input {...controlProps} ref={ref} value={value ?? ""} placeholder="policy_vector_index" />
          )}
        </FormField>

        {createNewIndex && fieldsInGroup("index").map((field) => renderField(field))}
        {!createNewIndex &&
          fieldsInGroup("index")
            .filter((field) => field.name === "mongodb_filter_fields")
            .map((field) =>
              renderField(
                field,
                "Must match the filter fields the existing index declares, otherwise MongoDB rejects the filter.",
              ),
            )}
      </Section>

      <Section
        title="Embedding model"
        hint="The model that produced the vectors in this collection. LiteLLM embeds every query with it."
      >
        {fieldsInGroup("embedding").map((field) =>
          renderField(
            field,
            verdictMessage ? (
              <span className={verdict.kind === "mismatch" ? "text-destructive" : "text-success"}>
                {verdictMessage}
              </span>
            ) : undefined,
          ),
        )}
      </Section>

      <Collapsible>
        <CollapsibleTrigger
          render={
            <Button type="button" variant="ghost" size="sm" className="group/advanced gap-1.5 px-0">
              <ChevronDown
                className={cn("size-4 transition-transform", "group-data-[panel-open]/advanced:rotate-180")}
              />
              Advanced
            </Button>
          }
        />
        <CollapsibleContent className="flex flex-col gap-3 pt-2">
          {fieldsInGroup("advanced")
            .filter((field) => !field.requiresCapability || hybridSupported)
            .map((field) => renderField(field))}
          {!hybridSupported && (
            <p className="text-sm text-muted-foreground">
              Hybrid search settings appear once Test connection reports a cluster that supports $rankFusion.
            </p>
          )}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

export default MongoDBStoreFields;
