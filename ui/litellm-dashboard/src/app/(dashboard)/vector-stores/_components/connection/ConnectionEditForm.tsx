"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { PlugZap } from "lucide-react";

import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { vectorStoreUpdateCall } from "@/components/networking";
import { getProviderSpecificFields } from "@/components/vector_store_providers";
import { Button } from "@/components/ui/button";
import { FieldGroup } from "@/components/ui/field";
import { TooltipProvider } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

import VectorStoreField, { type SelectOption } from "../fields/VectorStoreField";
import MongoDBStoreFields from "../mongodb/MongoDBStoreFields";
import { isSupportedProviderField, vectorStoreSchema } from "../vectorStoreFormSchema";
import ConnectionChecklist from "./ConnectionChecklist";
import { buildConnectionUpdateLitellmParams, connectionFormValuesFromLitellmParams } from "./connectionEditPayload";
import type { LitellmParams } from "./litellmParamsDisplay";
import { useVectorStoreConnectionTest } from "./useVectorStoreConnectionTest";

export interface ConnectionEditFormProps {
  vectorStoreId: string;
  provider: string;
  litellmParams: LitellmParams;
  accessToken: string | null;
  onCancel: () => void;
  onSaved: () => void;
}

/**
 * Edits a saved store's connection (its litellm_params), reusing the same field components the add
 * dialog uses for every provider, including MongoDB's discovery-backed fields, so nothing needs a
 * second implementation here. A secret field is prefilled with the redaction sentinel and, left
 * untouched, is sent back as that exact sentinel rather than the placeholder text; only a field the
 * admin actually changes is sent.
 */
export const ConnectionEditForm: React.FC<ConnectionEditFormProps> = ({
  vectorStoreId,
  provider,
  litellmParams,
  accessToken,
  onCancel,
  onSaved,
}) => {
  const initialValues = useMemo(
    () => connectionFormValuesFromLitellmParams(provider, litellmParams),
    [provider, litellmParams],
  );
  const form = useZodForm(vectorStoreSchema, {
    defaultValues: { custom_llm_provider: provider, vector_store_id: vectorStoreId, ...initialValues },
  });
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const connectionTest = useVectorStoreConnectionTest(accessToken);

  useEffect(() => {
    if (!accessToken) return;
    fetchAvailableModels(accessToken)
      .then((uniqueModels) => {
        if (uniqueModels.length > 0) setModelInfo(uniqueModels);
      })
      .catch((error) => console.error("Error fetching model info:", error));
  }, [accessToken]);

  const embeddingModelOptions: SelectOption[] = modelInfo
    .filter((option) => option.mode === "embedding" || option.mode === null)
    .map((option) => ({ value: option.model_group, label: option.model_group }));

  const currentLitellmParams = useCallback(
    () =>
      buildConnectionUpdateLitellmParams(
        provider,
        initialValues,
        form.getValues() as unknown as Record<string, string | undefined>,
      ),
    [provider, initialValues, form],
  );

  const runConnectionTest = useCallback(() => {
    void connectionTest.run({ vector_store_id: vectorStoreId, litellm_params: currentLitellmParams() });
  }, [connectionTest, vectorStoreId, currentLitellmParams]);

  const handleSave = async () => {
    if (!accessToken) return;
    setIsSaving(true);
    try {
      await vectorStoreUpdateCall(accessToken, {
        vector_store_id: vectorStoreId,
        litellm_params: currentLitellmParams(),
      });
      toast.success("Connection updated successfully");
      onSaved();
    } catch (error) {
      console.error("Error updating vector store connection:", error);
      toast.fromError("Error updating vector store connection: " + error);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <TooltipProvider>
      <form onSubmit={form.handleSubmit(handleSave)} className="flex flex-col gap-4">
        <FieldGroup>
          {provider === "mongodb" ? (
            <MongoDBStoreFields
              control={form.control}
              accessToken={accessToken}
              embeddingModelOptions={embeddingModelOptions}
              connectionTest={connectionTest}
              onRunConnectionTest={runConnectionTest}
            />
          ) : (
            <>
              {getProviderSpecificFields(provider)
                .filter(isSupportedProviderField)
                .map((field) => (
                  <VectorStoreField
                    key={field.name}
                    field={field}
                    control={form.control}
                    name={field.name}
                    fallbackOptions={embeddingModelOptions}
                  />
                ))}
              <div className="flex items-center gap-3">
                <Button type="button" variant="outline" disabled={connectionTest.isRunning} onClick={runConnectionTest}>
                  {connectionTest.isRunning ? <UiLoadingSpinner className="size-4" /> : <PlugZap className="size-4" />}
                  Test connection
                </Button>
              </div>
              {connectionTest.result && (
                <ConnectionChecklist
                  result={connectionTest.result}
                  curlCommand={connectionTest.curlCommand ?? undefined}
                />
              )}
            </>
          )}
        </FieldGroup>

        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onCancel} disabled={isSaving}>
            Cancel
          </Button>
          <Button type="submit" disabled={isSaving}>
            {isSaving && <UiLoadingSpinner className="size-4" />}
            Save connection
          </Button>
        </div>
      </form>
    </TooltipProvider>
  );
};

export default ConnectionEditForm;
