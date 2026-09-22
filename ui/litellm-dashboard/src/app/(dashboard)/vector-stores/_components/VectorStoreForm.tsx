import React, { useCallback, useState, useEffect } from "react";
import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useWatch } from "react-hook-form";
import { CredentialItem, vectorStoreCreateCall } from "@/components/networking";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  getVectorStoreProviderLogoAndName,
  isBetaVectorStoreProvider,
} from "@/components/vector_store_providers";
import BetaBadge from "@/components/BetaBadge";
import { Logo } from "@/components/molecules/logo/Logo";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";

import { useVectorStoreConnectionTest } from "./connection/useVectorStoreConnectionTest";
import VectorStoreField, { type SelectOption } from "./fields/VectorStoreField";
import MongoDBSetupAlert from "./mongodb/MongoDBSetupAlert";
import { supportsFeature } from "./mongodb/mongodbConnection";
import MongoDBStoreFields from "./mongodb/MongoDBStoreFields";
import { hasDeploymentDefaults, useMongoDBProviderDefaults } from "./mongodb/useMongoDBProviderDefaults";
import {
  buildVectorStoreLitellmParams,
  clearUnsupportedCapabilityFields,
  isSupportedProviderField,
  makeVectorStoreSchema,
  type VectorStoreFormValues,
} from "./vectorStoreFormSchema";

export { buildVectorStoreLitellmParams };

interface VectorStoreFormProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  credentials: CredentialItem[];
}

const VECTOR_STORE_ID_PLACEHOLDERS: Record<string, string> = {
  vertex_rag_engine: '6917529027641081856 (corpus ID from Vertex AI / "RAG Engine" console)',
  "vertex_ai/search_api": 'my-datastore_1234567890 (data store ID from Vertex AI / "Agent Search" console)',
  valkey: "my-search-index (FT index name in Valkey)",
  mongodb: "my-vector-index (MongoDB Vector Search index name)",
};

const VERTEX_SEARCH_API_WITH_ENGINE_PLACEHOLDER = "Any identifier you'll use to reference this in LiteLLM";

const DEFAULT_VECTOR_STORE_ID_PLACEHOLDER = "Enter vector store ID from your provider";

const EMPTY_VALUES: VectorStoreFormValues = {
  custom_llm_provider: "bedrock",
  vector_store_id: "",
  vertex_location: "global",
  mongodb_embedding_field: "embedding",
  mongodb_text_field: "text",
  valkey_port: "6379",
  valkey_ssl: "false",
  valkey_text_field: "text",
  valkey_embedding_field: "embedding",
};

interface CredentialOption {
  label: string;
  value: string | null;
}

const providerLabel = (provider: string, displayName: string): React.ReactNode =>
  isBetaVectorStoreProvider(provider) ? <BetaBadge>{displayName}</BetaBadge> : <span>{displayName}</span>;

const VectorStoreForm: React.FC<VectorStoreFormProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  accessToken,
  credentials,
}) => {
  const [metadataJson, setMetadataJson] = useState("{}");
  const [selectedProvider, setSelectedProvider] = useState("bedrock");
  // Gated the same way MongoDBStoreFields mounting used to gate this fetch, so selecting a
  // non-MongoDB provider never fires it and MongoDB's own discovery timing is unaffected.
  const providerDefaults = useMongoDBProviderDefaults(selectedProvider === "mongodb" ? accessToken : null);
  const form = useZodForm(makeVectorStoreSchema(hasDeploymentDefaults(providerDefaults)), {
    defaultValues: EMPTY_VALUES,
  });
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const vertexEngineId = useWatch({ control: form.control, name: "vertex_engine_id" });
  const connectionTest = useVectorStoreConnectionTest(accessToken);

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        if (uniqueModels.length > 0) {
          setModelInfo(uniqueModels);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
  }, [accessToken]);

  const embeddingModelOptions: SelectOption[] = modelInfo
    .filter((option) => option.mode === "embedding" || option.mode === null)
    .map((option) => ({ value: option.model_group, label: option.model_group }));

  const credentialOptions: CredentialOption[] = [
    { value: null, label: "None" },
    ...credentials.map((credential) => ({
      value: credential.credential_name,
      label: credential.credential_name,
    })),
  ];

  const makeProviderChangeHandler = (onChange: (provider: string) => void) => (provider: string | null) => {
    if (provider === null) return;
    onChange(provider);
    setSelectedProvider(provider);
    connectionTest.reset();
  };

  const runConnectionTest = useCallback(() => {
    const values = form.getValues();
    void connectionTest.run({
      custom_llm_provider: values.custom_llm_provider,
      litellm_params: buildVectorStoreLitellmParams(values.custom_llm_provider, values),
    });
  }, [connectionTest, form]);

  const handleCreate = async (formValues: VectorStoreFormValues) => {
    if (!accessToken) return;
    try {
      let metadata = {};
      try {
        metadata = metadataJson.trim() ? JSON.parse(metadataJson) : {};
      } catch (e) {
        toast.fromError("Invalid JSON in metadata field");
        return;
      }

      const submittedValues = clearUnsupportedCapabilityFields(
        formValues.custom_llm_provider,
        formValues,
        (capability) => supportsFeature(connectionTest.result, capability),
      );
      await vectorStoreCreateCall(accessToken, {
        vector_store_id: formValues.vector_store_id,
        custom_llm_provider: formValues.custom_llm_provider,
        vector_store_name: formValues.vector_store_name,
        vector_store_description: formValues.vector_store_description,
        vector_store_metadata: metadata,
        litellm_credential_name: formValues.litellm_credential_name,
        litellm_params: buildVectorStoreLitellmParams(formValues.custom_llm_provider, submittedValues),
      });
      toast.success("Vector store created successfully");
      form.reset(EMPTY_VALUES);
      setMetadataJson("{}");
      connectionTest.reset();
      onSuccess();
    } catch (error) {
      console.error("Error creating vector store:", error);
      toast.fromError("Error creating vector store: " + error);
    }
  };

  const handleCancel = () => {
    form.reset(EMPTY_VALUES);
    setMetadataJson("{}");
    setSelectedProvider("bedrock");
    connectionTest.reset();
    onCancel();
  };

  const isMongoDB = selectedProvider === "mongodb";

  const vectorStoreIdPlaceholder =
    selectedProvider === "vertex_ai/search_api" && vertexEngineId
      ? VERTEX_SEARCH_API_WITH_ENGINE_PLACEHOLDER
      : VECTOR_STORE_ID_PLACEHOLDERS[selectedProvider] ?? DEFAULT_VECTOR_STORE_ID_PLACEHOLDER;

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>Add New Vector Store</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleCreate)}>
            <FieldGroup>
              <FormField
                control={form.control}
                name="custom_llm_provider"
                label={labelWithHint("Provider", "Select the provider for this vector store")}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Select value={value} onValueChange={makeProviderChangeHandler(onChange)}>
                    <SelectTrigger
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      className="w-full"
                    >
                      <SelectValue>
                        {(provider: string) => {
                          const { displayName, logo } = getVectorStoreProviderLogoAndName(provider);
                          return (
                            <>
                              <Logo src={logo} label={displayName} className="w-5 h-5" />
                              {providerLabel(provider, displayName)}
                            </>
                          );
                        }}
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(VectorStoreProviders).map(([providerEnum, providerDisplayName]) => (
                        <SelectItem key={providerEnum} value={vectorStoreProviderMap[providerEnum]}>
                          <Logo
                            src={vectorStoreProviderLogoMap[providerDisplayName]}
                            label={providerDisplayName}
                            className="w-5 h-5"
                          />
                          {providerLabel(vectorStoreProviderMap[providerEnum], providerDisplayName)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </FormField>

              {selectedProvider === "pg_vector" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>PG Vector Setup Required</AlertTitle>
                  <AlertDescription>
                    <p>LiteLLM provides a server to connect to PG Vector. To use this provider:</p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        Deploy the litellm-pgvector server from:{" "}
                        <a href="https://github.com/BerriAI/litellm-pgvector" target="_blank" rel="noopener noreferrer">
                          https://github.com/BerriAI/litellm-pgvector
                        </a>
                      </li>
                      <li>Configure your PostgreSQL database with pgvector extension</li>
                      <li>Start the server and note the API base URL and API key</li>
                      <li>Enter those details in the fields below</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {isMongoDB && <MongoDBSetupAlert />}

              {selectedProvider === "valkey" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>Valkey Setup Required</AlertTitle>
                  <AlertDescription>
                    <p>
                      LiteLLM searches documents you have already stored in Valkey. It does not create the index or
                      upload documents for you. Before creating this vector store, make sure:
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        Your Valkey server has vector search enabled (the valkey-search module, included in the
                        valkey-bundle image and in AWS ElastiCache / MemoryDB for Valkey)
                      </li>
                      <li>
                        You have already created a search index and loaded your documents and their embeddings into it.
                        Enter that index name as the Vector Store ID
                      </li>
                      <li>
                        You know which embedding model created those stored embeddings. That model must be added to this
                        proxy under Models so you can pick it below. Using a different model returns wrong results
                      </li>
                      <li>
                        You know the field names your documents use for their text and their embedding. If they are not
                        &quot;text&quot; and &quot;embedding&quot;, set them below
                      </li>
                    </ol>
                    <p style={{ marginTop: "8px" }}>
                      When a query comes in, LiteLLM converts it to an embedding with the model below and returns the
                      closest matching documents from your index.
                    </p>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "vertex_rag_engine" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>Vertex AI RAG Engine Setup</AlertTitle>
                  <AlertDescription>
                    <p>To use Vertex AI RAG Engine:</p>
                    <p style={{ marginTop: "4px", fontStyle: "italic" }}>
                      Note: Google Cloud has renamed this to &quot;RAG Engine&quot; in its console — the steps below
                      still apply.
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        Set up your Vertex AI RAG Engine corpus following the guide:{" "}
                        <a
                          href="https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Vertex AI RAG Engine Overview
                        </a>
                      </li>
                      <li>Create a corpus in your Google Cloud project</li>
                      <li>
                        Note the corpus ID from the Vertex AI console (now labeled &quot;RAG Engine&quot; in Google
                        Cloud)
                      </li>
                      <li>Enter the corpus ID in the Vector Store ID field below</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "vertex_ai/search_api" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>Vertex AI Search Setup</AlertTitle>
                  <AlertDescription>
                    <p>To use Vertex AI Search (Discovery Engine):</p>
                    <p style={{ marginTop: "4px", fontStyle: "italic" }}>
                      Note: Google Cloud has renamed this to &quot;Agent Search&quot; in its console — the steps below
                      still apply.
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        Enable the Discovery Engine API on your Google Cloud project and create a data store following
                        the guide:{" "}
                        <a
                          href="https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es"
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ textDecoration: "underline" }}
                        >
                          Create a Vertex AI Search data store
                        </a>
                      </li>
                      <li>Pick a supported location: global, us, or eu</li>
                      <li>
                        For most data store types (Cloud Storage, BigQuery, Media): copy the data store ID and enter it
                        in the Vector Store ID field below.
                      </li>
                      <li>
                        For website, healthcare, and connector-based sources (Drive, Gmail, Slack, Jira, etc.): create a
                        search app on top of the data store, then copy the <strong>Engine ID</strong> and enter it in
                        the Engine ID field. The Vector Store ID is still required as the LiteLLM-side name for this
                        record, but it isn&apos;t used in the GCP URL when Engine ID is set.
                      </li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {!isMongoDB && (
                <FormField
                  control={form.control}
                  name="vector_store_id"
                  label={labelWithHint("Vector Store ID", "Enter the vector store ID from your api provider")}
                >
                  {({ ref, ...field }) => <Input {...field} ref={ref} placeholder={vectorStoreIdPlaceholder} />}
                </FormField>
              )}

              {isMongoDB ? (
                <MongoDBStoreFields
                  control={form.control}
                  setValue={form.setValue}
                  accessToken={accessToken}
                  providerDefaults={providerDefaults}
                  embeddingModelOptions={embeddingModelOptions}
                  connectionTest={connectionTest}
                  onRunConnectionTest={runConnectionTest}
                />
              ) : (
                getProviderSpecificFields(selectedProvider)
                  .filter(isSupportedProviderField)
                  .map((field) => (
                    <VectorStoreField
                      key={field.name}
                      field={field}
                      control={form.control}
                      name={field.name}
                      fallbackOptions={embeddingModelOptions}
                    />
                  ))
              )}

              <FormField
                control={form.control}
                name="vector_store_name"
                label={labelWithHint(
                  "Vector Store Name",
                  "Custom name you want to give to the vector store, this name will be rendered on the LiteLLM UI",
                )}
              >
                {({ ref, value, ...field }) => <Input {...field} ref={ref} value={value ?? ""} />}
              </FormField>

              <FormField control={form.control} name="vector_store_description" label="Description">
                {({ ref, value, ...field }) => <Textarea {...field} ref={ref} value={value ?? ""} rows={4} />}
              </FormField>

              <FormField
                control={form.control}
                name="litellm_credential_name"
                label={labelWithHint(
                  "Existing Credentials",
                  "Optionally select API provider credentials for this vector store eg. Bedrock API KEY",
                )}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Combobox
                    items={credentialOptions}
                    value={credentialOptions.find((option) => option.value === value) ?? null}
                    onValueChange={(option: CredentialOption | null) => onChange(option ? option.value : undefined)}
                    itemToStringLabel={(option: CredentialOption) => option.label}
                    isItemEqualToValue={(option: CredentialOption, selected: CredentialOption) =>
                      option.value === selected.value
                    }
                  >
                    <ComboboxInput
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      placeholder="Select or search for existing credentials"
                      className="w-full"
                      showClear={value !== undefined}
                    />
                    <ComboboxContent>
                      <ComboboxEmpty>No matching credentials</ComboboxEmpty>
                      <ComboboxList>
                        {(option: CredentialOption) => (
                          <ComboboxItem key={option.label} value={option}>
                            {option.label}
                          </ComboboxItem>
                        )}
                      </ComboboxList>
                    </ComboboxContent>
                  </Combobox>
                )}
              </FormField>

              <div role="group" className="flex w-full flex-col gap-3">
                <span className="flex w-fit gap-2 text-sm leading-snug font-medium">
                  {labelWithHint("Metadata", "JSON metadata for the vector store (optional)")}
                </span>
                <Textarea
                  rows={4}
                  value={metadataJson}
                  onChange={(event) => setMetadataJson(event.target.value)}
                  placeholder='{"key": "value"}'
                />
              </div>
            </FieldGroup>

            <div className="mt-6 flex justify-end space-x-3">
              <Button type="button" variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button type="submit">Create</Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default VectorStoreForm;
