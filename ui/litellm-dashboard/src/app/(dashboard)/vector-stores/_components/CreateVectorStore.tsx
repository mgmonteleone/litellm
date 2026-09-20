import React, { useEffect, useId, useState } from "react";
import { toast } from "@/lib/toast";
import { ChevronDown, CircleCheck, CircleHelp, Inbox, X } from "lucide-react";
import { v4 as uuidv4 } from "uuid";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { ragIngestCall } from "@/components/networking";
import { DocumentUpload, RAGIngestResponse } from "@/components/vector_store_management/types";
import DocumentsTable from "./DocumentsTable";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  VectorStoreFieldConfig,
} from "@/components/vector_store_providers";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { Logo } from "@/components/molecules/logo/Logo";
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import S3VectorsConfig from "./S3VectorsConfig";
import {
  buildChunkingStrategy,
  buildIngestProviderParams,
  chunkingError,
  describeIngestTotals,
  ingestTotals,
  type ChunkingInput,
} from "./ingestPayload";

const ACCEPTED_DOCUMENT_EXTENSIONS = ".pdf,.txt,.docx,.md,.doc";

const ACCEPTED_DOCUMENT_TYPES = [
  "application/pdf",
  "text/plain",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/msword",
  "text/markdown",
];

const MAX_DOCUMENT_BYTES = 50 * 1024 * 1024;

const RAG_INGEST_UNSUPPORTED_PROVIDERS = new Set(["valkey"]);

const providerItems = Object.entries(VectorStoreProviders)
  .filter(([providerEnum]) => !RAG_INGEST_UNSUPPORTED_PROVIDERS.has(vectorStoreProviderMap[providerEnum]))
  .map(([providerEnum, providerDisplayName]) => ({
    value: vectorStoreProviderMap[providerEnum],
    label: providerDisplayName,
  }));

const asText = (value: unknown): string => (typeof value === "string" ? value : "");

const IngestSuccessAlert: React.FC<{ ingestResults: RAGIngestResponse[]; onTestIt?: () => void }> = ({
  ingestResults,
  onTestIt,
}) => {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return null;
  }

  const vectorStoreId = ingestResults[0]?.vector_store_id;

  return (
    <Alert variant="success">
      <CircleCheck />
      <AlertTitle>Vector Store Created Successfully</AlertTitle>
      <AlertDescription>
        <div>
          <p>
            <strong>Vector Store ID:</strong> {vectorStoreId}
          </p>
          <p>
            <strong>Ingested:</strong> {describeIngestTotals(ingestTotals(ingestResults))}
          </p>
          {onTestIt && (
            <Button variant="link" size="sm" className="h-auto p-0" onClick={onTestIt}>
              Test it
            </Button>
          )}
        </div>
      </AlertDescription>
      <AlertAction>
        <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={() => setDismissed(true)}>
          <X className="size-4" />
        </Button>
      </AlertAction>
    </Alert>
  );
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

interface SelectOption {
  value: string;
  label: string;
}

const INPUT_TYPE_BY_FIELD_TYPE: Record<string, string> = { password: "password", number: "number" };

interface IngestProviderFieldProps {
  field: VectorStoreFieldConfig;
  value: string;
  onChange: (value: string) => void;
  embeddingModelOptions: readonly SelectOption[];
}

/**
 * The ingest tab rendered every field as free text, so the embedding model had to be typed from
 * memory and a typo only surfaced as a provider error much later. Select fields now get the same
 * combobox the add dialog uses.
 */
const IngestProviderField: React.FC<IngestProviderFieldProps> = ({ field, value, onChange, embeddingModelOptions }) => {
  const fieldId = `vector-store-${field.name}`;

  if (field.type === "select") {
    const options = field.options ?? embeddingModelOptions;
    return (
      <Field>
        <FieldLabel htmlFor={fieldId}>{labelWithHint(field.label, field.tooltip)}</FieldLabel>
        <Combobox
          items={options as SelectOption[]}
          value={options.find((option) => option.value === value) ?? null}
          onValueChange={(option: SelectOption | null) => onChange(option?.value ?? "")}
          itemToStringLabel={(option: SelectOption) => option.label}
          isItemEqualToValue={(option: SelectOption, selected: SelectOption) => option.value === selected.value}
        >
          <ComboboxInput id={fieldId} placeholder={field.placeholder} className="w-full" />
          <ComboboxContent>
            <ComboboxEmpty>No matching options</ComboboxEmpty>
            <ComboboxList>
              {(option: SelectOption) => (
                <ComboboxItem key={option.value} value={option}>
                  {option.label}
                </ComboboxItem>
              )}
            </ComboboxList>
          </ComboboxContent>
        </Combobox>
      </Field>
    );
  }

  const inputType = INPUT_TYPE_BY_FIELD_TYPE[field.type ?? "text"] ?? "text";

  return (
    <Field>
      <FieldLabel htmlFor={fieldId}>{labelWithHint(field.label, field.tooltip)}</FieldLabel>
      <Input
        id={fieldId}
        type={inputType}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={field.placeholder}
      />
    </Field>
  );
};

interface CreateVectorStoreProps {
  accessToken: string | null;
  onSuccess?: (vectorStoreId: string) => void;
  /** Opens the test tab with the freshly created store selected. */
  onTestVectorStore?: (vectorStoreId: string) => void;
}

const EMPTY_CHUNKING: ChunkingInput = { chunkSize: "", chunkOverlap: "" };

const CreateVectorStore: React.FC<CreateVectorStoreProps> = ({ accessToken, onSuccess, onTestVectorStore }) => {
  const [documents, setDocuments] = useState<DocumentUpload[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("bedrock");
  const [vectorStoreName, setVectorStoreName] = useState<string>("");
  const [vectorStoreDescription, setVectorStoreDescription] = useState<string>("");
  const [ingestResults, setIngestResults] = useState<RAGIngestResponse[]>([]);
  const [providerParams, setProviderParams] = useState<Record<string, unknown>>({});
  const [chunking, setChunking] = useState<ChunkingInput>(EMPTY_CHUNKING);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const documentsInputId = useId();

  useEffect(() => {
    if (!accessToken) return;
    fetchAvailableModels(accessToken)
      .then((models) => models.length > 0 && setModelInfo(models))
      .catch((error) => console.error("Error fetching model info:", error));
  }, [accessToken]);

  const embeddingModelOptions = modelInfo
    .filter((option) => option.mode === "embedding" || option.mode === null)
    .map((option) => ({ value: option.model_group, label: option.model_group }));

  const isSupportedDocument = (file: File): boolean => {
    if (!ACCEPTED_DOCUMENT_TYPES.includes(file.type)) {
      toast.error(`${file.name} is not a supported file type. Please upload PDF, TXT, DOCX, or MD files.`);
      return false;
    }
    if (file.size >= MAX_DOCUMENT_BYTES) {
      toast.error(`${file.name} must be smaller than 50MB!`);
      return false;
    }
    return true;
  };

  const handleAddDocuments = (files: readonly File[]) => {
    const accepted: DocumentUpload[] = files.filter(isSupportedDocument).map((file) => ({
      uid: uuidv4(),
      name: file.name,
      status: "done",
      size: file.size,
      type: file.type,
      originFileObj: file,
    }));

    if (accepted.length > 0) {
      setDocuments((prev) => [...prev, ...accepted]);
    }
  };

  const handleRemoveDocument = (uid: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.uid !== uid));
  };

  const handleCreateVectorStore = async () => {
    if (documents.length === 0) {
      toast.warning("Please upload at least one document");
      return;
    }

    if (!selectedProvider) {
      toast.warning("Please select a provider");
      return;
    }

    // Validate provider-specific required fields
    const requiredFields = getProviderSpecificFields(selectedProvider).filter((field) => field.required);
    for (const field of requiredFields) {
      if (!providerParams[field.name]) {
        toast.warning(`Please provide ${field.label}`);
        return;
      }
    }

    // S3 Vectors specific validation
    if (selectedProvider === "s3_vectors") {
      const bucketName = asText(providerParams.vector_bucket_name);
      const indexName = asText(providerParams.index_name);
      if (bucketName && bucketName.length < 3) {
        toast.warning("Vector bucket name must be at least 3 characters");
        return;
      }
      if (indexName && indexName.length > 0 && indexName.length < 3) {
        toast.warning("Index name must be at least 3 characters if provided");
        return;
      }
    }

    const chunkingProblem = chunkingError(chunking);
    if (chunkingProblem) {
      toast.warning(chunkingProblem);
      return;
    }

    if (!accessToken) {
      toast.error("No access token available");
      return;
    }

    setIsCreating(true);
    const results: RAGIngestResponse[] = [];
    let vectorStoreId: string | undefined;

    try {
      // Ingest each document
      for (const doc of documents) {
        if (!doc.originFileObj) continue;

        // Update document status to uploading
        setDocuments((prev) => prev.map((d) => (d.uid === doc.uid ? { ...d, status: "uploading" as const } : d)));

        try {
          const result = await ragIngestCall(
            accessToken,
            doc.originFileObj,
            selectedProvider,
            vectorStoreId, // Use the same vector store ID for subsequent uploads
            vectorStoreName || undefined,
            vectorStoreDescription || undefined,
            buildIngestProviderParams(selectedProvider, providerParams),
            buildChunkingStrategy(chunking),
          );

          // Store the vector store ID from the first successful ingest
          if (!vectorStoreId && result.vector_store_id) {
            vectorStoreId = result.vector_store_id;
          }

          results.push(result);

          // Update document status to done
          setDocuments((prev) => prev.map((d) => (d.uid === doc.uid ? { ...d, status: "done" as const } : d)));
        } catch (error) {
          console.error(`Error ingesting ${doc.name}:`, error);
          // Update document status to error
          setDocuments((prev) => prev.map((d) => (d.uid === doc.uid ? { ...d, status: "error" as const } : d)));
          throw error; // Stop processing on first error
        }
      }

      setIngestResults(results);
      toast.success(
        `Successfully created vector store with ${results.length} document(s). Vector Store ID: ${vectorStoreId}`,
      );

      if (onSuccess && vectorStoreId) {
        onSuccess(vectorStoreId);
      }

      // Clear documents after successful creation
      setTimeout(() => {
        setDocuments([]);
        setIngestResults([]);
      }, 3000);
    } catch (error) {
      console.error("Error creating vector store:", error);
      toast.fromError(`Failed to create vector store: ${error}`);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium">Create Vector Store</h3>
          <p className="text-sm text-muted-foreground">
            Upload documents and select a provider to create a new vector store with embedded content.
          </p>
        </div>

        {/* Upload Area */}
        <Card>
          <CardContent>
            <div className="mb-4">
              <p className="font-medium">Step 1: Upload Documents</p>
              <p className="text-sm text-muted-foreground block mt-1">
                Upload one or more documents (PDF, TXT, DOCX, MD). Maximum file size: 50MB per file.
              </p>
            </div>
            <label
              htmlFor={documentsInputId}
              className="flex cursor-pointer flex-col items-center gap-2 rounded-md border border-dashed border-input bg-muted/30 px-6 py-10 text-center transition-colors hover:border-primary hover:bg-muted/50 focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleAddDocuments(Array.from(event.dataTransfer.files));
              }}
            >
              <Inbox className="size-12 text-primary" />
              <span className="text-base">Click or drag files to this area to upload</span>
              <span className="text-sm text-muted-foreground">
                Support for single or bulk upload. Supported formats: PDF, TXT, DOCX, MD
              </span>
              <input
                id={documentsInputId}
                type="file"
                multiple
                accept={ACCEPTED_DOCUMENT_EXTENSIONS}
                className="sr-only"
                onChange={(event) => {
                  handleAddDocuments(Array.from(event.target.files ?? []));
                  event.target.value = "";
                }}
              />
            </label>
          </CardContent>
        </Card>

        {/* Documents Table */}
        {documents.length > 0 && (
          <Card>
            <CardContent>
              <div className="mb-4">
                <p className="font-medium">Uploaded Documents ({documents.length})</p>
              </div>
              <DocumentsTable documents={documents} onRemove={handleRemoveDocument} />
            </CardContent>
          </Card>
        )}

        {/* Provider Selection and Vector Store Details */}
        <Card>
          <CardContent className="space-y-4">
            <div>
              <p className="font-medium">Step 2: Configure Vector Store</p>
              <p className="text-sm text-muted-foreground block mt-1">
                Choose the provider and optionally provide a name and description for your vector store.
              </p>
            </div>

            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="vector-store-name">
                  {labelWithHint("Vector Store Name", "Optional: Give your vector store a meaningful name")}
                </FieldLabel>
                <Input
                  id="vector-store-name"
                  value={vectorStoreName}
                  onChange={(e) => setVectorStoreName(e.target.value)}
                  placeholder="e.g., Product Documentation, Customer Support KB"
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="vector-store-description">
                  {labelWithHint("Description", "Optional: Describe what this vector store contains")}
                </FieldLabel>
                <Textarea
                  id="vector-store-description"
                  value={vectorStoreDescription}
                  onChange={(e) => setVectorStoreDescription(e.target.value)}
                  placeholder="e.g., Contains all product documentation and user guides"
                  rows={2}
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="vector-store-provider">
                  {labelWithHint("Provider", "Select the provider for embedding and vector store operations")}
                </FieldLabel>
                <Select
                  items={providerItems}
                  value={selectedProvider}
                  onValueChange={(value: string | null) => value !== null && setSelectedProvider(value)}
                >
                  <SelectTrigger id="vector-store-provider" className="w-full">
                    <SelectValue placeholder="Select a provider" />
                  </SelectTrigger>
                  <SelectContent>
                    {providerItems.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        <Logo src={vectorStoreProviderLogoMap[item.label]} label={item.label} className="w-5 h-5" />
                        <span>{item.label}</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>

              {/* S3 Vectors Configuration */}
              {selectedProvider === "s3_vectors" && (
                <S3VectorsConfig
                  accessToken={accessToken}
                  providerParams={providerParams}
                  onParamsChange={setProviderParams}
                />
              )}

              {/* Other Provider-specific fields */}
              {selectedProvider !== "s3_vectors" &&
                getProviderSpecificFields(selectedProvider).map((field: VectorStoreFieldConfig) => (
                  <IngestProviderField
                    key={field.name}
                    field={field}
                    value={asText(providerParams[field.name])}
                    onChange={(value) => setProviderParams((prev) => ({ ...prev, [field.name]: value }))}
                    embeddingModelOptions={embeddingModelOptions}
                  />
                ))}

              <Collapsible>
                <CollapsibleTrigger
                  render={
                    <Button type="button" variant="ghost" size="sm" className="gap-1.5 px-0">
                      <ChevronDown className="size-4" />
                      Chunking
                    </Button>
                  }
                />
                <CollapsibleContent className="flex flex-col gap-3 pt-2">
                  <Field>
                    <FieldLabel htmlFor="chunk-size">
                      {labelWithHint(
                        "Chunk size",
                        "Characters per chunk before a document is embedded. Leave blank to use the backend default",
                      )}
                    </FieldLabel>
                    <Input
                      id="chunk-size"
                      type="number"
                      inputMode="numeric"
                      placeholder="1000"
                      value={chunking.chunkSize}
                      onChange={(e) => setChunking((prev) => ({ ...prev, chunkSize: e.target.value }))}
                    />
                  </Field>
                  <Field>
                    <FieldLabel htmlFor="chunk-overlap">
                      {labelWithHint(
                        "Chunk overlap",
                        "Characters each chunk repeats from the one before, so a sentence split across a boundary is still findable",
                      )}
                    </FieldLabel>
                    <Input
                      id="chunk-overlap"
                      type="number"
                      inputMode="numeric"
                      placeholder="200"
                      value={chunking.chunkOverlap}
                      onChange={(e) => setChunking((prev) => ({ ...prev, chunkOverlap: e.target.value }))}
                    />
                    <FieldDescription>Must be smaller than the chunk size.</FieldDescription>
                  </Field>
                </CollapsibleContent>
              </Collapsible>
            </FieldGroup>

            <div className="flex justify-end">
              <Button
                size="lg"
                onClick={handleCreateVectorStore}
                disabled={isCreating || documents.length === 0 || !selectedProvider}
              >
                {isCreating && <UiLoadingSpinner className="size-4" />}
                {isCreating ? "Creating Vector Store..." : "Create Vector Store"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Success Message */}
        {ingestResults.length > 0 && (
          <IngestSuccessAlert
            ingestResults={ingestResults}
            onTestIt={
              onTestVectorStore && ingestResults[0]?.vector_store_id
                ? () => onTestVectorStore(ingestResults[0].vector_store_id)
                : undefined
            }
          />
        )}
      </div>
    </TooltipProvider>
  );
};

export default CreateVectorStore;
