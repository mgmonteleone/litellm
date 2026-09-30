import { z } from "zod/v4";

import { getProviderSpecificFields, type VectorStoreFieldConfig } from "@/components/vector_store_providers";

const EMBEDDING_MODEL_RENAME_PROVIDERS = new Set(["milvus", "valkey", "mongodb"]);

/**
 * Every provider field name the form knows how to carry. A field missing from this list is silently
 * dropped on save, so adding a field to a provider means adding it here and to `vectorStoreShape`.
 */
export const PROVIDER_FIELD_NAMES = [
  "api_base",
  "api_key",
  "vertex_project",
  "vertex_location",
  "vertex_collection_id",
  "vertex_engine_id",
  "embedding_model",
  "vector_bucket_name",
  "index_name",
  "aws_region_name",
  "mongodb_database",
  "mongodb_collection",
  "mongodb_embedding_field",
  "mongodb_text_field",
  "mongodb_num_candidates",
  "mongodb_dimensions",
  "mongodb_similarity",
  "mongodb_filter_fields",
  "mongodb_text_index",
  "mongodb_hybrid_search",
  "mongodb_hybrid_weights",
  "mongodb_exact_search",
  "mongodb_score_threshold",
  "valkey_host",
  "valkey_port",
  "valkey_password",
  "valkey_ssl",
  "valkey_text_field",
  "valkey_embedding_field",
] as const;

export type ProviderFieldName = (typeof PROVIDER_FIELD_NAMES)[number];

export const isProviderFieldName = (name: string): name is ProviderFieldName =>
  (PROVIDER_FIELD_NAMES as readonly string[]).includes(name);

export type SupportedProviderField = VectorStoreFieldConfig & { name: ProviderFieldName };

export const isSupportedProviderField = (field: VectorStoreFieldConfig): field is SupportedProviderField =>
  isProviderFieldName(field.name);

const optionalText = z.string().optional();

/** Every field is held as text in the form; `buildVectorStoreLitellmParams` coerces on save. */
export const vectorStoreShape = {
  custom_llm_provider: z.string().min(1, "Please select a provider"),
  vector_store_id: z.string().min(1, "Please input the vector store ID from your api provider"),
  vector_store_name: optionalText,
  vector_store_description: optionalText,
  litellm_credential_name: z.string().nullable().optional(),
  api_base: optionalText,
  api_key: optionalText,
  vertex_project: optionalText,
  vertex_location: optionalText,
  vertex_collection_id: optionalText,
  vertex_engine_id: optionalText,
  embedding_model: optionalText,
  vector_bucket_name: optionalText,
  index_name: optionalText,
  aws_region_name: optionalText,
  mongodb_database: optionalText,
  mongodb_collection: optionalText,
  mongodb_embedding_field: optionalText,
  mongodb_text_field: optionalText,
  mongodb_num_candidates: optionalText,
  mongodb_dimensions: optionalText,
  mongodb_similarity: optionalText,
  mongodb_filter_fields: optionalText,
  mongodb_text_index: optionalText,
  mongodb_hybrid_search: optionalText,
  mongodb_hybrid_weights: optionalText,
  mongodb_exact_search: optionalText,
  mongodb_score_threshold: optionalText,
  valkey_host: optionalText,
  valkey_port: optionalText,
  valkey_password: optionalText,
  valkey_ssl: optionalText,
  valkey_text_field: optionalText,
  valkey_embedding_field: optionalText,
};

/**
 * mongodb's api_base/api_key are marked optional in its field config (`vector_store_providers.tsx`)
 * because a deployment can configure MONGODB_SIDECAR_API_BASE/_API_KEY once for every store. Without
 * that deployment default, though, they are the only way to reach the sidecar at all, so the schema
 * must require them itself rather than through the static field config.
 */
const MONGODB_CONNECTION_FIELD_NAMES: ReadonlySet<string> = new Set([
  "api_base",
  "api_key",
] satisfies ProviderFieldName[]);

const isFieldRequired = (field: VectorStoreFieldConfig, provider: string, hasDeploymentDefaults: boolean): boolean => {
  const requiresMongoDBConnectionField =
    provider === "mongodb" && !hasDeploymentDefaults && MONGODB_CONNECTION_FIELD_NAMES.has(field.name);
  return field.required || requiresMongoDBConnectionField;
};

const MONGODB_OVERRIDE_INCOMPLETE_MESSAGE = "A custom MongoDB sidecar connection needs both a URL and an API key.";

/**
 * With a deployment default, api_base/api_key are individually optional (the deployment's own
 * sidecar covers a blank pair), but the backend only ever sends the deployment's sidecar key to the
 * deployment's own sidecar (see MongoDBVectorStoreConfig.resolve_sidecar_api_key). Setting only one
 * of the pair would silently save a store the backend then rejects at request time, so the schema
 * itself requires both once either is touched.
 */
const requiresBothMongoDBOverrideFields = (values: Record<string, unknown>): boolean => {
  const apiBase = values.api_base;
  const apiKey = values.api_key;
  return values.custom_llm_provider === "mongodb" && Boolean(apiBase) !== Boolean(apiKey);
};

/**
 * `hasDeploymentDefaults` reflects whether the deployment already has a usable MongoDB sidecar
 * configured (see `useMongoDBProviderDefaults`). Callers outside a MongoDB-aware form that never
 * look this up pass `false`, the safe default: it can only make api_base/api_key required, never
 * looser.
 */
export const makeVectorStoreSchema = (hasDeploymentDefaults: boolean) =>
  z.object(vectorStoreShape).superRefine((values, ctx) => {
    getProviderSpecificFields(values.custom_llm_provider)
      .filter(
        (field): field is VectorStoreFieldConfig & { name: ProviderFieldName } =>
          isProviderFieldName(field.name) &&
          isFieldRequired(field, values.custom_llm_provider, hasDeploymentDefaults) &&
          !values[field.name],
      )
      .forEach((field) =>
        ctx.addIssue({
          code: "custom",
          path: [field.name],
          message:
            field.type === "select"
              ? `Please select the ${field.label.toLowerCase()}`
              : `Please input the ${field.label.toLowerCase()}`,
        }),
      );

    if (hasDeploymentDefaults && requiresBothMongoDBOverrideFields(values)) {
      ctx.addIssue({ code: "custom", path: ["api_base"], message: MONGODB_OVERRIDE_INCOMPLETE_MESSAGE });
      ctx.addIssue({ code: "custom", path: ["api_key"], message: MONGODB_OVERRIDE_INCOMPLETE_MESSAGE });
    }
  });

export const vectorStoreSchema = makeVectorStoreSchema(false);

export type VectorStoreFormValues = z.output<typeof vectorStoreSchema>;

const splitList = (value: string): string[] =>
  value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);

/**
 * Turns one form string into the JSON type the provider's `litellm_params` expects.
 * Returns undefined for anything blank so the backend keeps its own default.
 */
export const coerceFieldValue = (field: VectorStoreFieldConfig, raw: unknown): unknown => {
  if (typeof raw !== "string") return raw ?? undefined;
  const text = raw.trim();
  if (text === "") return undefined;

  switch (field.type) {
    case "number": {
      const parsed = Number(text);
      return Number.isFinite(parsed) ? parsed : undefined;
    }
    case "boolean":
      return text === "true";
    case "string-list": {
      const entries = splitList(text);
      return entries.length > 0 ? entries : undefined;
    }
    case "weight-split": {
      const vector = Number(text);
      return Number.isFinite(vector) && vector >= 0 && vector <= 1
        ? { vector, text: Number((1 - vector).toFixed(4)) }
        : undefined;
    }
    default:
      return raw;
  }
};

/** The key a field is stored under in litellm_params, which differs from its form name for a rename provider. */
export const paramName = (provider: string, field: VectorStoreFieldConfig): string =>
  EMBEDDING_MODEL_RENAME_PROVIDERS.has(provider) && field.name === "embedding_model"
    ? "litellm_embedding_model"
    : field.name;

export const buildVectorStoreLitellmParams = (
  provider: string,
  formValues: Record<string, unknown>,
): Record<string, unknown> =>
  Object.fromEntries(
    getProviderSpecificFields(provider)
      .filter(isSupportedProviderField)
      .map((field) => [paramName(provider, field), coerceFieldValue(field, formValues[field.name])] as const)
      .filter(([, value]) => value !== undefined),
  );

/**
 * A capability-gated field (e.g. hybrid search) stays registered in the form even while it's
 * hidden, so a value entered before the sidecar reported the capability unsupported would
 * otherwise still reach the saved store. Clearing it here, right before the params are built,
 * means what's saved always matches what the checklist most recently reported.
 */
export const clearUnsupportedCapabilityFields = (
  provider: string,
  formValues: Record<string, unknown>,
  isCapabilitySupported: (capability: string) => boolean,
): Record<string, unknown> => {
  const capabilityByField = new Map(
    getProviderSpecificFields(provider)
      .filter((field): field is VectorStoreFieldConfig & { requiresCapability: string } =>
        Boolean(field.requiresCapability),
      )
      .map((field) => [field.name, field.requiresCapability]),
  );
  return Object.fromEntries(
    Object.entries(formValues).map(([name, value]) => {
      const capability = capabilityByField.get(name);
      return [name, capability && !isCapabilitySupported(capability) ? undefined : value];
    }),
  );
};
