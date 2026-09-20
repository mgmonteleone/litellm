import { getProviderSpecificFields } from "@/components/vector_store_providers";

import { isSecretParamName, REDACTION_SENTINEL } from "./connectionCurl";

export interface DisplayParam {
  name: string;
  label: string;
  value: string;
  secret: boolean;
}

/** Providers whose embedding model is stored under the generic litellm key. */
const EMBEDDING_MODEL_RENAME_PROVIDERS = new Set(["milvus", "valkey", "mongodb"]);

const paramNameFor = (provider: string, fieldName: string): string =>
  EMBEDDING_MODEL_RENAME_PROVIDERS.has(provider) && fieldName === "embedding_model"
    ? "litellm_embedding_model"
    : fieldName;

const providerParamLabels = (provider: string): Map<string, string> =>
  new Map(getProviderSpecificFields(provider).map((field) => [paramNameFor(provider, field.name), field.label]));

/**
 * GenericLiteLLMParams stamps a handful of unset feature flags onto every saved store.
 * They carry no information about this store, so only values that say something are shown.
 */
const EMPTY_VALUES: readonly unknown[] = [null, undefined, "", false];

const saysSomething = (value: unknown): boolean => {
  if (EMPTY_VALUES.includes(value)) return false;
  return !Array.isArray(value) || value.length > 0;
};

export const formatParamValue = (value: unknown): string => {
  if (Array.isArray(value)) return value.map((entry) => String(entry)).join(", ");
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const toDisplayParam = (provider: string, labels: Map<string, string>, name: string, value: unknown): DisplayParam => {
  const secret = isSecretParamName(name) || value === REDACTION_SENTINEL;
  return {
    name,
    label: labels.get(name) ?? name,
    value: secret ? "Set, hidden" : formatParamValue(value),
    secret,
  };
};

/**
 * The saved connection in reading order: the provider's own fields first, as the dialog labelled
 * them, then whatever else the store carries. Secrets never reach the returned value.
 */
export const describeLitellmParams = (
  provider: string,
  params: Record<string, unknown> | null | undefined,
): readonly DisplayParam[] => {
  const entries = Object.entries(params ?? {});
  const labels = providerParamLabels(provider);
  const known = [...labels.keys()]
    .filter((name) => name in (params ?? {}))
    .map((name) => toDisplayParam(provider, labels, name, params?.[name]));
  const extra = entries
    .filter(([name, value]) => !labels.has(name) && saysSomething(value))
    .map(([name, value]) => toDisplayParam(provider, labels, name, value));
  return [...known, ...extra];
};
