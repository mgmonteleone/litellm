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

export type LitellmParams = Record<string, unknown> | string | null | undefined;

/**
 * `_redact_sensitive_litellm_params` re-serializes a config-registered store's params to a JSON
 * string, so every consumer of the saved params has to accept that shape and normalise it back to
 * a record before reading any key out of it.
 */
export const normalizeLitellmParams = (params: LitellmParams): Record<string, unknown> => {
  if (params == null) return {};
  if (typeof params !== "string") return params;
  try {
    const parsed: unknown = JSON.parse(params);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
};

/**
 * The saved connection in reading order: the provider's own fields first, as the dialog labelled
 * them, then whatever else the store carries. Secrets never reach the returned value.
 */
export const describeLitellmParams = (provider: string, params: LitellmParams): readonly DisplayParam[] => {
  const normalized = normalizeLitellmParams(params);
  const entries = Object.entries(normalized);
  const labels = providerParamLabels(provider);
  const known = [...labels.keys()]
    .filter((name) => name in normalized)
    .map((name) => toDisplayParam(provider, labels, name, normalized[name]));
  const extra = entries
    .filter(([name, value]) => !labels.has(name) && saysSomething(value))
    .map(([name, value]) => toDisplayParam(provider, labels, name, value));
  return [...known, ...extra];
};
