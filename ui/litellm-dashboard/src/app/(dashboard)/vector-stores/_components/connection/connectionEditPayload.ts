import { getProviderSpecificFields, type VectorStoreFieldConfig } from "@/components/vector_store_providers";

import { coerceFieldValue, isSupportedProviderField, paramName } from "../vectorStoreFormSchema";
import { isSecretParamName, REDACTION_SENTINEL } from "./connectionCurl";
import { normalizeLitellmParams, type LitellmParams } from "./litellmParamsDisplay";

/**
 * Text a supported provider field should hold in the edit form, given the value the store actually
 * has saved for it. This is `coerceFieldValue`'s inverse: the form always holds a string, whatever
 * shape the stored value takes. A saved secret already arrives as the redaction sentinel (the proxy
 * redacts it before the response leaves it), so it needs no special casing here.
 */
const formatFieldValueForForm = (field: VectorStoreFieldConfig, value: unknown): string => {
  if (value === undefined || value === null) return "";
  switch (field.type) {
    case "boolean":
      return value ? "true" : "false";
    case "string-list":
      return Array.isArray(value) ? value.join(", ") : String(value);
    case "weight-split":
      return value && typeof value === "object" && "vector" in value
        ? String((value as { vector: number }).vector)
        : "";
    default:
      return String(value);
  }
};

/** Prefills the connection edit form's fields from a store's saved (possibly redacted) litellm_params. */
export const connectionFormValuesFromLitellmParams = (
  provider: string,
  litellmParams: LitellmParams,
): Record<string, string> => {
  const normalized = normalizeLitellmParams(litellmParams);
  return Object.fromEntries(
    getProviderSpecificFields(provider)
      .filter(isSupportedProviderField)
      .map((field) => [field.name, formatFieldValueForForm(field, normalized[paramName(provider, field)])]),
  );
};

/**
 * Only a field the admin actually changed is sent, so the update never clobbers a saved value the
 * form merely round-tripped through its own type coercion. A secret field left untouched is the one
 * exception: it is still sent, but only ever as the redaction sentinel, matching what
 * /vector_store/update requires to mean "keep the saved secret" (see
 * litellm/proxy/vector_store_endpoints/management_endpoints.py). A field that was never saved and is
 * still blank is left out entirely rather than sent as an empty string.
 */
export const buildConnectionUpdateLitellmParams = (
  provider: string,
  initialValues: Record<string, string | undefined>,
  currentValues: Record<string, string | undefined>,
): Record<string, unknown> =>
  Object.fromEntries(
    getProviderSpecificFields(provider)
      .filter(isSupportedProviderField)
      .flatMap((field) => {
        const key = paramName(provider, field);
        const initial = initialValues[field.name] ?? "";
        const current = currentValues[field.name] ?? "";
        if (current === initial) {
          return isSecretParamName(key) && current === REDACTION_SENTINEL ? [[key, REDACTION_SENTINEL] as const] : [];
        }
        const coerced = coerceFieldValue(field, current);
        return coerced === undefined ? [] : [[key, coerced] as const];
      }),
  );
