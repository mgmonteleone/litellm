import type { VectorStoreDiscoverRequest } from "@/components/networking";

/** What the proxy sends back in place of a saved secret, and what it accepts back to mean "keep the saved one". */
export const REDACTION_SENTINEL = "REDACTED_BY_LITELM";

const SECRET_KEY_PATTERN = /(^|_)(key|password|secret|token|credential)$/i;

const SECRET_PLACEHOLDER = "<your-secret>";

const AUTH_PLACEHOLDER = "$LITELLM_API_KEY";

export const isSecretParamName = (name: string): boolean => SECRET_KEY_PATTERN.test(name);

/**
 * Replaces every secret-looking value with a placeholder. A copied command lands in a terminal,
 * a ticket, or a chat window, so it must never carry the real sidecar key.
 */
export const maskSecretParams = (params: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(params).map(([name, value]) =>
      isSecretParamName(name) && typeof value === "string" && value.length > 0
        ? [name, value === REDACTION_SENTINEL ? value : SECRET_PLACEHOLDER]
        : [name, value],
    ),
  );

const shellQuote = (value: string): string => `'${value.replaceAll("'", `'\\''`)}'`;

const curl = (url: string, body: Record<string, unknown>): string =>
  [
    `curl -X POST ${shellQuote(url)}`,
    `  -H ${shellQuote(`Authorization: Bearer ${AUTH_PLACEHOLDER}`)}`,
    `  -H 'Content-Type: application/json'`,
    `  -d ${shellQuote(JSON.stringify(body, null, 2))}`,
  ].join(" \\\n");

export interface TestConnectionRequestBody {
  vector_store_id?: string | null;
  custom_llm_provider?: string | null;
  litellm_params?: Record<string, unknown> | null;
}

/** Drops nulls so the copied body matches what the dashboard actually sent, and masks the secrets. */
const compactBody = (body: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(body)
      .filter(([, value]) => value !== undefined && value !== null)
      .map(([name, value]) =>
        name === "litellm_params" && value && typeof value === "object"
          ? [name, maskSecretParams(value as Record<string, unknown>)]
          : [name, value],
      ),
  );

export const buildTestConnectionCurl = (baseUrl: string, body: TestConnectionRequestBody): string =>
  curl(`${baseUrl}/vector_store/test_connection`, compactBody({ ...body }));

export const buildDiscoverCurl = (baseUrl: string, body: VectorStoreDiscoverRequest): string =>
  curl(`${baseUrl}/vector_store/discover`, compactBody({ ...body }));
