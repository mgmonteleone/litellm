import type { VectorStoreDiscoveryKind } from "@/components/vector_store_providers";

/**
 * Shapes returned by GET /v1/discovery/* on the MongoDB sidecar, proxied through
 * POST /vector_store/discover. Parsed defensively: the endpoint is provider-shaped JSON,
 * so an older sidecar can answer with fewer keys than the current one.
 */

export interface DiscoverySuggestion {
  /** The value written into the field. */
  value: string;
  /** One line of context, e.g. "1536-dim, 3 documents". */
  hint?: string;
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const asArray = (value: unknown): readonly unknown[] => (Array.isArray(value) ? value : []);

const asCount = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;

const countHint = (documents: unknown, noun: string): string | undefined => {
  const count = asCount(documents);
  return count === null ? undefined : `${count.toLocaleString()} ${noun}${count === 1 ? "" : "s"}`;
};

const joinHints = (...parts: (string | undefined)[]): string | undefined => {
  const present = parts.filter((part): part is string => Boolean(part));
  return present.length > 0 ? present.join(", ") : undefined;
};

const databases = (payload: Record<string, unknown>): DiscoverySuggestion[] =>
  asArray(payload.databases)
    .filter((name): name is string => typeof name === "string")
    .map((value) => ({ value }));

const collections = (payload: Record<string, unknown>): DiscoverySuggestion[] =>
  asArray(payload.collections)
    .map(asRecord)
    .filter((entry): entry is Record<string, unknown> => entry !== null && typeof entry.name === "string")
    .map((entry) => ({ value: String(entry.name), hint: countHint(entry.document_count, "document") }));

const vectorFields = (payload: Record<string, unknown>): DiscoverySuggestion[] =>
  asArray(payload.vector_fields)
    .map(asRecord)
    .filter((entry): entry is Record<string, unknown> => entry !== null && typeof entry.path === "string")
    .map((entry) => ({
      value: String(entry.path),
      hint: joinHints(
        asCount(entry.dimensions) === null ? undefined : `${entry.dimensions}-dim`,
        countHint(entry.documents, "document"),
      ),
    }));

const textFields = (payload: Record<string, unknown>): DiscoverySuggestion[] =>
  asArray(payload.text_fields)
    .map(asRecord)
    .filter((entry): entry is Record<string, unknown> => entry !== null && typeof entry.path === "string")
    .map((entry) => ({
      value: String(entry.path),
      hint: joinHints(
        asCount(entry.average_length) === null ? undefined : `~${entry.average_length} chars`,
        countHint(entry.documents, "document"),
      ),
    }));

const filterFields = (payload: Record<string, unknown>): DiscoverySuggestion[] =>
  asArray(payload.filter_candidates)
    .filter((path): path is string => typeof path === "string")
    .map((value) => ({ value }));

const PARSERS: Record<VectorStoreDiscoveryKind, (payload: Record<string, unknown>) => DiscoverySuggestion[]> = {
  databases,
  collections,
  vector_fields: vectorFields,
  text_fields: textFields,
  filter_fields: filterFields,
};

/** Discovery kinds "vector_fields", "text_fields" and "filter_fields" all come from one /discover call. */
export const discoverRequestKind = (kind: VectorStoreDiscoveryKind): "databases" | "collections" | "fields" =>
  kind === "databases" || kind === "collections" ? kind : "fields";

export const parseSuggestions = (kind: VectorStoreDiscoveryKind, payload: unknown): DiscoverySuggestion[] => {
  const record = asRecord(payload);
  return record ? PARSERS[kind](record) : [];
};

export interface MongoIndexSummary {
  name: string;
  queryable: boolean;
  status: string;
  dimensions: number | null;
  similarity: string | null;
  vectorPath: string | null;
  filterFields: readonly string[];
}

export const parseIndexes = (payload: unknown): MongoIndexSummary[] => {
  const record = asRecord(payload);
  if (!record) return [];
  return asArray(record.indexes)
    .map(asRecord)
    .filter((entry): entry is Record<string, unknown> => entry !== null && typeof entry.name === "string")
    .map((entry) => ({
      name: String(entry.name),
      queryable: entry.queryable === true,
      status: typeof entry.status === "string" ? entry.status : "unknown",
      dimensions: asCount(entry.dimensions),
      similarity: typeof entry.similarity === "string" ? entry.similarity : null,
      vectorPath: typeof entry.vector_path === "string" ? entry.vector_path : null,
      filterFields: asArray(entry.filter_fields).filter((path): path is string => typeof path === "string"),
    }));
};

export const indexSummaryHint = (index: MongoIndexSummary): string =>
  joinHints(
    index.dimensions === null ? undefined : `${index.dimensions}-dim`,
    index.similarity ?? undefined,
    index.queryable ? "ready" : index.status,
  ) ?? index.status;
