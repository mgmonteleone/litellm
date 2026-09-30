import { getProviderSpecificFields } from "@/components/vector_store_providers";
import type { RAGIngestResponse } from "@/components/vector_store_management/types";

import { coerceFieldValue } from "./vectorStoreFormSchema";

const EMBEDDING_MODEL_RENAME_PROVIDERS = new Set(["milvus", "valkey", "mongodb"]);

/**
 * The ingest endpoint takes the provider settings inside ingest_options.vector_store, under the
 * same names the vector store itself uses, so it needs the add dialog's rename and coercion.
 * Without the rename, a MongoDB ingest fails with "an embedding model is required".
 *
 * Keys the provider's field config does not describe are passed through untouched, because some
 * providers (S3 Vectors) collect their settings in a component of their own.
 */
export const buildIngestProviderParams = (
  provider: string,
  formValues: Record<string, unknown>,
): Record<string, unknown> => {
  const fields = new Map(getProviderSpecificFields(provider).map((field) => [field.name, field]));
  const renameEmbeddingModel = EMBEDDING_MODEL_RENAME_PROVIDERS.has(provider);
  return Object.fromEntries(
    Object.entries(formValues)
      .map(([name, value]) => {
        const field = fields.get(name);
        const paramName = renameEmbeddingModel && name === "embedding_model" ? "litellm_embedding_model" : name;
        return [paramName, field ? coerceFieldValue(field, value) : value] as const;
      })
      .filter(([, value]) => value !== undefined),
  );
};

export interface ChunkingInput {
  chunkSize: string;
  chunkOverlap: string;
}

const asPositiveInt = (raw: string): number | undefined => {
  const parsed = Number(raw.trim());
  return raw.trim() !== "" && Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
};

export const buildChunkingStrategy = (
  input: ChunkingInput,
): { chunk_size?: number; chunk_overlap?: number } | undefined => {
  const chunkSize = asPositiveInt(input.chunkSize);
  const chunkOverlap = asPositiveInt(input.chunkOverlap);
  const strategy = {
    ...(chunkSize !== undefined ? { chunk_size: chunkSize } : {}),
    ...(chunkOverlap !== undefined ? { chunk_overlap: chunkOverlap } : {}),
  };
  return Object.keys(strategy).length > 0 ? strategy : undefined;
};

/** The overlap has to fit inside the chunk, otherwise the splitter never advances. */
export const chunkingError = (input: ChunkingInput): string | null => {
  const chunkSize = asPositiveInt(input.chunkSize);
  const chunkOverlap = asPositiveInt(input.chunkOverlap);
  if (input.chunkSize.trim() !== "" && chunkSize === undefined) return "Chunk size must be a whole number above zero.";
  if (input.chunkOverlap.trim() !== "" && chunkOverlap === undefined) {
    return "Chunk overlap must be a whole number above zero.";
  }
  if (chunkSize !== undefined && chunkOverlap !== undefined && chunkOverlap >= chunkSize) {
    return "Chunk overlap must be smaller than the chunk size.";
  }
  return null;
};

export interface IngestTotals {
  documents: number;
  chunks: number | null;
  vectors: number | null;
}

const countFrom = (response: RAGIngestResponse, keys: readonly string[]): number | null => {
  const record = response as unknown as Record<string, unknown>;
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
};

/**
 * Chunk and vector counts are reported only by providers that bother to; a null total means
 * "not reported", which the alert leaves out rather than printing as zero.
 */
export const ingestTotals = (responses: readonly RAGIngestResponse[]): IngestTotals => {
  const sum = (keys: readonly string[]): number | null => {
    const counts = responses.map((response) => countFrom(response, keys)).filter((count) => count !== null);
    return counts.length > 0 ? counts.reduce((total, count) => total + count, 0) : null;
  };
  return {
    documents: responses.length,
    chunks: sum(["chunk_count", "chunks", "num_chunks"]),
    vectors: sum(["vector_count", "vectors", "num_vectors", "embedding_count"]),
  };
};

export const describeIngestTotals = (totals: IngestTotals): string =>
  [
    `${totals.documents} document${totals.documents === 1 ? "" : "s"}`,
    totals.chunks === null ? null : `${totals.chunks.toLocaleString()} chunks`,
    totals.vectors === null ? null : `${totals.vectors.toLocaleString()} vectors`,
  ]
    .filter((part): part is string => part !== null)
    .join(" · ");
