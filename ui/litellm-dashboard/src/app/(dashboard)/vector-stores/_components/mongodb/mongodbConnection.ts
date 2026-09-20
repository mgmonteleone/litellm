import type { VectorStoreTestConnectionResponse } from "@/components/networking";

const asRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const asPositiveInt = (value: unknown): number | null =>
  typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;

/**
 * What the sidecar says it can do, read off the sidecar_auth check. An empty map means the answer is
 * not known yet, which the dialog treats as "hide the capability-gated controls".
 */
export const sidecarFeatures = (result: VectorStoreTestConnectionResponse | null): Record<string, boolean> => {
  const auth = result?.checks?.find((check) => check.check === "sidecar_auth");
  const features = asRecord(asRecord(auth?.details)?.features);
  if (!features) return {};
  return Object.fromEntries(Object.entries(features).map(([name, value]) => [name, value === true]));
};

export const supportsFeature = (result: VectorStoreTestConnectionResponse | null, feature: string): boolean =>
  sidecarFeatures(result)[feature] === true;

/** Vector size the chosen embedding model actually returned during the test. */
export const embeddingDimensions = (result: VectorStoreTestConnectionResponse | null): number | null =>
  asPositiveInt(asRecord(result?.details)?.embedding_dimensions);

/** Vector size the MongoDB index declares. */
export const indexDimensions = (result: VectorStoreTestConnectionResponse | null): number | null =>
  asPositiveInt(asRecord(asRecord(result?.details)?.mongodb)?.index_dimensions);

export type DimensionVerdict =
  | { kind: "unknown" }
  | { kind: "match"; dimensions: number }
  | { kind: "mismatch"; embedding: number; index: number };

/**
 * A model whose vectors are a different size than the index is the single most common MongoDB
 * misconfiguration, and the search it produces fails late and obscurely, so the dialog says it up front.
 */
export const dimensionVerdict = (result: VectorStoreTestConnectionResponse | null): DimensionVerdict => {
  const embedding = embeddingDimensions(result);
  const index = indexDimensions(result);
  if (embedding === null || index === null) return { kind: "unknown" };
  return embedding === index ? { kind: "match", dimensions: index } : { kind: "mismatch", embedding, index };
};

export const dimensionVerdictMessage = (verdict: DimensionVerdict): string | null => {
  switch (verdict.kind) {
    case "unknown":
      return null;
    case "match":
      return `Matches the index (${verdict.dimensions}).`;
    case "mismatch":
      return `The index expects ${verdict.index}, this model returns ${verdict.embedding}.`;
  }
};
