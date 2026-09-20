import { describe, expect, it } from "vitest";

import type { VectorStoreTestConnectionResponse } from "@/components/networking";

import { dimensionVerdict, dimensionVerdictMessage, sidecarFeatures, supportsFeature } from "./mongodbConnection";

const resultWith = (overrides: Partial<VectorStoreTestConnectionResponse>): VectorStoreTestConnectionResponse => ({
  ok: true,
  summary: "All checks passed.",
  checks: [],
  ...overrides,
});

const AUTH_CHECK_WITH_HYBRID = resultWith({
  checks: [
    {
      check: "sidecar_auth",
      status: "pass",
      message: "Authenticated with sidecar v0.2.0 (MongoDB 8.2.11).",
      details: { sidecar_version: "0.2.0", features: { hybrid: true, discovery: false, filters: true } },
    },
  ],
});

describe("sidecarFeatures", () => {
  it("reads the feature flags off the sidecar_auth check", () => {
    expect(sidecarFeatures(AUTH_CHECK_WITH_HYBRID)).toEqual({ hybrid: true, discovery: false, filters: true });
  });

  it("reports nothing for a v0.1 sidecar whose auth check carries no features", () => {
    const v01 = resultWith({
      checks: [{ check: "sidecar_auth", status: "warn", message: "Connected to a v0.1 sidecar (search only)." }],
    });

    expect(sidecarFeatures(v01)).toEqual({});
    expect(supportsFeature(v01, "hybrid")).toBe(false);
  });

  it("reports nothing before a test has run", () => {
    expect(supportsFeature(null, "hybrid")).toBe(false);
  });

  it("treats a non-true flag as unsupported rather than truthy", () => {
    const odd = resultWith({
      checks: [{ check: "sidecar_auth", status: "pass", message: "ok", details: { features: { hybrid: "yes" } } }],
    });

    expect(supportsFeature(odd, "hybrid")).toBe(false);
  });
});

describe("dimensionVerdict", () => {
  it("confirms a model whose vectors are the size the index declares", () => {
    const matched = resultWith({ details: { embedding_dimensions: 1536, mongodb: { index_dimensions: 1536 } } });

    expect(dimensionVerdict(matched)).toEqual({ kind: "match", dimensions: 1536 });
    expect(dimensionVerdictMessage(dimensionVerdict(matched))).toBe("Matches the index (1536).");
  });

  it("names both sizes when the model does not fit the index", () => {
    const mismatched = resultWith({ details: { embedding_dimensions: 3072, mongodb: { index_dimensions: 1536 } } });

    expect(dimensionVerdict(mismatched)).toEqual({ kind: "mismatch", embedding: 3072, index: 1536 });
    expect(dimensionVerdictMessage(dimensionVerdict(mismatched))).toBe(
      "The index expects 1536, this model returns 3072.",
    );
  });

  it("says nothing when the index dimensions are unknown, rather than claiming a match", () => {
    const partial = resultWith({ details: { embedding_dimensions: 1536, mongodb: { index_dimensions: null } } });

    expect(dimensionVerdict(partial)).toEqual({ kind: "unknown" });
    expect(dimensionVerdictMessage(dimensionVerdict(partial))).toBeNull();
  });
});
