import { describe, expect, it } from "vitest";

import type { RAGIngestResponse } from "@/components/vector_store_management/types";

import {
  buildChunkingStrategy,
  buildIngestProviderParams,
  chunkingError,
  describeIngestTotals,
  ingestTotals,
} from "./ingestPayload";

const MONGODB_FORM = {
  api_base: "http://127.0.0.1:8080",
  api_key: "sidecar-key",
  mongodb_database: "knowledge",
  mongodb_collection: "policies",
  embedding_model: "text-embedding-3-small",
};

describe("buildIngestProviderParams", () => {
  it("renames the embedding model on the ingest path, which MongoDB ingest requires", () => {
    const params = buildIngestProviderParams("mongodb", MONGODB_FORM);

    expect(params.litellm_embedding_model).toBe("text-embedding-3-small");
    expect(params).not.toHaveProperty("embedding_model");
  });

  it("keeps embedding_model as-is for a provider outside the rename set", () => {
    const params = buildIngestProviderParams("s3_vectors", {
      vector_bucket_name: "bucket",
      aws_region_name: "us-west-2",
      embedding_model: "text-embedding-3-small",
    });

    expect(params.embedding_model).toBe("text-embedding-3-small");
  });

  it("passes a key the provider config does not describe through untouched, since S3 Vectors sets its own", () => {
    const params = buildIngestProviderParams("s3_vectors", { vector_bucket_name: "bucket", custom_thing: "kept" });

    expect(params.custom_thing).toBe("kept");
  });

  it("drops a field the admin cleared rather than sending an empty string", () => {
    const params = buildIngestProviderParams("mongodb", { ...MONGODB_FORM, mongodb_text_index: "" });

    expect(params).not.toHaveProperty("mongodb_text_index");
  });

  it("coerces the numeric candidate count the same way the add dialog does", () => {
    const params = buildIngestProviderParams("mongodb", { ...MONGODB_FORM, mongodb_num_candidates: "150" });

    expect(params.mongodb_num_candidates).toBe(150);
  });
});

describe("buildChunkingStrategy", () => {
  it("sends nothing when both boxes are empty, keeping the backend default", () => {
    expect(buildChunkingStrategy({ chunkSize: "", chunkOverlap: "" })).toBeUndefined();
  });

  it("sends whichever value was filled in, as a number", () => {
    expect(buildChunkingStrategy({ chunkSize: "800", chunkOverlap: "" })).toEqual({ chunk_size: 800 });
    expect(buildChunkingStrategy({ chunkSize: "800", chunkOverlap: "120" })).toEqual({
      chunk_size: 800,
      chunk_overlap: 120,
    });
  });
});

describe("chunkingError", () => {
  it("accepts empty boxes", () => {
    expect(chunkingError({ chunkSize: "", chunkOverlap: "" })).toBeNull();
  });

  it("rejects an overlap that is not smaller than the chunk, which never advances the splitter", () => {
    expect(chunkingError({ chunkSize: "500", chunkOverlap: "500" })).toBe(
      "Chunk overlap must be smaller than the chunk size.",
    );
    expect(chunkingError({ chunkSize: "500", chunkOverlap: "600" })).not.toBeNull();
  });

  it("rejects a value that is not a whole number above zero", () => {
    expect(chunkingError({ chunkSize: "0", chunkOverlap: "" })).toContain("Chunk size");
    expect(chunkingError({ chunkSize: "12.5", chunkOverlap: "" })).toContain("Chunk size");
    expect(chunkingError({ chunkSize: "", chunkOverlap: "-1" })).toContain("Chunk overlap");
  });

  it("accepts a valid pair", () => {
    expect(chunkingError({ chunkSize: "1000", chunkOverlap: "200" })).toBeNull();
  });
});

const ingestResponse = (extra: Record<string, unknown> = {}): RAGIngestResponse =>
  ({ id: "1", status: "completed", vector_store_id: "vs_1", file_id: "file_1", ...extra }) as RAGIngestResponse;

describe("ingestTotals", () => {
  it("counts the documents that were ingested", () => {
    expect(ingestTotals([ingestResponse(), ingestResponse()]).documents).toBe(2);
  });

  it("reports chunk and vector totals as unknown when the provider did not send any", () => {
    const totals = ingestTotals([ingestResponse()]);

    expect(totals.chunks).toBeNull();
    expect(totals.vectors).toBeNull();
    expect(describeIngestTotals(totals)).toBe("1 document");
  });

  it("sums the counts across files when the provider does send them", () => {
    const totals = ingestTotals([
      ingestResponse({ chunk_count: 12, vector_count: 12 }),
      ingestResponse({ chunk_count: 8, vector_count: 8 }),
    ]);

    expect(totals).toEqual({ documents: 2, chunks: 20, vectors: 20 });
    expect(describeIngestTotals(totals)).toBe("2 documents · 20 chunks · 20 vectors");
  });
});
