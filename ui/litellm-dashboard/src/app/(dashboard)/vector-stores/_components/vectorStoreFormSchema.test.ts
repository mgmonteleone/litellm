import { describe, expect, it } from "vitest";

import { getProviderSpecificFields } from "@/components/vector_store_providers";

import {
  buildVectorStoreLitellmParams,
  makeVectorStoreSchema,
  PROVIDER_FIELD_NAMES,
  vectorStoreShape,
} from "./vectorStoreFormSchema";

const MONGODB_BASE = {
  api_base: "http://127.0.0.1:8080",
  api_key: "sidecar-key",
  mongodb_database: "knowledge",
  mongodb_collection: "policies",
  embedding_model: "text-embedding-3-small",
};

describe("provider field allowlists", () => {
  it("carries every configured provider field, so none is silently dropped on save", () => {
    const configured = Object.values(
      Object.fromEntries(
        ["bedrock", "pg_vector", "vertex_ai/search_api", "openai", "azure", "milvus", "mongodb", "valkey", "s3_vectors"]
          .flatMap((provider) => getProviderSpecificFields(provider))
          .map((field) => [field.name, field.name]),
      ),
    );

    expect(PROVIDER_FIELD_NAMES).toEqual(expect.arrayContaining(configured));
  });

  it("gives every allowlisted field a slot in the form schema", () => {
    expect(Object.keys(vectorStoreShape)).toEqual(expect.arrayContaining([...PROVIDER_FIELD_NAMES]));
  });
});

describe("buildVectorStoreLitellmParams coercion", () => {
  it("sends mongodb_num_candidates as a number, not the string the input held", () => {
    const params = buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_num_candidates: "150" });

    expect(params.mongodb_num_candidates).toBe(150);
  });

  it("sends mongodb_score_threshold as a number so the sidecar can compare it", () => {
    const params = buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_score_threshold: "0.72" });

    expect(params.mongodb_score_threshold).toBe(0.72);
  });

  it("splits mongodb_filter_fields into the array the index definition needs", () => {
    const params = buildVectorStoreLitellmParams("mongodb", {
      ...MONGODB_BASE,
      mongodb_filter_fields: " metadata.department , metadata.year ,, ",
    });

    expect(params.mongodb_filter_fields).toEqual(["metadata.department", "metadata.year"]);
  });

  it("sends the hybrid toggle as a boolean in both positions", () => {
    expect(buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_hybrid_search: "true" })).toMatchObject({
      mongodb_hybrid_search: true,
    });
    expect(buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_hybrid_search: "false" })).toMatchObject(
      { mongodb_hybrid_search: false },
    );
  });

  it("expands the vector weight slider into the {vector, text} pair the sidecar ranks with", () => {
    const params = buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_hybrid_weights: "0.7" });

    expect(params.mongodb_hybrid_weights).toEqual({ vector: 0.7, text: 0.3 });
  });

  it("omits every advanced field the admin left alone", () => {
    const params = buildVectorStoreLitellmParams("mongodb", MONGODB_BASE);

    expect(Object.keys(params).sort()).toEqual([
      "api_base",
      "api_key",
      "litellm_embedding_model",
      "mongodb_collection",
      "mongodb_database",
    ]);
  });

  it("drops a non-numeric candidate count rather than sending a string the sidecar rejects", () => {
    const params = buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_num_candidates: "lots" });

    expect(params).not.toHaveProperty("mongodb_num_candidates");
  });

  it("keeps plain text fields as typed", () => {
    const params = buildVectorStoreLitellmParams("mongodb", { ...MONGODB_BASE, mongodb_text_index: "policy_text" });

    expect(params.mongodb_text_index).toBe("policy_text");
  });
});

describe("makeVectorStoreSchema mongodb connection requirement", () => {
  const MONGODB_WITHOUT_CONNECTION = {
    custom_llm_provider: "mongodb",
    vector_store_id: "policy_index",
    mongodb_database: "knowledge",
    mongodb_collection: "policies",
    embedding_model: "text-embedding-3-small",
  };

  it("requires api_base and api_key when the deployment has no defaults, so a store can never save with no connection", () => {
    const result = makeVectorStoreSchema(false).safeParse(MONGODB_WITHOUT_CONNECTION);

    expect(result.success).toBe(false);
    const paths = result.success ? [] : result.error.issues.map((issue) => issue.path.join("."));
    expect(paths).toEqual(expect.arrayContaining(["api_base", "api_key"]));
  });

  it("does not require api_base and api_key when the deployment already has defaults", () => {
    const result = makeVectorStoreSchema(true).safeParse(MONGODB_WITHOUT_CONNECTION);

    expect(result.success).toBe(true);
  });

  it("leaves non-mongodb providers alone regardless of hasDeploymentDefaults", () => {
    const bedrockValues = { custom_llm_provider: "bedrock", vector_store_id: "vs-1" };

    expect(makeVectorStoreSchema(false).safeParse(bedrockValues).success).toBe(true);
    expect(makeVectorStoreSchema(true).safeParse(bedrockValues).success).toBe(true);
  });

  it("rejects a custom api_base with no api_key when the deployment has defaults: the sidecar key is only sent to the deployment's own sidecar", () => {
    const result = makeVectorStoreSchema(true).safeParse({
      ...MONGODB_WITHOUT_CONNECTION,
      api_base: "https://tenant-sidecar.example",
    });

    expect(result.success).toBe(false);
    const paths = result.success ? [] : result.error.issues.map((issue) => issue.path.join("."));
    expect(paths).toEqual(expect.arrayContaining(["api_base", "api_key"]));
  });

  it("rejects a custom api_key with no api_base when the deployment has defaults", () => {
    const result = makeVectorStoreSchema(true).safeParse({
      ...MONGODB_WITHOUT_CONNECTION,
      api_key: "tenant-key",
    });

    expect(result.success).toBe(false);
    const paths = result.success ? [] : result.error.issues.map((issue) => issue.path.join("."));
    expect(paths).toEqual(expect.arrayContaining(["api_base", "api_key"]));
  });

  it("accepts a full override (both api_base and api_key) even when the deployment has defaults", () => {
    const result = makeVectorStoreSchema(true).safeParse({
      ...MONGODB_WITHOUT_CONNECTION,
      api_base: "https://tenant-sidecar.example",
      api_key: "tenant-key",
    });

    expect(result.success).toBe(true);
  });
});
