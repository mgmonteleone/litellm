import { describe, expect, it } from "vitest";

import { REDACTION_SENTINEL } from "./connectionCurl";
import { buildConnectionUpdateLitellmParams, connectionFormValuesFromLitellmParams } from "./connectionEditPayload";

describe("connectionFormValuesFromLitellmParams", () => {
  it("prefills text fields from the saved params and the secret field with the redaction sentinel", () => {
    const savedParams = {
      api_base: "http://127.0.0.1:8080",
      api_key: REDACTION_SENTINEL,
      mongodb_database: "knowledge",
      mongodb_collection: "policies",
    };
    const values = connectionFormValuesFromLitellmParams("mongodb", savedParams);

    expect(values.api_base).toBe("http://127.0.0.1:8080");
    expect(values.api_key).toBe(REDACTION_SENTINEL);
    expect(values.mongodb_database).toBe("knowledge");
    expect(values.mongodb_collection).toBe("policies");
  });

  it("leaves a field never saved as blank rather than the string 'undefined'", () => {
    const values = connectionFormValuesFromLitellmParams("mongodb", { api_base: "http://127.0.0.1:8080" });

    expect(values.mongodb_database).toBe("");
  });

  it("renames the stored litellm_embedding_model key back onto the embedding_model form field", () => {
    const values = connectionFormValuesFromLitellmParams("mongodb", {
      litellm_embedding_model: "text-embedding-3-small",
    });

    expect(values.embedding_model).toBe("text-embedding-3-small");
  });

  it("parses a JSON-string litellm_params the same way a config-registered store's redacted params arrive", () => {
    const values = connectionFormValuesFromLitellmParams("mongodb", JSON.stringify({ mongodb_database: "knowledge" }));

    expect(values.mongodb_database).toBe("knowledge");
  });
});

describe("buildConnectionUpdateLitellmParams", () => {
  const savedParams = {
    api_base: "http://127.0.0.1:8080",
    api_key: REDACTION_SENTINEL,
    mongodb_database: "knowledge",
    mongodb_collection: "old_collection",
  };
  const initial = connectionFormValuesFromLitellmParams("mongodb", savedParams);

  it("sends an untouched secret field back as the exact redaction sentinel", () => {
    const params = buildConnectionUpdateLitellmParams("mongodb", initial, { ...initial });

    expect(params.api_key).toBe(REDACTION_SENTINEL);
  });

  it("sends only the changed field, on a provider with no secret field to also round-trip", () => {
    const s3Initial = connectionFormValuesFromLitellmParams("s3_vectors", {
      vector_bucket_name: "my-bucket",
      aws_region_name: "us-west-2",
    });
    const s3Current = { ...s3Initial, aws_region_name: "us-east-1" };

    const params = buildConnectionUpdateLitellmParams("s3_vectors", s3Initial, s3Current);

    expect(params).toEqual({ aws_region_name: "us-east-1" });
  });

  it("sends a changed field's value on top of the untouched secret sentinel, and nothing else", () => {
    const current = { ...initial, mongodb_collection: "new_collection" };

    const params = buildConnectionUpdateLitellmParams("mongodb", initial, current);

    expect(params).toEqual({ api_key: REDACTION_SENTINEL, mongodb_collection: "new_collection" });
  });

  it("sends an os.environ/ reference typed into a secret field verbatim, not as the sentinel", () => {
    const current = { ...initial, api_key: "os.environ/MONGODB_SIDECAR_API_KEY" };

    const params = buildConnectionUpdateLitellmParams("mongodb", initial, current);

    expect(params.api_key).toBe("os.environ/MONGODB_SIDECAR_API_KEY");
  });

  it("omits a secret field that was never saved and is still blank", () => {
    const blankInitial = connectionFormValuesFromLitellmParams("mongodb", { mongodb_database: "knowledge" });

    const params = buildConnectionUpdateLitellmParams("mongodb", blankInitial, { ...blankInitial });

    expect(params).not.toHaveProperty("api_key");
  });

  it("coerces a changed numeric field the same way the create form does", () => {
    const current = { ...initial, mongodb_num_candidates: "250" };

    const params = buildConnectionUpdateLitellmParams("mongodb", initial, current);

    expect(params.mongodb_num_candidates).toBe(250);
  });
});
