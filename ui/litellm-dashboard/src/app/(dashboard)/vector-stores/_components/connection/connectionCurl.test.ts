import { describe, expect, it } from "vitest";

import { buildDiscoverCurl, buildTestConnectionCurl, maskSecretParams, REDACTION_SENTINEL } from "./connectionCurl";

describe("maskSecretParams", () => {
  it("replaces a real sidecar key so a copied command never carries the secret", () => {
    const masked = maskSecretParams({ api_base: "http://127.0.0.1:8080", api_key: "super-secret-key" });

    expect(masked.api_key).not.toContain("super-secret-key");
    expect(masked.api_base).toBe("http://127.0.0.1:8080");
  });

  it("keeps the redaction sentinel, which is what tells the proxy to reuse the saved secret", () => {
    expect(maskSecretParams({ api_key: REDACTION_SENTINEL }).api_key).toBe(REDACTION_SENTINEL);
  });

  it("masks every secret-shaped name, not only api_key", () => {
    const params = {
      valkey_password: "hunter2",
      aws_secret_access_key: "AKIA-secret",
      litellm_credential_name: "prod",
      mongodb_database: "knowledge",
    };
    const masked = maskSecretParams(params);

    expect(masked.valkey_password).not.toBe("hunter2");
    expect(masked.aws_secret_access_key).not.toBe("AKIA-secret");
    expect(masked.mongodb_database).toBe("knowledge");
  });
});

describe("buildTestConnectionCurl", () => {
  it("targets the proxy's test_connection route with the saved store id", () => {
    const command = buildTestConnectionCurl("http://localhost:4000", { vector_store_id: "policy_vector_index" });

    expect(command).toContain("curl -X POST 'http://localhost:4000/vector_store/test_connection'");
    expect(command).toContain('"vector_store_id": "policy_vector_index"');
  });

  it("never prints the sidecar key from an unsaved configuration", () => {
    const command = buildTestConnectionCurl("http://localhost:4000", {
      custom_llm_provider: "mongodb",
      litellm_params: { api_base: "http://127.0.0.1:8080", api_key: "leak-me" },
    });

    expect(command).not.toContain("leak-me");
    expect(command).toContain('"api_base": "http://127.0.0.1:8080"');
  });

  it("leaves out keys the dashboard did not send", () => {
    const command = buildTestConnectionCurl("http://localhost:4000", {
      vector_store_id: "policy_vector_index",
      custom_llm_provider: null,
      litellm_params: null,
    });

    expect(command).not.toContain("custom_llm_provider");
    expect(command).not.toContain("litellm_params");
  });

  it("escapes a single quote in a value so the copied command still parses in a shell", () => {
    const command = buildTestConnectionCurl("http://localhost:4000", { vector_store_id: "it's-fine" });

    expect(command).not.toMatch(/-d '[^']*it's/);
    expect(command).toContain(`'\\''`);
  });
});

describe("buildDiscoverCurl", () => {
  it("keeps the discovery kind and options in the copied body", () => {
    const command = buildDiscoverCurl("http://localhost:4000", {
      kind: "fields",
      custom_llm_provider: "mongodb",
      options: { mongodb_database: "knowledge", mongodb_collection: "policies" },
    });

    expect(command).toContain("/vector_store/discover");
    expect(command).toContain('"kind": "fields"');
    expect(command).toContain('"mongodb_collection": "policies"');
  });
});
