import { describe, expect, it } from "vitest";

import { describeLitellmParams, formatParamValue } from "./litellmParamsDisplay";
import { REDACTION_SENTINEL } from "./connectionCurl";

/** Exactly what POST /vector_store/info returns for a saved MongoDB store. */
const SAVED_MONGODB_PARAMS = {
  api_key: REDACTION_SENTINEL,
  api_base: "http://127.0.0.1:8080",
  use_xai_oauth: false,
  mongodb_database: "knowledge",
  use_litellm_proxy: false,
  mongodb_collection: "policies",
  use_in_pass_through: false,
  mongodb_filter_fields: ["metadata.department"],
  litellm_embedding_model: "text-embedding-3-small",
  allow_client_keepalive_override: false,
  merge_reasoning_content_in_choices: false,
};

describe("describeLitellmParams", () => {
  it("never reveals a secret, even one the proxy failed to redact", () => {
    const rows = describeLitellmParams("mongodb", { ...SAVED_MONGODB_PARAMS, api_key: "raw-sidecar-key" });
    const apiKey = rows.find((row) => row.name === "api_key");

    expect(apiKey?.secret).toBe(true);
    expect(apiKey?.value).toBe("Set, hidden");
    expect(rows.map((row) => row.value).join(" ")).not.toContain("raw-sidecar-key");
  });

  it("drops the GenericLiteLLMParams feature flags that say nothing about this store", () => {
    const names = describeLitellmParams("mongodb", SAVED_MONGODB_PARAMS).map((row) => row.name);

    expect(names).not.toContain("use_xai_oauth");
    expect(names).not.toContain("use_litellm_proxy");
    expect(names).not.toContain("merge_reasoning_content_in_choices");
  });

  it("labels each param the way the add dialog labelled it", () => {
    const rows = describeLitellmParams("mongodb", SAVED_MONGODB_PARAMS);
    const byName = (name: string) => rows.find((row) => row.name === name);

    expect(byName("api_base")?.label).toBe("Sidecar URL");
    expect(byName("mongodb_database")?.label).toBe("Database");
    expect(byName("litellm_embedding_model")?.label).toBe("Embedding Model");
  });

  it("lists the provider's own fields in dialog order, ahead of anything else", () => {
    const names = describeLitellmParams("mongodb", { ...SAVED_MONGODB_PARAMS, some_future_param: "kept" }).map(
      (row) => row.name,
    );

    expect(names.slice(0, 3)).toEqual(["api_base", "api_key", "mongodb_database"]);
    expect(names[names.length - 1]).toBe("some_future_param");
  });

  it("renders a filter field list as text rather than [object Object]", () => {
    const rows = describeLitellmParams("mongodb", SAVED_MONGODB_PARAMS);

    expect(rows.find((row) => row.name === "mongodb_filter_fields")?.value).toBe("metadata.department");
  });

  it("returns nothing for a store the proxy saved without any params", () => {
    expect(describeLitellmParams("bedrock", null)).toEqual([]);
    expect(describeLitellmParams("bedrock", {})).toEqual([]);
  });

  it("keeps an unknown provider's params visible under their raw names", () => {
    const rows = describeLitellmParams("some_new_provider", { widget_host: "widgets.example.com" });

    expect(rows).toEqual([{ name: "widget_host", label: "widget_host", value: "widgets.example.com", secret: false }]);
  });
});

describe("formatParamValue", () => {
  it("joins arrays, serialises objects and stringifies scalars", () => {
    expect(formatParamValue(["a", "b"])).toBe("a, b");
    expect(formatParamValue({ vector: 0.7, text: 0.3 })).toBe('{"vector":0.7,"text":0.3}');
    expect(formatParamValue(1536)).toBe("1536");
    expect(formatParamValue(true)).toBe("true");
  });
});
