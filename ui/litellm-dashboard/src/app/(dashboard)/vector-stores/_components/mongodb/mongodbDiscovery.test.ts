import { describe, expect, it } from "vitest";

import { discoverRequestKind, indexSummaryHint, parseIndexes, parseSuggestions } from "./mongodbDiscovery";

describe("discoverRequestKind", () => {
  it("asks the proxy for 'fields' for every field flavour, since one call answers all three", () => {
    expect(discoverRequestKind("vector_fields")).toBe("fields");
    expect(discoverRequestKind("text_fields")).toBe("fields");
    expect(discoverRequestKind("filter_fields")).toBe("fields");
    expect(discoverRequestKind("databases")).toBe("databases");
    expect(discoverRequestKind("collections")).toBe("collections");
  });
});

describe("parseSuggestions", () => {
  it("lists databases", () => {
    expect(parseSuggestions("databases", { databases: ["knowledge", "litellm_smoke"] })).toEqual([
      { value: "knowledge" },
      { value: "litellm_smoke" },
    ]);
  });

  it("shows the document count next to each collection", () => {
    const suggestions = parseSuggestions("collections", {
      mongodb_database: "knowledge",
      collections: [{ name: "policies", document_count: 3 }],
    });

    expect(suggestions).toEqual([{ value: "policies", hint: "3 documents" }]);
  });

  it("shows the inferred dimension next to a vector field", () => {
    const suggestions = parseSuggestions("vector_fields", {
      vector_fields: [{ path: "embedding", dimensions: 1536, documents: 12480 }],
    });

    expect(suggestions).toEqual([{ value: "embedding", hint: "1536-dim, 12,480 documents" }]);
  });

  it("describes a text field by its average length", () => {
    const suggestions = parseSuggestions("text_fields", {
      text_fields: [{ path: "text", average_length: 353, documents: 1 }],
    });

    expect(suggestions).toEqual([{ value: "text", hint: "~353 chars, 1 document" }]);
  });

  it("lists filter candidates", () => {
    expect(parseSuggestions("filter_fields", { filter_candidates: ["file_id", "metadata.department"] })).toEqual([
      { value: "file_id" },
      { value: "metadata.department" },
    ]);
  });

  it("returns nothing for a payload that is not an object, instead of throwing", () => {
    expect(parseSuggestions("databases", null)).toEqual([]);
    expect(parseSuggestions("collections", "nope")).toEqual([]);
  });

  it("skips entries an older sidecar left unnamed", () => {
    expect(parseSuggestions("collections", { collections: [{ document_count: 3 }, { name: "policies" }] })).toEqual([
      { value: "policies" },
    ]);
  });
});

describe("parseIndexes", () => {
  const PAYLOAD = {
    indexes: [
      {
        name: "policy_vector_index",
        type: "vectorSearch",
        status: "ready",
        queryable: true,
        vector_path: "embedding",
        dimensions: 1536,
        similarity: "cosine",
        filter_fields: ["metadata.department"],
      },
    ],
  };

  it("reads the index definition the dialog needs to prefill the form", () => {
    expect(parseIndexes(PAYLOAD)).toEqual([
      {
        name: "policy_vector_index",
        queryable: true,
        status: "ready",
        dimensions: 1536,
        similarity: "cosine",
        vectorPath: "embedding",
        filterFields: ["metadata.department"],
      },
    ]);
  });

  it("summarises a ready index in one line", () => {
    expect(indexSummaryHint(parseIndexes(PAYLOAD)[0])).toBe("1536-dim, cosine, ready");
  });

  it("shows the build status instead of 'ready' while MongoDB is still building", () => {
    const building = parseIndexes({
      indexes: [{ name: "new_index", status: "building", queryable: false, dimensions: 1536 }],
    });

    expect(indexSummaryHint(building[0])).toBe("1536-dim, building");
  });
});
