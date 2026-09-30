import { describe, expect, it } from "vitest";

import { buildSearchRequest, searchAsCurl, searchAsPython } from "./searchRequest";

const BASE_URL = "http://localhost:4000";

describe("buildSearchRequest", () => {
  it("sends only the query when nothing was configured", () => {
    expect(buildSearchRequest("expenses")).toEqual({ query: "expenses" });
  });

  it("puts the score threshold and the hybrid ranker together under ranking_options", () => {
    expect(buildSearchRequest("expenses", { scoreThreshold: 0.7, hybrid: true }).ranking_options).toEqual({
      score_threshold: 0.7,
      ranker: "hybrid",
    });
  });

  it("omits ranking_options entirely when neither ranking setting is on", () => {
    expect(buildSearchRequest("expenses", { maxNumResults: 5, hybrid: false })).toEqual({
      query: "expenses",
      max_num_results: 5,
    });
  });

  it("keeps a zero threshold, which is a real setting rather than an unset one", () => {
    expect(buildSearchRequest("expenses", { scoreThreshold: 0 }).ranking_options).toEqual({ score_threshold: 0 });
  });

  it("passes the filter through unchanged", () => {
    const filters = { type: "eq" as const, key: "metadata.department", value: "hr" };

    expect(buildSearchRequest("expenses", { filters }).filters).toBe(filters);
  });
});

describe("searchAsCurl", () => {
  it("targets the store's search route and carries the body", () => {
    const command = searchAsCurl(BASE_URL, "policy_vector_index", buildSearchRequest("expenses", { maxNumResults: 3 }));

    expect(command).toContain("curl -X POST 'http://localhost:4000/v1/vector_stores/policy_vector_index/search'");
    expect(command).toContain('"max_num_results": 3');
  });

  it("uses a placeholder rather than a real key", () => {
    const command = searchAsCurl(BASE_URL, "vs", buildSearchRequest("expenses"));

    expect(command).toContain("Bearer $LITELLM_API_KEY");
  });

  it("escapes a quote in the query so the copied command still parses", () => {
    const command = searchAsCurl(BASE_URL, "vs", buildSearchRequest("what's the policy"));

    expect(command).toContain(`'\\''`);
  });
});

describe("searchAsPython", () => {
  it("writes a runnable client.vector_stores.search call", () => {
    const code = searchAsPython(BASE_URL, "policy_vector_index", buildSearchRequest("expenses", { maxNumResults: 3 }));

    expect(code).toContain("from openai import OpenAI");
    expect(code).toContain('base_url="http://localhost:4000/v1"');
    expect(code).toContain('vector_store_id="policy_vector_index"');
    expect(code).toContain('query="expenses"');
    expect(code).toContain("max_num_results=3,");
  });

  it("uses Python's True rather than JSON's true inside a filter", () => {
    const code = searchAsPython(
      BASE_URL,
      "vs",
      buildSearchRequest("q", { filters: { type: "eq", key: "ok", value: true } }),
    );

    expect(code).toContain('"value": True');
    expect(code).not.toContain("true");
  });

  it("renders a nested compound filter as nested Python dicts and lists", () => {
    const code = searchAsPython(
      BASE_URL,
      "vs",
      buildSearchRequest("q", {
        filters: {
          type: "and",
          filters: [
            { type: "eq", key: "dept", value: "hr" },
            { type: "in", key: "year", value: [2025, 2026] },
          ],
        },
      }),
    );

    expect(code).toContain('"type": "and"');
    expect(code).toContain('"filters": [');
    expect(code).toContain("2025,");
  });
});
