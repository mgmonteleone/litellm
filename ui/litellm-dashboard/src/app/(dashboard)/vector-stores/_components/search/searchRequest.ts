import type { VectorStoreFilter } from "./searchFilters";

export interface SearchOptions {
  maxNumResults?: number;
  scoreThreshold?: number;
  hybrid?: boolean;
  filters?: VectorStoreFilter;
}

export interface SearchRequestBody {
  query: string;
  max_num_results?: number;
  filters?: VectorStoreFilter;
  ranking_options?: { score_threshold?: number; ranker?: "hybrid" };
}

const rankingOptions = (options: SearchOptions): SearchRequestBody["ranking_options"] => {
  const ranking = {
    ...(typeof options.scoreThreshold === "number" ? { score_threshold: options.scoreThreshold } : {}),
    ...(options.hybrid ? { ranker: "hybrid" as const } : {}),
  };
  return Object.keys(ranking).length > 0 ? ranking : undefined;
};

/** Only the options the admin actually set are sent, so the provider keeps its own defaults. */
export const buildSearchRequest = (query: string, options: SearchOptions = {}): SearchRequestBody => {
  const ranking = rankingOptions(options);
  return {
    query,
    ...(typeof options.maxNumResults === "number" ? { max_num_results: options.maxNumResults } : {}),
    ...(options.filters ? { filters: options.filters } : {}),
    ...(ranking ? { ranking_options: ranking } : {}),
  };
};

const shellQuote = (value: string): string => `'${value.replaceAll("'", `'\\''`)}'`;

export const searchAsCurl = (baseUrl: string, vectorStoreId: string, body: SearchRequestBody): string =>
  [
    `curl -X POST ${shellQuote(`${baseUrl}/v1/vector_stores/${vectorStoreId}/search`)}`,
    `  -H 'Authorization: Bearer $LITELLM_API_KEY'`,
    `  -H 'Content-Type: application/json'`,
    `  -d ${shellQuote(JSON.stringify(body, null, 2))}`,
  ].join(" \\\n");

/** Python literals differ from JSON for the three scalars a filter can carry. */
const toPythonLiteral = (value: unknown, indent: number): string => {
  const pad = " ".repeat(indent);
  const inner = " ".repeat(indent + 4);
  if (value === null) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    return `[\n${value.map((entry) => `${inner}${toPythonLiteral(entry, indent + 4)}`).join(",\n")},\n${pad}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) return "{}";
  return `{\n${entries
    .map(([key, entry]) => `${inner}${JSON.stringify(key)}: ${toPythonLiteral(entry, indent + 4)}`)
    .join(",\n")},\n${pad}}`;
};

export const searchAsPython = (baseUrl: string, vectorStoreId: string, body: SearchRequestBody): string => {
  const { query, ...rest } = body;
  const extras = Object.entries(rest).map(([key, value]) => `    ${key}=${toPythonLiteral(value, 4)},`);
  return [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url=${JSON.stringify(`${baseUrl}/v1`)}, api_key="LITELLM_API_KEY")`,
    "",
    "results = client.vector_stores.search(",
    `    vector_store_id=${JSON.stringify(vectorStoreId)},`,
    `    query=${JSON.stringify(query)},`,
    ...extras,
    ")",
    "",
    "for result in results.data:",
    "    print(result.score, result.filename)",
  ].join("\n");
};
