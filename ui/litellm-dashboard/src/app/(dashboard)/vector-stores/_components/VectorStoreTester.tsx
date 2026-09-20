"use client";

import React, { useMemo, useState } from "react";
import { Database, Send } from "lucide-react";

import { getProxyBaseUrl, vectorStoreSearchCall } from "@/components/networking";
import CopyButton from "@/components/shared/CopyButton";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

import { normalizeLitellmParams, type LitellmParams } from "./connection/litellmParamsDisplay";
import SearchOptionsBar, { type SearchOptionsState } from "./search/SearchOptionsBar";
import SearchResultCard, { type SearchResult } from "./search/SearchResultCard";
import { newCondition, newGroup, toOpenAIFilter } from "./search/searchFilters";
import { buildSearchRequest, searchAsCurl, searchAsPython } from "./search/searchRequest";

interface VectorStoreSearchResponse {
  object?: string;
  search_query?: string;
  data?: SearchResult[];
}

interface SearchEntry {
  query: string;
  response: VectorStoreSearchResponse | null;
  error: string | null;
  timestamp: number;
}

export interface VectorStoreTesterProps {
  vectorStoreId: string;
  accessToken: string;
  /** The store's saved params; supplies the filter fields and whether hybrid is configured. */
  litellmParams?: LitellmParams;
  className?: string;
}

const asStringList = (value: unknown): readonly string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];

const SearchEntryBody: React.FC<{ entry: SearchEntry; filterFields: readonly string[] }> = ({
  entry,
  filterFields,
}) => {
  if (entry.error) {
    return <p className="text-sm break-words text-destructive">{entry.error}</p>;
  }
  const results = entry.response?.data ?? [];
  if (results.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No results. Loosen the filters or lower the score threshold.</p>
    );
  }
  return (
    <ul className="space-y-2">
      {results.map((result, index) => (
        <SearchResultCard
          key={`${result.file_id ?? index}-${index}`}
          result={result}
          index={index}
          topScore={results[0].score}
          filterFields={filterFields}
        />
      ))}
    </ul>
  );
};

const initialOptions = (filterKeys: readonly string[]): SearchOptionsState => ({
  maxNumResults: undefined,
  scoreThreshold: "",
  hybrid: false,
  filters: newGroup([newCondition(filterKeys[0] ?? "")]),
});

export const VectorStoreTester: React.FC<VectorStoreTesterProps> = ({
  vectorStoreId,
  accessToken,
  litellmParams,
  className = "",
}) => {
  const params = useMemo(() => normalizeLitellmParams(litellmParams), [litellmParams]);
  const filterKeys = useMemo(() => asStringList(params.mongodb_filter_fields), [params]);
  const hybridSupported = params.mongodb_hybrid_search === true || Boolean(params.mongodb_text_index);

  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [options, setOptions] = useState<SearchOptionsState>(() => initialOptions(filterKeys));
  const [history, setHistory] = useState<SearchEntry[]>([]);

  const threshold = Number(options.scoreThreshold);
  const requestBody = buildSearchRequest(query || "your question here", {
    maxNumResults: options.maxNumResults,
    scoreThreshold: options.scoreThreshold.trim() !== "" && Number.isFinite(threshold) ? threshold : undefined,
    hybrid: options.hybrid,
    filters: toOpenAIFilter(options.filters),
  });

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.warning("Please enter a search query");
      return;
    }
    setIsLoading(true);
    const { query: _sentQuery, ...searchOptions } = requestBody;
    try {
      const response = await vectorStoreSearchCall(accessToken, vectorStoreId, query, searchOptions);
      setHistory((previous) => [{ query, response, error: null, timestamp: Date.now() }, ...previous]);
      setQuery("");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.fromError(message);
      setHistory((previous) => [{ query, response: null, error: message, timestamp: Date.now() }, ...previous]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSearch();
    }
  };

  return (
    <Card className={`w-full py-0 shadow-md ${className}`}>
      <div className="flex h-150 flex-col">
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center">
            <Database className="mr-2 size-4 text-primary" />
            <h4 className="text-base font-medium text-foreground">Test Vector Store</h4>
          </div>
          <div className="flex items-center gap-1">
            <CopyButton value={searchAsCurl(getProxyBaseUrl(), vectorStoreId, requestBody)} label="Copy as curl" />
            <CopyButton value={searchAsPython(getProxyBaseUrl(), vectorStoreId, requestBody)} label="Copy as Python" />
            {history.length > 0 && (
              <Button variant="outline" size="sm" onClick={() => setHistory([])}>
                Clear history
              </Button>
            )}
          </div>
        </div>

        <SearchOptionsBar
          value={options}
          onChange={setOptions}
          filterKeys={filterKeys}
          hybridSupported={hybridSupported}
        />

        <div className="flex-1 overflow-auto p-4">
          {history.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
              <Database className="mb-4 size-12" />
              <p className="text-sm">Test your vector store by entering a search query below</p>
            </div>
          ) : (
            <div className="space-y-5">
              {history.map((entry) => (
                <div key={entry.timestamp} className="space-y-2">
                  <div className="flex items-baseline justify-between gap-2">
                    <p className="text-sm font-medium">{entry.query}</p>
                    <span className="text-xs text-muted-foreground">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                  </div>

                  <SearchEntryBody entry={entry} filterFields={filterKeys} />
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="my-4 flex items-center justify-center">
              <UiLoadingSpinner className="size-6 text-primary" />
            </div>
          )}
        </div>

        <div className="border-t bg-card p-4">
          <div className="flex items-end space-x-2">
            <div className="flex-1">
              <Textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your search query... (Shift+Enter for new line)"
                disabled={isLoading}
                rows={1}
                className="field-sizing-fixed max-h-24 min-h-9 resize-none"
                aria-label="Search query"
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading || !query.trim()}>
              {isLoading ? <UiLoadingSpinner className="size-4" /> : <Send className="size-4" />}
              Search
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default VectorStoreTester;
