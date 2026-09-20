"use client";

import React from "react";

import { Badge } from "@/components/ui/badge";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";

export interface SearchResultContent {
  text?: string;
  type?: string;
}

export interface SearchResult {
  score: number;
  content?: SearchResultContent[];
  file_id?: string;
  filename?: string;
  attributes?: Record<string, unknown>;
  score_details?: Record<string, unknown>;
}

/** MongoDB's chunk index rides along in attributes; it belongs in the title, not in the chips. */
const TITLE_ATTRIBUTES = new Set(["chunk_index"]);

const resultTitle = (result: SearchResult, index: number): string => {
  const name = result.filename ?? result.file_id ?? `Result ${index + 1}`;
  const chunk = result.attributes?.chunk_index;
  return typeof chunk === "number" ? `${name} · chunk ${chunk}` : name;
};

const formatAttribute = (value: unknown): string =>
  value !== null && typeof value === "object" ? JSON.stringify(value) : String(value);

/** A hybrid result carries the per-pipeline contributions that produced its fused rank. */
export const hybridContributions = (
  scoreDetails: Record<string, unknown> | undefined,
): readonly { label: string; value: number }[] => {
  if (!scoreDetails) return [];
  return Object.entries(scoreDetails)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .map(([label, value]) => ({ label, value: value as number }));
};

interface SearchResultCardProps {
  result: SearchResult;
  index: number;
  /** The top score in this response, so the bars are relative to the best match. */
  topScore: number;
}

export const SearchResultCard: React.FC<SearchResultCardProps> = ({ result, index, topScore }) => {
  const ratio = topScore > 0 ? Math.min(result.score / topScore, 1) : 0;
  const contributions = hybridContributions(result.score_details);
  const chips = Object.entries(result.attributes ?? {}).filter(([key]) => !TITLE_ATTRIBUTES.has(key));

  return (
    <li className="space-y-2 rounded-lg border bg-card p-3">
      <div className="flex items-start justify-between gap-3">
        <p className="min-w-0 flex-1 truncate text-sm font-medium" title={resultTitle(result, index)}>
          {resultTitle(result, index)}
        </p>
        <span className="text-sm tabular-nums text-muted-foreground">{result.score.toFixed(4)}</span>
      </div>

      <Meter value={ratio * 100} aria-label={`Relative score for ${resultTitle(result, index)}`}>
        <MeterTrack>
          <MeterIndicator style={{ width: `${ratio * 100}%` }} />
        </MeterTrack>
      </Meter>

      {contributions.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {contributions.map((part) => `${part.label} ${part.value.toFixed(4)}`).join(" · ")}
        </p>
      )}

      {result.content?.map((content, contentIndex) => (
        <p
          key={contentIndex}
          className="max-h-40 overflow-y-auto rounded-sm border bg-muted/40 p-2 text-sm whitespace-pre-wrap"
        >
          {content.text}
        </p>
      ))}

      {chips.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {chips.map(([key, value]) => (
            <Badge key={key} variant="outline" className="font-normal">
              <span className="text-muted-foreground">{key}</span>
              <span className="ml-1">{formatAttribute(value)}</span>
            </Badge>
          ))}
        </div>
      )}
    </li>
  );
};

export default SearchResultCard;
