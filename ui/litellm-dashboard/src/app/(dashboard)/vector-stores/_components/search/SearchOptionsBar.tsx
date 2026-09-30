"use client";

import React from "react";
import { ChevronDown, Filter } from "lucide-react";

import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { Field, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

import SearchFilterBuilder from "./SearchFilterBuilder";
import type { FilterGroup } from "./searchFilters";
import { DEFAULT_MAX_NUM_RESULTS } from "./searchRequest";

export interface SearchOptionsState {
  /** Undefined until the admin moves the slider, so the provider's own default is sent as-is. */
  maxNumResults: number | undefined;
  scoreThreshold: string;
  hybrid: boolean;
  filters: FilterGroup;
}

interface SearchOptionsBarProps {
  value: SearchOptionsState;
  onChange: (next: SearchOptionsState) => void;
  /** Filter paths the index declares, from mongodb_filter_fields. */
  filterKeys: readonly string[];
  /** Hybrid is only offered when the store is configured for it. */
  hybridSupported: boolean;
}

export const SearchOptionsBar: React.FC<SearchOptionsBarProps> = ({ value, onChange, filterKeys, hybridSupported }) => (
  <div className="flex flex-col gap-3 border-b p-4">
    <div className="flex flex-wrap items-end gap-6">
      <Field className="w-64">
        <FieldLabel htmlFor="search-max-results">
          {labelWithHint("Max results", "How many matches the provider returns, at most")}
        </FieldLabel>
        <div className="flex items-center gap-3">
          <Slider
            id="search-max-results"
            min={1}
            max={50}
            step={1}
            value={value.maxNumResults ?? DEFAULT_MAX_NUM_RESULTS}
            onValueChange={(next: number | readonly number[]) =>
              onChange({ ...value, maxNumResults: Array.isArray(next) ? next[0] : (next as number) })
            }
          />
          <span className="w-6 text-sm tabular-nums text-muted-foreground">
            {value.maxNumResults ?? DEFAULT_MAX_NUM_RESULTS}
          </span>
        </div>
      </Field>

      <Field className="w-40">
        <FieldLabel htmlFor="search-score-threshold">
          {labelWithHint("Score threshold", "Drop results scoring below this. Leave blank to keep them all")}
        </FieldLabel>
        <Input
          id="search-score-threshold"
          type="number"
          inputMode="decimal"
          step={0.05}
          placeholder="none"
          value={value.scoreThreshold}
          onChange={(event) => onChange({ ...value, scoreThreshold: event.target.value })}
        />
      </Field>

      {hybridSupported && (
        <Field className="w-40">
          <FieldLabel htmlFor="search-hybrid">
            {labelWithHint("Hybrid", "Blend vector similarity with keyword matching for this query")}
          </FieldLabel>
          <Switch
            id="search-hybrid"
            checked={value.hybrid}
            onCheckedChange={(checked: boolean) => onChange({ ...value, hybrid: checked })}
          />
        </Field>
      )}
    </div>

    <Collapsible>
      <CollapsibleTrigger
        render={
          <Button type="button" variant="ghost" size="sm" className="gap-1.5 px-0">
            <Filter className="size-4" />
            Filters
            <ChevronDown className="size-4" />
          </Button>
        }
      />
      <CollapsibleContent className="pt-2">
        <SearchFilterBuilder
          value={value.filters}
          filterKeys={filterKeys}
          onChange={(filters) => onChange({ ...value, filters })}
        />
      </CollapsibleContent>
    </Collapsible>
  </div>
);

export default SearchOptionsBar;
