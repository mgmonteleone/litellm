"use client";

import React from "react";
import { Plus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/cva.config";

import {
  appendChild,
  FILTER_OPERATORS,
  isGroup,
  isListOperator,
  newCondition,
  newGroup,
  OPERATOR_LABELS,
  removeNode,
  replaceNode,
  type FilterCondition,
  type FilterGroup,
  type FilterJoin,
  type FilterOperator,
} from "./searchFilters";

interface ConditionRowProps {
  condition: FilterCondition;
  filterKeys: readonly string[];
  onChange: (next: FilterCondition) => void;
  onRemove: () => void;
}

const ConditionRow: React.FC<ConditionRowProps> = ({ condition, filterKeys, onChange, onRemove }) => (
  <div className="flex flex-wrap items-center gap-2">
    {filterKeys.length > 0 ? (
      <Select
        value={condition.key}
        onValueChange={(value: string | null) => value !== null && onChange({ ...condition, key: value })}
      >
        <SelectTrigger className="w-56" aria-label="Filter field">
          <SelectValue placeholder="Field" />
        </SelectTrigger>
        <SelectContent>
          {filterKeys.map((key) => (
            <SelectItem key={key} value={key}>
              {key}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    ) : (
      <Input
        className="w-56"
        aria-label="Filter field"
        placeholder="metadata.department"
        value={condition.key}
        onChange={(event) => onChange({ ...condition, key: event.target.value })}
      />
    )}

    <Select
      value={condition.operator}
      onValueChange={(value: string | null) =>
        value !== null && onChange({ ...condition, operator: value as FilterOperator })
      }
    >
      <SelectTrigger className="w-36" aria-label="Filter operator">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {FILTER_OPERATORS.map((operator) => (
          <SelectItem key={operator} value={operator}>
            {OPERATOR_LABELS[operator]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>

    <Input
      className="w-48 flex-1"
      aria-label="Filter value"
      placeholder={isListOperator(condition.operator) ? "hr, finance" : "hr"}
      value={condition.value}
      onChange={(event) => onChange({ ...condition, value: event.target.value })}
    />

    <Button type="button" variant="ghost" size="icon-sm" aria-label="Remove filter" onClick={onRemove}>
      <X className="size-4" />
    </Button>
  </div>
);

interface GroupEditorProps {
  group: FilterGroup;
  filterKeys: readonly string[];
  depth: number;
  onChange: (next: FilterGroup) => void;
  onRemove?: () => void;
}

const GroupEditor: React.FC<GroupEditorProps> = ({ group, filterKeys, depth, onChange, onRemove }) => (
  <div className={cn("flex flex-col gap-2", depth > 0 && "rounded-md border border-dashed p-3")}>
    <div className="flex items-center gap-2">
      <Select
        value={group.join}
        onValueChange={(value: string | null) => value !== null && onChange({ ...group, join: value as FilterJoin })}
      >
        <SelectTrigger size="sm" className="w-24" aria-label="Match">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="and">Match all</SelectItem>
          <SelectItem value="or">Match any</SelectItem>
        </SelectContent>
      </Select>
      <span className="text-xs text-muted-foreground">
        {group.join === "and" ? "every condition below must hold" : "any one condition below is enough"}
      </span>
      {onRemove && (
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Remove group" onClick={onRemove}>
          <X className="size-4" />
        </Button>
      )}
    </div>

    {group.children.map((child) =>
      isGroup(child) ? (
        <GroupEditor
          key={child.id}
          group={child}
          filterKeys={filterKeys}
          depth={depth + 1}
          onChange={(next) => onChange(replaceNode(group, child.id, () => next))}
          onRemove={() => onChange(removeNode(group, child.id))}
        />
      ) : (
        <ConditionRow
          key={child.id}
          condition={child}
          filterKeys={filterKeys}
          onChange={(next) => onChange(replaceNode(group, child.id, () => next))}
          onRemove={() => onChange(removeNode(group, child.id))}
        />
      ),
    )}

    <div className="flex gap-2">
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => onChange(appendChild(group, group.id, newCondition(filterKeys[0] ?? "")))}
      >
        <Plus className="size-4" />
        Condition
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={() => onChange(appendChild(group, group.id, newGroup([newCondition(filterKeys[0] ?? "")])))}
      >
        <Plus className="size-4" />
        Group
      </Button>
    </div>
  </div>
);

interface SearchFilterBuilderProps {
  value: FilterGroup;
  /** Fields the index can filter on; empty means free text, since MongoDB rejects the rest anyway. */
  filterKeys: readonly string[];
  onChange: (next: FilterGroup) => void;
}

export const SearchFilterBuilder: React.FC<SearchFilterBuilderProps> = ({ value, filterKeys, onChange }) => (
  <div className="flex flex-col gap-2">
    <GroupEditor group={value} filterKeys={filterKeys} depth={0} onChange={onChange} />
    {filterKeys.length === 0 && (
      <p className="text-xs text-muted-foreground">
        This store lists no filter fields, so type the path yourself. MongoDB only filters on fields the index declares.
      </p>
    )}
  </div>
);

export default SearchFilterBuilder;
