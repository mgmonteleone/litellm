/**
 * The OpenAI vector store filter schema, which the proxy translates to MQL:
 * a comparison is {type, key, value} and a compound is {type: "and" | "or", filters: [...]}.
 */

export const FILTER_OPERATORS = ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"] as const;

export type FilterOperator = (typeof FILTER_OPERATORS)[number];

export const OPERATOR_LABELS: Record<FilterOperator, string> = {
  eq: "is",
  ne: "is not",
  gt: "greater than",
  gte: "at least",
  lt: "less than",
  lte: "at most",
  in: "is one of",
  nin: "is none of",
};

/** in / nin take a list, so their value box is comma separated. */
export const isListOperator = (operator: FilterOperator): boolean => operator === "in" || operator === "nin";

export type FilterJoin = "and" | "or";

export interface FilterCondition {
  id: string;
  key: string;
  operator: FilterOperator;
  value: string;
}

export interface FilterGroup {
  id: string;
  join: FilterJoin;
  children: readonly (FilterCondition | FilterGroup)[];
}

export const isGroup = (node: FilterCondition | FilterGroup): node is FilterGroup => "children" in node;

export type ComparisonFilter = { type: FilterOperator; key: string; value: unknown };
export type CompoundFilter = { type: FilterJoin; filters: VectorStoreFilter[] };
export type VectorStoreFilter = ComparisonFilter | CompoundFilter;

/**
 * Attribute values arrive as text from an input but reach MongoDB as JSON, and a filter on a
 * numeric field only matches when the number is sent as a number.
 */
export const coerceFilterValue = (raw: string): unknown => {
  const text = raw.trim();
  if (text === "") return "";
  if (text === "true") return true;
  if (text === "false") return false;
  if (text === "null") return null;
  const asNumber = Number(text);
  return text !== "" && Number.isFinite(asNumber) ? asNumber : text;
};

const conditionFilter = (condition: FilterCondition): ComparisonFilter | null => {
  if (!condition.key.trim()) return null;
  if (isListOperator(condition.operator)) {
    const values = condition.value
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean)
      .map(coerceFilterValue);
    return values.length > 0 ? { type: condition.operator, key: condition.key.trim(), value: values } : null;
  }
  if (condition.value.trim() === "") return null;
  return { type: condition.operator, key: condition.key.trim(), value: coerceFilterValue(condition.value) };
};

/**
 * Incomplete rows are dropped rather than sent, so a half-typed filter never silently narrows
 * the search. A group with one child collapses to that child, which the schema allows.
 */
export const toOpenAIFilter = (group: FilterGroup): VectorStoreFilter | undefined => {
  const filters = group.children
    .map((child) => (isGroup(child) ? toOpenAIFilter(child) : conditionFilter(child)))
    .filter((filter): filter is VectorStoreFilter => filter !== null && filter !== undefined);
  if (filters.length === 0) return undefined;
  if (filters.length === 1) return filters[0];
  return { type: group.join, filters };
};

let nextId = 0;

export const newId = (): string => `filter-${(nextId += 1)}`;

export const newCondition = (key = ""): FilterCondition => ({ id: newId(), key, operator: "eq", value: "" });

export const newGroup = (children: readonly (FilterCondition | FilterGroup)[] = []): FilterGroup => ({
  id: newId(),
  join: "and",
  children,
});

/** Replaces one node anywhere in the tree, returning a new tree. */
export const replaceNode = (
  group: FilterGroup,
  id: string,
  update: (node: FilterCondition | FilterGroup) => FilterCondition | FilterGroup,
): FilterGroup => {
  const replaceChild = (child: FilterCondition | FilterGroup): FilterCondition | FilterGroup => {
    if (child.id === id) return update(child);
    return isGroup(child) ? replaceNode(child, id, update) : child;
  };
  return { ...group, children: group.children.map(replaceChild) };
};

export const removeNode = (group: FilterGroup, id: string): FilterGroup => ({
  ...group,
  children: group.children
    .filter((child) => child.id !== id)
    .map((child) => (isGroup(child) ? removeNode(child, id) : child)),
});

export const appendChild = (group: FilterGroup, parentId: string, child: FilterCondition | FilterGroup): FilterGroup =>
  group.id === parentId
    ? { ...group, children: [...group.children, child] }
    : {
        ...group,
        children: group.children.map((node) => (isGroup(node) ? appendChild(node, parentId, child) : node)),
      };
