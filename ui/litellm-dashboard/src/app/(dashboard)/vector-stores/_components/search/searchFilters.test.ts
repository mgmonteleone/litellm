import { describe, expect, it } from "vitest";

import {
  appendChild,
  coerceFilterValue,
  correctFilterKey,
  newCondition,
  newGroup,
  removeNode,
  replaceNode,
  toOpenAIFilter,
  type FilterCondition,
} from "./searchFilters";

const condition = (key: string, operator: FilterCondition["operator"], value: string): FilterCondition => ({
  ...newCondition(key),
  operator,
  value,
});

describe("correctFilterKey", () => {
  it("prefixes a bare key with metadata. when that full path is a declared filter field", () => {
    expect(correctFilterKey("department", ["metadata.department", "metadata.year"])).toBe("metadata.department");
  });

  it("leaves a key that is already a declared filter field untouched", () => {
    expect(correctFilterKey("metadata.department", ["metadata.department"])).toBe("metadata.department");
  });

  it("leaves a bare key alone when its metadata. form is not declared", () => {
    expect(correctFilterKey("department", ["metadata.year"])).toBe("department");
  });

  it("leaves a bare key alone when the store declares no filter fields at all", () => {
    expect(correctFilterKey("department", [])).toBe("department");
  });

  it("leaves a top-level (non-metadata) declared field as typed", () => {
    expect(correctFilterKey("chunk_index", ["chunk_index"])).toBe("chunk_index");
  });
});

describe("coerceFilterValue", () => {
  it("sends a numeric field as a number, since MongoDB will not match a stringified one", () => {
    expect(coerceFilterValue("2026")).toBe(2026);
    expect(coerceFilterValue(" 0.5 ")).toBe(0.5);
  });

  it("sends the three JSON scalars rather than their text", () => {
    expect(coerceFilterValue("true")).toBe(true);
    expect(coerceFilterValue("false")).toBe(false);
    expect(coerceFilterValue("null")).toBeNull();
  });

  it("leaves anything else as text", () => {
    expect(coerceFilterValue("hr")).toBe("hr");
    expect(coerceFilterValue("2026-01")).toBe("2026-01");
  });
});

describe("toOpenAIFilter", () => {
  it("returns nothing at all for an untouched builder, so no filter is sent", () => {
    expect(toOpenAIFilter(newGroup([newCondition("metadata.department")]))).toBeUndefined();
  });

  it("drops a row whose value is still blank instead of narrowing the search silently", () => {
    const group = newGroup([condition("metadata.department", "eq", "hr"), condition("metadata.year", "eq", "")]);

    expect(toOpenAIFilter(group)).toEqual({ type: "eq", key: "metadata.department", value: "hr" });
  });

  it("drops a row with a value but no field", () => {
    expect(toOpenAIFilter(newGroup([condition("", "eq", "hr")]))).toBeUndefined();
  });

  it("collapses a single condition to a bare comparison, which the schema allows", () => {
    expect(toOpenAIFilter(newGroup([condition("metadata.year", "gte", "2026")]))).toEqual({
      type: "gte",
      key: "metadata.year",
      value: 2026,
    });
  });

  it("joins several conditions with the group's operator", () => {
    const group = { ...newGroup([condition("a", "eq", "1"), condition("b", "ne", "2")]), join: "or" as const };

    expect(toOpenAIFilter(group)).toEqual({
      type: "or",
      filters: [
        { type: "eq", key: "a", value: 1 },
        { type: "ne", key: "b", value: 2 },
      ],
    });
  });

  it("nests a group inside its parent so AND over ORs is expressible", () => {
    const inner = { ...newGroup([condition("a", "eq", "1"), condition("b", "eq", "2")]), join: "or" as const };
    const outer = newGroup([condition("dept", "eq", "hr"), inner]);

    expect(toOpenAIFilter(outer)).toEqual({
      type: "and",
      filters: [
        { type: "eq", key: "dept", value: "hr" },
        {
          type: "or",
          filters: [
            { type: "eq", key: "a", value: 1 },
            { type: "eq", key: "b", value: 2 },
          ],
        },
      ],
    });
  });

  it("splits a list operator's value on commas and coerces each entry", () => {
    expect(toOpenAIFilter(newGroup([condition("dept", "in", "hr, finance , 2026")]))).toEqual({
      type: "in",
      key: "dept",
      value: ["hr", "finance", 2026],
    });
  });

  it("drops a list operator whose value holds nothing but separators", () => {
    expect(toOpenAIFilter(newGroup([condition("dept", "nin", " , , ")]))).toBeUndefined();
  });

  it("omits an empty nested group rather than sending an empty filters array", () => {
    const outer = newGroup([condition("dept", "eq", "hr"), newGroup([])]);

    expect(toOpenAIFilter(outer)).toEqual({ type: "eq", key: "dept", value: "hr" });
  });
});

describe("tree edits", () => {
  it("replaces a nested condition without touching its siblings", () => {
    const target = condition("a", "eq", "1");
    const sibling = condition("b", "eq", "2");
    const tree = newGroup([sibling, newGroup([target])]);

    const updated = replaceNode(tree, target.id, () => ({ ...target, value: "changed" }));

    expect(toOpenAIFilter(updated)).toMatchObject({
      filters: [
        { key: "b", value: 2 },
        { key: "a", value: "changed" },
      ],
    });
  });

  it("removes a nested condition", () => {
    const target = condition("a", "eq", "1");
    const tree = newGroup([condition("b", "eq", "2"), newGroup([target])]);

    expect(toOpenAIFilter(removeNode(tree, target.id))).toEqual({ type: "eq", key: "b", value: 2 });
  });

  it("appends to the group that was asked for, not to the root", () => {
    const inner = newGroup([condition("a", "eq", "1")]);
    const tree = newGroup([inner]);

    const updated = appendChild(tree, inner.id, condition("b", "eq", "2"));

    expect(toOpenAIFilter(updated)).toEqual({
      type: "and",
      filters: [
        { type: "eq", key: "a", value: 1 },
        { type: "eq", key: "b", value: 2 },
      ],
    });
  });

  it("leaves the original tree untouched, since the builder holds it in state", () => {
    const target = condition("a", "eq", "1");
    const tree = newGroup([target]);

    removeNode(tree, target.id);

    expect(tree.children).toHaveLength(1);
  });
});
