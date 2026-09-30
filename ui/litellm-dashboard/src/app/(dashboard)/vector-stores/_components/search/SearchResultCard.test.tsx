import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import SearchResultCard, { filterPathForAttribute, type SearchResult } from "./SearchResultCard";

const baseResult = (attributes: Record<string, unknown>): SearchResult => ({
  score: 0.9,
  file_id: "file-1",
  filename: "policy.txt",
  attributes,
});

describe("filterPathForAttribute", () => {
  it("resolves a flattened attribute key to the declared filter field it corresponds to", () => {
    expect(filterPathForAttribute("department", ["metadata.department", "metadata.year"])).toBe("metadata.department");
  });

  it("returns nothing when no declared filter field ends in that key", () => {
    expect(filterPathForAttribute("author", ["metadata.department"])).toBeNull();
  });

  it("never relabels the reserved chunk fields, even if a filter field happens to share the name", () => {
    expect(filterPathForAttribute("content_type", ["metadata.content_type"])).toBeNull();
    expect(filterPathForAttribute("ingested_at", ["metadata.ingested_at"])).toBeNull();
  });
});

describe("SearchResultCard chips", () => {
  it("labels a chip with the full filter path and a tooltip when it matches a declared filter field", async () => {
    const user = userEvent.setup();
    render(
      <SearchResultCard
        result={baseResult({ department: "hr" })}
        index={0}
        topScore={0.9}
        filterFields={["metadata.department"]}
      />,
    );

    expect(screen.getByText("metadata.department")).toBeInTheDocument();
    expect(screen.queryByText("department")).not.toBeInTheDocument();

    await user.hover(screen.getByText("metadata.department"));
    expect(await screen.findByText("filter on metadata.department")).toBeInTheDocument();
  });

  it("shows the raw key with no tooltip when it does not match any declared filter field", () => {
    render(
      <SearchResultCard
        result={baseResult({ source: "manual" })}
        index={0}
        topScore={0.9}
        filterFields={["metadata.department"]}
      />,
    );

    expect(screen.getByText("source")).toBeInTheDocument();
  });

  it("shows reserved chunk attributes by their raw key even when filter fields share the name", () => {
    render(
      <SearchResultCard
        result={baseResult({ content_type: "text/plain" })}
        index={0}
        topScore={0.9}
        filterFields={["metadata.content_type"]}
      />,
    );

    expect(screen.getByText("content_type")).toBeInTheDocument();
  });

  it("shows the raw key when the store has no declared filter fields at all", () => {
    render(<SearchResultCard result={baseResult({ department: "hr" })} index={0} topScore={0.9} />);

    expect(screen.getByText("department")).toBeInTheDocument();
  });
});
