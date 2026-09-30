import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import SearchFilterBuilder from "./SearchFilterBuilder";
import { newCondition, newGroup } from "./searchFilters";

const FILTER_KEYS = ["metadata.department", "metadata.year"];

describe("SearchFilterBuilder", () => {
  it("auto-corrects a bare key an admin types to the full declared filter path", () => {
    const onChange = vi.fn();
    render(<SearchFilterBuilder value={newGroup([newCondition("")])} filterKeys={FILTER_KEYS} onChange={onChange} />);

    fireEvent.change(screen.getByLabelText("Filter field"), { target: { value: "department" } });

    expect(onChange.mock.calls[0][0].children[0]).toMatchObject({ key: "metadata.department" });
  });

  it("leaves a typed field alone when its metadata. form is not a declared filter field", () => {
    const onChange = vi.fn();
    render(<SearchFilterBuilder value={newGroup([newCondition("")])} filterKeys={FILTER_KEYS} onChange={onChange} />);

    fireEvent.change(screen.getByLabelText("Filter field"), { target: { value: "region" } });

    expect(onChange.mock.calls[0][0].children[0]).toMatchObject({ key: "region" });
  });

  it("offers the store's declared filter fields as suggestions on the field input", () => {
    render(
      <SearchFilterBuilder
        value={newGroup([newCondition("metadata.department")])}
        filterKeys={FILTER_KEYS}
        onChange={vi.fn()}
      />,
    );

    const input = screen.getByLabelText("Filter field");
    const datalistId = input.getAttribute("list");
    expect(datalistId).toBeTruthy();
    const options = Array.from(document.querySelectorAll(`#${datalistId} option`)).map((option) =>
      option.getAttribute("value"),
    );
    expect(options).toEqual(FILTER_KEYS);
  });

  it("does not render a datalist when the store declares no filter fields", () => {
    render(<SearchFilterBuilder value={newGroup([newCondition("")])} filterKeys={[]} onChange={vi.fn()} />);

    expect(screen.getByLabelText("Filter field")).not.toHaveAttribute("list");
    expect(screen.getByText(/This store lists no filter fields/)).toBeInTheDocument();
  });
});
