import { describe, expect, it } from "vitest";

import type { VectorStoreTestConnectionResponse } from "@/components/networking";

import { connectionChips } from "./connectionStatus";

const withIndexStatus = (status: string): VectorStoreTestConnectionResponse => ({
  ok: true,
  summary: "All checks passed.",
  checks: [
    {
      check: "mongodb_index",
      status: "pass",
      message: `Index 'policy_vector_index' is ${status}.`,
      details: { status, queryable: status === "ready" },
    },
  ],
});

describe("connectionChips", () => {
  it("says the store has not been tested rather than implying it is healthy", () => {
    expect(connectionChips(null)).toEqual([{ label: "Not tested", tone: "neutral" }]);
  });

  it("reports the verdict with the summary as its tooltip", () => {
    const [verdict] = connectionChips({ ok: false, summary: "Cannot reach the sidecar.", checks: [] });

    expect(verdict).toEqual({ label: "Failed", tone: "error", tooltip: "Cannot reach the sidecar." });
  });

  it("adds a ready index chip in the success tone", () => {
    expect(connectionChips(withIndexStatus("ready"))[1]).toMatchObject({ label: "Index ready", tone: "success" });
  });

  it("warns while the index is still building, since search will not work yet", () => {
    expect(connectionChips(withIndexStatus("building"))[1]).toMatchObject({ label: "Index building", tone: "warning" });
  });

  it("uses the error tone for a failed index build", () => {
    expect(connectionChips(withIndexStatus("failed"))[1]).toMatchObject({ label: "Index failed", tone: "error" });
  });

  it("falls back to a neutral chip for an index state it has no tone for", () => {
    expect(connectionChips(withIndexStatus("quiescing"))[1]).toMatchObject({ tone: "neutral" });
  });

  it("shows only the verdict for a provider whose checklist has no index step", () => {
    const chips = connectionChips({
      ok: true,
      summary: "All checks passed.",
      checks: [{ check: "sidecar_reachable", status: "pass", message: "Ready." }],
    });

    expect(chips).toHaveLength(1);
  });
});
