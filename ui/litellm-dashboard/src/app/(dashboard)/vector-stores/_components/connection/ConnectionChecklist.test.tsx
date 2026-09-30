import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { VectorStoreTestConnectionResponse } from "@/components/networking";

import ConnectionChecklist from "./ConnectionChecklist";

const FAILING: VectorStoreTestConnectionResponse = {
  ok: false,
  summary: "Cannot reach the sidecar at http://127.0.0.1:8080 (ConnectError).",
  checks: [
    {
      check: "sidecar_reachable",
      status: "fail",
      message: "Cannot reach the sidecar at http://127.0.0.1:8080. Check the URL and that the container is running.",
    },
    {
      check: "embedding_model",
      status: "warn",
      message: "Embedding model returns 3072-dimensional vectors.",
      details: { model: "text-embedding-3-large", dimensions: 3072 },
    },
    { check: "hybrid_search", status: "skip", message: "Hybrid search is off for this store." },
  ],
};

describe("ConnectionChecklist", () => {
  it("leads with the failure summary so the admin sees the first thing to fix", () => {
    render(<ConnectionChecklist result={FAILING} />);

    expect(screen.getByText("Connection failed")).toBeInTheDocument();
    expect(screen.getByText(FAILING.summary as string)).toBeInTheDocument();
  });

  it("renders one row per check with its fix message", () => {
    render(<ConnectionChecklist result={FAILING} />);

    expect(screen.getByText("Sidecar reachable")).toBeInTheDocument();
    expect(screen.getByText(/Check the URL and that the container is running/)).toBeInTheDocument();
    expect(screen.getByText("Hybrid search")).toBeInTheDocument();
  });

  it("marks each row with its own status so failures stand apart from warnings and skips", () => {
    render(<ConnectionChecklist result={FAILING} />);

    expect(screen.getByLabelText("Failed")).toBeInTheDocument();
    expect(screen.getByLabelText("Warning")).toBeInTheDocument();
    expect(screen.getByLabelText("Skipped")).toBeInTheDocument();
  });

  it("hides the raw details behind Show details, and only for rows that have any", () => {
    render(<ConnectionChecklist result={FAILING} />);

    expect(screen.getAllByRole("button", { name: "Show details" })).toHaveLength(1);
    expect(screen.queryByText(/text-embedding-3-large/)).not.toBeInTheDocument();
  });

  it("reveals the details payload when Show details is clicked", async () => {
    const user = userEvent.setup();
    render(<ConnectionChecklist result={FAILING} />);

    await user.click(screen.getByRole("button", { name: "Show details" }));

    expect(screen.getByText(/text-embedding-3-large/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide details" })).toBeInTheDocument();
  });

  it("offers the curl only when the caller supplies one", () => {
    const { rerender } = render(<ConnectionChecklist result={FAILING} />);
    expect(screen.queryByRole("button", { name: "Copy as curl" })).not.toBeInTheDocument();

    rerender(<ConnectionChecklist result={FAILING} curlCommand="curl -X POST ..." />);
    expect(screen.getByRole("button", { name: "Copy as curl" })).toBeInTheDocument();
  });

  it("reports success when every check passed", () => {
    render(
      <ConnectionChecklist
        result={{
          ok: true,
          summary: "All checks passed.",
          checks: [{ check: "mongodb_ping", status: "pass", message: "MongoDB is reachable." }],
        }}
      />,
    );

    expect(screen.getByText("Connection verified")).toBeInTheDocument();
    expect(screen.getByLabelText("Passed")).toBeInTheDocument();
  });

  it("reads as success with a next-step summary when the sidecar checked out but the namespace checks were skipped", () => {
    render(
      <ConnectionChecklist
        result={{
          ok: true,
          summary: "Choose a database and collection to finish the checks.",
          checks: [
            { check: "sidecar_auth", status: "pass", message: "Authenticated with sidecar v0.2.0." },
            {
              check: "mongodb_collection",
              status: "skip",
              message: "Choose a database and collection to check the index.",
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("Sidecar connected")).toBeInTheDocument();
    expect(screen.queryByText("Connection verified")).not.toBeInTheDocument();
    expect(screen.getByText("Choose a database and collection to finish the checks.")).toBeInTheDocument();
    expect(screen.getByText("Choose a database and collection to check the index.")).toBeInTheDocument();
    expect(screen.getByLabelText("Skipped")).toBeInTheDocument();
  });

  it("reads as verified, not 'choose a database', when the namespace is already chosen and only the index checks are skipped", () => {
    render(
      <ConnectionChecklist
        result={{
          ok: true,
          summary: "Connected. The index is created when you save this store.",
          checks: [
            { check: "sidecar_auth", status: "pass", message: "Authenticated with sidecar v0.2.0." },
            {
              check: "mongodb_collection",
              status: "pass",
              message: "Collection 'policies' exists with 120 documents.",
            },
            { check: "mongodb_index", status: "skip", message: "The index is created when you save this store." },
          ],
        }}
      />,
    );

    expect(screen.getByText("Connection verified")).toBeInTheDocument();
    expect(screen.queryByText("Sidecar connected")).not.toBeInTheDocument();
    expect(screen.getByText("The index is created when you save this store.")).toBeInTheDocument();
  });

  it("stays a failure when a real failure is reported alongside skipped rows", () => {
    render(
      <ConnectionChecklist
        result={{
          ok: false,
          summary: "Collection 'policies' does not exist.",
          checks: [
            { check: "mongodb_collection", status: "fail", message: "Collection 'policies' does not exist." },
            { check: "mongodb_index", status: "skip", message: "The index is created when you save this store." },
          ],
        }}
      />,
    );

    expect(screen.getByText("Connection failed")).toBeInTheDocument();
    expect(screen.getAllByText("Collection 'policies' does not exist.")).toHaveLength(2);
  });
});
