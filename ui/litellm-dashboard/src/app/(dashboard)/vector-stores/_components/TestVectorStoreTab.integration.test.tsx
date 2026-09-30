import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreSearchCall } from "@/components/networking";
import { VectorStore } from "@/components/vector_store_management/types";

import TestVectorStoreTab from "./TestVectorStoreTab";

vi.mock("@/components/networking", () => ({
  vectorStoreSearchCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

const mockSearch = vi.mocked(vectorStoreSearchCall);

const MONGODB_STORE: VectorStore = {
  vector_store_id: "vs_mongo",
  custom_llm_provider: "mongodb",
  vector_store_name: "MongoDB store",
  litellm_params: {
    mongodb_filter_fields: ["metadata.department"],
    mongodb_hybrid_search: true,
    mongodb_text_index: "policy_text_index",
  },
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const OPENAI_STORE: VectorStore = {
  vector_store_id: "vs_openai",
  custom_llm_provider: "openai",
  vector_store_name: "OpenAI store",
  litellm_params: {},
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const runSearch = async (user: ReturnType<typeof userEvent.setup>) => {
  fireEvent.change(screen.getByPlaceholderText(/enter your search query/i), { target: { value: "hello" } });
  await user.click(screen.getByRole("button", { name: /^search$/i }));
};

const lastSearchOptions = () => mockSearch.mock.calls[mockSearch.mock.calls.length - 1][3];

describe("TestVectorStoreTab switching stores", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearch.mockResolvedValue({ object: "vector_store.search_results.page", data: [] });
  });

  it("drops hybrid ranking and filters from the previous store once a different store is selected", async () => {
    const user = userEvent.setup();
    render(
      <TestVectorStoreTab
        accessToken="test-token"
        vectorStores={[MONGODB_STORE, OPENAI_STORE]}
        preselectedVectorStoreId="vs_mongo"
      />,
    );

    await user.click(screen.getByRole("switch", { name: /Hybrid/ }));
    await user.click(screen.getByRole("button", { name: /Filters/ }));
    fireEvent.change(screen.getByLabelText("Filter value"), { target: { value: "hr" } });
    await runSearch(user);

    expect(lastSearchOptions()).toMatchObject({ ranking_options: { ranker: "hybrid" } });
    expect(lastSearchOptions()).toHaveProperty("filters");

    await user.click(screen.getByPlaceholderText("Select a vector store"));
    await user.click(await screen.findByText("OpenAI store"));
    await runSearch(user);

    expect(lastSearchOptions()).not.toHaveProperty("ranking_options");
    expect(lastSearchOptions()).not.toHaveProperty("filters");
  });
});
