import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreSearchCall } from "@/components/networking";

import { VectorStoreTester } from "./VectorStoreTester";
import { toast } from "@/lib/toast";

vi.mock("@/components/networking", () => ({
  vectorStoreSearchCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

const mockWarning = vi.mocked(toast.warning);
const mockFromBackend = vi.mocked(toast.fromError);

const mockSearch = vi.mocked(vectorStoreSearchCall);

const searchResponse = {
  object: "vector_store.search_results.page",
  search_query: "hello",
  data: [
    {
      score: 0.91234,
      content: [{ text: "the quick brown fox", type: "text" }],
      file_id: "file-1",
      filename: "notes.txt",
      attributes: { source: "manual", chunk_index: 2 },
    },
  ],
};

const MONGODB_PARAMS = {
  mongodb_filter_fields: ["metadata.department", "metadata.year"],
};

const EMPTY_STATE = "Test your vector store by entering a search query below";

const renderTester = (litellmParams?: Record<string, unknown>) =>
  render(<VectorStoreTester vectorStoreId="vs_123" accessToken="sk-test" litellmParams={litellmParams} />);

const queryInput = () => screen.getByPlaceholderText(/enter your search query/i);
const searchButton = () => screen.getByRole("button", { name: /^search$/i });
const searchOptions = () => mockSearch.mock.calls[0][3];

describe("VectorStoreTester", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearch.mockResolvedValue(searchResponse);
  });

  it("shows the empty state before any search has run", () => {
    renderTester();
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /clear history/i })).not.toBeInTheDocument();
  });

  it("does not search until a non-blank query is entered", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.click(searchButton());
    expect(mockSearch).not.toHaveBeenCalled();

    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    expect(mockSearch.mock.calls[0].slice(0, 3)).toEqual(["sk-test", "vs_123", "hello"]);
  });

  it("warns instead of searching when the query is only whitespace", async () => {
    const user = userEvent.setup();
    renderTester();

    fireEvent.change(queryInput(), { target: { value: "   " } });
    await user.type(queryInput(), "{Enter}");

    expect(mockWarning).toHaveBeenCalledWith("Please enter a search query");
    expect(mockSearch).not.toHaveBeenCalled();
  });

  it("submits on Enter but not on Shift+Enter", async () => {
    const user = userEvent.setup();
    renderTester();

    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.type(queryInput(), "{Shift>}{Enter}{/Shift}");
    expect(mockSearch).not.toHaveBeenCalled();

    await user.type(queryInput(), "{Enter}");
    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
  });
});

describe("VectorStoreTester results", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearch.mockResolvedValue(searchResponse);
  });

  const runSearch = async (user: ReturnType<typeof userEvent.setup>) => {
    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());
  };

  it("titles a result with its filename and chunk index, and shows the text without expanding", async () => {
    const user = userEvent.setup();
    renderTester();

    await runSearch(user);

    expect(await screen.findByText("notes.txt · chunk 2")).toBeInTheDocument();
    expect(screen.getByText("the quick brown fox")).toBeInTheDocument();
  });

  it("renders the score both as a number and as a bar relative to the top match", async () => {
    const user = userEvent.setup();
    mockSearch.mockResolvedValue({
      ...searchResponse,
      data: [searchResponse.data[0], { ...searchResponse.data[0], score: 0.45617, filename: "second.txt" }],
    });
    renderTester();

    await runSearch(user);

    expect(await screen.findByText("0.9123")).toBeInTheDocument();
    expect(screen.getByText("0.4562")).toBeInTheDocument();
    const bars = screen.getAllByRole("meter");
    expect(bars[0]).toHaveAttribute("aria-valuenow", "100");
    expect(bars[1]).toHaveAttribute("aria-valuenow", "50");
  });

  it("shows attributes as chips but keeps the chunk index out of them", async () => {
    const user = userEvent.setup();
    renderTester();

    await runSearch(user);

    expect(await screen.findByText("source")).toBeInTheDocument();
    expect(screen.getByText("manual")).toBeInTheDocument();
    expect(screen.queryByText("chunk_index")).not.toBeInTheDocument();
  });

  it("breaks a hybrid result down by its vector and text contributions", async () => {
    const user = userEvent.setup();
    mockSearch.mockResolvedValue({
      ...searchResponse,
      data: [{ ...searchResponse.data[0], score_details: { vector: 0.0163, text: 0.0084 } }],
    });
    renderTester();

    await runSearch(user);

    expect(await screen.findByText("vector 0.0163 · text 0.0084")).toBeInTheDocument();
  });

  it("shows the backend error in the history and keeps the query for a retry", async () => {
    const user = userEvent.setup();
    const errorBody = '{"error":{"message":"OpenAIException - api_key is required"}}';
    mockSearch.mockRejectedValue(new Error(errorBody));
    renderTester();

    await runSearch(user);

    await waitFor(() => expect(mockFromBackend).toHaveBeenCalledWith(errorBody));
    expect(screen.getByText(errorBody)).toBeInTheDocument();
    expect(queryInput()).toHaveValue("hello");
  });

  it("suggests loosening the search rather than reporting an error for an empty result set", async () => {
    const user = userEvent.setup();
    mockSearch.mockResolvedValue({ object: "vector_store.search_results.page", search_query: "hello", data: [] });
    renderTester();

    await runSearch(user);

    expect(await screen.findByText(/Loosen the filters or lower the score threshold/)).toBeInTheDocument();
  });

  it("clears the search history", async () => {
    const user = userEvent.setup();
    renderTester();

    await runSearch(user);
    expect(await screen.findByText("notes.txt · chunk 2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /clear history/i }));

    expect(screen.queryByText("notes.txt · chunk 2")).not.toBeInTheDocument();
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
  });
});

describe("VectorStoreTester options", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearch.mockResolvedValue(searchResponse);
  });

  it("sends the default result count and nothing else the admin did not set", async () => {
    const user = userEvent.setup();
    renderTester();

    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    expect(searchOptions()).toEqual({ max_num_results: 5 });
  });

  it("sends a score threshold as a number under ranking_options", async () => {
    const user = userEvent.setup();
    renderTester();

    fireEvent.change(screen.getByLabelText(/Score threshold/), { target: { value: "0.7" } });
    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    expect(searchOptions()).toMatchObject({ ranking_options: { score_threshold: 0.7 } });
  });

  it("hides the hybrid toggle for a store that is not configured for hybrid search", () => {
    renderTester(MONGODB_PARAMS);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
  });

  it("offers hybrid and sends the hybrid ranker once the store has a text index", async () => {
    const user = userEvent.setup();
    renderTester({ ...MONGODB_PARAMS, mongodb_text_index: "policy_text_index" });

    await user.click(screen.getByRole("switch", { name: /Hybrid/ }));
    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    expect(searchOptions()).toMatchObject({ ranking_options: { ranker: "hybrid" } });
  });

  it("sends a filter built from the store's own filter fields", async () => {
    const user = userEvent.setup();
    renderTester(MONGODB_PARAMS);

    await user.click(screen.getByRole("button", { name: /Filters/ }));
    fireEvent.change(screen.getByLabelText("Filter value"), { target: { value: "hr" } });
    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    expect(searchOptions()).toMatchObject({
      filters: { type: "eq", key: "metadata.department", value: "hr" },
    });
  });

  it("leaves filters out entirely while the row is still half-typed", async () => {
    const user = userEvent.setup();
    renderTester(MONGODB_PARAMS);

    fireEvent.change(queryInput(), { target: { value: "hello" } });
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    expect(searchOptions()).not.toHaveProperty("filters");
  });
});
