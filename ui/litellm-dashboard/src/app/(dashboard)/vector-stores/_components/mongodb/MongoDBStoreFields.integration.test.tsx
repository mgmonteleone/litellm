import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreCreateCall, vectorStoreDiscoverCall, vectorStoreTestConnectionCall } from "@/components/networking";
import { ApiError } from "@/lib/http/client";

import VectorStoreForm from "../VectorStoreForm";

vi.mock("@/components/networking", () => ({
  vectorStoreCreateCall: vi.fn(),
  vectorStoreTestConnectionCall: vi.fn(),
  vectorStoreDiscoverCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "text-embedding-3-small", mode: "embedding" }]),
}));

const mockCreate = vi.mocked(vectorStoreCreateCall);
const mockTest = vi.mocked(vectorStoreTestConnectionCall);
const mockDiscover = vi.mocked(vectorStoreDiscoverCall);

const SIDECAR_URL = "http://127.0.0.1:8080";

const PASSING_RESULT = {
  ok: true,
  supported: true,
  custom_llm_provider: "mongodb",
  summary: "All checks passed.",
  checks: [
    {
      check: "sidecar_auth",
      status: "pass" as const,
      message: "Authenticated with sidecar v0.2.0 (MongoDB 8.2.11).",
      details: { features: { hybrid: true, discovery: true } },
    },
  ],
  details: { embedding_dimensions: 1536, mongodb: { index_dimensions: 1536 } },
};

const discoveryPayloads: Record<string, unknown> = {
  databases: { databases: ["knowledge", "litellm_smoke"] },
  collections: { collections: [{ name: "policies", document_count: 3 }] },
  fields: {
    vector_fields: [{ path: "embedding", dimensions: 1536, documents: 3 }],
    text_fields: [{ path: "text", average_length: 353, documents: 3 }],
    filter_candidates: ["metadata.department"],
  },
};

const setupUser = () => userEvent.setup({ pointerEventsCheck: 0 });

const renderForm = () =>
  render(
    <VectorStoreForm
      isVisible={true}
      onCancel={vi.fn()}
      onSuccess={vi.fn()}
      accessToken="test-token"
      credentials={[]}
    />,
  );

const chooseMongoDB = async (user: ReturnType<typeof userEvent.setup>) => {
  const trigger = screen.getAllByRole("combobox")[0];
  await user.click(trigger);
  const options = await screen.findAllByText("MongoDB");
  await user.click(options[options.length - 1]);
};

const chooseEmbeddingModel = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByLabelText(/Embedding Model/));
  const options = await screen.findAllByText("text-embedding-3-small");
  await user.click(options[options.length - 1]);
};

const fillConnection = () => {
  fireEvent.change(screen.getByPlaceholderText(SIDECAR_URL), { target: { value: SIDECAR_URL } });
  fireEvent.change(screen.getByPlaceholderText("Enter sidecar API key"), { target: { value: "sidecar-key" } });
};

describe("MongoDB vector store dialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue(undefined);
    mockTest.mockResolvedValue(PASSING_RESULT);
    mockDiscover.mockImplementation(async (_token, body) => discoveryPayloads[body.kind] ?? {});
  });

  it("shows the sidecar setup guidance with a copyable docker run", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);

    expect(screen.getByText("MongoDB Atlas Setup")).toBeInTheDocument();
    expect(screen.getByText(/docker run -d -p 8080:8080/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy the docker run command" })).toBeInTheDocument();
  });

  it("keeps Test connection disabled until the sidecar URL and key are both filled", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    expect(screen.getByRole("button", { name: /Test connection/ })).toBeDisabled();

    fillConnection();

    expect(await screen.findByRole("button", { name: /Test connection/ })).toBeEnabled();
  });

  it("runs the checklist against the unsaved configuration and renders every row", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));

    expect(await screen.findByText("Connection verified")).toBeInTheDocument();
    expect(mockTest).toHaveBeenCalledWith("test-token", {
      custom_llm_provider: "mongodb",
      litellm_params: expect.objectContaining({ api_base: SIDECAR_URL, api_key: "sidecar-key" }),
    });
    expect(screen.getByText("Sidecar auth")).toBeInTheDocument();
  });

  it("offers a curl for the checklist that carries the endpoint but not the key", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));

    expect(await screen.findByRole("button", { name: "Copy as curl" })).toBeInTheDocument();
  });

  it("fills the Database field from discovery once the sidecar details are in", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();

    await vi.waitFor(() =>
      expect(mockDiscover).toHaveBeenCalledWith("test-token", expect.objectContaining({ kind: "databases" })),
    );
    expect(await screen.findByRole("button", { name: "Enter a value that is not listed" })).toBeInTheDocument();
  });

  it("falls back to plain text entry and names the env var when the sidecar has discovery turned off", async () => {
    mockDiscover.mockRejectedValue(new ApiError("Discovery is disabled", 403, null));
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();

    expect(await screen.findAllByText(/MONGODB_SIDECAR_ALLOW_DISCOVERY=true/)).not.toHaveLength(0);
    expect(screen.queryByRole("button", { name: "Enter a value that is not listed" })).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("sample_mflix")).toBeInTheDocument();
  });

  it("hides the hybrid settings until the sidecar reports that the cluster supports them", async () => {
    mockTest.mockResolvedValue({
      ...PASSING_RESULT,
      checks: [{ ...PASSING_RESULT.checks[0], details: { features: { hybrid: false } } }],
    });
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));
    await screen.findByText("Connection verified");
    await user.click(screen.getByRole("button", { name: "Advanced" }));

    expect(screen.getByText(/Hybrid search settings appear once Test connection/)).toBeInTheDocument();
    expect(screen.queryByText("Hybrid Search")).not.toBeInTheDocument();
  });

  it("shows the hybrid toggle, text index and weight slider once the cluster supports $rankFusion", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));
    await screen.findByText("Connection verified");
    await user.click(screen.getByRole("button", { name: "Advanced" }));

    expect(screen.getByText("Hybrid Search")).toBeInTheDocument();
    expect(screen.getByText("Text Index")).toBeInTheDocument();
    expect(screen.getByText("Vector Weight")).toBeInTheDocument();
  });

  it("reports the embedding dimension against the index after a test", async () => {
    mockTest.mockResolvedValue({
      ...PASSING_RESULT,
      details: { embedding_dimensions: 3072, mongodb: { index_dimensions: 1536 } },
    });
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));

    expect(await screen.findByText("The index expects 1536, this model returns 3072.")).toBeInTheDocument();
  });

  it("creates the store with the index name and coerced advanced params", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    fireEvent.change(screen.getByPlaceholderText("policy_vector_index"), { target: { value: "policy_index" } });
    fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
    fireEvent.change(screen.getByPlaceholderText("embedded_movies"), { target: { value: "policies" } });
    await chooseEmbeddingModel(user);
    await user.click(screen.getByRole("button", { name: /Test connection/ }));
    await screen.findByText("Connection verified");
    await user.click(screen.getByRole("button", { name: "Advanced" }));
    fireEvent.change(screen.getByPlaceholderText("100"), { target: { value: "250" } });
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    const payload = mockCreate.mock.calls[0][1];
    expect(payload.vector_store_id).toBe("policy_index");
    const expectedParams = {
      api_base: SIDECAR_URL,
      mongodb_database: "knowledge",
      mongodb_collection: "policies",
      mongodb_num_candidates: 250,
      litellm_embedding_model: "text-embedding-3-small",
    };
    expect(payload.litellm_params).toMatchObject(expectedParams);
    expect(payload.litellm_params).not.toHaveProperty("embedding_model");
  });

  it("offers the create-a-new-index fields only on the create branch", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    expect(screen.queryByText("Index Dimensions")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Create a new index" }));

    expect(screen.getByText("Index Dimensions")).toBeInTheDocument();
    expect(screen.getByText("Similarity")).toBeInTheDocument();
  });
});
