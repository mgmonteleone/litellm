import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  vectorStoreCreateCall,
  vectorStoreDiscoverCall,
  vectorStoreProviderDefaultsCall,
  vectorStoreTestConnectionCall,
} from "@/components/networking";
import { ApiError } from "@/lib/http/client";

import { REDACTION_SENTINEL } from "../connection/connectionCurl";
import VectorStoreForm from "../VectorStoreForm";

vi.mock("@/components/networking", () => ({
  vectorStoreCreateCall: vi.fn(),
  vectorStoreTestConnectionCall: vi.fn(),
  vectorStoreDiscoverCall: vi.fn(),
  vectorStoreProviderDefaultsCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "text-embedding-3-small", mode: "embedding" }]),
}));

const mockCreate = vi.mocked(vectorStoreCreateCall);
const mockTest = vi.mocked(vectorStoreTestConnectionCall);
const mockDiscover = vi.mocked(vectorStoreDiscoverCall);
const mockProviderDefaults = vi.mocked(vectorStoreProviderDefaultsCall);

const NO_DEPLOYMENT_DEFAULTS = { custom_llm_provider: "mongodb", api_base: null, api_key_configured: false };
const WITH_DEPLOYMENT_DEFAULTS = {
  custom_llm_provider: "mongodb",
  api_base: "https://deployment-sidecar.example",
  api_key_configured: true,
};

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

const CONNECTION_ONLY_RESULT = {
  ok: true,
  supported: true,
  custom_llm_provider: "mongodb",
  summary: "Choose a database and collection to finish the checks.",
  checks: [
    { check: "sidecar_auth", status: "pass" as const, message: "Authenticated with sidecar v0.2.0." },
    {
      check: "mongodb_collection",
      status: "skip" as const,
      message: "Choose a database and collection to check the index.",
    },
  ],
  details: {},
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
    mockProviderDefaults.mockResolvedValue(NO_DEPLOYMENT_DEFAULTS);
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

  it("copies a curl for the checklist that carries the endpoint but never the raw key", async () => {
    const user = setupUser();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));
    await user.click(await screen.findByRole("button", { name: "Copy as curl" }));

    await vi.waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    const copied = writeText.mock.calls[0][0] as string;
    expect(copied).toContain(SIDECAR_URL);
    // The unsaved key is masked to a placeholder; only a value that was already the proxy's
    // redaction sentinel is ever echoed back verbatim, and the raw key never appears either way.
    expect(copied).not.toContain("sidecar-key");
    expect(copied).not.toContain(REDACTION_SENTINEL);
  });

  it("reads as a passing connection and keeps the Database combobox usable when Test connection runs before a database is chosen", async () => {
    mockTest.mockResolvedValue(CONNECTION_ONLY_RESULT);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await user.click(screen.getByRole("button", { name: /Test connection/ }));

    expect(await screen.findByText("Sidecar connected")).toBeInTheDocument();
    expect(screen.getByText("Choose a database and collection to check the index.")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Enter a value that is not listed" })).toBeInTheDocument();
  });

  it("fills the Database field from discovery once the sidecar details are in", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();

    await vi.waitFor(() =>
      expect(mockDiscover).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({ kind: "databases" }),
        expect.anything(),
      ),
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

  it("blocks Create without a sidecar URL and key when the deployment has no defaults", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    // Deliberately not calling fillConnection(): nothing else can reach the sidecar without it.
    fireEvent.change(screen.getByPlaceholderText("policy_vector_index"), { target: { value: "policy_index" } });
    fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
    fireEvent.change(screen.getByPlaceholderText("embedded_movies"), { target: { value: "policies" } });
    await chooseEmbeddingModel(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(await screen.findByText("Please input the sidecar url")).toBeInTheDocument();
    expect(screen.getByText("Please input the sidecar api key")).toBeInTheDocument();
    expect(mockCreate).not.toHaveBeenCalled();
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

  it("drops the hybrid fields from the saved params once the sidecar stops reporting hybrid support", async () => {
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

    await user.click(screen.getByRole("switch", { name: /Hybrid Search/ }));
    fireEvent.change(screen.getByPlaceholderText("policy_text_index"), { target: { value: "policy_text_index" } });

    // The cluster (or connection) changes and a second checklist run reports hybrid unsupported.
    mockTest.mockResolvedValue({
      ...PASSING_RESULT,
      checks: [{ ...PASSING_RESULT.checks[0], details: { features: { hybrid: false } } }],
    });
    await user.click(screen.getByRole("button", { name: /Test connection/ }));
    await screen.findByText(/Hybrid search settings appear once Test connection/);

    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    const payload = mockCreate.mock.calls[0][1];
    expect(payload.litellm_params).not.toHaveProperty("mongodb_hybrid_search");
    expect(payload.litellm_params).not.toHaveProperty("mongodb_text_index");
    expect(payload.litellm_params).not.toHaveProperty("mongodb_hybrid_weights");
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

describe("MongoDB deployment-configured sidecar defaults", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue(undefined);
    mockTest.mockResolvedValue(PASSING_RESULT);
    mockDiscover.mockImplementation(async (_token, body) => discoveryPayloads[body.kind] ?? {});
  });

  it("hides the sidecar fields and omits them from the saved params when the deployment has defaults", async () => {
    mockProviderDefaults.mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);

    expect(await screen.findByText(/Using this deployment's MongoDB sidecar at/)).toBeInTheDocument();
    expect(screen.getByText("https://deployment-sidecar.example")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(SIDECAR_URL)).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Enter sidecar API key")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Test connection/ })).toBeEnabled();

    fireEvent.change(screen.getByPlaceholderText("policy_vector_index"), { target: { value: "policy_index" } });
    fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
    fireEvent.change(screen.getByPlaceholderText("embedded_movies"), { target: { value: "policies" } });
    await chooseEmbeddingModel(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    const payload = mockCreate.mock.calls[0][1];
    expect(payload.litellm_params).not.toHaveProperty("api_base");
    expect(payload.litellm_params).not.toHaveProperty("api_key");
  });

  it("hides the MongoDB Atlas Setup guidance once the deployment has default sidecar credentials", async () => {
    mockProviderDefaults.mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);

    await screen.findByText(/Using this deployment's MongoDB sidecar at/);
    expect(screen.queryByText("MongoDB Atlas Setup")).not.toBeInTheDocument();
  });

  it("keeps the MongoDB Atlas Setup guidance hidden even after the admin opens the override", async () => {
    mockProviderDefaults.mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    await screen.findByText(/Using this deployment's MongoDB sidecar at/);
    await user.click(screen.getByRole("button", { name: "Override sidecar connection" }));

    expect(screen.queryByText("MongoDB Atlas Setup")).not.toBeInTheDocument();
  });

  it("sends the typed override once the admin opens it", async () => {
    mockProviderDefaults.mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    await screen.findByText(/Using this deployment's MongoDB sidecar at/);
    await user.click(screen.getByRole("button", { name: "Override sidecar connection" }));
    fillConnection();

    fireEvent.change(screen.getByPlaceholderText("policy_vector_index"), { target: { value: "policy_index" } });
    fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
    fireEvent.change(screen.getByPlaceholderText("embedded_movies"), { target: { value: "policies" } });
    await chooseEmbeddingModel(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][1].litellm_params).toMatchObject({
      api_base: SIDECAR_URL,
      api_key: "sidecar-key",
    });
  });

  it("hides the override fields again, and clears any value typed into them, once collapsed back", async () => {
    mockProviderDefaults.mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    await screen.findByText(/Using this deployment's MongoDB sidecar at/);
    await user.click(screen.getByRole("button", { name: "Override sidecar connection" }));
    fillConnection();
    await user.click(screen.getByRole("button", { name: "Override sidecar connection" }));

    expect(screen.queryByPlaceholderText(SIDECAR_URL)).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("policy_vector_index"), { target: { value: "policy_index" } });
    fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
    fireEvent.change(screen.getByPlaceholderText("embedded_movies"), { target: { value: "policies" } });
    await chooseEmbeddingModel(user);
    await user.click(screen.getByRole("button", { name: "Create" }));

    await vi.waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][1].litellm_params).not.toHaveProperty("api_base");
    expect(mockCreate.mock.calls[0][1].litellm_params).not.toHaveProperty("api_key");
  });

  it("renders the sidecar fields plainly when the deployment has no defaults configured", async () => {
    mockProviderDefaults.mockResolvedValue(NO_DEPLOYMENT_DEFAULTS);
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);

    expect(screen.queryByText(/Using this deployment's MongoDB sidecar at/)).not.toBeInTheDocument();
    expect(await screen.findByPlaceholderText(SIDECAR_URL)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter sidecar API key")).toBeInTheDocument();
  });

  it("keeps a value typed before deployment defaults resolve visible in an open override, not hidden in a collapsed one", async () => {
    let resolveDefaults: (value: typeof WITH_DEPLOYMENT_DEFAULTS) => void = () => {};
    mockProviderDefaults.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveDefaults = resolve;
        }),
    );
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    // The deployment-defaults fetch is still pending, so the fields render unconditionally.
    fillConnection();

    act(() => {
      resolveDefaults(WITH_DEPLOYMENT_DEFAULTS);
    });
    await screen.findByText(/Using this deployment's MongoDB sidecar at/);

    // The override section opening is a second render past the one that resolves
    // usingDeploymentDefaults (see the effect in MongoDBStoreFields), so this polls rather than
    // asserting immediately: findBy, not getBy.
    expect(await screen.findByPlaceholderText(SIDECAR_URL)).toHaveValue(SIDECAR_URL);
    expect(screen.getByPlaceholderText("Enter sidecar API key")).toHaveValue("sidecar-key");
  });
});

describe("MongoDB discovery debouncing and scoping", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue(undefined);
    mockTest.mockResolvedValue(PASSING_RESULT);
    mockDiscover.mockImplementation(async (_token, body) => discoveryPayloads[body.kind] ?? {});
    mockProviderDefaults.mockResolvedValue(NO_DEPLOYMENT_DEFAULTS);
  });

  it("debounces five fast keystrokes in Database into a single discover call, and never re-fires Databases", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    await vi.waitFor(() =>
      expect(mockDiscover).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({ kind: "databases" }),
        expect.anything(),
      ),
    );
    // Discovery has answered, so the Database field is now a combobox over its suggestions;
    // switch to free text the way an admin typing a database that isn't listed yet would.
    await user.click(await screen.findByRole("button", { name: "Enter a value that is not listed" }));
    mockDiscover.mockClear();

    const databaseInput = screen.getByPlaceholderText("sample_mflix");
    act(() => {
      "knowl".split("").forEach((_, index) => {
        fireEvent.change(databaseInput, { target: { value: "knowl".slice(0, index + 1) } });
      });
    });

    await vi.waitFor(
      () => expect(mockDiscover.mock.calls.filter((call) => call[1].kind === "collections")).toHaveLength(1),
      { timeout: 2000 },
    );
    expect(mockDiscover.mock.calls.filter((call) => call[1].kind === "databases")).toHaveLength(0);
    expect(mockDiscover.mock.calls.find((call) => call[1].kind === "collections")?.[1].litellm_params).toMatchObject({
      mongodb_database: "knowl",
    });
  });

  it("does not refetch collections when only the vector field name changes", async () => {
    const user = setupUser();
    renderForm();

    await chooseMongoDB(user);
    fillConnection();
    fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
    await vi.waitFor(() =>
      expect(mockDiscover).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({ kind: "collections" }),
        expect.anything(),
      ),
    );
    mockDiscover.mockClear();

    fireEvent.change(screen.getByPlaceholderText("embedding"), { target: { value: "embedding_v2" } });
    await new Promise((resolve) => setTimeout(resolve, 350));

    expect(mockDiscover.mock.calls.filter((call) => call[1].kind === "collections")).toHaveLength(0);
    expect(mockDiscover.mock.calls.filter((call) => call[1].kind === "databases")).toHaveLength(0);
  });
});
