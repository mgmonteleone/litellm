import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  credentialListCall,
  vectorStoreDiscoverCall,
  vectorStoreInfoCall,
  vectorStoreProviderDefaultsCall,
  vectorStoreTestConnectionCall,
  vectorStoreUpdateCall,
} from "@/components/networking";
import { toast } from "@/lib/toast";

import VectorStoreInfoView from "./vector_store_info";

vi.mock("@/components/networking", () => ({
  vectorStoreInfoCall: vi.fn(),
  vectorStoreUpdateCall: vi.fn(),
  credentialListCall: vi.fn(),
  vectorStoreTestConnectionCall: vi.fn(),
  vectorStoreDiscoverCall: vi.fn(),
  vectorStoreProviderDefaultsCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("./VectorStoreTester", () => ({ __esModule: true, default: () => null }));

const mockInfo = vi.mocked(vectorStoreInfoCall);
const mockUpdate = vi.mocked(vectorStoreUpdateCall);
const mockCredentials = vi.mocked(credentialListCall);
const mockToast = vi.mocked(toast);
const mockTestConnection = vi.mocked(vectorStoreTestConnectionCall);
const mockDiscover = vi.mocked(vectorStoreDiscoverCall);
const mockProviderDefaults = vi.mocked(vectorStoreProviderDefaultsCall);

const MONGODB_RECORD = {
  vector_store_id: "policy_vector_index",
  vector_store_name: "company-policies",
  vector_store_description: "HR and finance policies",
  custom_llm_provider: "mongodb",
  vector_store_metadata: {
    ingested_files: [
      {
        file_id: "file_fdbe36ef",
        filename: "travel-policy.txt",
        file_size: 401,
        content_type: "text/plain",
        ingested_at: "2026-09-20T16:26:42.014534+00:00",
      },
    ],
  },
  litellm_params: {
    api_key: "REDACTED_BY_LITELM",
    api_base: "http://127.0.0.1:8080",
    use_xai_oauth: false,
    mongodb_database: "knowledge",
    mongodb_collection: "policies",
    litellm_embedding_model: "text-embedding-3-small",
  },
  created_at: "2026-09-20T16:26:41Z",
  updated_at: "2026-09-20T17:02:56Z",
};

const PASSING_CHECKLIST = {
  ok: true,
  supported: true,
  custom_llm_provider: "mongodb",
  summary: "All checks passed.",
  checks: [
    {
      check: "mongodb_index",
      status: "pass" as const,
      message: "Index 'policy_vector_index' is ready.",
      details: { status: "ready", queryable: true },
    },
  ],
};

const serverRecord = {
  vector_store_id: "vs-1",
  vector_store_name: "support-docs-store",
  vector_store_description: "Docs for support",
  custom_llm_provider: "bedrock",
  vector_store_metadata: { tier: "gold" },
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-02-02T00:00:00Z",
  litellm_credential_name: "bedrock-prod",
};

const renderView = (editVectorStore: boolean) =>
  render(
    <VectorStoreInfoView
      vectorStoreId="vs-1"
      onClose={vi.fn()}
      accessToken="sk-test"
      is_admin={true}
      editVectorStore={editVectorStore}
    />,
  );

const renderViewAsNonAdmin = (editVectorStore: boolean) =>
  render(
    <VectorStoreInfoView
      vectorStoreId="vs-1"
      onClose={vi.fn()}
      accessToken="sk-test"
      is_admin={false}
      editVectorStore={editVectorStore}
    />,
  );

const savedPayload = () => mockUpdate.mock.calls[0][1];

describe("VectorStoreInfoView save payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockInfo.mockResolvedValue({ vector_store: serverRecord });
    mockCredentials.mockResolvedValue({ credentials: [{ credential_name: "bedrock-prod" }] });
    mockUpdate.mockResolvedValue({});
    mockProviderDefaults.mockResolvedValue({
      custom_llm_provider: "mongodb",
      api_base: null,
      api_key_configured: false,
    });
  });

  it("still saves when the server left the nullable name and description null", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({
      vector_store: { ...serverRecord, vector_store_name: null, vector_store_description: null },
    });
    renderView(true);
    await screen.findByRole("button", { name: "Save Changes" });

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: null,
      vector_store_description: null,
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("sends only the five editable keys and drops every server-only field", async () => {
    const user = userEvent.setup();
    renderView(true);

    const nameInput = await screen.findByDisplayValue("support-docs-store");
    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "renamed-store" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0][0]).toBe("sk-test");
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "renamed-store",
      vector_store_description: "Docs for support",
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("sends the same five keys when editing is entered from the details view", async () => {
    const user = userEvent.setup();
    renderView(false);

    const editButtons = await screen.findAllByRole("button", { name: "Edit Vector Store" });
    await user.click(editButtons[0]);
    const descriptionInput = await screen.findByDisplayValue("Docs for support");
    await user.clear(descriptionInput);
    fireEvent.change(descriptionInput, { target: { value: "new description" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual({
      vector_store_id: "vs-1",
      custom_llm_provider: "bedrock",
      vector_store_name: "support-docs-store",
      vector_store_description: "new description",
      vector_store_metadata: { tier: "gold" },
    });
  });

  it("keeps the credential field out of the payload even after it is picked", async () => {
    const user = userEvent.setup();
    renderView(true);

    await screen.findByDisplayValue("support-docs-store");
    await user.click(screen.getAllByRole("combobox")[1]);
    const options = await screen.findAllByText("bedrock-prod");
    await user.click(options[options.length - 1]);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(Object.keys(savedPayload())).toStrictEqual([
      "vector_store_id",
      "custom_llm_provider",
      "vector_store_name",
      "vector_store_description",
      "vector_store_metadata",
    ]);
  });

  it("blocks the request and reports invalid metadata JSON instead of saving", async () => {
    const user = userEvent.setup();
    renderView(true);

    const metadataInput = await screen.findByPlaceholderText('{"key": "value"}');
    await user.clear(metadataInput);
    fireEvent.change(metadataInput, { target: { value: "not json" } });
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await vi.waitFor(() => expect(mockToast.fromError).toHaveBeenCalledWith("Invalid JSON in metadata field"));
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("keeps the required-field message that blocks saving without a vector store id", async () => {
    const user = userEvent.setup();
    mockInfo.mockResolvedValue({ vector_store: { ...serverRecord, vector_store_id: "" } });
    renderView(true);

    await screen.findByDisplayValue("support-docs-store");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    expect(await screen.findByText("Please input a vector store ID")).toBeInTheDocument();
    expect(mockUpdate).not.toHaveBeenCalled();
  });
});

describe("VectorStoreInfoView connection card", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockInfo.mockResolvedValue({ vector_store: MONGODB_RECORD });
    mockCredentials.mockResolvedValue({ credentials: [] });
    mockTestConnection.mockResolvedValue(PASSING_CHECKLIST);
    mockDiscover.mockResolvedValue({});
    mockUpdate.mockResolvedValue({ status: "success" });
    mockProviderDefaults.mockResolvedValue({
      custom_llm_provider: "mongodb",
      api_base: null,
      api_key_configured: false,
    });
  });

  it("shows the saved connection so database and collection are visible after save", async () => {
    renderView(false);

    expect(await screen.findByText("Connection")).toBeInTheDocument();
    expect(screen.getByText("http://127.0.0.1:8080")).toBeInTheDocument();
    expect(screen.getByText("knowledge")).toBeInTheDocument();
    expect(screen.getByText("policies")).toBeInTheDocument();
    expect(screen.getByText("text-embedding-3-small")).toBeInTheDocument();
  });

  it("hides the sidecar key rather than printing the redaction sentinel", async () => {
    renderView(false);

    await screen.findByText("Connection");
    expect(screen.getByText("Set, hidden")).toBeInTheDocument();
    expect(screen.queryByText("REDACTED_BY_LITELM")).not.toBeInTheDocument();
  });

  it("starts out saying the connection has not been tested", async () => {
    renderView(false);

    expect(await screen.findByText("Not tested")).toBeInTheDocument();
  });

  it("tests the saved store by id, so the proxy reuses the stored secret", async () => {
    const user = userEvent.setup();
    renderView(false);

    await user.click(await screen.findByRole("button", { name: /Test connection/ }));

    await vi.waitFor(() => expect(mockTestConnection).toHaveBeenCalledTimes(1));
    expect(mockTestConnection.mock.calls[0]).toEqual(["sk-test", { vector_store_id: "vs-1" }]);
  });

  it("hides Test connection from a non-admin, since /vector_store/test_connection is admin-only", async () => {
    renderViewAsNonAdmin(false);

    expect(await screen.findByText("Connection")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Test connection/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Edit connection/ })).not.toBeInTheDocument();
  });

  it("renders the checklist and the index status chip once the test returns", async () => {
    const user = userEvent.setup();
    renderView(false);

    await user.click(await screen.findByRole("button", { name: /Test connection/ }));

    expect(await screen.findByText("Connection verified")).toBeInTheDocument();
    expect(screen.getByText("Index ready")).toBeInTheDocument();
    expect(screen.getByText("Index 'policy_vector_index' is ready.")).toBeInTheDocument();
  });

  it("lists the ingested files with their size and type", async () => {
    renderView(false);

    expect(await screen.findByText("travel-policy.txt")).toBeInTheDocument();
    expect(screen.getByText(/401 B/)).toBeInTheDocument();
    expect(screen.getByText("file_fdbe36ef")).toBeInTheDocument();
  });

  it("says so plainly when nothing has been ingested through LiteLLM", async () => {
    mockInfo.mockResolvedValue({ vector_store: { ...MONGODB_RECORD, vector_store_metadata: {} } });
    renderView(false);

    expect(await screen.findByText(/No documents have been ingested through LiteLLM/)).toBeInTheDocument();
  });

  it("offers every vector store provider in the edit form, not only Bedrock", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderView(true);

    await user.click((await screen.findAllByRole("combobox"))[0]);

    expect(await screen.findByText("Valkey")).toBeInTheDocument();
    expect(screen.getByText("Milvus")).toBeInTheDocument();
    expect(screen.getAllByText("MongoDB").length).toBeGreaterThan(0);
  });

  it("edits the connection: prefills the saved fields, sends only the changed key plus the untouched secret sentinel, and refreshes on save", async () => {
    const user = userEvent.setup();
    renderView(false);

    await user.click(await screen.findByRole("button", { name: "Edit connection" }));

    expect(screen.getByLabelText(/Sidecar URL/)).toHaveValue("http://127.0.0.1:8080");
    expect(screen.getByLabelText(/Sidecar API Key/)).toHaveValue("REDACTED_BY_LITELM");

    fireEvent.change(screen.getByLabelText(/Collection/), { target: { value: "policies_v2" } });
    mockInfo.mockResolvedValueOnce({
      vector_store: {
        ...MONGODB_RECORD,
        litellm_params: { ...MONGODB_RECORD.litellm_params, mongodb_collection: "policies_v2" },
      },
    });
    await user.click(screen.getByRole("button", { name: "Save connection" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0]).toEqual([
      "sk-test",
      {
        vector_store_id: "policy_vector_index",
        litellm_params: { mongodb_collection: "policies_v2", api_key: "REDACTED_BY_LITELM" },
      },
    ]);

    // Back to the read-only card, refreshed with the saved change, and Test connection is offered again.
    await vi.waitFor(() => expect(mockInfo).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("policies_v2")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Test connection/ })).toBeInTheDocument();
  });

  it("cancels the connection edit without saving", async () => {
    const user = userEvent.setup();
    renderView(false);

    await user.click(await screen.findByRole("button", { name: "Edit connection" }));
    fireEvent.change(screen.getByLabelText(/Collection/), { target: { value: "abandoned" } });
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(mockUpdate).not.toHaveBeenCalled();
    expect(await screen.findByText("policies")).toBeInTheDocument();
  });
});
