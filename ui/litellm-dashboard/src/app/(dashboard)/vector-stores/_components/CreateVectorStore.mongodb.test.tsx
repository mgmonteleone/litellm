import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";

import * as networking from "@/components/networking";
import * as fetchModels from "@/components/llm_calls/fetch_models";
import { toast } from "@/lib/toast";

import CreateVectorStore from "./CreateVectorStore";

vi.mock("@/components/networking", () => ({
  ragIngestCall: vi.fn(),
  vectorStoreProviderDefaultsCall: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

const SIDECAR_URL = "http://127.0.0.1:8080";
const WITH_DEPLOYMENT_DEFAULTS = {
  custom_llm_provider: "mongodb",
  api_base: "https://deployment-sidecar.example",
  api_key_configured: true,
};
const SUCCESSFUL_INGEST_RESULT = {
  id: "test-id",
  status: "completed" as const,
  vector_store_id: "vs_123",
  file_id: "file_123",
};

const uploadFile = async () => {
  const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
  const uploadInput = screen.getByLabelText(/Click or drag files to this area to upload/);
  await act(async () => {
    fireEvent.change(uploadInput, { target: { files: [file] } });
  });
  await screen.findByText(/Uploaded Documents \(1\)/);
};

const pickMongoDB = async () => {
  const user = userEvent.setup({ pointerEventsCheck: 0 });
  const trigger = screen.getAllByRole("combobox")[0];
  await user.click(trigger);
  const option = await screen.findByText("MongoDB");
  await user.click(option);
};

const chooseEmbeddingModel = async () => {
  const user = userEvent.setup({ pointerEventsCheck: 0 });
  await user.click(screen.getByLabelText(/Embedding Model/));
  const options = await screen.findAllByText("text-embedding-3-small");
  await user.click(options[options.length - 1]);
};

const fillRequiredDataFields = () => {
  fireEvent.change(screen.getByPlaceholderText("sample_mflix"), { target: { value: "knowledge" } });
  fireEvent.change(screen.getByPlaceholderText("embedded_movies"), { target: { value: "policies" } });
};

const clickCreate = async () => {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: /Create Vector Store/i }));
  });
};

describe("CreateVectorStore MongoDB sidecar connection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchModels.fetchAvailableModels).mockResolvedValue([
      { model_group: "text-embedding-3-small", mode: "embedding" },
    ] as Awaited<ReturnType<typeof fetchModels.fetchAvailableModels>>);
    vi.mocked(networking.ragIngestCall).mockResolvedValue(SUCCESSFUL_INGEST_RESULT);
  });

  it("uses the deployment's default sidecar: hides the connection fields and omits them from the ingest request", async () => {
    vi.mocked(networking.vectorStoreProviderDefaultsCall).mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickMongoDB();

    expect(await screen.findByText(/Using this deployment's MongoDB sidecar at/)).toBeInTheDocument();
    expect(screen.getByText("https://deployment-sidecar.example")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(SIDECAR_URL)).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Enter sidecar API key")).not.toBeInTheDocument();

    fillRequiredDataFields();
    await chooseEmbeddingModel();
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    const providerParams = vi.mocked(networking.ragIngestCall).mock.calls[0][6] as Record<string, unknown>;
    expect(providerParams).not.toHaveProperty("api_base");
    expect(providerParams).not.toHaveProperty("api_key");
  });

  it("sends the typed override once the admin opens 'Override sidecar connection'", async () => {
    vi.mocked(networking.vectorStoreProviderDefaultsCall).mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickMongoDB();

    await screen.findByText(/Using this deployment's MongoDB sidecar at/);
    await user.click(screen.getByRole("button", { name: "Override sidecar connection" }));
    fireEvent.change(screen.getByPlaceholderText(SIDECAR_URL), { target: { value: SIDECAR_URL } });
    fireEvent.change(screen.getByPlaceholderText("Enter sidecar API key"), { target: { value: "sidecar-key" } });

    fillRequiredDataFields();
    await chooseEmbeddingModel();
    await clickCreate();

    await waitFor(() => expect(networking.ragIngestCall).toHaveBeenCalledTimes(1));
    const providerParams = vi.mocked(networking.ragIngestCall).mock.calls[0][6] as Record<string, unknown>;
    expect(providerParams).toMatchObject({ api_base: SIDECAR_URL, api_key: "sidecar-key" });
  });

  it("renders and requires the sidecar fields when the deployment has no configured defaults", async () => {
    vi.mocked(networking.vectorStoreProviderDefaultsCall).mockRejectedValue(new Error("Forbidden"));
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickMongoDB();

    expect(await screen.findByPlaceholderText(SIDECAR_URL)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter sidecar API key")).toBeInTheDocument();
    expect(screen.queryByText(/Using this deployment's MongoDB sidecar at/)).not.toBeInTheDocument();

    fillRequiredDataFields();
    await chooseEmbeddingModel();
    await clickCreate();

    expect(networking.ragIngestCall).not.toHaveBeenCalled();
  });

  it("rejects a partial override: the sidecar key is only sent to the deployment's own sidecar", async () => {
    vi.mocked(networking.vectorStoreProviderDefaultsCall).mockResolvedValue(WITH_DEPLOYMENT_DEFAULTS);
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();
    await pickMongoDB();

    await screen.findByText(/Using this deployment's MongoDB sidecar at/);
    await user.click(screen.getByRole("button", { name: "Override sidecar connection" }));
    fireEvent.change(screen.getByPlaceholderText(SIDECAR_URL), { target: { value: SIDECAR_URL } });

    fillRequiredDataFields();
    await chooseEmbeddingModel();
    await clickCreate();

    expect(toast.warning).toHaveBeenCalledWith("A custom MongoDB sidecar connection needs both a URL and an API key");
    expect(networking.ragIngestCall).not.toHaveBeenCalled();
  });
});
