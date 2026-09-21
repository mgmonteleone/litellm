import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreDiscoverCall, vectorStoreUpdateCall } from "@/components/networking";

import ConnectionEditForm from "./ConnectionEditForm";
import { REDACTION_SENTINEL } from "./connectionCurl";

vi.mock("@/components/networking", () => ({
  vectorStoreUpdateCall: vi.fn(),
  vectorStoreTestConnectionCall: vi.fn(),
  vectorStoreDiscoverCall: vi.fn(),
  getProxyBaseUrl: () => "http://localhost:4000",
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const mockUpdate = vi.mocked(vectorStoreUpdateCall);
const mockDiscover = vi.mocked(vectorStoreDiscoverCall);

const renderForm = (overrides: Partial<React.ComponentProps<typeof ConnectionEditForm>> = {}) =>
  render(
    <ConnectionEditForm
      vectorStoreId="vs-1"
      provider="azure"
      litellmParams={{
        api_key: REDACTION_SENTINEL,
        api_base: "https://my-resource.openai.azure.com/",
      }}
      accessToken="test-token"
      onCancel={vi.fn()}
      onSaved={vi.fn()}
      {...overrides}
    />,
  );

describe("ConnectionEditForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUpdate.mockResolvedValue({ status: "success" });
    mockDiscover.mockResolvedValue({});
  });

  it("prefills a non-secret field with its saved value and a secret field with the redaction sentinel", () => {
    renderForm();

    expect(screen.getByLabelText(/API Base/)).toHaveValue("https://my-resource.openai.azure.com/");
    expect(screen.getByLabelText(/API Key/)).toHaveValue(REDACTION_SENTINEL);
  });

  it("sends only the changed field, and the untouched secret as the exact sentinel", async () => {
    const user = userEvent.setup();
    renderForm();

    fireEvent.change(screen.getByLabelText(/API Base/), {
      target: { value: "https://renamed-resource.openai.azure.com/" },
    });
    await user.click(screen.getByRole("button", { name: "Save connection" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const [, payload] = mockUpdate.mock.calls[0];
    expect(payload).toEqual({
      vector_store_id: "vs-1",
      litellm_params: {
        api_base: "https://renamed-resource.openai.azure.com/",
        api_key: REDACTION_SENTINEL,
      },
    });
  });

  it("leaves the saved secret alone when nothing on the form changes", async () => {
    const user = userEvent.setup();
    renderForm();

    await user.click(screen.getByRole("button", { name: "Save connection" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const [, payload] = mockUpdate.mock.calls[0];
    expect(payload.litellm_params).toEqual({ api_key: REDACTION_SENTINEL });
  });

  it("sends an os.environ/ reference typed into the secret field verbatim, not the sentinel", async () => {
    const user = userEvent.setup();
    renderForm();

    fireEvent.change(screen.getByLabelText(/API Key/), { target: { value: "os.environ/AZURE_OPENAI_API_KEY" } });
    await user.click(screen.getByRole("button", { name: "Save connection" }));

    await vi.waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    const [, payload] = mockUpdate.mock.calls[0];
    expect(payload.litellm_params).toEqual({ api_key: "os.environ/AZURE_OPENAI_API_KEY" });
  });

  it("calls onCancel without saving when Cancel is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderForm({ onCancel });

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("renders the reused MongoDB field components for a mongodb store", () => {
    renderForm({
      provider: "mongodb",
      litellmParams: {
        api_base: "http://127.0.0.1:8080",
        api_key: REDACTION_SENTINEL,
        mongodb_database: "knowledge",
        mongodb_collection: "policies",
      },
    });

    expect(screen.getByLabelText(/Database/)).toHaveValue("knowledge");
    expect(screen.getByLabelText(/Collection/)).toHaveValue("policies");
  });
});
