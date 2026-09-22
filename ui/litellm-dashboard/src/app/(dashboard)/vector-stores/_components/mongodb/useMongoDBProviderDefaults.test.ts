import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreProviderDefaultsCall } from "@/components/networking";
import { ApiError } from "@/lib/http/client";

import { hasDeploymentDefaults, useMongoDBProviderDefaults } from "./useMongoDBProviderDefaults";

vi.mock("@/components/networking", () => ({
  vectorStoreProviderDefaultsCall: vi.fn(),
}));

const mockProviderDefaults = vi.mocked(vectorStoreProviderDefaultsCall);

describe("useMongoDBProviderDefaults", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("reports the deployment's sidecar defaults once the call resolves", async () => {
    mockProviderDefaults.mockResolvedValue({
      custom_llm_provider: "mongodb",
      api_base: "https://deployment-sidecar.example",
      api_key_configured: true,
    });

    const { result } = renderHook(() => useMongoDBProviderDefaults("test-token"));

    expect(result.current.status).toBe("loading");
    await vi.waitFor(() =>
      expect(result.current).toEqual({
        status: "ready",
        apiBase: "https://deployment-sidecar.example",
        apiKeyConfigured: true,
      }),
    );
    expect(mockProviderDefaults).toHaveBeenCalledWith("test-token", "mongodb");
  });

  it("reports no usable defaults when the deployment has not configured a sidecar", async () => {
    mockProviderDefaults.mockResolvedValue({
      custom_llm_provider: "mongodb",
      api_base: null,
      api_key_configured: false,
    });

    const { result } = renderHook(() => useMongoDBProviderDefaults("test-token"));

    await vi.waitFor(() => expect(result.current.status).toBe("ready"));
    expect(hasDeploymentDefaults(result.current)).toBe(false);
  });

  it("treats a failed lookup as unavailable rather than throwing", async () => {
    mockProviderDefaults.mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useMongoDBProviderDefaults("test-token"));

    await vi.waitFor(() => expect(result.current.status).toBe("unavailable"));
  });

  it("treats a 403 (a non-admin caller) as unavailable too, so the form falls back to plain fields", async () => {
    mockProviderDefaults.mockRejectedValue(
      new ApiError("Only proxy admins can read provider defaults for vector store connections.", 403, null),
    );

    const { result } = renderHook(() => useMongoDBProviderDefaults("test-token"));

    await vi.waitFor(() => expect(result.current.status).toBe("unavailable"));
    expect(hasDeploymentDefaults(result.current)).toBe(false);
  });

  it("stays unavailable and never calls the endpoint without an access token", () => {
    const { result } = renderHook(() => useMongoDBProviderDefaults(null));

    expect(result.current.status).toBe("unavailable");
    expect(mockProviderDefaults).not.toHaveBeenCalled();
  });
});

describe("hasDeploymentDefaults", () => {
  it("is true only once both an api_base and a configured api_key are reported", () => {
    expect(hasDeploymentDefaults({ status: "ready", apiBase: "https://x", apiKeyConfigured: true })).toBe(true);
    expect(hasDeploymentDefaults({ status: "ready", apiBase: null, apiKeyConfigured: true })).toBe(false);
    expect(hasDeploymentDefaults({ status: "ready", apiBase: "https://x", apiKeyConfigured: false })).toBe(false);
    expect(hasDeploymentDefaults({ status: "loading" })).toBe(false);
    expect(hasDeploymentDefaults({ status: "unavailable" })).toBe(false);
  });
});
