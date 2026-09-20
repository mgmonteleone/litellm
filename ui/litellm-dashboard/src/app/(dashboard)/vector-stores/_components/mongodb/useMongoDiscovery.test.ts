import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreDiscoverCall } from "@/components/networking";
import { ApiError } from "@/lib/http/client";

import { useMongoDiscovery } from "./useMongoDiscovery";

vi.mock("@/components/networking", () => ({
  vectorStoreDiscoverCall: vi.fn(),
}));

const mockDiscover = vi.mocked(vectorStoreDiscoverCall);

describe("useMongoDiscovery", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("debounces five fast changes to the request into a single discover call", async () => {
    mockDiscover.mockResolvedValue({ databases: ["knowledge"] });

    const { rerender } = renderHook(
      ({ database }: { database: string }) =>
        useMongoDiscovery(
          "test-token",
          "collections",
          { litellmParams: { api_base: "http://127.0.0.1:8080", api_key: "key", mongodb_database: database } },
          Boolean(database),
        ),
      { initialProps: { database: "" } },
    );

    "knowl".split("").forEach((_, index) => rerender({ database: "knowl".slice(0, index + 1) }));

    await vi.waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(1));
    expect(mockDiscover.mock.calls[0][1]).toMatchObject({
      kind: "collections",
      litellm_params: { mongodb_database: "knowl" },
    });
  });

  it("coalesces further edits made before the debounce settles into a single follow-up call", async () => {
    mockDiscover.mockResolvedValue({ databases: [] });

    const { rerender } = renderHook(
      ({ database }: { database: string }) =>
        useMongoDiscovery("test-token", "collections", { litellmParams: { mongodb_database: database } }, true),
      { initialProps: { database: "k" } },
    );

    // Enabling with a value fires once immediately; only edits made after that are debounced.
    await vi.waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(1));
    mockDiscover.mockClear();

    rerender({ database: "kn" });
    rerender({ database: "kno" });
    expect(mockDiscover).not.toHaveBeenCalled();

    await vi.waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(1));
    expect(mockDiscover.mock.calls[0][1]).toMatchObject({ litellm_params: { mongodb_database: "kno" } });
  });

  it("passes an abort signal to the discover call and aborts it once superseded by a newer request", async () => {
    mockDiscover.mockResolvedValue({ databases: [] });

    const { rerender } = renderHook(
      ({ database }: { database: string }) =>
        useMongoDiscovery("test-token", "collections", { litellmParams: { mongodb_database: database } }, true),
      { initialProps: { database: "first" } },
    );

    await vi.waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(1));
    const [, , firstSignal] = mockDiscover.mock.calls[0];
    expect(firstSignal).toBeInstanceOf(AbortSignal);
    expect(firstSignal?.aborted).toBe(false);

    rerender({ database: "second" });
    await vi.waitFor(() => expect(mockDiscover).toHaveBeenCalledTimes(2));

    expect(firstSignal?.aborted).toBe(true);
  });

  it("reports the sidecar's discovery-disabled message on a 403 without treating it as a hard failure", async () => {
    mockDiscover.mockRejectedValue(new ApiError("Forbidden", 403, null));

    const { result } = renderHook(() =>
      useMongoDiscovery("test-token", "databases", { litellmParams: { api_base: "http://x", api_key: "k" } }, true),
    );

    await vi.waitFor(() => expect(result.current.status).toBe("unavailable"));
    expect(result.current.message).toMatch(/MONGODB_SIDECAR_ALLOW_DISCOVERY=true/);
  });

  it("stays idle without calling discover when disabled", () => {
    const { result } = renderHook(() => useMongoDiscovery("test-token", "databases", { litellmParams: {} }, false));

    expect(result.current.status).toBe("idle");
    expect(mockDiscover).not.toHaveBeenCalled();
  });
});
