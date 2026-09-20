"use client";

import { useEffect, useMemo, useState } from "react";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";

import { vectorStoreDiscoverCall } from "@/components/networking";
import type { VectorStoreDiscoveryKind } from "@/components/vector_store_providers";
import { ApiError, extractProxyErrorMessage } from "@/lib/http/client";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";

import { discoverRequestKind, parseSuggestions, type DiscoverySuggestion } from "./mongodbDiscovery";

export type DiscoveryStatus = "idle" | "loading" | "ready" | "unavailable";

export interface DiscoveryState {
  status: DiscoveryStatus;
  suggestions: readonly DiscoverySuggestion[];
  /** Why discovery produced nothing, so the field can say so instead of looking broken. */
  message: string | null;
}

export interface MongoDiscoveryRequest {
  vectorStoreId?: string | null;
  litellmParams?: Record<string, unknown>;
  options?: Record<string, unknown>;
}

const IDLE: DiscoveryState = { status: "idle", suggestions: [], message: null };

const LOADING: DiscoveryState = { status: "loading", suggestions: [], message: null };

/**
 * The sidecar ships with MONGODB_SIDECAR_ALLOW_DISCOVERY off, so a 403 is the normal answer
 * rather than a failure: the caller keeps plain text entry and says why.
 */
const DISCOVERY_DISABLED_MESSAGE =
  "Discovery is turned off on this sidecar (set MONGODB_SIDECAR_ALLOW_DISCOVERY=true to list them). Type the value instead.";

export const useMongoDiscovery = (
  accessToken: string | null,
  kind: VectorStoreDiscoveryKind,
  request: MongoDiscoveryRequest,
  enabled: boolean,
): DiscoveryState => {
  /** Keyed so a result from a superseded request cannot be shown against the current one. */
  const [answered, setAnswered] = useState<{ key: string; state: DiscoveryState } | null>(null);
  /**
   * `enabled` travels through the same debounce as the request: a database field going from
   * empty to one character flips it true on the very next keystroke, and without debouncing that
   * transition too, the effect would fire once immediately with the stale (pre-keystroke) request
   * and again after the debounce settles.
   */
  const payloadKey = useMemo(() => JSON.stringify({ enabled, request }), [enabled, request]);
  /** Waits for the admin to stop typing before firing, so one field doesn't send a request per keystroke. */
  const [debouncedPayloadKey] = useDebouncedValue(payloadKey, { wait: DEBOUNCE_WAIT_MS });
  const requestKey = `${kind}:${debouncedPayloadKey}`;

  useEffect(() => {
    const { enabled: debouncedEnabled, request: body } = JSON.parse(debouncedPayloadKey) as {
      enabled: boolean;
      request: MongoDiscoveryRequest;
    };
    if (!accessToken || !debouncedEnabled) return;
    const controller = new AbortController();
    const resolve = (state: DiscoveryState) => {
      if (!controller.signal.aborted) setAnswered({ key: `${kind}:${debouncedPayloadKey}`, state });
    };

    const discoverRequest = {
      kind: discoverRequestKind(kind),
      custom_llm_provider: "mongodb",
      vector_store_id: body.vectorStoreId ?? null,
      litellm_params: body.litellmParams ?? null,
      options: body.options ?? {},
    };

    vectorStoreDiscoverCall(accessToken, discoverRequest, controller.signal)
      .then((payload) => {
        const suggestions = parseSuggestions(kind, payload);
        resolve({
          status: "ready",
          suggestions,
          message: suggestions.length === 0 ? "Nothing found. Type the value instead." : null,
        });
      })
      .catch((error: unknown) => {
        const disabled = error instanceof ApiError && (error.status === 403 || error.status === 404);
        resolve({
          status: "unavailable",
          suggestions: [],
          message: disabled ? DISCOVERY_DISABLED_MESSAGE : extractProxyErrorMessage(error),
        });
      });

    return () => controller.abort();
  }, [accessToken, kind, debouncedPayloadKey]);

  if (!accessToken || !enabled) return IDLE;
  return answered?.key === requestKey ? answered.state : LOADING;
};
