"use client";

import { useEffect, useState } from "react";

import { vectorStoreProviderDefaultsCall } from "@/components/networking";

export type MongoDBProviderDefaultsState =
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "ready"; apiBase: string | null; apiKeyConfigured: boolean };

const LOADING: MongoDBProviderDefaultsState = { status: "loading" };
const UNAVAILABLE: MongoDBProviderDefaultsState = { status: "unavailable" };

/** Whether the deployment already has a usable sidecar configured, so the form can use it by default. */
export const hasDeploymentDefaults = (
  state: MongoDBProviderDefaultsState,
): state is { status: "ready"; apiBase: string; apiKeyConfigured: true } =>
  state.status === "ready" && Boolean(state.apiBase) && state.apiKeyConfigured;

/**
 * Fetches the deployment's configured MongoDB sidecar defaults once per access token, so the form can
 * offer to use them instead of asking the admin to re-enter values the deployment already sets.
 */
export const useMongoDBProviderDefaults = (accessToken: string | null): MongoDBProviderDefaultsState => {
  /** Keyed so a result from a superseded request (an earlier access token) cannot be shown against the current one. */
  const [answered, setAnswered] = useState<{ token: string; state: MongoDBProviderDefaultsState } | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    vectorStoreProviderDefaultsCall(accessToken, "mongodb")
      .then((response) => {
        if (cancelled) return;
        setAnswered({
          token: accessToken,
          state: {
            status: "ready",
            apiBase: response.api_base ?? null,
            apiKeyConfigured: Boolean(response.api_key_configured),
          },
        });
      })
      .catch(() => {
        if (!cancelled) setAnswered({ token: accessToken, state: UNAVAILABLE });
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  if (!accessToken) return UNAVAILABLE;
  return answered?.token === accessToken ? answered.state : LOADING;
};
