"use client";

import { useCallback, useState } from "react";

import {
  getProxyBaseUrl,
  vectorStoreTestConnectionCall,
  type VectorStoreTestConnectionResponse,
} from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";

import { buildTestConnectionCurl, type TestConnectionRequestBody } from "./connectionCurl";

/** The proxy answers a failed checklist with 200 and ok: false, so only transport errors land here. */
const asFailedResult = (error: unknown): VectorStoreTestConnectionResponse => ({
  ok: false,
  supported: true,
  summary: extractProxyErrorMessage(error),
  checks: [
    {
      check: "test_connection_request",
      status: "fail",
      message: `The proxy could not run the checklist: ${extractProxyErrorMessage(error)}`,
    },
  ],
});

export interface ConnectionTestState {
  isRunning: boolean;
  result: VectorStoreTestConnectionResponse | null;
  curlCommand: string | null;
  run: (body: TestConnectionRequestBody) => Promise<VectorStoreTestConnectionResponse | null>;
  reset: () => void;
}

export const useVectorStoreConnectionTest = (accessToken: string | null): ConnectionTestState => {
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<VectorStoreTestConnectionResponse | null>(null);
  const [curlCommand, setCurlCommand] = useState<string | null>(null);

  const run = useCallback(
    async (body: TestConnectionRequestBody) => {
      if (!accessToken) return null;
      setIsRunning(true);
      setCurlCommand(buildTestConnectionCurl(getProxyBaseUrl(), body));
      try {
        const response = await vectorStoreTestConnectionCall(accessToken, body);
        setResult(response);
        return response;
      } catch (error) {
        const failure = asFailedResult(error);
        setResult(failure);
        return failure;
      } finally {
        setIsRunning(false);
      }
    },
    [accessToken],
  );

  const reset = useCallback(() => {
    setResult(null);
    setCurlCommand(null);
  }, []);

  return { isRunning, result, curlCommand, run, reset };
};
