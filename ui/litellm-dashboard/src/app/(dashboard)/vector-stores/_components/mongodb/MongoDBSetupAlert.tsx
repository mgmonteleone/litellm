import React from "react";
import { Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import CopyButton from "@/components/shared/CopyButton";

const SIDECAR_DOCKER_RUN = [
  "docker run -d -p 8080:8080 \\",
  '  -e MONGODB_CONNECTION_STRING="mongodb+srv://..." \\',
  '  -e MONGODB_SIDECAR_API_KEY="pick-a-long-random-string" \\',
  "  ghcr.io/berriai/litellm-mongodb:latest",
].join("\n");

export const MongoDBSetupAlert: React.FC = () => (
  <Alert variant="info">
    <Info />
    <AlertTitle>MongoDB Atlas Setup</AlertTitle>
    <AlertDescription>
      <p>
        LiteLLM talks to Atlas through a small sidecar so the proxy never needs a MongoDB driver or your connection
        string. Run the sidecar next to LiteLLM, point it at your cluster, and give LiteLLM its URL and API key below.
      </p>
      <div className="relative mt-2 w-full">
        <pre className="overflow-x-auto rounded-sm border bg-card p-3 pr-10 font-mono text-xs text-foreground">
          {SIDECAR_DOCKER_RUN}
        </pre>
        <div className="absolute top-1.5 right-1.5">
          <CopyButton value={SIDECAR_DOCKER_RUN} label="Copy the docker run command" />
        </div>
      </div>
      <p className="mt-2">
        Then click Test connection. Every step is checked in order and each failure names its fix. See the{" "}
        <a
          href="https://docs.litellm.ai/docs/completion/knowledgebase"
          target="_blank"
          rel="noopener noreferrer"
          className="underline"
        >
          vector store docs
        </a>{" "}
        for the Atlas roles the sidecar needs.
      </p>
    </AlertDescription>
  </Alert>
);

export default MongoDBSetupAlert;
