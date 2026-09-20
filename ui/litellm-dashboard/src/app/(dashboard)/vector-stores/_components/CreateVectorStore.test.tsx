import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import CreateVectorStore from "./CreateVectorStore";
import * as networking from "@/components/networking";
import { getProviderSpecificFields } from "@/components/vector_store_providers";

// Mock the networking module
vi.mock("@/components/networking", () => ({
  ragIngestCall: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

// Mock vector_store_providers
vi.mock("@/components/vector_store_providers", () => ({
  VectorStoreProviders: {
    BEDROCK: "Amazon Bedrock",
    OPENAI: "OpenAI",
    AZURE_OPENAI: "Azure OpenAI",
    S3Vectors: "AWS S3 Vectors",
    Valkey: "Valkey",
  },
  vectorStoreProviderMap: {
    BEDROCK: "bedrock",
    OPENAI: "openai",
    AZURE_OPENAI: "azure_openai",
    S3Vectors: "s3_vectors",
    Valkey: "valkey",
  },
  vectorStoreProviderLogoMap: {
    "Amazon Bedrock": "https://example.com/bedrock.png",
    OpenAI: "https://example.com/openai.png",
    "Azure OpenAI": "https://example.com/azure.png",
    "AWS S3 Vectors": "https://example.com/aws.png",
    Valkey: "https://example.com/valkey.svg",
  },
  getProviderSpecificFields: vi.fn((provider: string) => {
    if (provider === "s3_vectors") {
      return [
        {
          name: "vector_bucket_name",
          label: "Vector Bucket Name",
          tooltip: "S3 bucket name for vector storage",
          placeholder: "my-vector-bucket",
          required: true,
          type: "text",
        },
        {
          name: "aws_region_name",
          label: "AWS Region",
          tooltip: "AWS region",
          placeholder: "us-west-2",
          required: true,
          type: "text",
        },
        {
          name: "embedding_model",
          label: "Embedding Model",
          tooltip: "Embedding model to use",
          placeholder: "text-embedding-3-small",
          required: true,
          type: "select",
        },
      ];
    }
    return [];
  }),
}));

describe("CreateVectorStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the component successfully", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getAllByText("Create Vector Store").length).toBeGreaterThan(0);
    expect(screen.getByText("Step 1: Upload Documents")).toBeInTheDocument();
    expect(screen.getByText("Step 2: Configure Vector Store")).toBeInTheDocument();
  });

  it("should display upload area with correct text", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getByText("Click or drag files to this area to upload")).toBeInTheDocument();
    expect(screen.getByText(/Support for single or bulk upload/)).toBeInTheDocument();
  });

  it("should show the provider display name on the trigger rather than its wire value", async () => {
    const user = userEvent.setup();
    render(<CreateVectorStore accessToken="test-token" />);

    const providerSelect = screen.getByRole("combobox", { name: /Provider/ });
    expect(providerSelect).toHaveTextContent("Amazon Bedrock");

    await user.click(providerSelect);
    await user.click(await screen.findByText("AWS S3 Vectors"));

    expect(screen.getByRole("combobox", { name: /Provider/ })).toHaveTextContent("AWS S3 Vectors");
  });

  it("should have provider selection dropdown", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getByText("Provider")).toBeInTheDocument();
  });

  it("should have create button disabled initially when no documents", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });
    expect(createButton).toBeDisabled();
  });

  it("should show uploaded documents table when files are added", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    // Create a mock file
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });

    // Find the upload input (it's hidden but accessible)
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });
  });

  it("should call ragIngestCall when create button is clicked", async () => {
    const mockRagIngestCall = vi.spyOn(networking, "ragIngestCall");
    mockRagIngestCall.mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });

    const onSuccess = vi.fn();
    render(<CreateVectorStore accessToken="test-token" onSuccess={onSuccess} />);

    // Create a mock file
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    // Wait for file to be added
    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });

    // Click create button
    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });

    await act(async () => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(mockRagIngestCall).toHaveBeenCalledWith(
        "test-token",
        expect.any(File),
        "bedrock",
        undefined,
        undefined,
        undefined,
        {},
        undefined,
      );
    });
  });

  it("should display success message after successful creation", async () => {
    const mockRagIngestCall = vi.spyOn(networking, "ragIngestCall");
    mockRagIngestCall.mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });

    render(<CreateVectorStore accessToken="test-token" />);

    // Create and upload a mock file
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });

    // Click create button
    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });

    await act(async () => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Vector Store Created Successfully")).toBeInTheDocument();
    });
  });

  it("should exclude valkey from the provider dropdown since it has no RAG ingestion", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    const providerSelect = screen.getByRole("combobox");

    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      expect(screen.getByText("AWS S3 Vectors")).toBeInTheDocument();
    });
    expect(screen.queryByText("Valkey")).not.toBeInTheDocument();
  });

  it("should display S3 Vectors provider-specific fields when selected", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    // Find and click the provider dropdown
    const providerSelect = screen.getByRole("combobox");

    await userEvent.click(providerSelect);

    // Wait for dropdown options to appear
    await userEvent.click(await screen.findByText("AWS S3 Vectors"));

    // Check if S3-specific fields are displayed
    await waitFor(() => {
      expect(screen.getByText("Vector Bucket Name")).toBeInTheDocument();
      expect(screen.getByText("AWS Region")).toBeInTheDocument();
      expect(screen.getByText("Embedding Model")).toBeInTheDocument();
    });
  });

  it("should validate S3 Vectors required fields before submission", async () => {
    render(<CreateVectorStore accessToken="test-token" />);

    // Upload a file first
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;

    await act(async () => {
      if (uploadInput) {
        fireEvent.change(uploadInput, { target: { files: [file] } });
      }
    });

    await waitFor(() => {
      expect(screen.getByText("Uploaded Documents (1)")).toBeInTheDocument();
    });

    // Select S3 Vectors provider
    const providerSelect = screen.getByRole("combobox");

    await userEvent.click(providerSelect);

    await userEvent.click(await screen.findByText("AWS S3 Vectors"));

    // Try to create without filling required fields
    const createButton = screen.getByRole("button", { name: /Create Vector Store/i });

    await act(async () => {
      fireEvent.click(createButton);
    });

    // Should show validation warning (mocked message.warning would be called)
    // The actual validation happens in the component
  });
});

describe("CreateVectorStore boolean, list and weight provider fields", () => {
  const BOOLEAN_LIST_WEIGHT_FIELDS = [
    {
      name: "hybrid_enabled",
      label: "Hybrid Search",
      tooltip: "Blend vector similarity with keyword matching",
      required: false,
      type: "boolean" as const,
    },
    {
      name: "filter_fields",
      label: "Filter Fields",
      tooltip: "Fields to filter on, comma separated",
      placeholder: "metadata.department, metadata.year",
      required: false,
      type: "string-list" as const,
    },
    {
      name: "vector_weight",
      label: "Vector Weight",
      tooltip: "How much of the rank comes from vector similarity",
      required: false,
      type: "weight-split" as const,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    // The default selected provider (bedrock) gets these fields for this describe block only.
    vi.mocked(getProviderSpecificFields).mockImplementation((provider: string) =>
      provider === "bedrock" ? BOOLEAN_LIST_WEIGHT_FIELDS : [],
    );
  });

  const uploadFile = async () => {
    const file = new File(["test content"], "test.pdf", { type: "application/pdf" });
    const uploadInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    await act(async () => {
      fireEvent.change(uploadInput, { target: { files: [file] } });
    });
    expect(await screen.findByText("Uploaded Documents (1)")).toBeInTheDocument();
  };

  it("renders a switch for a boolean field and a slider for a weight-split field, not bare text inputs", () => {
    render(<CreateVectorStore accessToken="test-token" />);

    expect(screen.getByRole("switch", { name: /Hybrid Search/ })).toBeInTheDocument();
    expect(screen.getAllByRole("slider", { hidden: true }).length).toBeGreaterThan(0);
    expect(screen.getByPlaceholderText("metadata.department, metadata.year")).toBeInTheDocument();
  });

  it("submits a boolean field as true and a string-list field as an array, not as literal text", async () => {
    const mockRagIngestCall = vi.spyOn(networking, "ragIngestCall");
    mockRagIngestCall.mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });
    const user = userEvent.setup({ pointerEventsCheck: 0 });

    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();

    await user.click(screen.getByRole("switch", { name: /Hybrid Search/ }));
    fireEvent.change(screen.getByPlaceholderText("metadata.department, metadata.year"), {
      target: { value: "metadata.department, metadata.year" },
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Create Vector Store/i }));
    });

    await waitFor(() => expect(mockRagIngestCall).toHaveBeenCalledTimes(1));
    const providerParams = mockRagIngestCall.mock.calls[0][6];
    expect(providerParams).toMatchObject({
      hybrid_enabled: true,
      filter_fields: ["metadata.department", "metadata.year"],
    });
  });

  it("clamps the weight slider to [0, 1] and submits it as a vector/text split object", async () => {
    const mockRagIngestCall = vi.spyOn(networking, "ragIngestCall");
    mockRagIngestCall.mockResolvedValue({
      id: "test-id",
      status: "completed",
      vector_store_id: "vs_123",
      file_id: "file_123",
    });

    render(<CreateVectorStore accessToken="test-token" />);
    await uploadFile();

    const slider = screen.getAllByRole("slider", { hidden: true })[0];
    fireEvent.keyDown(slider, { key: "ArrowRight", keyCode: 39, which: 39 });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Create Vector Store/i }));
    });

    await waitFor(() => expect(mockRagIngestCall).toHaveBeenCalledTimes(1));
    const providerParams = mockRagIngestCall.mock.calls[0][6] as Record<string, unknown>;
    const weight = providerParams.vector_weight as { vector: number; text: number };
    expect(weight.vector).toBeGreaterThanOrEqual(0);
    expect(weight.vector).toBeLessThanOrEqual(1);
    expect(weight.vector + weight.text).toBeCloseTo(1);
  });
});
