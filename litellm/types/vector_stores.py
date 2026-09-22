from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, Field
from typing_extensions import ReadOnly, TypedDict


class SupportedVectorStoreIntegrations(str, Enum):
    """Supported vector store integrations."""

    BEDROCK = "bedrock"
    RAGFLOW = "ragflow"


class LiteLLM_VectorStoreConfig(TypedDict, total=False):
    """Parameters for initializing a vector store on Litellm proxy config.yaml"""

    vector_store_name: str
    litellm_params: dict[str, Any] | None


class LiteLLM_ManagedVectorStore(TypedDict, total=False):
    """LiteLLM managed vector store object - this is is the object stored in the database"""

    vector_store_id: str
    custom_llm_provider: str

    vector_store_name: str | None
    vector_store_description: str | None
    vector_store_metadata: dict[str, Any] | str | None
    created_at: datetime | None
    updated_at: datetime | None

    # credential fields
    litellm_credential_name: str | None

    # litellm_params
    litellm_params: dict[str, Any] | None

    # access control fields
    team_id: str | None
    user_id: str | None


class LiteLLM_ManagedVectorStoreListResponse(TypedDict, total=False):
    """Response format for listing vector stores"""

    object: Literal["list"]  # Always "list"
    data: list[LiteLLM_ManagedVectorStore]
    total_count: int | None
    current_page: int | None
    total_pages: int | None


class VectorStoreUpdateRequest(BaseModel):
    """Request litellm_params merge over the saved ones; a value equal to the redaction sentinel keeps the
    saved secret, exactly as VectorStoreTestConnectionRequest.litellm_params does for test_connection."""

    vector_store_id: str
    custom_llm_provider: str | None = None
    vector_store_name: str | None = None
    vector_store_description: str | None = None
    vector_store_metadata: dict | None = None
    litellm_params: Mapping[str, object] | None = None


class VectorStoreDeleteRequest(BaseModel):
    vector_store_id: str


class VectorStoreInfoRequest(BaseModel):
    vector_store_id: str


class VectorStoreResultContent(TypedDict, total=False):
    """Content of a vector store result"""

    text: str | None
    type: str | None


class VectorStoreSearchResult(TypedDict, total=False):
    """Result of a vector store search"""

    score: float | None
    content: list[VectorStoreResultContent] | None
    file_id: str | None
    filename: str | None
    attributes: dict | None


class VectorStoreSearchResponse(TypedDict, total=False):
    """Response after searching a vector store"""

    object: Literal["vector_store.search_results.page"]  # Always "vector_store.search_results.page"
    search_query: str | None
    data: list[VectorStoreSearchResult] | None


VectorStoreSearchFailureMode = Literal["annotate", "error"]


class VectorStoreSearchFailure(TypedDict):
    """A configured vector store whose search failed, as reported back to the API caller"""

    vector_store_id: ReadOnly[str]
    custom_llm_provider: ReadOnly[str | None]
    error: ReadOnly[str]


class VectorStoreComparisonFilter(TypedDict, total=False):
    """OpenAI vector store comparison filter: {"type": "eq", "key": "department", "value": "hr"}"""

    type: ReadOnly[Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]]
    key: ReadOnly[str]
    value: ReadOnly[
        str | int | float | bool | None | list[str | int | float | bool | None]
    ]  # mutable-ok: OpenAI JSON schema


class VectorStoreCompoundFilter(TypedDict, total=False):
    """OpenAI vector store compound filter: {"type": "and", "filters": [...]}"""

    type: ReadOnly[Literal["and", "or"]]
    filters: ReadOnly[list["VectorStoreComparisonFilter | VectorStoreCompoundFilter"]]  # mutable-ok: OpenAI JSON schema


class VectorStoreRankingOptions(TypedDict, total=False):
    """OpenAI vector store ranking options; providers may accept additional ranker names (MongoDB: "hybrid")."""

    ranker: ReadOnly[str | None]
    score_threshold: ReadOnly[float | None]


CheckStatus = Literal["pass", "warn", "fail", "skip"]


class VectorStoreConnectionCheck(TypedDict, total=False):
    """One step of a provider's connection checklist, with a message that names the fix when it fails."""

    check: ReadOnly[str]
    status: ReadOnly[CheckStatus]
    message: ReadOnly[str]
    details: ReadOnly[dict | None]  # mutable-ok: JSON response


class VectorStoreTestConnectionResponse(TypedDict, total=False):
    """Result of POST /vector_store/test_connection"""

    ok: ReadOnly[bool]
    supported: ReadOnly[bool]
    custom_llm_provider: ReadOnly[str]
    summary: ReadOnly[str]
    checks: ReadOnly[list[VectorStoreConnectionCheck]]  # mutable-ok: JSON response
    details: ReadOnly[dict | None]  # mutable-ok: JSON response


class VectorStoreTestConnectionRequest(BaseModel):
    """Test a saved store (vector_store_id) or an unsaved configuration (custom_llm_provider + litellm_params).

    Request litellm_params override the saved ones; a value equal to the redaction sentinel keeps the saved secret.
    """

    vector_store_id: str | None = None
    custom_llm_provider: str | None = None
    litellm_params: dict[str, Any] | None = None  # mutable-ok: request body
    litellm_credential_name: str | None = None


class VectorStoreDiscoverRequest(VectorStoreTestConnectionRequest):
    """Ask a provider to list databases, collections, indexes, or suggested fields for the dashboard."""

    kind: Literal["databases", "collections", "indexes", "fields"]
    options: dict[str, Any] = Field(default_factory=dict)  # mutable-ok: request body


class VectorStoreProviderDefaultsResponse(TypedDict, total=False):
    """Result of GET /vector_store/provider_defaults. The sidecar's own API key is never returned, only
    whether the deployment has one configured, so the dashboard can offer to use it without displaying it."""

    custom_llm_provider: ReadOnly[str]
    api_base: ReadOnly[str | None]
    api_key_configured: ReadOnly[bool]


class VectorStoreSearchOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the vector store search API."""

    filters: dict | None
    max_num_results: int | None
    ranking_options: dict | None
    rewrite_query: bool | None


class VectorStoreSearchRequest(VectorStoreSearchOptionalRequestParams, total=False):
    """Request body for searching a vector store"""

    query: str | list[str]


class VertexSearchDataStoreExtraBody(TypedDict, total=False):
    """
    Native Discovery Engine ``SearchRequest`` fields callers may forward via
    ``extra_body`` when searching a Vertex AI Search **data store** serving
    config (``.../dataStores/{id}/servingConfigs/default_config``).

    The data store is scoped by the request URL path, so target-selecting
    fields (``servingConfig``, ``branch``, ``entity``) are intentionally
    omitted and rejected by the transformation layer. Engine/app-only fields
    such as ``dataStoreSpecs`` and ``numResultsPerDataStore`` live on
    ``VertexSearchEngineExtraBody`` instead.
    """

    query: str
    pageSize: int
    pageToken: str
    offset: int
    oneBoxPageSize: int
    pageCategories: Sequence[str]
    imageQuery: Mapping[str, object]
    filter: str
    canonicalFilter: str
    orderBy: str
    userInfo: Mapping[str, object]
    languageCode: str
    facetSpecs: Sequence[Mapping[str, object]]
    boostSpec: Mapping[str, object]
    params: Mapping[str, object]
    queryExpansionSpec: Mapping[str, object]
    spellCorrectionSpec: Mapping[str, object]
    userPseudoId: str
    contentSearchSpec: Mapping[str, object]
    rankingExpression: str
    rankingExpressionBackend: str
    safeSearch: bool
    userLabels: Mapping[str, str]
    naturalLanguageQueryUnderstandingSpec: Mapping[str, object]
    searchAsYouTypeSpec: Mapping[str, object]
    displaySpec: Mapping[str, object]
    crowdingSpecs: Sequence[Mapping[str, object]]
    relevanceThreshold: str
    relevanceScoreSpec: Mapping[str, object]
    customRankingParams: Mapping[str, object]


class VertexSearchEngineExtraBody(VertexSearchDataStoreExtraBody, total=False):
    """
    Native Discovery Engine ``SearchRequest`` fields callers may forward via
    ``extra_body`` when searching a Vertex AI Search **engine/app** serving
    config (``.../engines/{id}/servingConfigs/default_serving_config``).

    Inherits every data-store field and adds fields that only make sense when
    an app fans out across multiple member data stores, e.g. ``dataStoreSpecs``
    (per-store scoping/filtering) and ``numResultsPerDataStore``.
    """

    dataStoreSpecs: Sequence[Mapping[str, object]]
    numResultsPerDataStore: int


# Vector Store Creation Types
class VectorStoreExpirationPolicy(TypedDict, total=False):
    """The expiration policy for a vector store"""

    anchor: Literal["last_active_at"]  # Anchor timestamp after which the expiration policy applies
    days: int  # Number of days after anchor time that the vector store will expire


class VectorStoreAutoChunkingStrategy(TypedDict, total=False):
    """Auto chunking strategy configuration"""

    type: Literal["auto"]  # Always "auto"


class VectorStoreStaticChunkingStrategyConfig(TypedDict, total=False):
    """Static chunking strategy configuration"""

    max_chunk_size_tokens: int  # Maximum number of tokens per chunk
    chunk_overlap_tokens: int  # Number of tokens to overlap between chunks


class VectorStoreStaticChunkingStrategy(TypedDict, total=False):
    """Static chunking strategy"""

    type: Literal["static"]  # Always "static"
    static: VectorStoreStaticChunkingStrategyConfig


class VectorStoreChunkingStrategy(TypedDict, total=False):
    """Union type for chunking strategies"""

    # This can be either auto or static
    type: Literal["auto", "static"]
    static: VectorStoreStaticChunkingStrategyConfig | None


class VectorStoreFileCounts(TypedDict, total=False):
    """File counts for a vector store"""

    in_progress: int
    completed: int
    failed: int
    cancelled: int
    total: int


class VectorStoreCreateOptionalRequestParams(TypedDict, total=False):
    """TypedDict for Optional parameters supported by the vector store create API."""

    name: str | None  # Name of the vector store
    file_ids: list[str] | None  # List of File IDs that the vector store should use
    expires_after: VectorStoreExpirationPolicy | None  # Expiration policy for the vector store
    chunking_strategy: VectorStoreChunkingStrategy | None  # Chunking strategy for the files
    metadata: dict[str, str] | None  # Set of key-value pairs for metadata


class VectorStoreCreateRequest(VectorStoreCreateOptionalRequestParams, total=False):
    """Request body for creating a vector store"""

    # All fields are optional for vector store creation


class VectorStoreCreateResponse(TypedDict, total=False):
    """Response after creating a vector store"""

    id: str  # ID of the vector store
    object: Literal["vector_store"]  # Always "vector_store"
    created_at: int  # Unix timestamp of when the vector store was created
    name: str | None  # Name of the vector store
    bytes: int  # Size of the vector store in bytes
    file_counts: VectorStoreFileCounts  # File counts for the vector store
    status: Literal["expired", "in_progress", "completed"]  # Status of the vector store
    expires_after: VectorStoreExpirationPolicy | None  # Expiration policy
    expires_at: int | None  # Unix timestamp of when the vector store expires
    last_active_at: int | None  # Unix timestamp of when the vector store was last active
    metadata: dict[str, str] | None  # Metadata associated with the vector store


class IndexCreateLiteLLMParams(BaseModel):
    vector_store_index: str
    vector_store_name: str


class IndexCreateRequest(BaseModel):
    index_name: str
    litellm_params: IndexCreateLiteLLMParams
    index_info: dict[str, object] | None = None


class BaseVectorStoreAuthCredentials(TypedDict, total=False):
    headers: dict
    query_params: dict


class LiteLLM_ManagedVectorStoreIndex(BaseModel):
    """LiteLLM managed vector store index object - this is is the object stored in the database"""

    id: str
    index_name: str
    litellm_params: IndexCreateLiteLLMParams
    index_info: dict[str, object] | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class IndexListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: tuple[LiteLLM_ManagedVectorStoreIndex, ...]


class VectorStoreIndexType(str, Enum):
    """Type of vector store index"""

    READ = "read"
    WRITE = "write"


class VectorStoreIndexEndpoints(TypedDict):
    """Endpoints for vector store index"""

    read: list[
        tuple[Literal["GET", "POST", "PUT", "DELETE", "PATCH"], str]
    ]  # endpoints for reading a vector store index
    write: list[
        tuple[Literal["GET", "POST", "PUT", "DELETE", "PATCH"], str]
    ]  # endpoints for writing a vector store index


MANAGED_STORE_CALLER_OPTIONS: Final = frozenset(
    {
        "vector_store_id",
        "data_source_id",
        "wait_for_ingestion",
        "ingestion_timeout",
        "custom_metadata",
        "file_description",
        "max_embedding_requests_per_min",
    }
)
"""Per-request ingest options a caller may send for a managed store, as opposed to the
store's own configuration. Providers treat them as recognised keys rather than typos."""

NON_ADMIN_VECTOR_STORE_PARAMS: Final = frozenset(
    {
        *MANAGED_STORE_CALLER_OPTIONS,
        "litellm_embedding_model",
        "embedding_model",
        "ttl_days",
        "wait_for_import",
        "import_timeout",
        "s3_prefix",
        "vector_bucket_name",
        "index_name",
        "dimension",
        "distance_metric",
        "non_filterable_metadata_keys",
        "vertex_collection_id",
        "vertex_engine_id",
        "azure_search_vector_field",
        "milvus_db_name",
        "milvus_partition_names",
        "milvus_text_field",
        "valkey_text_field",
        "valkey_embedding_field",
        "mongodb_database",
        "mongodb_collection",
        "mongodb_text_field",
        "mongodb_embedding_field",
        "mongodb_num_candidates",
        "mongodb_dimensions",
        "mongodb_similarity",
        "mongodb_filter_fields",
        "mongodb_text_index",
        "mongodb_hybrid_search",
        "mongodb_hybrid_weights",
        "mongodb_exact_search",
        "mongodb_score_threshold",
    }
)
"""The only litellm_params a non-admin may set, change or clear on a vector store: what data to read and how to
shape it, never where traffic goes or what signs it. Each one only reaches a request path or body. Values must be
scalars or lists of scalars, except the keys in NON_ADMIN_VECTOR_STORE_MAPPING_PARAMS."""

NON_ADMIN_VECTOR_STORE_MAPPING_PARAMS: Final = frozenset({"custom_metadata", "mongodb_hybrid_weights"})

VECTOR_STORE_ENDPOINT_KEYS: Final = frozenset(
    {
        "api_base",
        "base_url",
        "endpoint",
        "azure_endpoint",
        "azure_search_service_name",
        "aws_bedrock_runtime_endpoint",
        "aws_sts_endpoint",
        "aws_region_name",
        "aws_role_name",
        "aws_profile_name",
        "aws_session_name",
        "aws_external_id",
        "aws_web_identity_token",
        "tenant_id",
        "client_id",
        "vertex_project",
        "vertex_ai_project",
        "vertex_location",
        "vertex_ai_location",
        "valkey_host",
        "valkey_port",
        "valkey_ssl",
    }
)
"""Params that decide where a vector store sends its traffic, or which cloud identity signs it. Only proxy admins
may set them, and a managed store's saved values win over anything a request carries."""


VECTOR_STORE_OPENAI_PARAMS = Literal[
    "filters",
    "max_num_results",
    "ranking_options",
    "rewrite_query",
]


@dataclass
class VectorStoreToolParams:
    """Parameters extracted from a file_search tool definition"""

    filters: dict | None = None
    max_num_results: int | None = None
    ranking_options: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values"""
        return {
            k: v
            for k, v in {
                "filters": self.filters,
                "max_num_results": self.max_num_results,
                "ranking_options": self.ranking_options,
            }.items()
            if v is not None
        }
