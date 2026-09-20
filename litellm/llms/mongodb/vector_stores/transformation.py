from collections.abc import Mapping, Sequence
from ipaddress import ip_address
from math import isfinite
from time import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import AuthenticationError, BadRequestError, ServiceUnavailableError, Timeout
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    LiteLLMVectorStoreEmbeddingExecutor,
    VectorStoreEmbeddingExecutor,
)
from litellm.llms.mongodb.vector_stores.filters import translate_filters
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreFileCounts,
    VectorStoreIndexEndpoints,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreTestConnectionResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_EMBEDDING_FIELD_NAME: Final = "embedding"
DEFAULT_TEXT_FIELD_NAME: Final = "text"
DEFAULT_SIMILARITY: Final = "cosine"
DEFAULT_MAX_NUM_RESULTS: Final = 10
MIN_MAX_NUM_RESULTS: Final = 1
MAX_MAX_NUM_RESULTS: Final = 50
NUM_CANDIDATES_MULTIPLIER: Final = 10
MIN_NUM_CANDIDATES: Final = 100
MAX_NUM_CANDIDATES: Final = 10_000
MAX_QUERY_CHARACTERS: Final = 32_000
MAX_DIMENSIONS: Final = 8192
DIMENSION_PROBE_TEXT: Final = "LiteLLM embedding dimension probe"
_EMPTY_EMBEDDING_CONFIG: Final = MappingProxyType({})
_SIDECAR_TOO_OLD_MESSAGE: Final = (
    "The MongoDB sidecar does not support this operation. Creating and ingesting vector stores "
    "requires litellm-mongodb v0.2 or later; upgrade the sidecar image."
)

Similarity = Literal["cosine", "euclidean", "dotProduct"]
HYBRID_RANKER: Final = "hybrid"
_SUPPORTED_OPENAI_PARAMS: Final = ("filters", "max_num_results", "ranking_options")


def config_error(message: str) -> BadRequestError:
    return BadRequestError(message=message, model=None, llm_provider="mongodb")


class _Content(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    type: Literal["text"]
    text: str


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False, extra="ignore")
    score: float | None
    content: Sequence[_Content]
    file_id: str | None
    filename: str | None
    attributes: Mapping[str, str | int | float | bool | None] | None = None


class _SearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    object: Literal["vector_store.search_results.page"]
    search_query: str
    data: Sequence[_Result]


class _CreateResponse(BaseModel):
    """The sidecar's index status document, validated strictly before it becomes an OpenAI-shaped response."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    index_name: str
    mongodb_database: str
    mongodb_collection: str
    status: Literal["ready", "building", "failed", "missing"]
    queryable: bool
    document_count: int | None
    created: bool


class MongoDBVectorStoreParams(BaseModel):
    """Typed view over the vector store's litellm_params; unrelated keys are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_embedding_model: str | None = None
    litellm_embedding_config: Mapping[str, object] | None = None
    mongodb_database: str | None = None
    mongodb_collection: str | None = None
    mongodb_text_field: str | None = None
    mongodb_embedding_field: str | None = None
    mongodb_num_candidates: int | None = None
    mongodb_dimensions: int | None = None
    mongodb_similarity: Similarity | None = None
    mongodb_filter_fields: Sequence[str] | None = None
    mongodb_text_index: str | None = None
    mongodb_hybrid_search: bool | None = None
    mongodb_hybrid_weights: Mapping[str, float] | None = None
    mongodb_exact_search: bool | None = None
    mongodb_score_threshold: float | None = None

    @property
    def text_field(self) -> str:
        return self.mongodb_text_field or DEFAULT_TEXT_FIELD_NAME

    @property
    def embedding_field(self) -> str:
        return self.mongodb_embedding_field or DEFAULT_EMBEDDING_FIELD_NAME

    @property
    def similarity(self) -> Similarity:
        return self.mongodb_similarity or DEFAULT_SIMILARITY

    @property
    def filter_fields(self) -> tuple[str, ...]:
        return tuple(self.mongodb_filter_fields or ())

    def require_text_index(self) -> str:
        if not self.mongodb_text_index:
            raise config_error(
                "Hybrid search needs an Atlas Search text index. Set mongodb_text_index in litellm_params "
                "(it is created automatically alongside the vector index when the store is created through LiteLLM)."
            )
        return self.mongodb_text_index

    def hybrid_weights(self) -> tuple[float, float]:
        weights: Final = self.mongodb_hybrid_weights or _EMPTY_EMBEDDING_CONFIG
        vector: Final = weights.get("vector", 1.0)
        text: Final = weights.get("text", 1.0)
        for name, value in (("vector", vector), ("text", text)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) or value < 0:
                raise config_error(f"mongodb_hybrid_weights.{name} must be a non-negative number")
        if vector == 0 and text == 0:
            raise config_error("mongodb_hybrid_weights must give vector or text a positive weight")
        return float(vector), float(text)

    def require_embedding_model(self) -> str:
        if not self.litellm_embedding_model:
            raise config_error(
                "litellm_embedding_model is required in litellm_params for the MongoDB vector store. "
                "It must be the same model that produced the vectors stored in "
                f"'{self.mongodb_collection or '<collection>'}.{self.embedding_field}', or search results "
                "will be meaningless. Example: litellm_embedding_model: openai/text-embedding-3-small"
            )
        return self.litellm_embedding_model

    def require_database(self) -> str:
        if not self.mongodb_database:
            raise config_error(
                "mongodb_database is required in litellm_params for the MongoDB vector store. "
                "Example: mongodb_database: sample_mflix"
            )
        return self.mongodb_database

    def require_collection(self) -> str:
        if not self.mongodb_collection:
            raise config_error(
                "mongodb_collection is required in litellm_params for the MongoDB vector store. "
                "Example: mongodb_collection: embedded_movies"
            )
        return self.mongodb_collection


_MONGODB_PARAM_PREFIX: Final = "mongodb_"
_KNOWN_MONGODB_PARAMS: Final = frozenset(
    name for name in MongoDBVectorStoreParams.model_fields if name.startswith(_MONGODB_PARAM_PREFIX)
)
_RESPONSE_ADAPTER: Final = TypeAdapter(VectorStoreSearchResponse)


def reject_unknown_params(litellm_params: Mapping[str, object]) -> None:
    """Without this a mistyped mongodb_collection reads as 'mongodb_collection is required',
    naming a key the reader can see they have set."""
    if litellm_params.get("mongodb_connection_string") is not None:
        raise config_error(
            "MongoDB vector stores now use the BETA sidecar. Move mongodb_connection_string to "
            "MONGODB_CONNECTION_STRING in the sidecar, remove it from LiteLLM, and configure api_base and api_key."
        )
    unknown: Final = sorted(
        key for key in litellm_params if key.startswith(_MONGODB_PARAM_PREFIX) and key not in _KNOWN_MONGODB_PARAMS
    )
    if unknown:
        raise config_error(
            f"Unrecognised MongoDB vector store parameter(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(_KNOWN_MONGODB_PARAMS))}."
        )


def validated_params(litellm_params: Mapping[str, object]) -> MongoDBVectorStoreParams:
    """Validate litellm_params for any MongoDB operation; search, create, and ingest share these rules."""
    reject_unknown_params(litellm_params)
    try:
        params: Final = MongoDBVectorStoreParams.model_validate(litellm_params)
    except ValidationError:
        raise config_error(
            "Invalid MongoDB vector-store configuration. Check the database, collection, fields, "
            "similarity, filter fields, and candidate count."
        ) from None
    params.require_database()
    params.require_collection()
    if params.mongodb_dimensions is not None and not 1 <= params.mongodb_dimensions <= MAX_DIMENSIONS:
        raise config_error(
            f"mongodb_dimensions must be between 1 and {MAX_DIMENSIONS}, got {params.mongodb_dimensions}"
        )
    for field in params.filter_fields:
        if not field.strip() or field.startswith("$"):
            raise config_error("mongodb_filter_fields entries must be nonblank field paths that do not start with $")
    if params.mongodb_score_threshold is not None and not 0.0 <= params.mongodb_score_threshold <= 1.0:
        raise config_error("mongodb_score_threshold must be between 0 and 1")
    return params


def score_threshold(params: MongoDBVectorStoreParams, ranking_options: Mapping[str, object] | None) -> float | None:
    requested: Final = ranking_options.get("score_threshold") if ranking_options else None
    if requested is None:
        return params.mongodb_score_threshold
    if isinstance(requested, bool) or not isinstance(requested, (int, float)) or not 0.0 <= requested <= 1.0:
        raise config_error("ranking_options.score_threshold must be a number between 0 and 1")
    return float(requested)


def hybrid_requested(params: MongoDBVectorStoreParams, ranking_options: Mapping[str, object] | None) -> bool:
    ranker: Final = ranking_options.get("ranker") if ranking_options else None
    if ranker is not None and ranker not in ("auto", "default-2024-11-15", HYBRID_RANKER):
        raise config_error(f"ranking_options.ranker {ranker!r} is not supported; use 'auto' or 'hybrid'")
    return ranker == HYBRID_RANKER or bool(params.mongodb_hybrid_search)


def embedding_vector(embedding_response: EmbeddingResponse) -> tuple[float, ...]:
    if not embedding_response.data:
        raise config_error("The embedding model returned no embedding. Check litellm_embedding_model.")
    vector: Final = embedding_response.data[0]["embedding"]
    if not vector or any(not isinstance(value, (float, int)) or not isfinite(value) for value in vector):
        raise config_error("The embedding model must return a non-empty, finite vector.")
    return tuple(vector)


class MongoDBVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig):
    def __init__(self, embedding_executor: VectorStoreEmbeddingExecutor | None = None) -> None:
        self.embedding_executor: Final = embedding_executor or LiteLLMVectorStoreEmbeddingExecutor()

    def get_auth_credentials(self, litellm_params: Mapping[str, object]) -> BaseVectorStoreAuthCredentials:
        return BaseVectorStoreAuthCredentials()

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return VectorStoreIndexEndpoints(read=[], write=[])  # mutable-ok: the TypedDict declares list fields

    def get_supported_openai_params(self, model: str) -> list[VECTOR_STORE_OPENAI_PARAMS]:  # mutable-ok: base contract
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: the base contract returns a list

    @staticmethod
    def _reject_unknown_params(litellm_params: Mapping[str, object]) -> None:
        reject_unknown_params(litellm_params)

    @staticmethod
    def _query_text(query: str | Sequence[str]) -> str:
        text: Final = query if isinstance(query, str) else " ".join(query)
        if not text.strip():
            raise config_error("query must not be empty")
        if len(text) > MAX_QUERY_CHARACTERS:
            raise config_error(f"query must be at most {MAX_QUERY_CHARACTERS} characters, got {len(text)}")
        return text

    @staticmethod
    def _limit(vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams) -> int:
        requested: Final = vector_store_search_optional_params.get("max_num_results")
        if requested is None:
            return DEFAULT_MAX_NUM_RESULTS
        if not MIN_MAX_NUM_RESULTS <= requested <= MAX_MAX_NUM_RESULTS:
            raise config_error(
                f"max_num_results must be between {MIN_MAX_NUM_RESULTS} and {MAX_MAX_NUM_RESULTS}, got {requested}"
            )
        return requested

    @staticmethod
    def _num_candidates(limit: int, configured: int | None) -> int:
        if configured is not None:
            if not limit <= configured <= MAX_NUM_CANDIDATES:
                raise config_error(
                    f"mongodb_num_candidates must be between max_num_results ({limit}) and "
                    f"{MAX_NUM_CANDIDATES}, got {configured}"
                )
            return configured
        return min(max(limit * NUM_CANDIDATES_MULTIPLIER, MIN_NUM_CANDIDATES), MAX_NUM_CANDIDATES)

    def validate_environment(
        self, headers: Mapping[str, object], litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, object]:  # mutable-ok: the shared HTTP handler requires writable headers
        if litellm_params is None:
            raise config_error("Configure api_base and api_key for the MongoDB BETA sidecar.")
        self._reject_unknown_params(MappingProxyType(dict(litellm_params)))
        api_key: Final = litellm_params.api_key or get_secret_str("MONGODB_SIDECAR_API_KEY")
        if not api_key:
            raise config_error("MongoDB sidecar api_key is required. Set api_key or MONGODB_SIDECAR_API_KEY.")
        return {
            **headers,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }  # mutable-ok: writable HTTP headers

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        if not api_base:
            raise config_error("MongoDB sidecar api_base is required, for example http://127.0.0.1:8080.")
        try:
            parsed: Final = urlsplit(api_base)
            valid: Final = parsed.scheme in ("http", "https") and bool(parsed.hostname) and parsed.port != 0
        except ValueError:
            raise config_error("MongoDB sidecar api_base must be a valid HTTP or HTTPS URL.") from None
        if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise config_error(
                "MongoDB sidecar api_base must be an HTTP or HTTPS URL without credentials, query, or fragment."
            )
        if parsed.scheme == "http":
            try:
                loopback: Final = ip_address(parsed.hostname or "").is_loopback
            except ValueError:
                raise config_error(
                    "MongoDB sidecar requires HTTPS. HTTP is supported only for a loopback IP such as 127.0.0.1."
                ) from None
            if not loopback:
                raise config_error(
                    "MongoDB sidecar requires HTTPS. HTTP is supported only for a loopback IP such as 127.0.0.1."
                )
        return api_base.rstrip("/")

    @staticmethod
    def _timeout_ms(value: object) -> int:
        seconds: Final = value.read if isinstance(value, httpx.Timeout) else value
        if seconds is None:
            return 30_000
        if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds <= 0:
            raise config_error("MongoDB search timeout must be a positive finite number.")
        try:
            return max(1, int(seconds * 1000))
        except (ValueError, OverflowError):
            raise config_error("MongoDB search timeout must be a positive finite number.") from None

    @classmethod
    def _params(
        cls,
        litellm_params: Mapping[str, object],
        optional_params: VectorStoreSearchOptionalRequestParams,
        extra_body: Mapping[str, object] | None,
    ) -> MongoDBVectorStoreParams:
        if extra_body:
            raise config_error("MongoDB vector store does not support extra_body overrides.")
        if optional_params.get("rewrite_query") is not None:
            raise config_error("MongoDB vector store does not support the rewrite_query parameter.")
        params: Final = validated_params(litellm_params)
        params.require_embedding_model()
        cls._num_candidates(cls._limit(optional_params), params.mongodb_num_candidates)
        cls._timeout_ms(litellm_params.get("timeout"))
        translate_filters(optional_params.get("filters"))
        ranking: Final = optional_params.get("ranking_options")
        if hybrid_requested(params, ranking):
            params.require_text_index()
            params.hybrid_weights()
            if score_threshold(params, ranking) is not None:
                raise config_error(
                    "score_threshold applies to vector similarity and cannot be combined with hybrid ranking"
                )
        else:
            score_threshold(params, ranking)
        return params

    @classmethod
    def _request(
        cls,
        vector_store_id: str,
        query_text: str,
        params: MongoDBVectorStoreParams,
        optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        embedding_response: EmbeddingResponse,
        timeout: object,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        vector: Final = embedding_vector(embedding_response)
        limit: Final = cls._limit(optional_params)
        ranking: Final = optional_params.get("ranking_options")
        body: dict[str, object] = {  # mutable-ok: JSON transport requires a dict
            "query": query_text,
            "query_vector": vector,
            "mongodb_database": params.require_database(),
            "mongodb_collection": params.require_collection(),
            "mongodb_embedding_field": params.embedding_field,
            "mongodb_text_field": params.text_field,
            "mongodb_num_candidates": cls._num_candidates(limit, params.mongodb_num_candidates),
            "max_num_results": limit,
            "include_metadata": True,
            "timeout_ms": cls._timeout_ms(timeout),
        }
        mql: Final = translate_filters(optional_params.get("filters"))
        if mql is not None:
            body["filter"] = mql
        if params.mongodb_exact_search:
            body["exact"] = True
        if hybrid_requested(params, ranking):
            vector_weight, text_weight = params.hybrid_weights()
            body["hybrid"] = {  # mutable-ok: JSON transport requires a dict
                "text_index": params.require_text_index(),
                "vector_weight": vector_weight,
                "text_weight": text_weight,
            }
        else:
            threshold: Final = score_threshold(params, ranking)
            if threshold is not None:
                body["score_threshold"] = threshold
        return (f"{api_base}/v1/vector_stores/{quote(vector_store_id, safe='')}/search", body)

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        params: Final = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        query_text: Final = self._query_text(query)
        response: Final = (embedding_executor or self.embedding_executor).embed(
            params.require_embedding_model(), query_text, params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG
        )
        return self._request(
            vector_store_id,
            query_text,
            params,
            vector_store_search_optional_params,
            api_base,
            response,
            litellm_params.get("timeout"),
        )

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        params: Final = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        query_text: Final = self._query_text(query)
        response: Final = await (embedding_executor or self.embedding_executor).aembed(
            params.require_embedding_model(), query_text, params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG
        )
        return self._request(
            vector_store_id,
            query_text,
            params,
            vector_store_search_optional_params,
            api_base,
            response,
            litellm_params.get("timeout"),
        )

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: "LiteLLMLoggingObj"
    ) -> VectorStoreSearchResponse:
        try:
            validated: Final = _SearchResponse.model_validate_json(response.content)
            # Older sidecars send no attributes; keep their response shape unchanged by omitting the null key.
            data: Final = tuple(
                {  # mutable-ok: JSON transport
                    key: value for key, value in row.model_dump().items() if key != "attributes" or value is not None
                }
                for row in validated.data
            )
            return _RESPONSE_ADAPTER.validate_python(
                {  # mutable-ok: JSON transport
                    "object": validated.object,
                    "search_query": validated.search_query,
                    "data": data,
                }  # mutable-ok: JSON transport
            )
        except ValidationError:
            raise ServiceUnavailableError(
                message="MongoDB sidecar returned an invalid search response. Check the sidecar version and deployment.",
                model=None,
                llm_provider="mongodb",
            ) from None

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, object] | httpx.Headers
    ) -> BaseLLMException:
        if status_code in (400, 409):
            raise config_error(error_message)
        if status_code == 401:
            raise AuthenticationError(message="MongoDB sidecar rejected api_key.", model=None, llm_provider="mongodb")
        if status_code in (404, 405):
            raise config_error(_SIDECAR_TOO_OLD_MESSAGE)
        if status_code == 408:
            raise Timeout(message=error_message, model=None, llm_provider="mongodb")
        raise ServiceUnavailableError(
            message="MongoDB sidecar is unavailable. Check its address, health, and logs.",
            model=None,
            llm_provider="mongodb",
        )

    # ---- diagnostics --------------------------------------------------------------------------------------------

    async def atest_connection(
        self,
        litellm_params: Mapping[str, object],
        vector_store_id: str | None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> VectorStoreTestConnectionResponse:
        from litellm.llms.mongodb.vector_stores.diagnostics import run_test_connection

        return await run_test_connection(self, litellm_params, vector_store_id, embedding_executor)

    async def adiscover(
        self, kind: str, litellm_params: Mapping[str, object], options: Mapping[str, object]
    ) -> Mapping[str, object]:
        from litellm.llms.mongodb.vector_stores.diagnostics import run_discovery

        return await run_discovery(self, kind, litellm_params, options)

    # ---- create -------------------------------------------------------------------------------------------------

    def validate_create_vector_store(self) -> None:
        return None

    @staticmethod
    def _index_name(vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams) -> str:
        name: Final = vector_store_create_optional_params.get("name")
        if not name or not name.strip() or name.startswith("$"):
            raise config_error(
                "name is required when creating a MongoDB vector store: it becomes the MongoDB Vector Search index "
                "name and the vector store ID. Example: name: policy_vector_index"
            )
        return name

    @staticmethod
    def _create_body(
        index_name: str, params: MongoDBVectorStoreParams, dimensions: int, timeout: object
    ) -> dict[str, object]:  # mutable-ok: the provider contract returns a writable JSON request body
        return {  # mutable-ok: JSON transport requires a dict
            "index_name": index_name,
            "mongodb_database": params.require_database(),
            "mongodb_collection": params.require_collection(),
            "mongodb_embedding_field": params.embedding_field,
            "mongodb_text_field": params.text_field,
            "dimensions": dimensions,
            "similarity": params.similarity,
            "filter_fields": params.filter_fields,
            "text_index_name": params.mongodb_text_index,
            "timeout_ms": MongoDBVectorStoreConfig._timeout_ms(timeout),
        }

    def transform_create_vector_store_request(
        self, vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams, api_base: str
    ) -> tuple[str, dict]:  # mutable-ok: the provider contract returns a writable JSON request body
        raise config_error(
            "Creating a MongoDB vector store needs litellm_params (database, collection, and embedding model). "
            "Call litellm.vector_stores.create with custom_llm_provider='mongodb' and those parameters."
        )

    def transform_create_vector_store_request_with_litellm_params(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict]:  # mutable-ok: the provider contract returns a writable JSON request body
        params: Final = validated_params(litellm_params)
        index_name: Final = self._index_name(vector_store_create_optional_params)
        dimensions: Final = params.mongodb_dimensions or len(
            embedding_vector(
                self.embedding_executor.embed(
                    params.require_embedding_model(),
                    DIMENSION_PROBE_TEXT,
                    params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG,
                )
            )
        )
        return (
            f"{api_base}/v1/vector_stores",
            self._create_body(index_name, params, dimensions, litellm_params.get("timeout")),
        )

    async def atransform_create_vector_store_request_with_litellm_params(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict]:  # mutable-ok: the provider contract returns a writable JSON request body
        params: Final = validated_params(litellm_params)
        index_name: Final = self._index_name(vector_store_create_optional_params)
        dimensions: Final = params.mongodb_dimensions or len(
            embedding_vector(
                await self.embedding_executor.aembed(
                    params.require_embedding_model(),
                    DIMENSION_PROBE_TEXT,
                    params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG,
                )
            )
        )
        return (
            f"{api_base}/v1/vector_stores",
            self._create_body(index_name, params, dimensions, litellm_params.get("timeout")),
        )

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        try:
            status: Final = _CreateResponse.model_validate_json(response.content)
        except ValidationError:
            raise ServiceUnavailableError(
                message="MongoDB sidecar returned an invalid create response. Check the sidecar version and deployment.",
                model=None,
                llm_provider="mongodb",
            ) from None
        return VectorStoreCreateResponse(
            id=status.index_name,
            object="vector_store",
            created_at=int(time()),
            name=status.index_name,
            bytes=0,
            file_counts=VectorStoreFileCounts(in_progress=0, completed=0, failed=0, cancelled=0, total=0),
            status="completed" if status.queryable else "in_progress",
            expires_after=None,
            expires_at=None,
            last_active_at=None,
            metadata={  # mutable-ok: the TypedDict declares a dict field
                "mongodb_database": status.mongodb_database,
                "mongodb_collection": status.mongodb_collection,
                "mongodb_index_status": status.status,
                "mongodb_index_created": str(status.created).lower(),
                "mongodb_document_count": str(status.document_count if status.document_count is not None else 0),
            },
        )
