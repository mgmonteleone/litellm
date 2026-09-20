import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.mongodb.vector_stores.transformation import MongoDBVectorStoreConfig
from litellm.rag.ingestion.mongodb_ingestion import MongoDBRAGIngestion
from litellm.rag.main import get_ingestion_class
from litellm.types.utils import EmbeddingResponse

BASE_PARAMS: Final = {
    "api_base": "https://sidecar.example",
    "api_key": "test-sidecar-key",
    "litellm_embedding_model": "embedding-alias",
    "mongodb_database": "knowledge",
    "mongodb_collection": "policies",
}
INDEX_STATUS: Final = {
    "index_name": "policy_index",
    "mongodb_database": "knowledge",
    "mongodb_collection": "policies",
    "status": "building",
    "queryable": False,
    "definition": {"fields": []},
    "document_count": 0,
    "created": True,
}


class RecordingEmbeddingExecutor:
    def __init__(self, dimensions: int = 3) -> None:
        self.call: Final = MagicMock(return_value=EmbeddingResponse(data=[{"embedding": [0.1] * dimensions}]))

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("configured_dimensions", [None, 1536])
@pytest.mark.asyncio
async def test_create_request_probes_dimensions_only_when_not_configured(
    asynchronous: bool, configured_dimensions: int | None
) -> None:
    executor: Final = RecordingEmbeddingExecutor(dimensions=3)
    config: Final = MongoDBVectorStoreConfig(executor)
    params: Final = {
        **BASE_PARAMS,
        "mongodb_similarity": "dotProduct",
        "mongodb_filter_fields": ["metadata.department", "metadata.year"],
        "mongodb_dimensions": configured_dimensions,
        "timeout": 12,
    }
    kwargs: Final = {
        "vector_store_create_optional_params": {"name": "policy index"},
        "api_base": "https://sidecar.example",
        "litellm_params": params,
    }
    if asynchronous:
        url, body = await config.atransform_create_vector_store_request_with_litellm_params(**kwargs)
    else:
        url, body = config.transform_create_vector_store_request_with_litellm_params(**kwargs)
    assert url == "https://sidecar.example/v1/vector_stores"
    assert body == {
        "index_name": "policy index",
        "mongodb_database": "knowledge",
        "mongodb_collection": "policies",
        "mongodb_embedding_field": "embedding",
        "mongodb_text_field": "text",
        "dimensions": configured_dimensions or 3,
        "similarity": "dotProduct",
        "filter_fields": ("metadata.department", "metadata.year"),
        "text_index_name": None,
        "timeout_ms": 12_000,
    }
    if configured_dimensions is None:
        executor.call.assert_called_once()
        assert executor.call.call_args.args[0] == "embedding-alias"
    else:
        executor.call.assert_not_called()


@pytest.mark.parametrize(
    "overrides,create_params",
    [
        ({}, {}),
        ({}, {"name": " "}),
        ({}, {"name": "$system"}),
        ({"mongodb_database": None}, {"name": "idx"}),
        ({"mongodb_similarity": "manhattan"}, {"name": "idx"}),
        ({"mongodb_dimensions": 0}, {"name": "idx"}),
        ({"mongodb_filter_fields": ["$bad"]}, {"name": "idx"}),
        ({"mongodb_quantization": "scalar"}, {"name": "idx"}),
        ({"litellm_embedding_model": None, "mongodb_dimensions": None}, {"name": "idx"}),
    ],
)
def test_invalid_create_is_rejected_before_any_probe(
    overrides: Mapping[str, object], create_params: Mapping[str, object]
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    config: Final = MongoDBVectorStoreConfig(executor)
    with pytest.raises(litellm.BadRequestError):
        config.transform_create_vector_store_request_with_litellm_params(
            vector_store_create_optional_params=dict(create_params),
            api_base="https://sidecar.example",
            litellm_params={**BASE_PARAMS, **overrides},
        )
    executor.call.assert_not_called()


def test_create_response_maps_index_status_to_openai_shape() -> None:
    config: Final = MongoDBVectorStoreConfig(RecordingEmbeddingExecutor())
    building: Final = config.transform_create_vector_store_response(httpx.Response(201, json=INDEX_STATUS))
    assert building["id"] == "policy_index"
    assert building["status"] == "in_progress"
    assert building["metadata"] == {
        "mongodb_database": "knowledge",
        "mongodb_collection": "policies",
        "mongodb_index_status": "building",
        "mongodb_index_created": "true",
        "mongodb_document_count": "0",
    }
    ready: Final = config.transform_create_vector_store_response(
        httpx.Response(200, json={**INDEX_STATUS, "status": "ready", "queryable": True, "created": False})
    )
    assert ready["status"] == "completed"
    with pytest.raises(litellm.ServiceUnavailableError):
        config.transform_create_vector_store_response(httpx.Response(200, json={"unexpected": True}))


@pytest.mark.parametrize(
    "status,error_type,match",
    [
        (404, litellm.BadRequestError, "v0.2 or later"),
        (405, litellm.BadRequestError, "v0.2 or later"),
        (409, litellm.BadRequestError, "already exists"),
        (401, litellm.AuthenticationError, "rejected"),
        (503, litellm.ServiceUnavailableError, "unavailable"),
    ],
)
def test_sidecar_errors_explain_the_fix(status: int, error_type: type[Exception], match: str) -> None:
    config: Final = MongoDBVectorStoreConfig(RecordingEmbeddingExecutor())
    with pytest.raises(error_type, match=match):
        config.get_error_class("Search index 'x' already exists with a different definition.", status, {})


@pytest.mark.asyncio
async def test_public_sdk_create_posts_to_sidecar_and_returns_openai_shape() -> None:
    executor: Final = RecordingEmbeddingExecutor(dimensions=4)
    seen: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://sidecar.example/v1/vector_stores"
        assert request.headers["authorization"] == "Bearer test-sidecar-key"
        seen.append(json.loads(request.content))
        return httpx.Response(201, json=INDEX_STATUS)

    client: Final = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch(
        "litellm.utils.ProviderConfigManager.get_provider_vector_stores_config",
        return_value=MongoDBVectorStoreConfig(executor),
    ):
        response: Final = await litellm.vector_stores.acreate(
            name="policy_index", custom_llm_provider="mongodb", client=client, **BASE_PARAMS
        )
    assert response["id"] == "policy_index"
    assert response["status"] == "in_progress"
    assert seen == [
        {
            "index_name": "policy_index",
            "mongodb_database": "knowledge",
            "mongodb_collection": "policies",
            "mongodb_embedding_field": "embedding",
            "mongodb_text_field": "text",
            "dimensions": 4,
            "similarity": "cosine",
            "filter_fields": [],
            "text_index_name": None,
            "timeout_ms": seen[0]["timeout_ms"],
        }
    ]


def test_mongodb_is_registered_for_rag_ingestion() -> None:
    assert get_ingestion_class("mongodb") is MongoDBRAGIngestion
    assert MongoDBRAGIngestion.supports_existing_file_id is False


@pytest.mark.asyncio
async def test_ingestion_embeds_locally_then_creates_index_and_posts_batches() -> None:
    requests: list[tuple[str, dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), json.loads(request.content)))
        assert request.headers["authorization"] == "Bearer test-sidecar-key"
        if request.url.path == "/v1/vector_stores":
            return httpx.Response(201, json=INDEX_STATUS)
        return httpx.Response(
            200, json={"file_id": "ignored", "inserted": 1, "deleted": 0, "ingested_at": "2026-09-20T00:00:00Z"}
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    chunks: Final = [f"chunk {index}" for index in range(201)]
    embeddings: Final = [[float(index), 0.5] for index in range(201)]
    ingestion: Final = MongoDBRAGIngestion(
        {
            "vector_store": {
                "custom_llm_provider": "mongodb",
                "vector_store_id": "policy_index",
                **BASE_PARAMS,
                "mongodb_filter_fields": ["metadata.department"],
            }
        }
    )
    with patch("litellm.rag.ingestion.mongodb_ingestion.get_async_httpx_client", return_value=handler):
        index_name, file_id = await ingestion.store(
            file_content=b"raw",
            filename="9f1c2b4e.txt",
            content_type="text/markdown",
            chunks=chunks,
            embeddings=embeddings,
            display_filename="travel.md",
        )
    assert index_name == "policy_index"
    assert file_id is not None and file_id.startswith("file_")
    urls: Final = [url for url, _ in requests]
    assert urls == [
        "https://sidecar.example/v1/vector_stores",
        "https://sidecar.example/v1/vector_stores/policy_index/documents",
        "https://sidecar.example/v1/vector_stores/policy_index/documents",
    ]
    create_body: Final = requests[0][1]
    assert (create_body["dimensions"], create_body["filter_fields"]) == (2, ["metadata.department"])
    first, second = requests[1][1], requests[2][1]
    assert (len(first["documents"]), len(second["documents"])) == (200, 1)
    assert (first["replace_existing"], second["replace_existing"]) == (True, False)
    assert first["documents"][0] == {"chunk_index": 0, "text": "chunk 0", "embedding": [0.0, 0.5], "metadata": {}}
    assert file_id == "file_" + __import__("hashlib").sha256(b"raw").hexdigest()[:32]
    assert second["documents"][0]["chunk_index"] == 200
    assert first["file_id"] == file_id and first["filename"] == "travel.md"
    assert second["filename"] == "travel.md"


@pytest.mark.asyncio
async def test_ingestion_falls_back_to_the_storage_name_when_no_display_name_is_given() -> None:
    """URL and SDK ingests have no separate display name, so the stored chunks keep the only name there is."""
    bodies: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/vector_stores":
            return httpx.Response(201, json=INDEX_STATUS)
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"file_id": "ignored", "inserted": 1, "deleted": 0, "ingested_at": "2026-09-20T00:00:00Z"}
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ingestion: Final = MongoDBRAGIngestion({"vector_store": {**BASE_PARAMS, "vector_store_id": "policy_index"}})
    with patch("litellm.rag.ingestion.mongodb_ingestion.get_async_httpx_client", return_value=handler):
        await ingestion.store(
            file_content=b"raw",
            filename="handbook.pdf",
            content_type="application/pdf",
            chunks=["a"],
            embeddings=[[1.0]],
        )
    assert bodies[0]["filename"] == "handbook.pdf"


@pytest.mark.asyncio
async def test_ingestion_prefers_registration_embedding_model_and_router() -> None:
    router: Final = MagicMock()

    async def aembedding(model: str, input: list[str], **kwargs: object) -> EmbeddingResponse:
        assert model == "embedding-alias" and kwargs == {"dimensions": 256}
        return EmbeddingResponse(data=[{"embedding": [1.0, 2.0]} for _ in input])

    router.aembedding = aembedding
    ingestion: Final = MongoDBRAGIngestion(
        {
            "embedding": {"model": "ignored-fallback"},
            "vector_store": {**BASE_PARAMS, "litellm_embedding_config": {"dimensions": 256}},
        },
        router=router,
    )
    assert await ingestion.embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert await ingestion.embed([]) is None


@pytest.mark.asyncio
async def test_ingestion_without_an_embedding_model_or_text_fails_clearly() -> None:
    missing_model: Final = MongoDBRAGIngestion({"vector_store": {**BASE_PARAMS, "litellm_embedding_model": None}})
    with pytest.raises(litellm.BadRequestError, match="embedding model is required"):
        await missing_model.embed(["a"])
    empty: Final = MongoDBRAGIngestion({"vector_store": BASE_PARAMS})
    with pytest.raises(ValueError, match="No text content"):
        await empty.store(
            file_content=b"", filename="x.pdf", content_type="application/pdf", chunks=[], embeddings=None
        )


@pytest.mark.asyncio
async def test_ingestion_surfaces_old_sidecar_as_actionable_error() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ingestion: Final = MongoDBRAGIngestion({"vector_store": BASE_PARAMS})
    with patch("litellm.rag.ingestion.mongodb_ingestion.get_async_httpx_client", return_value=handler):
        with pytest.raises(litellm.BadRequestError, match=r"v0\.2 or later"):
            await ingestion.store(
                file_content=b"raw", filename="a.md", content_type="text/markdown", chunks=["a"], embeddings=[[1.0]]
            )


def test_custom_metadata_becomes_filterable_chunk_metadata() -> None:
    from litellm.rag.ingestion.mongodb_ingestion import chunk_metadata, deterministic_file_id

    assert dict(chunk_metadata({"department": "hr", "year": 2026, "$bad": 1, "a.b": 2, "nested": {"x": 1}})) == {
        "department": "hr",
        "year": 2026,
    }
    assert dict(chunk_metadata(None)) == {}
    assert deterministic_file_id("a.md", b"x") == deterministic_file_id("renamed-by-proxy.txt", b"x")
    assert deterministic_file_id("a.md", b"x") != deterministic_file_id("a.md", b"y")
    assert deterministic_file_id("a.md", None).startswith("file_")


def test_embedding_config_drops_reserved_kwargs_and_vectors_are_floats() -> None:
    ingestion: Final = MongoDBRAGIngestion(
        {"vector_store": {**BASE_PARAMS, "litellm_embedding_config": {"model": "x", "input": "y", "dimensions": 8}}}
    )
    assert dict(ingestion._embedding_config()) == {"dimensions": 8}
    from litellm.rag.ingestion.mongodb_ingestion import _documents

    (document,) = _documents(["a"], [[1, 0]], {}, 0, 1)
    assert document["embedding"] == (1.0, 0.0) and all(isinstance(v, float) for v in document["embedding"])
