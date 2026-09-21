import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.mongodb.vector_stores.transformation import MongoDBVectorStoreConfig
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import VectorStoreSearchOptionalRequestParams, VectorStoreSearchResponse

BASE_PARAMS: Final = {
    "api_base": "https://sidecar.example/prefix",
    "api_key": "test-sidecar-key",
    "litellm_embedding_model": "embedding-alias",
    "mongodb_database": "policies",
    "mongodb_collection": "documents",
}
RESULT: Final = {
    "object": "vector_store.search_results.page",
    "search_query": "travel policy",
    "data": [
        {"score": 0.9, "file_id": "123", "filename": "123", "content": [{"type": "text", "text": "Use code BLUE-42"}]}
    ],
}


class RecordingEmbeddingExecutor:
    def __init__(self) -> None:
        self.call: Final = MagicMock(return_value=EmbeddingResponse(data=[{"embedding": [0.1, 0.2, 0.3]}]))

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit,candidates", [(None, 100), (1, 100), (50, 500)])
@pytest.mark.asyncio
async def test_search_preserves_embedding_and_http_contract(
    asynchronous: bool, limit: int | None, candidates: int
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    config: Final = MongoDBVectorStoreConfig(executor)
    params: Final = {
        **BASE_PARAMS,
        "mongodb_text_field": "metadata.body",
        "mongodb_embedding_field": "stored_vector",
        "litellm_embedding_config": {"dimensions": 3},
        "timeout": 0.75,
    }
    kwargs: Final = {
        "vector_store_id": "exact index",
        "query": ["travel", "policy"],
        "vector_store_search_optional_params": {"max_num_results": limit},
        "api_base": BASE_PARAMS["api_base"],
        "litellm_logging_obj": MagicMock(),
        "litellm_params": params,
    }
    if asynchronous:
        url, body = await config.atransform_search_vector_store_request(**kwargs)
    else:
        url, body = config.transform_search_vector_store_request(**kwargs)
    assert url == "https://sidecar.example/prefix/v1/vector_stores/exact%20index/search"
    assert body == {
        "query": "travel policy",
        "query_vector": (0.1, 0.2, 0.3),
        "mongodb_database": "policies",
        "mongodb_collection": "documents",
        "mongodb_text_field": "metadata.body",
        "mongodb_embedding_field": "stored_vector",
        "mongodb_num_candidates": candidates,
        "max_num_results": limit or 10,
        "include_metadata": True,
        "timeout_ms": 750,
    }
    executor.call.assert_called_once_with("embedding-alias", "travel policy", {"dimensions": 3})
    assert config.transform_search_vector_store_response(httpx.Response(200, json=RESULT), MagicMock()) == RESULT


@pytest.mark.parametrize(
    "query,overrides,options",
    [
        ("", {}, {}),
        ("  ", {}, {}),
        ("x" * 32_001, {}, {}),
        ("travel", {"litellm_embedding_model": None}, {}),
        ("travel", {"mongodb_database": None}, {}),
        ("travel", {"mongodb_collection": None}, {}),
        ("travel", {"mongodb_connection_string": "mongodb://obsolete-secret"}, {}),
        ("travel", {"mongodb_filter": {"private": True}}, {}),
        ("travel", {"mongodb_num_candidates": 9}, {}),
        ("travel", {"mongodb_num_candidates": 10_001}, {}),
        ("travel", {}, {"max_num_results": 0}),
        ("travel", {}, {"max_num_results": 51}),
        ("travel", {}, {"filters": {}}),
        ("travel", {}, {"filters": {"type": "regex", "key": "a", "value": "b"}}),
        ("travel", {}, {"ranking_options": {"ranker": "bm25"}}),
        ("travel", {}, {"ranking_options": {"score_threshold": 2}}),
        ("travel", {"mongodb_hybrid_search": True}, {}),
        (
            "travel",
            {"mongodb_hybrid_search": True, "mongodb_text_index": "t"},
            {"ranking_options": {"score_threshold": 0.5}},
        ),
        ("travel", {}, {"rewrite_query": False}),
    ],
)
def test_invalid_search_is_rejected_before_embedding(
    query: str, overrides: Mapping[str, object], options: VectorStoreSearchOptionalRequestParams
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    config: Final = MongoDBVectorStoreConfig(executor)
    with pytest.raises(litellm.BadRequestError) as error:
        config.transform_search_vector_store_request(
            vector_store_id="policy_index",
            query=query,
            vector_store_search_optional_params=options,
            api_base=BASE_PARAMS["api_base"],
            litellm_logging_obj=MagicMock(),
            litellm_params={**BASE_PARAMS, **overrides},
        )
    assert "obsolete-secret" not in str(error.value)
    executor.call.assert_not_called()


@pytest.mark.parametrize(
    "status,body,error_type",
    [
        (400, {"error": {"message": "Index is not queryable"}}, litellm.BadRequestError),
        (401, {}, litellm.AuthenticationError),
        (408, {}, litellm.Timeout),
        (503, {}, litellm.ServiceUnavailableError),
        (200, {}, litellm.ServiceUnavailableError),
        (200, {**RESULT, "data": [{"score": "wrong"}]}, litellm.ServiceUnavailableError),
        (0, {}, litellm.Timeout),
        (-1, {}, litellm.BadRequestError),
        (-2, {"api_base": "http://sidecar.example"}, litellm.BadRequestError),
        (-2, {"api_base": "http://10.0.0.10:8080"}, litellm.BadRequestError),
        (-2, {"api_base": "http://localhost:8080"}, litellm.BadRequestError),
        (200, RESULT, None),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("timeout", [0.75, 120.0])
@pytest.mark.parametrize("api_base", ["https://sidecar.example/prefix", "http://127.0.0.1:8080", "http://[::1]:8080"])
@pytest.mark.asyncio
async def test_public_sdk_preserves_http_errors_response_and_timeout(
    status: int,
    body: Mapping[str, object],
    error_type: type[Exception] | None,
    asynchronous: bool,
    timeout: float,
    api_base: str,
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    if status == -1:
        if asynchronous:
            with pytest.raises(litellm.BadRequestError, match=r"api_(key|base) is required"):
                await litellm.vector_stores.acreate(custom_llm_provider="mongodb")
        else:
            with pytest.raises(litellm.BadRequestError, match=r"api_(key|base) is required"):
                litellm.vector_stores.create(custom_llm_provider="mongodb")
        return
    if status == -2:
        rejected_params: Final = {**BASE_PARAMS, "api_base": str(body["api_base"])}
        if asynchronous:
            with pytest.raises(litellm.BadRequestError, match="requires HTTPS"):
                await litellm.vector_stores.asearch(
                    vector_store_id="policy_index",
                    query="travel policy",
                    custom_llm_provider="mongodb",
                    _direct_vector_store_embedding_executor=executor,
                    **rejected_params,
                )
        else:
            with pytest.raises(litellm.BadRequestError, match="requires HTTPS"):
                litellm.vector_stores.search(
                    vector_store_id="policy_index",
                    query="travel policy",
                    custom_llm_provider="mongodb",
                    _direct_vector_store_embedding_executor=executor,
                    **rejected_params,
                )
        executor.call.assert_not_called()
        return

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{api_base}/v1/vector_stores/policy_index/search"
        assert request.headers["authorization"] == "Bearer test-sidecar-key"
        assert request.extensions["timeout"]["read"] == timeout
        payload: Final = json.loads(request.content)
        assert payload["timeout_ms"] == int(timeout * 1000)
        assert payload["query_vector"] == [0.1, 0.2, 0.3]
        if status == 0:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(status, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as async_transport:
        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            client: Final = AsyncHTTPHandler() if asynchronous else HTTPHandler(client=transport)
            if isinstance(client, AsyncHTTPHandler):
                await client.client.aclose()
                client.client = async_transport

            async def search() -> VectorStoreSearchResponse:
                kwargs: Final = {
                    **BASE_PARAMS,
                    "api_base": api_base,
                    "vector_store_id": "policy_index",
                    "query": "travel policy",
                    "custom_llm_provider": "mongodb",
                    "_direct_vector_store_embedding_executor": executor,
                    "client": client,
                    "timeout": timeout,
                }
                if asynchronous:
                    return await litellm.vector_stores.asearch(**kwargs)
                return litellm.vector_stores.search(**kwargs)

            if error_type is not None:
                with pytest.raises(error_type):
                    await search()
            else:
                assert await search() == RESULT
    executor.call.assert_called_once_with("embedding-alias", "travel policy", {})


@pytest.mark.parametrize(
    ("typo", "suggestion"),
    [
        ("embedding_model", "litellm_embedding_model"),
        ("mongodb_databse", "mongodb_database"),
        ("mongodb_collections", "mongodb_collection"),
        ("hybrid_search", "mongodb_hybrid_search"),
        ("filter_fields", "mongodb_filter_fields"),
    ],
)
def test_unknown_store_params_are_named_with_the_closest_supported_key(typo: str, suggestion: str) -> None:
    """These used to be dropped silently and resurfaced later as 'litellm_embedding_model is required'."""
    from litellm.llms.mongodb.vector_stores.transformation import validated_params

    with pytest.raises(litellm.BadRequestError) as error:
        validated_params({**BASE_PARAMS, typo: "value"})

    assert f"'{typo}'" in str(error.value)
    assert f"did you mean '{suggestion}'" in str(error.value)


def test_unknown_store_param_without_a_close_match_is_still_named() -> None:
    """A garbled mongodb_ field is always a typo in this store's own namespace, close match or not."""
    from litellm.llms.mongodb.vector_stores.transformation import validated_params

    with pytest.raises(litellm.BadRequestError, match=r"'mongodb_zzzzzzzz'"):
        validated_params({**BASE_PARAMS, "mongodb_zzzzzzzz": 1})


@pytest.mark.parametrize(
    ("typo", "suggestion"),
    [
        ("embedding_model", "litellm_embedding_model"),
        ("dimensions", "mongodb_dimensions"),
    ],
)
def test_proxy_plumbing_does_not_mask_real_typos(typo: str, suggestion: str) -> None:
    """The plumbing tolerance below must not swallow a genuine typo made alongside it."""
    from litellm.types.router import GenericLiteLLMParams

    litellm_params: Final = GenericLiteLLMParams(
        **{**BASE_PARAMS, "model": None, "user": "some-user", "disable_fallbacks": True, typo: "x"}
    )

    with pytest.raises(litellm.BadRequestError, match=rf"did you mean '{suggestion}'"):
        MongoDBVectorStoreConfig().validate_environment(headers={}, litellm_params=litellm_params)


def test_proxy_plumbing_params_are_not_mistaken_for_typos() -> None:
    """common_request_processing.py writes model=None into every vector store search's litellm_params, and
    litellm_pre_call_utils.py merges in user and disable_fallbacks from the virtual key. None of the three
    are MongoDB parameters, and a MongoDB store search must not 400 because of them."""
    from litellm.types.router import GenericLiteLLMParams

    litellm_params: Final = GenericLiteLLMParams(
        **{**BASE_PARAMS, "model": None, "user": "some-user", "disable_fallbacks": True}
    )

    result: Final = MongoDBVectorStoreConfig().validate_environment(headers={}, litellm_params=litellm_params)

    assert result["Authorization"] == "Bearer test-sidecar-key"


def test_validate_environment_rejects_unknown_params_too() -> None:
    """test_connection and ingest reach the provider through validate_environment."""
    from litellm.types.router import GenericLiteLLMParams

    with pytest.raises(litellm.BadRequestError, match=r"'embeddding_model'"):
        MongoDBVectorStoreConfig().validate_environment(
            headers={},
            litellm_params=GenericLiteLLMParams(**{**BASE_PARAMS, "embeddding_model": "x"}),
        )


@pytest.mark.asyncio
async def test_search_rejects_an_unknown_param_before_embedding_or_any_request() -> None:
    executor: Final = RecordingEmbeddingExecutor()
    client: Final = MagicMock()

    with pytest.raises(litellm.BadRequestError, match=r"'dimensions'"):
        await litellm.vector_stores.asearch(
            **BASE_PARAMS,
            dimensions=1536,
            vector_store_id="policy_index",
            query="travel policy",
            custom_llm_provider="mongodb",
            _direct_vector_store_embedding_executor=executor,
            client=client,
        )

    executor.call.assert_not_called()


def test_get_complete_url_falls_back_to_the_deployment_sidecar_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example/")

    assert MongoDBVectorStoreConfig().get_complete_url(api_base=None, litellm_params={}) == (
        "https://deployment-sidecar.example"
    )


def test_get_complete_url_prefers_a_store_specific_override_over_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example")

    assert (
        MongoDBVectorStoreConfig().get_complete_url(api_base="https://per-store-override.example", litellm_params={})
        == "https://per-store-override.example"
    )


def test_get_complete_url_without_either_source_names_both_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONGODB_SIDECAR_API_BASE", raising=False)

    with pytest.raises(litellm.BadRequestError, match="api_base or MONGODB_SIDECAR_API_BASE"):
        MongoDBVectorStoreConfig().get_complete_url(api_base=None, litellm_params={})


@pytest.mark.asyncio
async def test_search_resolves_api_base_from_the_deployment_sidecar_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """An admin should not need to know the sidecar's address: the deployment configures it once."""
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example")
    executor: Final = RecordingEmbeddingExecutor()
    params_without_api_base: Final = {key: value for key, value in BASE_PARAMS.items() if key != "api_base"}

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://deployment-sidecar.example/v1/vector_stores/policy_index/search"
        return httpx.Response(200, json=RESULT)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client: Final = HTTPHandler(client=transport)
        response: Final = litellm.vector_stores.search(
            vector_store_id="policy_index",
            query="travel policy",
            custom_llm_provider="mongodb",
            _direct_vector_store_embedding_executor=executor,
            client=client,
            **params_without_api_base,
        )
    assert response == RESULT
