import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.mongodb.vector_stores.transformation import MongoDBVectorStoreConfig
from litellm.types.utils import EmbeddingResponse

BASE_PARAMS: Final = {
    "api_base": "https://sidecar.example",
    "api_key": "test-sidecar-key",
    "litellm_embedding_model": "embedding-alias",
    "mongodb_database": "knowledge",
    "mongodb_collection": "policies",
}
CAPABILITIES: Final = {
    "version": "0.2.0b1",
    "features": {"create": True, "hybrid": False, "discovery": True},
    "mongodb": {"server_version": "8.2.11", "rank_fusion_supported": False},
}
SIDECAR_REPORT: Final = {
    "ok": True,
    "server_version": "8.2.11",
    "index_dimensions": 3,
    "document_count": 12,
    "checks": [
        {"check": "mongodb_ping", "status": "pass", "message": "MongoDB is reachable."},
        {
            "check": "mongodb_index",
            "status": "warn",
            "message": "Index 'policy_index' is building.",
            "details": {"status": "building"},
        },
    ],
}


class Executor:
    def __init__(self, dimensions: int = 3, fail: bool = False) -> None:
        self.dimensions: Final = dimensions
        self.fail: Final = fail

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        if self.fail:
            raise litellm.AuthenticationError(message="bad provider key", model=model, llm_provider="openai")
        return EmbeddingResponse(data=[{"embedding": [0.1] * self.dimensions}])

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.embed(model, query, configuration)


def sidecar(
    readiness: int = 200, capabilities: int = 200, report: Mapping[str, object] | None = None
) -> tuple[AsyncHTTPHandler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path: Final = request.url.path
        if path == "/health/readiness":
            return httpx.Response(readiness, json={"status": "ok" if readiness == 200 else "unavailable"})
        assert request.headers["authorization"] == "Bearer test-sidecar-key"
        if path == "/v1/capabilities":
            return httpx.Response(capabilities, json=CAPABILITIES if capabilities == 200 else {"detail": "x"})
        if path == "/v1/test_connection":
            return httpx.Response(200, json=report or SIDECAR_REPORT)
        if path.startswith("/v1/discovery/"):
            return httpx.Response(200, json={"kind": path.rsplit("/", 1)[-1], "query": dict(request.url.params)})
        return httpx.Response(404)

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return handler, seen


def statuses(response: Mapping[str, object]) -> dict[str, str]:
    return {row["check"]: row["status"] for row in response["checks"]}  # type: ignore[index]


@pytest.mark.asyncio
async def test_full_checklist_merges_sidecar_report_and_embedding_probe() -> None:
    handler, seen = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index")
    assert response["ok"] is True and response["supported"] is True
    assert statuses(response) == {
        "sidecar_reachable": "pass",
        "sidecar_auth": "pass",
        "embedding_model": "pass",
        "mongodb_ping": "pass",
        "mongodb_index": "warn",
    }
    assert response["summary"].startswith("Connected with 1 warning")
    posted: Final = json.loads(next(r for r in seen if r.url.path == "/v1/test_connection").content)
    assert posted["index_name"] == "policy_index" and posted["expected_dimensions"] == 3
    assert response["details"]["embedding_dimensions"] == 3
    assert response["details"]["mongodb"]["index_dimensions"] == 3
    assert "test-sidecar-key" not in json.dumps(response)


@pytest.mark.asyncio
async def test_unreachable_sidecar_stops_early_with_a_fix() -> None:
    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    )
    config: Final = MongoDBVectorStoreConfig(Executor())
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index")
    assert response["ok"] is False
    assert statuses(response) == {"sidecar_reachable": "fail"}
    assert "container is running" in response["summary"]


@pytest.mark.asyncio
async def test_old_sidecar_warns_and_skips_mongodb_checks() -> None:
    handler, seen = sidecar(capabilities=404)
    config: Final = MongoDBVectorStoreConfig(Executor())
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index")
    assert response["ok"] is True
    assert statuses(response) == {"sidecar_reachable": "pass", "sidecar_auth": "warn", "embedding_model": "pass"}
    assert "v0.1" in response["summary"]
    assert not any(r.url.path == "/v1/test_connection" for r in seen)


@pytest.mark.asyncio
async def test_bad_key_and_bad_embedding_model_both_fail_clearly() -> None:
    handler, _ = sidecar(capabilities=401, readiness=503)
    config: Final = MongoDBVectorStoreConfig(Executor(fail=True))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, None)
    assert response["ok"] is False
    assert statuses(response) == {"sidecar_reachable": "warn", "sidecar_auth": "fail", "embedding_model": "fail"}
    assert "rejected the API key" in response["summary"]


NAMESPACE_LESS_REPORT: Final = {
    "ok": True,
    "server_version": "8.2.11",
    "index_dimensions": None,
    "document_count": None,
    "checks": [
        {"check": "mongodb_ping", "status": "pass", "message": "MongoDB is reachable."},
        {
            "check": "mongodb_version",
            "status": "pass",
            "message": "MongoDB 8.2.11.",
            "details": {"server_version": "8.2.11"},
        },
        {
            "check": "mongodb_collection",
            "status": "skip",
            "message": "Provide mongodb_database and mongodb_collection to check the collection.",
        },
        {
            "check": "mongodb_sample_document",
            "status": "skip",
            "message": "Provide mongodb_database and mongodb_collection to check the collection.",
        },
        {
            "check": "mongodb_index",
            "status": "skip",
            "message": "Provide mongodb_database and mongodb_collection to check the collection.",
        },
        {
            "check": "mongodb_index_definition",
            "status": "skip",
            "message": "Provide mongodb_database and mongodb_collection to check the collection.",
        },
        {
            "check": "mongodb_dimensions",
            "status": "skip",
            "message": "Provide mongodb_database and mongodb_collection to check the collection.",
        },
    ],
}


@pytest.mark.asyncio
async def test_missing_database_and_collection_still_verifies_the_sidecar_connection() -> None:
    """The bug this guards: clicking Test connection before picking a database/collection used to raise
    'mongodb_database is required in litellm_params' instead of running the checks it actually can."""
    params: Final = {
        key: value for key, value in BASE_PARAMS.items() if key not in ("mongodb_database", "mongodb_collection")
    }
    handler, seen = sidecar(report=NAMESPACE_LESS_REPORT)
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(params, "policy_index")
    assert response["ok"] is True
    assert statuses(response) == {
        "sidecar_reachable": "pass",
        "sidecar_auth": "pass",
        "embedding_model": "pass",
        "mongodb_ping": "pass",
        "mongodb_version": "pass",
        "mongodb_collection": "skip",
        "mongodb_sample_document": "skip",
        "mongodb_index": "skip",
        "mongodb_index_definition": "skip",
        "mongodb_dimensions": "skip",
    }
    messages: Final = {row["check"]: row["message"] for row in response["checks"]}
    assert messages["mongodb_collection"] == "Choose a database and collection to check the index."
    assert response["summary"] == "Choose a database and collection to finish the checks."
    posted: Final = json.loads(next(r for r in seen if r.url.path == "/v1/test_connection").content)
    assert "mongodb_database" not in posted
    assert "mongodb_collection" not in posted


@pytest.mark.asyncio
async def test_database_without_collection_also_skips_the_namespace_checks() -> None:
    params: Final = {key: value for key, value in BASE_PARAMS.items() if key != "mongodb_collection"}
    handler, seen = sidecar(report=NAMESPACE_LESS_REPORT)
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(params, "policy_index")
    assert response["ok"] is True
    assert statuses(response)["mongodb_collection"] == "skip"
    posted: Final = json.loads(next(r for r in seen if r.url.path == "/v1/test_connection").content)
    assert posted["mongodb_database"] == "knowledge"
    assert "mongodb_collection" not in posted


INDEX_NOT_CREATED_REPORT: Final = {
    "ok": True,
    "server_version": "8.2.11",
    "index_dimensions": None,
    "document_count": 120,
    "checks": [
        {"check": "mongodb_ping", "status": "pass", "message": "MongoDB is reachable."},
        {
            "check": "mongodb_collection",
            "status": "pass",
            "message": "Collection 'policies' exists with 120 documents.",
        },
        {"check": "mongodb_index", "status": "skip", "message": "No index named 'policy_index' exists yet."},
        {
            "check": "mongodb_index_definition",
            "status": "skip",
            "message": "No index named 'policy_index' exists yet.",
        },
        {"check": "mongodb_dimensions", "status": "skip", "message": "No index named 'policy_index' exists yet."},
    ],
}


@pytest.mark.asyncio
async def test_index_checks_skip_with_their_own_message_when_the_namespace_is_already_chosen() -> None:
    """Regression: every skip, including these (which only mean the index does not exist yet because the
    store has not been saved), used to be reported with the 'choose a database and collection' message
    and summary even though the admin had already chosen both."""
    handler, _ = sidecar(report=INDEX_NOT_CREATED_REPORT)
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index")
    assert response["ok"] is True
    assert statuses(response)["mongodb_index"] == "skip"
    message: Final = next(r["message"] for r in response["checks"] if r["check"] == "mongodb_index")
    assert message == "The index is created when you save this store."
    assert response["summary"] == "Connected. The index is created when you save this store."


@pytest.mark.asyncio
async def test_a_real_collection_not_found_failure_with_a_chosen_namespace_stays_a_failure() -> None:
    report: Final = {
        **INDEX_NOT_CREATED_REPORT,
        "checks": [
            {"check": "mongodb_ping", "status": "pass", "message": "MongoDB is reachable."},
            {"check": "mongodb_collection", "status": "fail", "message": "Collection 'policies' does not exist."},
        ],
    }
    handler, _ = sidecar(report=report)
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index")
    assert response["ok"] is False
    assert statuses(response)["mongodb_collection"] == "fail"
    assert response["summary"] == "Collection 'policies' does not exist."


@pytest.mark.asyncio
async def test_a_failure_alongside_skipped_checks_still_reports_failure() -> None:
    report: Final = {
        **INDEX_NOT_CREATED_REPORT,
        "checks": [
            {"check": "mongodb_ping", "status": "fail", "message": "MongoDB is unreachable from the sidecar."},
            {"check": "mongodb_index", "status": "skip", "message": "No index named 'policy_index' exists yet."},
        ],
    }
    handler, _ = sidecar(report=report)
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index")
    assert response["ok"] is False
    assert response["summary"] == "MongoDB is unreachable from the sidecar."


@pytest.mark.asyncio
async def test_missing_embedding_model_is_skipped_not_failed_before_the_admin_picks_one() -> None:
    params: Final = {
        key: value
        for key, value in BASE_PARAMS.items()
        if key not in ("mongodb_database", "mongodb_collection", "litellm_embedding_model")
    }
    handler, _ = sidecar(report=NAMESPACE_LESS_REPORT)
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(params, "policy_index")
    assert response["ok"] is True
    assert statuses(response)["embedding_model"] == "skip"
    assert response["details"]["embedding_dimensions"] is None
    message: Final = next(r["message"] for r in response["checks"] if r["check"] == "embedding_model")
    assert message == "Choose an embedding model to check its output dimensions."


@pytest.mark.asyncio
async def test_saved_store_missing_a_collection_fails_instead_of_skipping() -> None:
    """Regression: a saved store that lost its database/collection reported ok=True with a skip
    instead of surfacing the real problem, because the add-dialog relaxation (see the two tests above)
    applied to every request regardless of whether the store was already saved and meant to be complete."""
    params: Final = {
        key: value for key, value in BASE_PARAMS.items() if key not in ("mongodb_database", "mongodb_collection")
    }
    config: Final = MongoDBVectorStoreConfig(Executor())
    response: Final = await config.atest_connection(params, "policy_index", is_saved_store=True)
    assert response["ok"] is False
    assert statuses(response) == {"configuration": "fail"}
    assert "mongodb_database" in response["summary"]


@pytest.mark.asyncio
async def test_saved_store_missing_an_embedding_model_fails_instead_of_skipping() -> None:
    params: Final = {key: value for key, value in BASE_PARAMS.items() if key != "litellm_embedding_model"}
    handler, _ = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor())
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(params, "policy_index", is_saved_store=True)
    assert response["ok"] is False
    assert statuses(response)["embedding_model"] == "fail"
    message: Final = next(r["message"] for r in response["checks"] if r["check"] == "embedding_model")
    assert "litellm_embedding_model" in message


@pytest.mark.asyncio
async def test_saved_store_with_complete_configuration_still_passes() -> None:
    """The strict path must not regress a saved store that is actually complete."""
    handler, _ = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor(dimensions=3))
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(BASE_PARAMS, "policy_index", is_saved_store=True)
    assert response["ok"] is True
    assert statuses(response)["embedding_model"] == "pass"


@pytest.mark.asyncio
async def test_unrecognised_param_still_fails_without_a_database_or_collection() -> None:
    params: Final = {
        key: value for key, value in BASE_PARAMS.items() if key not in ("mongodb_database", "mongodb_collection")
    } | {"mongodb_bogus": "x"}
    config: Final = MongoDBVectorStoreConfig(Executor())
    response: Final = await config.atest_connection(params, "policy_index")
    assert response["ok"] is False
    assert statuses(response) == {"configuration": "fail"}
    assert "mongodb_bogus" in response["summary"]


@pytest.mark.asyncio
async def test_bad_dimensions_still_fails_without_a_database_or_collection() -> None:
    params: Final = {
        key: value for key, value in BASE_PARAMS.items() if key not in ("mongodb_database", "mongodb_collection")
    } | {"mongodb_dimensions": 999_999}
    config: Final = MongoDBVectorStoreConfig(Executor())
    response: Final = await config.atest_connection(params, "policy_index")
    assert response["ok"] is False
    assert statuses(response) == {"configuration": "fail"}
    assert "mongodb_dimensions must be between" in response["summary"]


@pytest.mark.asyncio
async def test_hybrid_prerequisites_are_checked() -> None:
    handler, _ = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor())
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(
            {**BASE_PARAMS, "mongodb_hybrid_search": True, "mongodb_text_index": "t"}, "policy_index"
        )
    assert statuses(response)["hybrid_search"] == "fail"
    assert "rankFusion" in next(r["message"] for r in response["checks"] if r["check"] == "hybrid_search")


@pytest.mark.asyncio
async def test_connection_resolves_api_base_from_the_deployment_sidecar_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admin should not need to know the sidecar's address: the deployment configures it once."""
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example")
    handler, _ = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor())
    params_without_api_base: Final = {key: value for key, value in BASE_PARAMS.items() if key != "api_base"}
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        response: Final = await config.atest_connection(params_without_api_base, "policy_index")
    assert response["ok"] is True
    assert statuses(response)["sidecar_reachable"] == "pass"


@pytest.mark.asyncio
async def test_invalid_configuration_is_reported_not_raised() -> None:
    config: Final = MongoDBVectorStoreConfig(Executor())
    response: Final = await config.atest_connection({**BASE_PARAMS, "api_base": "http://not-loopback:8080"}, None)
    assert response["ok"] is False
    assert statuses(response) == {"configuration": "fail"}
    assert "HTTPS" in response["summary"]


@pytest.mark.asyncio
async def test_connection_without_an_api_base_names_both_ways_to_provide_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONGODB_SIDECAR_API_BASE", raising=False)
    config: Final = MongoDBVectorStoreConfig(Executor())
    params_without_api_base: Final = {key: value for key, value in BASE_PARAMS.items() if key != "api_base"}
    response: Final = await config.atest_connection(params_without_api_base, None)
    assert response["ok"] is False
    assert "api_base or MONGODB_SIDECAR_API_BASE" in response["summary"]


@pytest.mark.asyncio
async def test_discovery_resolves_api_base_from_the_deployment_sidecar_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery shares get_complete_url with search, create, ingest, and test_connection, so it must
    fall back to the deployment's MONGODB_SIDECAR_API_BASE the same way they do."""
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example")
    handler, seen = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor())
    params_without_api_base: Final = {key: value for key, value in BASE_PARAMS.items() if key != "api_base"}
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        databases: Final = await config.adiscover("databases", params_without_api_base, {})
    assert dict(databases) == {"kind": "databases", "query": {}}
    assert [str(r.url) for r in seen] == ["https://deployment-sidecar.example/v1/discovery/databases"]


@pytest.mark.asyncio
async def test_discovery_forwards_kind_and_scope_to_the_sidecar() -> None:
    handler, seen = sidecar()
    config: Final = MongoDBVectorStoreConfig(Executor())
    with patch("litellm.llms.mongodb.vector_stores.diagnostics.get_async_httpx_client", return_value=handler):
        fields: Final = await config.adiscover("fields", BASE_PARAMS, {"sample_size": 5})
        databases: Final = await config.adiscover("databases", BASE_PARAMS, {})
        with pytest.raises(litellm.BadRequestError, match="mongodb_collection is required"):
            await config.adiscover("indexes", {**BASE_PARAMS, "mongodb_collection": None}, {})
        with pytest.raises(litellm.BadRequestError, match="Unknown discovery kind"):
            await config.adiscover("users", BASE_PARAMS, {})
    assert dict(fields) == {
        "kind": "fields",
        "query": {"mongodb_database": "knowledge", "mongodb_collection": "policies", "sample_size": "5"},
    }
    assert dict(databases) == {"kind": "databases", "query": {}}
    assert [r.url.path for r in seen] == ["/v1/discovery/fields", "/v1/discovery/databases"]


def test_default_provider_hook_reports_unsupported() -> None:
    import asyncio

    from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig

    response: Final = asyncio.run(OpenAIVectorStoreConfig().atest_connection({}, None))
    assert response["supported"] is False and response["ok"] is False
    with pytest.raises(litellm.BadRequestError, match="not available"):
        asyncio.run(OpenAIVectorStoreConfig().adiscover("databases", {}, {}))
