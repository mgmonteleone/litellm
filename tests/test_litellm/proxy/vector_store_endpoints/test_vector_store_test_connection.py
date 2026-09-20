from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.management_endpoints import (
    discover_vector_store_resources,
    vector_store_test_connection,
)
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    VectorStoreDiscoverRequest,
    VectorStoreTestConnectionRequest,
    VectorStoreTestConnectionResponse,
)
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin", api_key="sk-admin")
VIEWER: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, user_id="v", api_key="sk-v")


class FakeConfig:
    def __init__(self) -> None:
        self.test_calls: list[tuple[Mapping[str, object], str | None]] = []
        self.discover_calls: list[tuple[str, Mapping[str, object], Mapping[str, object]]] = []
        self.executors: list[object] = []

    async def atest_connection(self, litellm_params, vector_store_id, embedding_executor=None):
        self.test_calls.append((dict(litellm_params), vector_store_id))
        self.executors.append(embedding_executor)
        return VectorStoreTestConnectionResponse(
            ok=True,
            supported=True,
            custom_llm_provider="mongodb",
            summary="All checks passed.",
            checks=[],
            details=None,
        )

    async def adiscover(self, kind, litellm_params, options):
        self.discover_calls.append((kind, dict(litellm_params), dict(options)))
        if kind == "fields":
            raise litellm.BadRequestError(message="mongodb_collection is required", model=None, llm_provider="mongodb")
        return {"databases": ["knowledge"]}


@pytest.fixture
def fake_config():
    config: Final = FakeConfig()
    with (
        patch("litellm.utils.ProviderConfigManager.get_provider_vector_stores_config", return_value=config),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch("litellm.proxy.vector_store_endpoints.management_endpoints.check_feature_access_for_user", AsyncMock()),
    ):
        yield config


@pytest.fixture
def registry_store():
    store: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="policy_index",
        custom_llm_provider="mongodb",
        litellm_params={
            "api_base": "http://127.0.0.1:8080",
            "api_key": "saved-secret",
            "mongodb_database": "knowledge",
            "mongodb_collection": "policies",
        },
    )
    registry: Final = VectorStoreRegistry(vector_stores=[store])
    with patch.object(litellm, "vector_store_registry", registry):
        yield store


@pytest.mark.asyncio
async def test_non_admins_are_refused(fake_config) -> None:
    with pytest.raises(HTTPException) as error:
        await vector_store_test_connection(
            VectorStoreTestConnectionRequest(custom_llm_provider="mongodb", litellm_params={}), VIEWER
        )
    assert error.value.status_code == 403
    assert fake_config.test_calls == []


@pytest.mark.asyncio
async def test_unsaved_configuration_is_passed_through(fake_config) -> None:
    params: Final = {"api_base": "http://127.0.0.1:8080", "api_key": "k", "mongodb_database": "d"}
    response: Final = await vector_store_test_connection(
        VectorStoreTestConnectionRequest(custom_llm_provider="mongodb", vector_store_id=None, litellm_params=params),
        ADMIN,
    )
    assert response["ok"] is True
    assert fake_config.test_calls == [(params, None)]


@pytest.mark.asyncio
async def test_saved_store_keeps_its_secret_when_request_sends_the_redaction_sentinel(
    fake_config, registry_store
) -> None:
    await vector_store_test_connection(
        VectorStoreTestConnectionRequest(
            vector_store_id="policy_index",
            litellm_params={"api_key": REDACTED_BY_LITELM_STRING, "mongodb_collection": "other"},
        ),
        ADMIN,
    )
    merged, vector_store_id = fake_config.test_calls[0]
    assert vector_store_id == "policy_index"
    assert merged["api_key"] == "saved-secret"
    assert merged["mongodb_collection"] == "other"
    assert merged["mongodb_database"] == "knowledge"
    assert "vector_store_id" not in merged and "custom_llm_provider" not in merged


@pytest.mark.asyncio
async def test_environment_references_and_unknown_stores_are_rejected(fake_config) -> None:
    with pytest.raises(HTTPException) as env_error:
        await vector_store_test_connection(
            VectorStoreTestConnectionRequest(
                custom_llm_provider="mongodb", litellm_params={"api_key": "os.environ/SECRET"}
            ),
            ADMIN,
        )
    assert env_error.value.status_code == 400
    with patch.object(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=[])):
        with pytest.raises(HTTPException) as missing:
            await vector_store_test_connection(VectorStoreTestConnectionRequest(vector_store_id="nope"), ADMIN)
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as no_provider:
        await vector_store_test_connection(VectorStoreTestConnectionRequest(litellm_params={}), ADMIN)
    assert no_provider.value.status_code == 400


@pytest.mark.asyncio
async def test_unsupported_provider_returns_400(fake_config) -> None:
    with patch("litellm.utils.ProviderConfigManager.get_provider_vector_stores_config", return_value=None):
        with pytest.raises(HTTPException) as error:
            await vector_store_test_connection(
                VectorStoreTestConnectionRequest(custom_llm_provider="openai", litellm_params={}), ADMIN
            )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_discovery_proxies_kind_and_options_and_maps_errors(fake_config, registry_store) -> None:
    databases: Final = await discover_vector_store_resources(
        VectorStoreDiscoverRequest(vector_store_id="policy_index", kind="databases"), ADMIN
    )
    assert dict(databases) == {"databases": ["knowledge"]}
    kind, merged, options = fake_config.discover_calls[0]
    assert kind == "databases" and merged["api_key"] == "saved-secret" and options == {}
    with pytest.raises(HTTPException) as error:
        await discover_vector_store_resources(
            VectorStoreDiscoverRequest(vector_store_id="policy_index", kind="fields", options={"sample_size": 5}), ADMIN
        )
    assert error.value.status_code == 400 and "mongodb_collection" in str(error.value.detail)
    with pytest.raises(HTTPException) as forbidden:
        await discover_vector_store_resources(
            VectorStoreDiscoverRequest(vector_store_id="policy_index", kind="databases"), VIEWER
        )
    assert forbidden.value.status_code == 403


@pytest.mark.asyncio
async def test_embedding_probe_goes_through_the_proxy_router_when_available(fake_config) -> None:
    from litellm.llms.base_llm.vector_store.transformation import RouterVectorStoreEmbeddingExecutor

    router: Final = litellm.Router(model_list=[])
    with patch("litellm.proxy.proxy_server.llm_router", router):
        await vector_store_test_connection(
            VectorStoreTestConnectionRequest(custom_llm_provider="mongodb", litellm_params={"api_key": "k"}), ADMIN
        )
    with patch("litellm.proxy.proxy_server.llm_router", None):
        await vector_store_test_connection(
            VectorStoreTestConnectionRequest(custom_llm_provider="mongodb", litellm_params={"api_key": "k"}), ADMIN
        )
    routed, direct = fake_config.executors
    assert isinstance(routed, RouterVectorStoreEmbeddingExecutor) and routed.router is router
    assert direct is None


@pytest.mark.asyncio
async def test_nested_environment_references_are_rejected(fake_config) -> None:
    with pytest.raises(HTTPException) as error:
        await vector_store_test_connection(
            VectorStoreTestConnectionRequest(
                custom_llm_provider="mongodb",
                litellm_params={"litellm_embedding_config": {"api_key": "os.environ/SECRET"}},
            ),
            ADMIN,
        )
    assert error.value.status_code == 400
    assert fake_config.test_calls == []


@pytest.mark.asyncio
async def test_saved_environment_references_are_resolved_before_the_provider_sees_them(
    fake_config, monkeypatch
) -> None:
    monkeypatch.setenv("SIDECAR_KEY_FOR_TEST", "from-environment")
    store: Final = LiteLLM_ManagedVectorStore(
        vector_store_id="policy_index",
        custom_llm_provider="mongodb",
        litellm_params={"api_base": "http://127.0.0.1:8080", "api_key": "os.environ/SIDECAR_KEY_FOR_TEST"},
    )
    with patch.object(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=[store])):
        await vector_store_test_connection(VectorStoreTestConnectionRequest(vector_store_id="policy_index"), ADMIN)
    merged, _ = fake_config.test_calls[0]
    assert merged["api_key"] == "from-environment"
