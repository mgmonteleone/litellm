import json
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.vector_store_endpoints.management_endpoints import vector_store_provider_defaults

ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin", api_key="sk-admin")
VIEWER: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, user_id="v", api_key="sk-v")


@pytest.fixture
def feature_access_allowed():
    with patch("litellm.proxy.vector_store_endpoints.management_endpoints.check_feature_access_for_user", AsyncMock()):
        yield


@pytest.mark.asyncio
async def test_admin_gets_the_deployments_mongodb_defaults(feature_access_allowed, monkeypatch) -> None:
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example")
    monkeypatch.setenv("MONGODB_SIDECAR_API_KEY", "the-deployment-secret")

    response: Final = await vector_store_provider_defaults("mongodb", ADMIN)

    assert dict(response) == {
        "custom_llm_provider": "mongodb",
        "api_base": "https://deployment-sidecar.example",
        "api_key_configured": True,
    }
    assert "the-deployment-secret" not in json.dumps(response)


@pytest.mark.asyncio
async def test_admin_gets_nulls_when_the_deployment_has_not_configured_mongodb(
    feature_access_allowed, monkeypatch
) -> None:
    monkeypatch.delenv("MONGODB_SIDECAR_API_BASE", raising=False)
    monkeypatch.delenv("MONGODB_SIDECAR_API_KEY", raising=False)

    response: Final = await vector_store_provider_defaults("mongodb", ADMIN)

    assert dict(response) == {"custom_llm_provider": "mongodb", "api_base": None, "api_key_configured": False}


@pytest.mark.asyncio
async def test_other_providers_report_no_defaults(feature_access_allowed, monkeypatch) -> None:
    monkeypatch.setenv("MONGODB_SIDECAR_API_BASE", "https://deployment-sidecar.example")
    monkeypatch.setenv("MONGODB_SIDECAR_API_KEY", "the-deployment-secret")

    response: Final = await vector_store_provider_defaults("openai", ADMIN)

    assert dict(response) == {"custom_llm_provider": "openai", "api_base": None, "api_key_configured": False}


@pytest.mark.asyncio
async def test_non_admins_are_refused(feature_access_allowed, monkeypatch) -> None:
    monkeypatch.setenv("MONGODB_SIDECAR_API_KEY", "the-deployment-secret")

    with pytest.raises(HTTPException) as error:
        await vector_store_provider_defaults("mongodb", VIEWER)

    assert error.value.status_code == 403
