import json
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from fastapi import HTTPException, Request

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.vector_store.transformation import (
    ModelKind,
    model_not_configured_message,
    router_serves_model,
)
from litellm.proxy._experimental.mcp_server.ui_session_utils import (
    is_ui_session_credential,
    resolve_ui_session_team_ids,
)
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.types.utils import LlmProviders
from litellm.types.vector_stores import (
    NON_ADMIN_VECTOR_STORE_MAPPING_PARAMS,
    NON_ADMIN_VECTOR_STORE_PARAMS,
    VECTOR_STORE_ENDPOINT_KEYS,
    LiteLLM_ManagedVectorStore,
)
from litellm.utils import ProviderConfigManager
from litellm.vector_stores.vector_store_registry import contains_env_reference, nested_param_entries

if TYPE_CHECKING:
    from litellm.router import Router


def _normalize_litellm_params(
    vector_store: LiteLLM_ManagedVectorStore,
) -> LiteLLM_ManagedVectorStore:
    litellm_params: Final = vector_store.get("litellm_params")
    if isinstance(litellm_params, str):
        normalized: Final = LiteLLM_ManagedVectorStore(**dict(vector_store))
        try:
            parsed: Final = json.loads(litellm_params)
            normalized["litellm_params"] = parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            normalized["litellm_params"] = {}
        return normalized
    return vector_store


def is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    )


def assert_proxy_admin_for_vector_store_index_management(
    user_api_key_dict: UserAPIKeyAuth,
    *,
    operation: Literal["create", "delete", "update", "list"] = "create",
) -> None:
    """Raise 403 unless the caller is a proxy admin."""
    if is_proxy_admin(user_api_key_dict):
        return
    raise HTTPException(
        status_code=403,
        detail=(f"Only proxy admins can {operation} vector store indexes. Contact your LiteLLM administrator."),
    )


def assert_proxy_admin_for_env_references(params: object, user_api_key_dict: UserAPIKeyAuth) -> None:
    if is_proxy_admin(user_api_key_dict) or not contains_env_reference(params):
        return
    raise HTTPException(
        status_code=403,
        detail="Only proxy admins can save or change vector store settings that contain os.environ/ references. "
        "Enter the value itself, or ask a proxy admin to make this change.",
    )


CREDENTIAL_NAME_KEY: Final = "litellm_credential_name"
_EMPTY_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


def _raise_admin_only(keys: Iterable[str]) -> NoReturn:
    raise HTTPException(
        status_code=403,
        detail="Only proxy admins can set, change or clear these vector store settings: "
        f"{', '.join(sorted(keys))}. Leave these fields as they are, or ask a proxy admin to make this change.",
    )


def _non_admin_may_set(key: str, value: object) -> bool:
    if key not in NON_ADMIN_VECTOR_STORE_PARAMS:
        return False
    if key in NON_ADMIN_VECTOR_STORE_MAPPING_PARAMS:
        return True
    entries: Final = nested_param_entries(value)
    return (
        entries is not None
        and not isinstance(value, Mapping)
        and all(len(path) <= 1 and not isinstance(item, Mapping) for path, item in entries)
    )


def _admin_only_params(params: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {key: value for key, value in params.items() if value is not None and not _non_admin_may_set(key, value)}
    )


def _changed_admin_only_keys(requested: Mapping[str, object], saved: Mapping[str, object]) -> frozenset[str]:
    if nested_param_entries(requested) is None or nested_param_entries(saved) is None:
        return frozenset({"litellm_params"})
    requested_admin_only: Final = _admin_only_params(requested)
    saved_admin_only: Final = _admin_only_params(saved)
    return frozenset(
        key
        for key in requested_admin_only.keys() | saved_admin_only.keys()
        if requested_admin_only.get(key) != saved_admin_only.get(key)
    )


def assert_proxy_admin_for_vector_store_params(
    params: Mapping[str, object] | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    saved_params: Mapping[str, object] | None = None,
) -> None:
    """A non-admin may only set, change or clear NON_ADMIN_VECTOR_STORE_PARAMS. Keeping exactly what
    ``saved_params`` holds is allowed, so an owner can edit their store without touching what an admin set."""
    if is_proxy_admin(user_api_key_dict):
        return
    changed: Final = _changed_admin_only_keys(params or _EMPTY_PARAMS, saved_params or _EMPTY_PARAMS)
    if changed:
        _raise_admin_only(changed)


_REQUEST_ADMIN_ONLY_KEYS: Final = VECTOR_STORE_ENDPOINT_KEYS | frozenset({CREDENTIAL_NAME_KEY})


def assert_proxy_admin_for_request_endpoints(payload: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth) -> None:
    """A search or query body picks endpoints only at its top level; nested values such as filters are data."""
    if is_proxy_admin(user_api_key_dict):
        return
    requested: Final = frozenset(
        key for key, value in payload.items() if key in _REQUEST_ADMIN_ONLY_KEYS and value is not None
    )
    if requested:
        _raise_admin_only(requested)


STORE_EMBEDDING_MODEL_KEYS: Final = ("litellm_embedding_model", "embedding_model")


def store_embedding_models(
    params: Mapping[str, object] | None, saved_params: Mapping[str, object] | None = None
) -> tuple[tuple[str, ModelKind], ...]:
    """The embedding models ``params`` sets on a store, skipping any that ``saved_params`` already holds."""
    requested: Final = params or _EMPTY_PARAMS
    saved: Final = saved_params or _EMPTY_PARAMS
    return tuple(
        (model, "embedding")
        for key in STORE_EMBEDDING_MODEL_KEYS
        if isinstance(model := requested.get(key), str) and model and model != saved.get(key)
    )


async def assert_caller_can_use_models(
    models: Iterable[tuple[str, ModelKind]],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router | None",
) -> None:
    """A non-admin may only name models this proxy serves and their key and team may call."""
    if is_proxy_admin(user_api_key_dict):
        return
    from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

    for model, kind in models:
        if llm_router is None or not router_serves_model(llm_router, model, user_api_key_dict.team_id):
            raise HTTPException(status_code=400, detail=model_not_configured_message(model, kind))
        try:
            await can_key_call_resolved_model(
                model=model, llm_model_list=None, valid_token=user_api_key_dict, llm_router=llm_router
            )
        except ProxyException as denial:
            raise HTTPException(status_code=403, detail=denial.message) from None


def _suffix_after_index_name(request_path: str, index_name: str) -> str | None:
    """Return the path suffix after ``/indexes/{index_name}``, or None if absent."""
    match: Final = re.search(rf"/indexes/{re.escape(index_name)}(?=$|[/?])", request_path)
    if match is None:
        return None
    return request_path[match.end() :]


def _is_vector_store_index_lifecycle_request(
    request_method: str,
    request_path: str,
    index_name: str,
) -> bool:
    """
    True when the request creates or deletes a search index itself (not documents).

    Examples (admin-only):
    - DELETE /azure_ai/indexes/my-index
    - PUT /azure_ai/indexes/my-index
    - POST /azure_ai/indexes
    """
    if request_method not in ("POST", "PUT", "DELETE", "PATCH"):
        return False

    suffix: Final = _suffix_after_index_name(request_path, index_name)
    if suffix is not None:
        # Document operations live under /indexes/{name}/docs/...
        if suffix.startswith("/docs"):
            return False
        # DELETE/PUT/PATCH on /indexes/{name} itself is index lifecycle.
        if suffix == "" or suffix.startswith("?"):
            return True

    # POST /indexes (create index at service level; no index name in path).
    normalized: Final = request_path.split("?", 1)[0].rstrip("/")
    if request_method == "POST" and normalized.endswith("/indexes"):
        return True

    return False


def _object_permission_allows_vector_store(
    object_permission: LiteLLM_ObjectPermissionTable | None,
    vector_store_id: str,
) -> bool:
    """Returns True if an object permission explicitly allowlists the vector store."""
    if object_permission is None:
        return False
    allowed: Final = object_permission.vector_stores
    if not allowed:
        return False
    return vector_store_id in allowed


async def _get_object_permission_for_id(
    object_permission_id: str | None,
) -> LiteLLM_ObjectPermissionTable | None:
    """Load an object permission record by id, using the shared cache/DB helper."""
    if not object_permission_id:
        return None

    from litellm.proxy.auth.auth_checks import get_object_permission
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        return None

    try:
        return await get_object_permission(
            object_permission_id=object_permission_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            "Failed to load object_permission id=%s: %s",
            object_permission_id,
            e,
        )
        return None


async def can_user_access_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """
    Returns True if the caller is allowed to access this managed vector store.

    Access is granted (first match wins) when any of the following is true:
    1. The caller's role is PROXY_ADMIN.
    2. The vector store has no team_id (legacy behavior - accessible to all).
    3. The caller's key-level object_permission.vector_stores explicitly lists
       this vector store id.
    4. The caller's team-level object_permission.vector_stores explicitly lists
       this vector store id.
    5. The caller's team_id matches the vector store's team_id.

    A dashboard session credential is evaluated against the same effective
    contexts as listing (its own grants plus each real team of the user).
    Otherwise access is denied.
    """
    if is_proxy_admin(user_api_key_dict):
        return True

    if vector_store.get("team_id") is None:
        return True

    auth_contexts: Final = await _vector_store_auth_contexts(user_api_key_dict)
    return await _is_vector_store_granted_to_any(vector_store, auth_contexts)


async def _is_vector_store_granted(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    vector_store_id: Final = vector_store.get("vector_store_id") or ""

    key_object_permission = user_api_key_dict.object_permission
    if key_object_permission is None:
        key_object_permission = await _get_object_permission_for_id(user_api_key_dict.object_permission_id)
    if _object_permission_allows_vector_store(key_object_permission, vector_store_id):
        return True

    team_object_permission: LiteLLM_ObjectPermissionTable | None = user_api_key_dict.team_object_permission
    if team_object_permission is None:
        team_object_permission = await _get_object_permission_for_id(user_api_key_dict.team_object_permission_id)
    if _object_permission_allows_vector_store(team_object_permission, vector_store_id):
        return True

    return user_api_key_dict.team_id is not None and user_api_key_dict.team_id == vector_store.get("team_id")


async def _team_auth_context(team_id: str, user_api_key_dict: UserAPIKeyAuth) -> UserAPIKeyAuth:
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    team: Final = await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    return user_api_key_dict.model_copy(
        update=MappingProxyType(
            {
                "team_id": team_id,
                "team_object_permission": team.object_permission,
                "team_object_permission_id": team.object_permission_id,
            }
        )
    )


async def _vector_store_auth_contexts(
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[UserAPIKeyAuth, ...]:
    if not is_ui_session_credential(user_api_key_dict):
        return (user_api_key_dict,)
    session_key_context: Final = user_api_key_dict.model_copy(
        update=MappingProxyType({"team_id": None, "team_object_permission": None, "team_object_permission_id": None})
    )
    team_ids: Final = await resolve_ui_session_team_ids(user_api_key_dict)
    team_contexts: Final = tuple([await _team_auth_context(team_id, user_api_key_dict) for team_id in team_ids])
    return (session_key_context, *team_contexts)


async def _is_vector_store_granted_to_any(
    vector_store: LiteLLM_ManagedVectorStore,
    auth_contexts: tuple[UserAPIKeyAuth, ...],
) -> bool:
    for auth_context in auth_contexts:
        if await _is_vector_store_granted(vector_store, auth_context):
            return True
    return False


async def filter_listable_vector_stores(
    vector_stores: Iterable[LiteLLM_ManagedVectorStore],
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[LiteLLM_ManagedVectorStore, ...]:
    """Non-admins only see stores their key, one of their teams' object_permission, or team ownership grants."""
    if is_proxy_admin(user_api_key_dict):
        return tuple(vector_stores)

    auth_contexts: Final = await _vector_store_auth_contexts(user_api_key_dict)
    return tuple([vs for vs in vector_stores if await _is_vector_store_granted_to_any(vs, auth_contexts)])


async def get_litellm_managed_vector_store(
    vector_store_id: str,
) -> LiteLLM_ManagedVectorStore | None:
    """
    Resolve a LiteLLM-managed vector store from the registry or shared cache.

    Provider-native vector store IDs will not be present in either location and
    return None, preserving direct provider behavior while still protecting
    LiteLLM-managed multi-tenant stores.
    """
    if not vector_store_id:
        return None

    if litellm.vector_store_registry is not None:
        try:
            vector_store: Final = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=vector_store_id
            )
            if vector_store is not None:
                return _normalize_litellm_params(vector_store)
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to resolve vector store id=%s from registry: %s",
                vector_store_id,
                e,
            )
            raise HTTPException(
                status_code=500,
                detail="Unable to validate vector store access",
            ) from e

    try:
        from litellm.proxy.auth.auth_checks import (
            get_managed_vector_store_rows_by_uuids,
        )
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if prisma_client is None:
            return None
        rows: Final = await get_managed_vector_store_rows_by_uuids(
            uuids=[vector_store_id],
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if not rows:
            return None
        return _normalize_litellm_params(LiteLLM_ManagedVectorStore(**rows[0].model_dump()))
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to resolve vector store id=%s from shared cache: %s",
            vector_store_id,
            e,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to validate vector store access",
        ) from e


async def assert_user_can_access_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
    detail: str = "Access denied: You do not have permission to access this vector store",
) -> None:
    """Raise 403 unless the caller can access the resolved vector store."""
    if not await can_user_access_vector_store(vector_store, user_api_key_dict):
        raise HTTPException(status_code=403, detail=detail)


async def assert_user_can_access_vector_store_id(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    detail: str = "Access denied: You do not have permission to access this vector store",
) -> LiteLLM_ManagedVectorStore | None:
    """
    Resolve a managed vector store id and enforce ownership if it exists.

    Unknown ids are treated as provider-native ids and are not rejected here.
    """
    vector_store: Final = await get_litellm_managed_vector_store(vector_store_id=vector_store_id)
    if vector_store is not None:
        await assert_user_can_access_vector_store(
            vector_store=vector_store,
            user_api_key_dict=user_api_key_dict,
            detail=detail,
        )
    return vector_store


def _does_endpoint_match(endpoint_path: str, request_path: str) -> bool:
    if endpoint_path in request_path:
        return True
    if "{" in endpoint_path:
        prefix: Final = endpoint_path.split("{", 1)[0]
        if prefix and prefix in request_path:
            return True
    return False


def check_vector_store_permission(
    index_name: str,
    permission: str,
    key_metadata: Mapping[str, object] | None,
    team_metadata: Mapping[str, object] | None,
) -> bool:
    """
    Check if a specific permission is allowed for a given vector store index.

    Args:
        index_name: The name of the vector store index
        permission: The permission to check (e.g., "read", "write")
        key_metadata: Metadata from the API key
        team_metadata: Metadata from the team

    Returns:
        True if the permission is allowed, False otherwise

    Example metadata format:
        "metadata": {
            "allowed_vector_store_indexes": [
                {
                    "index_name": "dall-e-3",
                    "index_permissions": ["write"]
                }
            ]
        }
    """
    # Check both key_metadata and team_metadata
    for metadata in [key_metadata, team_metadata]:
        if metadata is None:
            continue

        allowed_indexes = metadata.get("allowed_vector_store_indexes")
        if not allowed_indexes or not isinstance(allowed_indexes, list):
            continue

        # Look for matching index
        for index_config in allowed_indexes:
            if not isinstance(index_config, dict):
                continue

            if index_config.get("index_name") == index_name:
                index_permissions = index_config.get("index_permissions", [])
                if isinstance(index_permissions, list) and permission in index_permissions:
                    return True

    return False


def is_allowed_to_call_vector_store_endpoint(
    provider: LlmProviders,
    index_name: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Literal[True] | None:
    """
    Check if the user is allowed to call the vector store endpoint.

    Cover:
    1. Creating a vector store index
    2. Reading a vector store index (Search / List / Get)
    """
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True
    # check what allowed permissions are for the key
    key_metadata: Final = user_api_key_dict.metadata
    team_metadata: Final = user_api_key_dict.team_metadata

    provider_config: Final = ProviderConfigManager.get_provider_vector_stores_config(provider=provider)
    if provider_config is None:
        return None

    provider_vector_store_endpoints: Final = provider_config.get_vector_store_endpoints_by_type()

    # Inline import — auth_utils participates in a proxy import cycle.
    from litellm.proxy.auth.auth_utils import get_request_route  # noqa: PLC0415

    request_route: Final = get_request_route(request)

    if _is_vector_store_index_lifecycle_request(
        request_method=request.method,
        request_path=request_route,
        index_name=index_name,
    ):
        operation_label: Literal["create", "delete", "update"] = "create"
        if request.method == "DELETE":
            operation_label = "delete"
        elif request.method in ("PUT", "PATCH"):
            operation_label = "update"
        assert_proxy_admin_for_vector_store_index_management(
            user_api_key_dict,
            operation=operation_label,
        )
        return True

    # Writes are classified before reads so a path matching both patterns
    # requires the stronger grant (e.g. the azure batch write on an index
    # named "analyze*" also contains the "/analyze" read fragment)
    permission_type = None
    for endpoint in provider_vector_store_endpoints["write"]:
        if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
            permission_type = "write"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints["read"]:
            if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
                permission_type = "read"
                break

    if permission_type is None:
        raise HTTPException(
            status_code=403,
            detail=(
                f"User does not have permission to call vector store endpoint "
                f"{index_name}. Ask your administrator to add the necessary "
                "permissions to your API key/Team."
            ),
        )

    # Check if key has specific permission for allowed_vector_store_indexes
    has_permission: Final = check_vector_store_permission(
        index_name=index_name,
        permission=permission_type,
        key_metadata=key_metadata,
        team_metadata=team_metadata,
    )

    if not has_permission:
        raise HTTPException(
            status_code=403,
            detail=f"User does not have permission to call vector store endpoint {index_name}. Ask your administrator to add the necessary permissions to your API key/Team.",
        )

    return has_permission


def is_allowed_to_call_vector_store_files_endpoint(
    provider: LlmProviders,
    vector_store_id: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Literal[True] | None:
    if (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return True

    key_metadata: Final = user_api_key_dict.metadata
    team_metadata: Final = user_api_key_dict.team_metadata

    provider_config: Final = ProviderConfigManager.get_provider_vector_store_files_config(provider=provider)
    if provider_config is None:
        return None

    provider_vector_store_endpoints: Final = provider_config.get_vector_store_file_endpoints_by_type()

    # Inline import — auth_utils participates in a proxy import cycle.
    from litellm.proxy.auth.auth_utils import get_request_route  # noqa: PLC0415

    request_route: Final = get_request_route(request)

    permission_type: str | None = None
    for endpoint in provider_vector_store_endpoints.get("write", ()):
        if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
            permission_type = "write"
            break

    if permission_type is None:
        for endpoint in provider_vector_store_endpoints.get("read", ()):
            if request.method == endpoint[0] and _does_endpoint_match(endpoint[1], request_route):
                permission_type = "read"
                break

    if permission_type is None:
        return None

    has_permission: Final = check_vector_store_permission(
        index_name=vector_store_id,
        permission=permission_type,
        key_metadata=key_metadata,
        team_metadata=team_metadata,
    )

    if not has_permission:
        raise HTTPException(
            status_code=403,
            detail=f"User does not have permission to call vector store file endpoint {vector_store_id}. Ask your administrator to add the necessary permissions to your API key/Team.",
        )

    return has_permission
