"""
VECTOR STORE MANAGEMENT

All /vector_store management endpoints

/vector_store/new
/vector_store/delete
/vector_store/list
"""

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from fastapi import APIRouter, Depends, HTTPException

if TYPE_CHECKING:
    from prisma.models import LiteLLM_ManagedVectorStoresTable as _VectorStoreRow

    from litellm.llms.base_llm.vector_store.transformation import (
        BaseVectorStoreConfig,
        RouterVectorStoreEmbeddingExecutor,
    )
    from litellm.proxy.utils import PrismaClient
import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import (
    LiteLLM_ManagedVectorStoresTable,
    LitellmUserRoles,
    ResponseLiteLLM_ManagedVectorStore,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.rbac_utils import check_feature_access_for_user
from litellm.proxy.vector_store_endpoints.utils import (
    can_user_access_vector_store,
    filter_listable_vector_stores,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import ManagedVectorStoresRepository
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    VectorStoreDeleteRequest,
    VectorStoreDiscoverRequest,
    VectorStoreInfoRequest,
    VectorStoreTestConnectionRequest,
    VectorStoreTestConnectionResponse,
    VectorStoreUpdateRequest,
)
from litellm.vector_stores.vector_store_registry import (
    VectorStoreRegistry,
    resolve_litellm_params_references,
)

router: Final = APIRouter()


def _vector_store_table(prisma_client: "PrismaClient") -> "TableActions[_VectorStoreRow]":
    return ManagedVectorStoresRepository(prisma_client).table


def _row_to_vector_store(row: "_VectorStoreRow") -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(**row.model_dump())


_LITELLM_PARAMS_MASKER: Final = SensitiveDataMasker(extra_sensitive_patterns=frozenset(("connection",)))
_EMPTY_PARAMS: Final = MappingProxyType({})

# The redaction placeholder every read path echoes back for a saved secret, and the only string an update
# request can send to mean "keep the saved value" (see ``update_vector_store``). It is REDACTED_BY_LITELM,
# with a single L, not REDACTED_BY_LITELLM.
_REDACTION_SENTINEL_TYPO_PATTERN: Final = re.compile(r"^REDACTED_BY_LITEL+M+$", re.IGNORECASE)


_REDACT_LITELLM_PARAMS_MAX_DEPTH: Final = 10


def _redact_sensitive_litellm_params(litellm_params: object, _depth: int = 0) -> Any:
    """
    Replace credential-bearing values in ``litellm_params`` with
    ``REDACTED_BY_LITELM`` while preserving non-secret keys (``api_base``,
    ``region``, ``model``, ``api_version``).

    Handles three input shapes:

    * ``dict`` — recurse into nested dicts (e.g. ``litellm_embedding_config``
      which itself carries ``api_key`` / ``aws_*`` / ``vertex_credentials``).
    * ``str`` — the in-memory registry occasionally holds the params as a
      JSON-serialized string. Parse, redact, re-serialize. If parsing
      fails, return the redaction sentinel rather than echo the value
      back verbatim.
    * Anything else, or ``None`` — passed through.

    Recursion depth is bounded by ``_REDACT_LITELLM_PARAMS_MAX_DEPTH`` —
    matching the convention of other allowlisted recursive helpers in the
    repo (see ``tests/code_coverage_tests/recursive_detector.py``).
    """
    if _depth >= _REDACT_LITELLM_PARAMS_MAX_DEPTH:
        return REDACTED_BY_LITELM_STRING
    if litellm_params is None:
        return None
    if isinstance(litellm_params, str):
        try:
            parsed: Final = json.loads(litellm_params)
        except (TypeError, ValueError):
            return REDACTED_BY_LITELM_STRING
        return json.dumps(_redact_sensitive_litellm_params(parsed, _depth + 1))
    if not isinstance(litellm_params, dict):
        return litellm_params
    out: Final[dict[str, object]] = {}
    for k, v in litellm_params.items():
        if _LITELLM_PARAMS_MASKER.is_sensitive_key(k):
            out[k] = REDACTED_BY_LITELM_STRING
        elif isinstance(v, dict):
            out[k] = _redact_sensitive_litellm_params(v, _depth + 1)
        else:
            out[k] = v
    return out


def _registry_vector_store(vector_store_id: str) -> LiteLLM_ManagedVectorStore | None:
    if litellm.vector_store_registry is None:
        return None
    return litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(vector_store_id=vector_store_id)


def _parse_stored_json_field(raw: object, field_name: str) -> object:
    """The database (and config-registered stores) may hold this field as a JSON string; parse it before it
    reaches response validation, which requires a mapping and would otherwise fail as an unhandled 500."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"The saved vector store has malformed {field_name}.") from None


def _vector_store_info(vector_store: LiteLLM_ManagedVectorStore) -> LiteLLM_ManagedVectorStoresTable:
    """Build the info response, parsing fields the database may hold as JSON strings."""
    metadata: Final = _parse_stored_json_field(vector_store.get("vector_store_metadata"), "vector_store_metadata")
    litellm_params: Final = _parse_stored_json_field(vector_store.get("litellm_params"), "litellm_params")
    return LiteLLM_ManagedVectorStoresTable(
        vector_store_id=vector_store.get("vector_store_id") or "",
        custom_llm_provider=vector_store.get("custom_llm_provider") or "",
        vector_store_name=vector_store.get("vector_store_name") or None,
        vector_store_description=vector_store.get("vector_store_description") or None,
        vector_store_metadata=metadata if isinstance(metadata, dict) else None,
        created_at=vector_store.get("created_at") or None,
        updated_at=vector_store.get("updated_at") or None,
        litellm_credential_name=vector_store.get("litellm_credential_name"),
        litellm_params=_redact_sensitive_litellm_params(litellm_params),
        team_id=vector_store.get("team_id") or None,
        user_id=vector_store.get("user_id") or None,
    )


async def _fetch_and_authorize_vector_store(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: "PrismaClient",
) -> "LiteLLM_ManagedVectorStore":
    """
    Look up a vector store by id and confirm the caller can access it.
    Raises HTTPException(404) on miss and HTTPException(403) on access
    denial.
    """
    row: Final = await _vector_store_table(prisma_client).find_unique(where={"vector_store_id": vector_store_id})
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store with ID {vector_store_id} not found",
        )
    typed: Final = _row_to_vector_store(row)
    if not await _check_vector_store_access(typed, user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="Access denied: You do not have permission to access this vector store",
        )
    return typed


########################################################
# Helper Functions
########################################################
async def _check_vector_store_access(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """
    Check if the user has access to the vector store.

    Delegates to :func:`can_user_access_vector_store`, which honors:
    - PROXY_ADMIN bypass
    - legacy vector stores with no team_id
    - key-level and team-level ``object_permission.vector_stores`` allowlists
    - team_id match between key and store
    """
    return await can_user_access_vector_store(vector_store=vector_store, user_api_key_dict=user_api_key_dict)


async def create_vector_store_in_db(
    vector_store_id: str,
    custom_llm_provider: str,
    prisma_client: "PrismaClient | None",
    vector_store_name: str | None = None,
    vector_store_description: str | None = None,
    vector_store_metadata: dict | None = None,
    litellm_params: dict | None = None,
    litellm_credential_name: str | None = None,
    team_id: str | None = None,
    user_id: str | None = None,
) -> LiteLLM_ManagedVectorStore:
    """
    Helper function to create a vector store in the database.

    This function handles:
    - Checking if vector store already exists
    - Creating the vector store in the database
    - Adding it to the vector store registry

    Returns:
        LiteLLM_ManagedVectorStore: The created vector store object

    Raises:
        HTTPException: If vector store already exists or database error occurs
    """
    from litellm.types.router import GenericLiteLLMParams

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    # Check if vector store already exists
    existing_vector_store: Final = await _vector_store_table(prisma_client).find_unique(
        where={"vector_store_id": vector_store_id}
    )
    if existing_vector_store is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store with ID {vector_store_id} already exists",
        )

    # Prepare data for database
    data_to_create: Final[dict[str, object]] = {
        "vector_store_id": vector_store_id,
        "custom_llm_provider": custom_llm_provider,
    }

    if vector_store_name is not None:
        data_to_create["vector_store_name"] = vector_store_name
    if vector_store_description is not None:
        data_to_create["vector_store_description"] = vector_store_description
    if vector_store_metadata is not None:
        data_to_create["vector_store_metadata"] = safe_dumps(vector_store_metadata)
    if litellm_credential_name is not None:
        data_to_create["litellm_credential_name"] = litellm_credential_name
    if team_id is not None:
        data_to_create["team_id"] = team_id
    if user_id is not None:
        data_to_create["user_id"] = user_id

    # Handle litellm_params - always provide at least an empty dict.
    # The earlier behaviour resolved ``litellm_embedding_config`` from the
    # admin-configured router/DB model and persisted the cleartext result
    # (``api_key``, ``api_base``, ``api_version``) into this row. That
    # exposed every env-stored embedding-model credential on the
    # ``/vector_store/{new,info,update,list}`` responses. Keep the user's
    # raw ``litellm_embedding_model`` reference; each search embeds the
    # query through the router at request time, so the credentials stay
    # on the deployment and never reach the database.
    if litellm_params:
        litellm_params_dict: Final = GenericLiteLLMParams(**litellm_params).model_dump(exclude_none=True)
        data_to_create["litellm_params"] = safe_dumps(litellm_params_dict)
    else:
        # Provide empty dict if no litellm_params provided
        data_to_create["litellm_params"] = safe_dumps({})

    # Create in database
    _new_vector_store: Final = await _vector_store_table(prisma_client).create(data=data_to_create)

    new_vector_store: Final[LiteLLM_ManagedVectorStore] = _row_to_vector_store(_new_vector_store)

    # Add vector store to registry
    if litellm.vector_store_registry is not None:
        litellm.vector_store_registry.add_vector_store_to_registry(vector_store=new_vector_store)

    verbose_proxy_logger.info("Vector store %s created in database successfully", vector_store_id)

    return new_vector_store


########################################################
# Management Endpoints
########################################################
@router.post(
    "/vector_store/new",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new vector store.

    Parameters:
    - vector_store_id: str - Unique identifier for the vector store
    - custom_llm_provider: str - Provider of the vector store
    - vector_store_name: Optional[str] - Name of the vector store
    - vector_store_description: Optional[str] - Description of the vector store
    - vector_store_metadata: Optional[Dict] - Additional metadata for the vector store
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    try:
        vector_store_id: Final = vector_store.get("vector_store_id")
        custom_llm_provider: Final = vector_store.get("custom_llm_provider")

        if not vector_store_id or not custom_llm_provider:
            raise HTTPException(
                status_code=400,
                detail="vector_store_id and custom_llm_provider are required",
            )

        # Extract and validate metadata
        metadata: Final = vector_store.get("vector_store_metadata")
        validated_metadata: dict | None = None
        if metadata is not None and isinstance(metadata, dict):
            validated_metadata = metadata

        new_vector_store: Final = await create_vector_store_in_db(
            vector_store_id=vector_store_id,
            custom_llm_provider=custom_llm_provider,
            prisma_client=prisma_client,
            vector_store_name=vector_store.get("vector_store_name"),
            vector_store_description=vector_store.get("vector_store_description"),
            vector_store_metadata=validated_metadata,
            litellm_params=vector_store.get("litellm_params"),
            litellm_credential_name=vector_store.get("litellm_credential_name"),
            team_id=user_api_key_dict.team_id,
            user_id=user_api_key_dict.user_id,
        )

        # Apply the same litellm_params redaction the list / info / update
        # endpoints already use, so a caller-supplied credential or a
        # cleartext value persisted by an earlier proxy version doesn't
        # come back in the response.
        response_vs: Final = LiteLLM_ManagedVectorStore(**new_vector_store)
        response_vs["litellm_params"] = _redact_sensitive_litellm_params(new_vector_store.get("litellm_params"))

        return {
            "status": "success",
            "message": f"Vector store {vector_store.get('vector_store_id')} created successfully",
            "vector_store": response_vs,
        }
    except Exception as e:
        verbose_proxy_logger.exception("Error creating vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
@router.get(
    "/v1/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
async def list_vector_stores(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = 1,
    page_size: int = 100,
):
    """
    List all available vector stores with optional filtering and pagination.
    Combines both in-memory vector stores and those stored in the database.
    Database is the source of truth - deleted stores are removed from memory, updated stores sync to memory.

    Parameters:
    - page: int - Page number for pagination (default: 1)
    - page_size: int - Number of items per page (default: 100)
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    try:
        vector_stores_from_db: Final = await VectorStoreRegistry._get_vector_stores_from_db(prisma_client=prisma_client)
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.sync_with_db(vector_stores_from_db)
        # Config-registered stores exist only in memory; database rows win for any id in both.
        vector_store_map: Final = {
            store_id: store
            for source in (
                litellm.vector_store_registry.vector_stores if litellm.vector_store_registry is not None else (),
                vector_stores_from_db,
            )
            for store in source
            if (store_id := store.get("vector_store_id"))
        }

        # Filter vector stores based on access control
        accessible_vector_stores: Final = []
        for vs in await filter_listable_vector_stores(vector_store_map.values(), user_api_key_dict):
            redacted = LiteLLM_ManagedVectorStore(**vs)
            redacted["litellm_params"] = _redact_sensitive_litellm_params(vs.get("litellm_params"))
            accessible_vector_stores.append(redacted)

        total_count: Final = len(accessible_vector_stores)
        total_pages: Final = (total_count + page_size - 1) // page_size

        # Format response using LiteLLM_ManagedVectorStoreListResponse
        response: Final = LiteLLM_ManagedVectorStoreListResponse(
            object="list",
            data=accessible_vector_stores,
            total_count=total_count,
            current_page=page,
            total_pages=total_pages,
        )

        return response
    except Exception as e:
        verbose_proxy_logger.exception("Error listing vector stores: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/delete",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_vector_store(
    data: VectorStoreDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a vector store from both database and in-memory registry.

    Parameters:
    - vector_store_id: str - ID of the vector store to delete
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if vector store exists in database or in-memory registry
        db_vector_store_exists = False
        memory_vector_store_exists = False
        vector_store_to_check = None

        existing_vector_store: Final = await _vector_store_table(prisma_client).find_unique(
            where={"vector_store_id": data.vector_store_id}
        )
        if existing_vector_store is not None:
            db_vector_store_exists = True
            vector_store_to_check = _row_to_vector_store(existing_vector_store)

        # Check in-memory registry
        if litellm.vector_store_registry is not None:
            memory_vector_store: Final = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
            if memory_vector_store is not None:
                memory_vector_store_exists = True
                if vector_store_to_check is None:
                    vector_store_to_check = memory_vector_store

        # If not found in either location, raise 404
        if not db_vector_store_exists and not memory_vector_store_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {data.vector_store_id} not found",
            )

        # Check access control
        if vector_store_to_check and not await _check_vector_store_access(vector_store_to_check, user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You do not have permission to delete this vector store",
            )

        # Delete from database if exists
        if db_vector_store_exists:
            await _vector_store_table(prisma_client).delete(where={"vector_store_id": data.vector_store_id})

        # Delete from in-memory registry if exists
        if memory_vector_store_exists and litellm.vector_store_registry is not None:
            litellm.vector_store_registry.delete_vector_store_from_registry(vector_store_id=data.vector_store_id)

        return {
            "status": "success",
            "message": f"Vector store {data.vector_store_id} deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/info",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ResponseLiteLLM_ManagedVectorStore,
)
async def get_vector_store_info(
    data: VectorStoreInfoRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return a single vector store's details"""
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # The database is the source of truth. Memory only answers for config-registered stores, which
        # have no row, so info never serves a copy the ten-second registry sync has not caught up with.
        row: Final = await _vector_store_table(prisma_client).find_unique(
            where={"vector_store_id": data.vector_store_id}
        )
        vector_store: Final = (
            _row_to_vector_store(row) if row is not None else _registry_vector_store(data.vector_store_id)
        )
        if vector_store is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {data.vector_store_id} not found",
            )
        if not await _check_vector_store_access(vector_store, user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You do not have permission to access this vector store",
            )
        return {"vector_store": _vector_store_info(vector_store)}
    except HTTPException:
        # Preserve 403/404 from the access-control / not-found checks above;
        # the catch-all below would otherwise rewrite them as 500.
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting vector store info: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/update",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_vector_store(
    data: VectorStoreUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update vector store details in both database and in-memory registry.
    The updated data is immediately synchronized to the in-memory registry.

    A ``litellm_params`` value equal to the redaction sentinel ``REDACTED_BY_LITELM`` (single L) keeps the
    saved secret instead of overwriting it; a near-miss like the double-L ``REDACTED_BY_LITELLM`` is rejected
    with a 400 rather than persisted as the literal credential.
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client
    from litellm.types.router import GenericLiteLLMParams

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        update_data: Final = data.model_dump(exclude_unset=True)
        vector_store_id: Final[str] = update_data.pop("vector_store_id")

        # Per-store access control: anyone authenticated who passes the
        # premium-feature gate could otherwise update *any* vector store —
        # including stores belonging to other teams.
        saved_store: Final = await _fetch_and_authorize_vector_store(
            vector_store_id=vector_store_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )

        # Handle metadata serialization
        if update_data.get("vector_store_metadata") is not None:
            update_data["vector_store_metadata"] = safe_dumps(update_data["vector_store_metadata"])

        # Merge request litellm_params over the saved ones, the same way /vector_store/new persists them: request
        # keys win, a value equal to the redaction sentinel keeps the saved secret instead of overwriting it, and
        # an os.environ/ reference is stored as-is and resolved later by resolve_litellm_params_references (see
        # build_request_data_from_managed_vector_store), matching how /vector_store/new already behaves. Only the
        # ad hoc test_connection/discover path (_resolve_connection_target) still rejects unresolved references,
        # since those run against the value immediately rather than persisting it. This stores the raw params (no
        # credential resolution), since each search embeds the query through the router at request time.
        if "litellm_params" in update_data:
            _reject_misspelled_redaction_sentinel(update_data.get("litellm_params") or _EMPTY_PARAMS)
            request_litellm_params: Final = {
                key: value
                for key, value in (update_data.get("litellm_params") or _EMPTY_PARAMS).items()
                if value != REDACTED_BY_LITELM_STRING
            }
            merged_litellm_params: Final = {**_saved_raw_litellm_params(saved_store), **request_litellm_params}
            litellm_params_dict: Final = GenericLiteLLMParams.model_validate(merged_litellm_params).model_dump(
                exclude_none=True
            )
            update_data["litellm_params"] = safe_dumps(litellm_params_dict)

        # Update in database
        updated: Final = await _vector_store_table(prisma_client).update(
            where={"vector_store_id": vector_store_id},
            data=update_data,
        )

        if updated is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {vector_store_id} not found",
            )

        updated_vs: Final = _row_to_vector_store(updated)

        # Immediately update in-memory registry to keep it in sync
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.update_vector_store_in_registry(
                vector_store_id=vector_store_id,
                updated_data=updated_vs,
            )
            verbose_proxy_logger.debug(
                "Updated vector store %s in both database and in-memory registry", vector_store_id
            )

        # The DB row is returned in full, so the response would otherwise
        # echo the persisted ``litellm_params`` (including provider
        # credentials) back to the caller — even when the caller only
        # changed unrelated fields like ``vector_store_description``.
        response_vs: Final = LiteLLM_ManagedVectorStore(**updated_vs)
        response_vs["litellm_params"] = _redact_sensitive_litellm_params(
            _parse_stored_json_field(updated_vs.get("litellm_params"), "litellm_params")
        )
        return {
            "status": "success",
            "message": f"Vector store {vector_store_id} updated successfully",
            "vector_store": response_vs,
        }
    except HTTPException:
        # Preserve 403/404 responses from the access-control / not-found
        # checks above; the catch-all below would otherwise rewrite them
        # as 500 with the original status code embedded in the detail.
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------------------------------------------
# Test connection and discovery
# ---------------------------------------------------------------------------------------------------------------


def _assert_proxy_admin(user_api_key_dict: UserAPIKeyAuth, action: str) -> None:
    """These endpoints probe arbitrary hosts with caller-supplied credentials, so they are admin only."""
    if user_api_key_dict.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value):
        return
    raise HTTPException(status_code=403, detail=f"Only proxy admins can {action} vector store connections.")


def _reject_environment_references(params: Mapping[str, object]) -> None:
    """Request-supplied values must already be resolved; nested references are rejected like top-level ones."""
    from litellm.proxy.health_endpoints._health_endpoints import (
        _reject_os_environ_references,  # pyright: ignore[reportPrivateUsage]  # shared nested-walk guard
    )

    _reject_os_environ_references(dict(params))  # mutable-ok: the shared guard takes a dict


def _reject_misspelled_redaction_sentinel(params: Mapping[str, object]) -> None:
    """update_vector_store persists an os.environ/ api_key as-is (unlike _resolve_connection_target above), so a
    fat-fingered sentinel like ``REDACTED_BY_LITELLM`` no longer gets caught by the "reject env references" guard;
    it would instead be saved verbatim as the literal api_key value, silently clobbering the real secret."""
    api_key: Final = params.get("api_key")
    looks_like_sentinel: Final = isinstance(api_key, str) and bool(_REDACTION_SENTINEL_TYPO_PATTERN.match(api_key))
    if looks_like_sentinel and api_key != REDACTED_BY_LITELM_STRING:
        raise HTTPException(
            status_code=400,
            detail=f"'api_key' looks like a misspelled redaction sentinel. Use the exact string "
            f"'{REDACTED_BY_LITELM_STRING}' to keep the saved secret.",
        )


def _saved_litellm_params(vector_store: LiteLLM_ManagedVectorStore) -> dict[str, object]:  # mutable-ok: merged copy
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor

    raw: Final = vector_store.get("litellm_params")
    try:
        parsed: Final = json.loads(raw) if isinstance(raw, str) else (raw or _EMPTY_PARAMS)
    except ValueError:
        raise HTTPException(status_code=400, detail="The saved vector store has malformed litellm_params.") from None
    merged: Final = dict(  # mutable-ok: merged copy
        resolve_litellm_params_references(parsed if isinstance(parsed, Mapping) else None)
    )
    credential_name: Final = vector_store.get("litellm_credential_name")
    if credential_name and litellm.credential_list:
        merged.update(CredentialAccessor.get_credential_values(credential_name))
    return merged


def _saved_raw_litellm_params(vector_store: LiteLLM_ManagedVectorStore) -> dict[str, object]:  # mutable-ok: merged copy
    """The saved litellm_params as persisted, with no environment/credential resolution: an update should
    merge over what is actually stored (which may itself hold an os.environ/ reference), not a live-connection
    value only meant for the test_connection and search paths."""
    parsed: Final = _parse_stored_json_field(vector_store.get("litellm_params"), "litellm_params")
    return dict(parsed) if isinstance(parsed, Mapping) else {}  # mutable-ok: merged copy


async def _resolve_connection_target(
    data: VectorStoreTestConnectionRequest, user_api_key_dict: UserAPIKeyAuth
) -> tuple[str, dict[str, object], str | None]:  # mutable-ok: merged copy
    """Merge a saved store (when vector_store_id is given) with request overrides.

    A request value equal to the redaction sentinel keeps the saved secret, so the dashboard can re-test a
    store without asking the admin to paste the key again.
    """
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.proxy.proxy_server import prisma_client

    request_params: Final = {  # mutable-ok: merged copy
        key: value
        for key, value in (data.litellm_params or _EMPTY_PARAMS).items()
        if value != REDACTED_BY_LITELM_STRING
    }
    _reject_environment_references(request_params)
    saved: dict[str, object] = {}  # mutable-ok: merged copy
    provider: str | None = data.custom_llm_provider
    if data.vector_store_id:
        store: LiteLLM_ManagedVectorStore | None = None
        if litellm.vector_store_registry is not None:
            store = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
        if store is None and prisma_client is not None:
            store = await _fetch_and_authorize_vector_store(
                vector_store_id=data.vector_store_id, user_api_key_dict=user_api_key_dict, prisma_client=prisma_client
            )
        if store is None:
            raise HTTPException(status_code=404, detail=f"Vector store {data.vector_store_id} not found")
        saved = _saved_litellm_params(store)
        provider = provider or store.get("custom_llm_provider")
    if data.litellm_credential_name and litellm.credential_list:
        saved.update(CredentialAccessor.get_credential_values(data.litellm_credential_name))
    if not provider:
        raise HTTPException(status_code=400, detail="custom_llm_provider is required when no vector_store_id is given")
    merged: Final = {  # mutable-ok: merged copy
        key: value
        for key, value in {**saved, **request_params}.items()  # mutable-ok: merged copy
        if key not in ("vector_store_id", "custom_llm_provider")
    }
    requested_id: Final = request_params.get("vector_store_id")
    vector_store_id: Final = data.vector_store_id or (requested_id if isinstance(requested_id, str) else None)
    return provider, merged, vector_store_id


def _router_embedding_executor(user_api_key_dict: UserAPIKeyAuth) -> "RouterVectorStoreEmbeddingExecutor | None":
    """Embed through the proxy's router so model aliases from config or the DB resolve like a real search would."""
    from litellm.llms.base_llm.vector_store.transformation import RouterVectorStoreEmbeddingExecutor
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return None
    metadata: Final = (
        MappingProxyType({"user_api_key_team_id": user_api_key_dict.team_id})
        if user_api_key_dict.team_id
        else _EMPTY_PARAMS
    )
    return RouterVectorStoreEmbeddingExecutor(router=llm_router, metadata=metadata)


def _lookup_provider_config(provider: str) -> "BaseVectorStoreConfig | None":
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    try:
        return ProviderConfigManager.get_provider_vector_stores_config(provider=LlmProviders(provider))
    except ValueError:
        return None


def _provider_config(provider: str) -> "BaseVectorStoreConfig":
    config: Final = _lookup_provider_config(provider)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Vector store provider '{provider}' is not supported")
    return config


@router.post(
    "/vector_store/test_connection",
    tags=["vector store management"],  # mutable-ok: FastAPI route metadata
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata
    response_model=VectorStoreTestConnectionResponse,
)
@router.post(
    "/v1/vector_store/test_connection",
    tags=["vector store management"],  # mutable-ok: FastAPI route metadata
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata
    response_model=VectorStoreTestConnectionResponse,
)
async def vector_store_test_connection(
    data: VectorStoreTestConnectionRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI dependency injection
) -> VectorStoreTestConnectionResponse:
    """
    Run the provider's connection checklist for a saved store or an unsaved configuration.

    Each check reports pass, warn, fail, or skip with a message that names the fix. Proxy admins only.

    Example request (unsaved configuration):
    ```json
    {"custom_llm_provider": "mongodb", "vector_store_id": "policy_index",
     "litellm_params": {"api_base": "http://127.0.0.1:8080", "api_key": "...", "mongodb_database": "knowledge",
                        "mongodb_collection": "policies", "litellm_embedding_model": "text-embedding-3-small"}}
    ```
    Example request (saved store, keep the saved secret): `{"vector_store_id": "policy_index"}`
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")
    _assert_proxy_admin(user_api_key_dict, "test")
    provider, litellm_params, vector_store_id = await _resolve_connection_target(data, user_api_key_dict)
    config: Final = _provider_config(provider)
    try:
        return await config.atest_connection(
            litellm_params=litellm_params,
            vector_store_id=vector_store_id,
            embedding_executor=_router_embedding_executor(user_api_key_dict),
        )
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001  # a diagnostic endpoint reports failures instead of raising
        verbose_proxy_logger.exception("Vector store test connection failed: %s", error)
        return VectorStoreTestConnectionResponse(
            ok=False,
            supported=True,
            custom_llm_provider=provider,
            summary=f"Test connection failed unexpectedly ({type(error).__name__}); see the proxy logs.",
            checks=[],  # mutable-ok: the TypedDict declares a list field
            details=None,
        )


@router.post(
    "/vector_store/discover",
    tags=["vector store management"],  # mutable-ok: FastAPI route metadata
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata
)
@router.post(
    "/v1/vector_store/discover",
    tags=["vector store management"],  # mutable-ok: FastAPI route metadata
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route metadata
)
async def discover_vector_store_resources(
    data: VectorStoreDiscoverRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI dependency injection
) -> Mapping[str, object]:
    """
    List databases, collections, indexes, or suggested fields for a provider so the dashboard can offer
    dropdowns instead of free-text inputs. Proxy admins only.

    Example: `{"custom_llm_provider": "mongodb", "kind": "collections",
               "litellm_params": {"api_base": "http://127.0.0.1:8080", "api_key": "..."},
               "options": {"mongodb_database": "knowledge"}}`
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")
    _assert_proxy_admin(user_api_key_dict, "discover")
    provider, litellm_params, _ = await _resolve_connection_target(data, user_api_key_dict)
    config: Final = _provider_config(provider)
    try:
        return await config.adiscover(kind=data.kind, litellm_params=litellm_params, options=data.options)
    except HTTPException:
        raise
    except litellm.BadRequestError as error:
        raise HTTPException(status_code=400, detail=str(error.message))
    except litellm.AuthenticationError as error:
        raise HTTPException(status_code=401, detail=str(error.message))
    except Exception as error:  # noqa: BLE001  # surface provider failures as a clean 502
        verbose_proxy_logger.exception("Vector store discovery failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)[:500])
