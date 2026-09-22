"""Test Connection checklist and discovery for MongoDB vector stores, run from LiteLLM against the sidecar.

LiteLLM contributes the checks only it can perform (sidecar reachability and auth, the embedding
model's vector size, hybrid prerequisites) and merges the sidecar's own MongoDB checklist.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

import httpx

from litellm.exceptions import BadRequestError
from litellm.llms.base_llm.vector_store.transformation import VectorStoreEmbeddingExecutor
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.llms.mongodb.vector_stores.transformation import (
    DIMENSION_PROBE_TEXT,
    MongoDBVectorStoreConfig,
    MongoDBVectorStoreParams,
    embedding_vector,
    sidecar_error_message,
    validated_params,
    validated_test_connection_params,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.vector_stores import CheckStatus, VectorStoreConnectionCheck, VectorStoreTestConnectionResponse

PROBE_TIMEOUT_SECONDS: Final = 15.0
DISCOVERY_KINDS: Final = frozenset({"databases", "collections", "indexes", "fields"})
_EMPTY: Final = MappingProxyType({})
_STATUSES: Final = MappingProxyType({"pass": "pass", "warn": "warn", "fail": "fail", "skip": "skip"})
_NAMESPACE_CHECKS: Final = frozenset(
    {"mongodb_collection", "mongodb_sample_document", "mongodb_index", "mongodb_index_definition", "mongodb_dimensions"}
)
_CHOOSE_NAMESPACE_MESSAGE: Final = "Choose a database and collection to check the index."
_CHOOSE_EMBEDDING_MODEL_MESSAGE: Final = "Choose an embedding model to check its output dimensions."


def as_status(value: object) -> CheckStatus:
    resolved: Final = _STATUSES.get(str(value), "warn")
    return resolved  # pyright: ignore[reportReturnType]  # the mapping's values are exactly the literal set


def check(
    name: str, status: CheckStatus, message: str, details: Mapping[str, object] | None = None
) -> VectorStoreConnectionCheck:
    return VectorStoreConnectionCheck(
        check=name,
        status=status,
        message=message,
        details=dict(details) if details is not None else None,  # mutable-ok: JSON response
    )


def _resolve(
    config: MongoDBVectorStoreConfig, litellm_params: Mapping[str, object], is_saved_store: bool
) -> tuple[str, dict[str, object], MongoDBVectorStoreParams]:  # mutable-ok: writable HTTP headers
    """A saved store keeps the pre-relaxation strict rule (missing database/collection fails the whole
    check) since it is already meant to be complete; an unsaved configuration may still be missing them
    while the admin is filling in the add dialog, so those checks report skip instead."""
    generic: Final = GenericLiteLLMParams.model_validate(dict(litellm_params))  # mutable-ok: pydantic input copy
    headers: Final = config.validate_environment(headers=_EMPTY, litellm_params=generic)
    api_base: Final = config.get_complete_url(api_base=generic.api_base, litellm_params=litellm_params)
    validate: Final = validated_params if is_saved_store else validated_test_connection_params
    return api_base, headers, validate(litellm_params)


async def _capabilities(
    client: AsyncHTTPHandler,
    api_base: str,
    headers: dict[str, object],  # mutable-ok: httpx headers
) -> tuple[VectorStoreConnectionCheck, Mapping[str, object] | None]:
    """Authenticate against the sidecar and learn what it supports. None means a v0.1 (search-only) sidecar."""
    try:
        response: Final = await client.get(
            f"{api_base}/v1/capabilities", headers=headers, timeout=PROBE_TIMEOUT_SECONDS
        )
    except httpx.HTTPError as error:
        return check("sidecar_auth", "fail", f"Could not reach the sidecar: {type(error).__name__}."), None
    if response.status_code == 401:
        return check("sidecar_auth", "fail", "The sidecar rejected the API key. Check MONGODB_SIDECAR_API_KEY."), None
    if response.status_code == 404:
        return (
            check(
                "sidecar_auth",
                "warn",
                "Connected to a v0.1 sidecar (search only). Upgrade to litellm-mongodb v0.2 for create, ingest, "
                "filters, hybrid search, and connection checks.",
            ),
            None,
        )
    if response.status_code != 200:
        return check(
            "sidecar_auth", "fail", f"Sidecar returned {response.status_code}: {sidecar_error_message(response)}"
        ), None
    payload: Final = response.json()
    if not isinstance(payload, Mapping):
        return check("sidecar_auth", "fail", "The sidecar returned an unexpected capabilities payload."), None
    raw_mongodb: Final = payload.get("mongodb")
    mongodb: Final = raw_mongodb if isinstance(raw_mongodb, Mapping) else _EMPTY
    return (
        check(
            "sidecar_auth",
            "pass",
            f"Authenticated with sidecar v{payload.get('version', '?')} "
            f"(MongoDB {mongodb.get('server_version') or 'version unknown'}).",
            {  # mutable-ok: JSON response
                "sidecar_version": payload.get("version"),
                "features": payload.get("features"),
                "mongodb": mongodb,
            },
        ),
        payload,
    )


async def run_test_connection(
    config: MongoDBVectorStoreConfig,
    litellm_params: Mapping[str, object],
    vector_store_id: str | None,
    embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    *,
    is_saved_store: bool = False,
) -> VectorStoreTestConnectionResponse:
    checks: list[VectorStoreConnectionCheck] = []  # mutable-ok: accumulated checklist
    details: dict[str, object] = {}  # mutable-ok: JSON response
    try:
        api_base, headers, params = _resolve(config, litellm_params, is_saved_store)
    except BadRequestError as error:
        return _finish("mongodb", (check("configuration", "fail", str(error.message)),), details)
    client: Final = get_async_httpx_client(llm_provider=LlmProviders.MONGODB)

    try:
        readiness: Final = await client.get(f"{api_base}/health/readiness", timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        checks.append(
            check(
                "sidecar_reachable",
                "fail",
                f"Cannot reach the sidecar at {api_base} ({type(error).__name__}). Check the URL, that the "
                "sidecar container is running, and that LiteLLM can reach it (HTTPS, or HTTP on loopback).",
            )
        )
        return _finish("mongodb", checks, details)
    if readiness.status_code == 200:
        checks.append(check("sidecar_reachable", "pass", f"Sidecar at {api_base} is ready."))
    elif readiness.status_code == 503:
        checks.append(
            check(
                "sidecar_reachable",
                "warn",
                "The sidecar is running but reports MongoDB as unavailable. Check MONGODB_CONNECTION_STRING, "
                "the Atlas IP access list, and the sidecar logs.",
            )
        )
    else:
        checks.append(
            check("sidecar_reachable", "fail", f"Unexpected readiness status {readiness.status_code} from {api_base}.")
        )

    auth_check, capabilities = await _capabilities(client, api_base, headers)
    checks.append(auth_check)
    if capabilities is not None:
        details["sidecar"] = dict(capabilities)  # mutable-ok: JSON response

    embedding_dimensions: int | None = None
    if params.litellm_embedding_model:
        try:
            probe: Final = await (embedding_executor or config.embedding_executor).aembed(
                params.require_embedding_model(), DIMENSION_PROBE_TEXT, params.litellm_embedding_config or _EMPTY
            )
            embedding_dimensions = len(embedding_vector(probe))
            checks.append(
                check(
                    "embedding_model",
                    "pass",
                    f"Embedding model '{params.litellm_embedding_model}' returns "
                    f"{embedding_dimensions}-dimensional vectors.",
                    {  # mutable-ok: JSON response
                        "model": params.litellm_embedding_model,
                        "dimensions": embedding_dimensions,
                    },  # mutable-ok: JSON response
                )
            )
        except Exception as error:  # noqa: BLE001  # any provider failure is a diagnostic result, not a crash
            checks.append(
                check(
                    "embedding_model",
                    "fail",
                    f"Embedding model '{params.litellm_embedding_model}' failed: {str(error)[:300]}",
                )
            )
    elif is_saved_store:
        try:
            params.require_embedding_model()
        except BadRequestError as error:
            checks.append(check("embedding_model", "fail", str(error.message)))
    else:
        checks.append(check("embedding_model", "skip", _CHOOSE_EMBEDDING_MODEL_MESSAGE))
    details["embedding_dimensions"] = embedding_dimensions

    if auth_check.get("status") == "pass" and capabilities is not None:
        namespace_chosen: Final = bool(params.mongodb_database) and bool(params.mongodb_collection)
        body: dict[str, object] = {  # mutable-ok: JSON transport
            "index_name": vector_store_id,
            "mongodb_embedding_field": params.embedding_field,
            "mongodb_text_field": params.text_field,
            "expected_dimensions": embedding_dimensions,
            "timeout_ms": int(PROBE_TIMEOUT_SECONDS * 1000),
        }
        if params.mongodb_database:
            body["mongodb_database"] = params.mongodb_database
        if params.mongodb_collection:
            body["mongodb_collection"] = params.mongodb_collection
        try:
            report: Final = await client.post(
                f"{api_base}/v1/test_connection", headers=headers, json=body, timeout=PROBE_TIMEOUT_SECONDS
            )
            payload: Final = report.json()
            rows: Final = payload.get("checks", ()) if isinstance(payload, Mapping) else ()
            checks.extend(_sidecar_check(row, namespace_chosen) for row in rows if isinstance(row, Mapping))
            if isinstance(payload, Mapping):
                details["mongodb"] = {  # mutable-ok: JSON response
                    key: payload.get(key)  # mutable-ok: JSON response
                    for key in ("server_version", "index_dimensions", "document_count")
                }
        except httpx.HTTPStatusError as error:
            checks.append(check("mongodb_checklist", "fail", sidecar_error_message(error.response)))
        except (httpx.HTTPError, ValueError) as error:
            checks.append(check("mongodb_checklist", "fail", f"The sidecar checklist failed: {type(error).__name__}."))

        raw_features: Final = capabilities.get("features")
        features: Final = raw_features if isinstance(raw_features, Mapping) else _EMPTY
        if params.mongodb_hybrid_search:
            if not features.get("hybrid"):
                checks.append(
                    check(
                        "hybrid_search",
                        "fail",
                        "Hybrid search is enabled for this store but the cluster does not support $rankFusion "
                        "(MongoDB 8.1+ required). Disable mongodb_hybrid_search or upgrade the cluster.",
                    )
                )
            elif not params.mongodb_text_index:
                checks.append(
                    check(
                        "hybrid_search",
                        "fail",
                        "Hybrid search is enabled but mongodb_text_index is not set. Name the Atlas Search text "
                        "index (it is created with the vector index when the store is created through LiteLLM).",
                    )
                )
            else:
                checks.append(
                    check(
                        "hybrid_search",
                        "pass",
                        f"Hybrid search available via text index '{params.mongodb_text_index}'.",
                    )
                )
    return _finish("mongodb", checks, details)


def _sidecar_check(row: Mapping[str, object], namespace_chosen: bool) -> VectorStoreConnectionCheck:
    """One row of the sidecar's own checklist, keeping its verdict as-is except for the checks that need a
    database and collection: those are reported as our own skip rather than whatever the sidecar sent, since
    an admin who has not chosen a namespace yet has not failed anything."""
    name: Final = str(row.get("check"))
    if not namespace_chosen and name in _NAMESPACE_CHECKS:
        return check(name, "skip", _CHOOSE_NAMESPACE_MESSAGE)
    details: Final = row.get("details")
    return check(
        name,
        as_status(row.get("status")),
        str(row.get("message")),
        details if isinstance(details, Mapping) else None,
    )


def _summary(
    failures: Sequence[VectorStoreConnectionCheck],
    warnings: Sequence[VectorStoreConnectionCheck],
    skips: Sequence[VectorStoreConnectionCheck],
) -> str:
    if failures:
        return str(failures[0].get("message"))
    if warnings:
        return f"Connected with {len(warnings)} warning(s): {warnings[0].get('message')}"
    if skips:
        return "Choose a database and collection to finish the checks."
    return "All checks passed."


def _finish(
    provider: str, checks: Sequence[VectorStoreConnectionCheck], details: Mapping[str, object]
) -> VectorStoreTestConnectionResponse:
    failures: Final = tuple(row for row in checks if row.get("status") == "fail")
    warnings: Final = tuple(row for row in checks if row.get("status") == "warn")
    skips: Final = tuple(row for row in checks if row.get("status") == "skip")
    summary: Final = _summary(failures, warnings, skips)
    return VectorStoreTestConnectionResponse(
        ok=not failures,
        supported=True,
        custom_llm_provider=provider,
        summary=summary,
        checks=list(checks),  # mutable-ok: the TypedDict declares a list field
        details=dict(details) if details else None,  # mutable-ok: JSON response
    )


async def run_discovery(
    config: MongoDBVectorStoreConfig, kind: str, litellm_params: Mapping[str, object], options: Mapping[str, object]
) -> Mapping[str, object]:
    if kind not in DISCOVERY_KINDS:
        raise BadRequestError(
            message=f"Unknown discovery kind {kind!r}; expected one of {', '.join(sorted(DISCOVERY_KINDS))}.",
            model=None,
            llm_provider="mongodb",
        )
    generic: Final = GenericLiteLLMParams.model_validate(dict(litellm_params))  # mutable-ok: pydantic input copy
    headers: Final = config.validate_environment(headers=_EMPTY, litellm_params=generic)
    api_base: Final = config.get_complete_url(api_base=generic.api_base, litellm_params=litellm_params)
    query: dict[str, str] = {}  # mutable-ok: httpx query params
    database: Final = options.get("mongodb_database", litellm_params.get("mongodb_database"))
    collection: Final = options.get("mongodb_collection", litellm_params.get("mongodb_collection"))
    if kind != "databases":
        if not isinstance(database, str) or not database:
            raise BadRequestError(
                message="mongodb_database is required for this discovery.", model=None, llm_provider="mongodb"
            )
        query["mongodb_database"] = database
    if kind in ("indexes", "fields"):
        if not isinstance(collection, str) or not collection:
            raise BadRequestError(
                message="mongodb_collection is required for this discovery.", model=None, llm_provider="mongodb"
            )
        query["mongodb_collection"] = collection
    if kind == "fields" and options.get("sample_size") is not None:
        query["sample_size"] = str(options["sample_size"])
    client: Final = get_async_httpx_client(llm_provider=LlmProviders.MONGODB)
    url: Final = f"{api_base}/v1/discovery/{quote(kind, safe='')}"
    response: Final = await client.get(url, params=query, headers=headers, timeout=PROBE_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise config.get_error_class(sidecar_error_message(response), response.status_code, response.headers)
    payload: Final = response.json()
    if not isinstance(payload, Mapping):
        raise BadRequestError(
            message="The sidecar returned an unexpected discovery payload.", model=None, llm_provider="mongodb"
        )
    return MappingProxyType(dict(payload))  # mutable-ok: JSON response wrapped read-only
