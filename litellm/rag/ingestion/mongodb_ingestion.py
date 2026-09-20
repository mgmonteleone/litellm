"""
MongoDB Atlas Vector Search ingestion through the LiteLLM MongoDB sidecar.

LiteLLM chunks and embeds locally, then posts text plus vectors to the sidecar,
which owns the MongoDB driver. No MongoDB dependency is imported here.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import quote
from uuid import uuid4

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.llms.mongodb.vector_stores.transformation import (
    MongoDBVectorStoreConfig,
    MongoDBVectorStoreParams,
    config_error,
    validated_params,
)
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

INGEST_BATCH_SIZE: Final = 200
INGEST_TIMEOUT_MS: Final = 60_000
_EMPTY_CONFIG: Final = MappingProxyType({})


class MongoDBRAGIngestion(BaseRAGIngestion):
    """Ingest documents into a MongoDB collection that the sidecar can search."""

    supports_existing_file_id: bool = False

    def _embedding_model(self) -> str:
        configured: Final = self.vector_store_config.get("litellm_embedding_model")
        if isinstance(configured, str) and configured:
            return configured
        fallback: Final = (self.embedding_config or _EMPTY_CONFIG).get("model")
        if isinstance(fallback, str) and fallback:
            return fallback
        raise config_error(
            "An embedding model is required to ingest into MongoDB. Set litellm_embedding_model on the vector "
            "store registration (or ingest_options.embedding.model); the same model is used for every search."
        )

    def _embedding_config(self) -> Mapping[str, object]:
        configured: Final = self.vector_store_config.get("litellm_embedding_config")
        return configured if isinstance(configured, Mapping) else _EMPTY_CONFIG

    async def embed(self, chunks: list[str]) -> list[list[float]] | None:  # mutable-ok: BaseRAGIngestion contract
        if not chunks:
            return None
        model: Final = self._embedding_model()
        verbose_logger.debug("MongoDB ingest: embedding %s chunks with %s", len(chunks), model)
        if self.router is not None:
            response = await self.router.aembedding(model=model, input=chunks, **self._embedding_config())
        else:
            response = await litellm.aembedding(model=model, input=chunks, **self._embedding_config())
        return [item["embedding"] for item in response.data]  # mutable-ok: BaseRAGIngestion.embed returns a list

    def _sidecar(self) -> tuple[str, dict[str, object], MongoDBVectorStoreParams]:  # mutable-ok: writable HTTP headers
        """Resolve and validate the sidecar URL, auth headers, and MongoDB parameters."""
        config: Final = MongoDBVectorStoreConfig()
        litellm_params: Final = GenericLiteLLMParams(**self.vector_store_config)
        headers: Final = config.validate_environment(headers=_EMPTY_CONFIG, litellm_params=litellm_params)
        api_base: Final = config.get_complete_url(
            api_base=litellm_params.api_base,
            litellm_params=dict(litellm_params),  # mutable-ok: get_complete_url takes a plain mapping copy
        )
        params: Final = validated_params(self.vector_store_config)
        return api_base, headers, params

    @staticmethod
    async def _post(
        client: AsyncHTTPHandler,
        url: str,
        headers: dict[str, object],  # mutable-ok: writable HTTP headers
        body: dict[str, object],  # mutable-ok: httpx JSON body
    ) -> httpx.Response:
        try:
            return await client.post(url=url, headers=headers, json=body)
        except httpx.HTTPStatusError as error:
            failed: Final = error.response
            raise MongoDBVectorStoreConfig().get_error_class(
                _error_message(failed), failed.status_code, failed.headers
            ) from None

    async def store(
        self,
        file_content: bytes | None,
        filename: str | None,
        content_type: str | None,
        chunks: list[str],  # mutable-ok: BaseRAGIngestion contract
        embeddings: list[list[float]] | None,  # mutable-ok: BaseRAGIngestion contract
        existing_file_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        if not chunks or not embeddings:
            raise ValueError(
                "No text content could be extracted from the file for embedding. "
                "Possible causes:\n"
                "  1. PDF files require OCR - add 'ocr' config with a vision model\n"
                "  2. Binary files cannot be processed - convert to text first\n"
                "  3. File is empty or contains no extractable text"
            )
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk and embedding counts differ ({len(chunks)} vs {len(embeddings)})")

        api_base, headers, params = self._sidecar()
        configured_index: Final = self.vector_store_config.get("vector_store_id")
        index_name: Final = (
            configured_index
            if isinstance(configured_index, str) and configured_index
            else f"litellm_{uuid4().hex[:12]}"
        )
        dimensions: Final = len(embeddings[0])
        client: Final = get_async_httpx_client(llm_provider=LlmProviders.MONGODB)

        create_body: Final = {  # mutable-ok: JSON transport requires a dict
            "index_name": index_name,
            "mongodb_database": params.require_database(),
            "mongodb_collection": params.require_collection(),
            "mongodb_embedding_field": params.embedding_field,
            "mongodb_text_field": params.text_field,
            "dimensions": params.mongodb_dimensions or dimensions,
            "similarity": params.similarity,
            "filter_fields": params.filter_fields,
            "timeout_ms": INGEST_TIMEOUT_MS,
        }
        created: Final = await self._post(client, f"{api_base}/v1/vector_stores", headers, create_body)
        verbose_logger.info(
            "MongoDB ingest: index %s on %s.%s is %s",
            index_name,
            params.mongodb_database,
            params.mongodb_collection,
            created.json().get("status"),
        )

        file_id: Final = f"file_{uuid4().hex}"
        documents_url: Final = f"{api_base}/v1/vector_stores/{quote(index_name, safe='')}/documents"
        for start in range(0, len(chunks), INGEST_BATCH_SIZE):
            batch = _documents(chunks, embeddings, start, start + INGEST_BATCH_SIZE)
            await self._post(
                client,
                documents_url,
                headers,
                {  # mutable-ok: JSON transport requires a dict
                    "mongodb_database": params.require_database(),
                    "mongodb_collection": params.require_collection(),
                    "mongodb_embedding_field": params.embedding_field,
                    "mongodb_text_field": params.text_field,
                    "file_id": file_id,
                    "filename": filename,
                    "content_type": content_type,
                    "documents": batch,
                    "replace_existing": start == 0,
                    "timeout_ms": INGEST_TIMEOUT_MS,
                },
            )
        verbose_logger.info("MongoDB ingest: stored %s chunks for %s as %s", len(chunks), filename, file_id)
        return index_name, file_id


def _documents(
    chunks: Sequence[str], embeddings: Sequence[Sequence[float]], start: int, stop: int
) -> tuple[dict[str, object], ...]:  # mutable-ok: httpx JSON body
    return tuple(
        {  # mutable-ok: JSON transport requires a dict
            "chunk_index": position,
            "text": chunks[position],
            "embedding": tuple(embeddings[position]),
            "metadata": {},  # mutable-ok: JSON transport requires a dict
        }
        for position in range(start, min(stop, len(chunks)))
    )


def _error_message(response: httpx.Response) -> str:
    try:
        payload: Final = response.json()
    except ValueError:
        return response.text[:500]
    error: Final = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        return str(error["message"])
    return response.text[:500]
