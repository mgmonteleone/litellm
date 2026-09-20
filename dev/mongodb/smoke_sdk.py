"""SDK-level smoke test: LiteLLM -> sidecar (127.0.0.1:8080) -> MongoDB (Atlas Local).

Runs create -> ingest -> search through the public LiteLLM SDK. Embeddings go to the
upstream LiteLLM router named by UPSTREAM_LITELLM_API_BASE / UPSTREAM_LITELLM_API_KEY
in .env. Set FAKE_EMBEDDINGS=1 (default when that key is empty) to replace the embedding
model with a deterministic hash-based vector so the chain runs without any key.

    uv run --no-sync python dev/mongodb/smoke_sdk.py
"""

import asyncio
import hashlib
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final
from unittest.mock import patch

from dotenv import dotenv_values

import litellm
from litellm.types.utils import EmbeddingResponse

ROOT: Final = Path(__file__).resolve().parents[2]
ENV: Final = {**dotenv_values(ROOT / ".env"), **os.environ}
SIDECAR: Final = "http://127.0.0.1:8080"
DIMENSIONS: Final = 64
EMBEDDING_MODEL: Final = ENV.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_BASE: Final = ENV.get("EMBEDDING_API_BASE", ENV.get("UPSTREAM_LITELLM_API_BASE", ""))
EMBEDDING_KEY: Final = ENV.get("EMBEDDING_API_KEY", ENV.get("UPSTREAM_LITELLM_API_KEY", ""))
# One index per embedding model: an index is bound to the vector size of the model that fills it.
INDEX: Final = (
    "litellm_smoke_fake"
    if not EMBEDDING_KEY
    else "litellm_smoke_" + "".join(c if c.isalnum() else "_" for c in EMBEDDING_MODEL.split("/")[-1])
)
STORE: Final = {
    "api_base": SIDECAR,
    "api_key": ENV.get("MONGODB_SIDECAR_API_KEY", ""),
    "mongodb_database": "litellm_smoke",
    "mongodb_collection": "policies",
    "litellm_embedding_model": EMBEDDING_MODEL,
    # Forwarded as kwargs to litellm.embedding, so the call lands on the configured embedding server.
    "litellm_embedding_config": {"api_base": EMBEDDING_BASE, "api_key": EMBEDDING_KEY},
    "mongodb_filter_fields": ["metadata.department"],
}
DOCUMENTS: Final = {
    "travel.md": ("hr", "Employees fly economy on trips under six hours. Business class needs VP approval."),
    "expenses.md": ("finance", "Submit expense reports within thirty days. Receipts are required above twenty-five dollars."),
    "remote.md": ("hr", "Remote work is allowed three days per week. Core hours are ten to three in your local time zone."),
}


def fake_vector(text: str) -> list[float]:
    """Bag-of-words hashing embedding: deterministic, and similar texts share dimensions."""
    vector: Final = [0.0] * DIMENSIONS
    for token in text.lower().split():
        bucket: Final = int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIMENSIONS
        vector[bucket] += 1.0
    norm: Final = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


async def fake_aembedding(model: str, input: list[str] | str, **_: object) -> EmbeddingResponse:
    texts: Final = [input] if isinstance(input, str) else list(input)
    return EmbeddingResponse(data=[{"embedding": fake_vector(text), "index": i} for i, text in enumerate(texts)])


class FakeExecutor:
    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return EmbeddingResponse(data=[{"embedding": fake_vector(query)}])

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.embed(model, query, configuration)


async def run(fake: bool) -> None:
    extra: dict[str, object] = {"mongodb_dimensions": DIMENSIONS} if fake else {}
    print(f"[1/4] create index {INDEX} via litellm.vector_stores.acreate (fake_embeddings={fake})")
    created: Final = await litellm.vector_stores.acreate(name=INDEX, custom_llm_provider="mongodb", **STORE, **extra)
    print("      ->", created["status"], created["metadata"])

    print("[2/4] ingest documents via litellm.aingest")
    for filename, (department, text) in DOCUMENTS.items():
        result: Final = await litellm.aingest(
            ingest_options={
                "vector_store": {
                    "custom_llm_provider": "mongodb",
                    "vector_store_id": INDEX,
                    "custom_metadata": {"department": department},
                    **STORE,
                    **extra,
                },
                "chunking_strategy": {"type": "static", "chunk_size": 120, "chunk_overlap": 20},
            },
            file_data=(filename, text.encode(), "text/markdown"),
        )
        print(f"      -> {filename}: {result['status']} file_id={result.get('file_id')} error={result.get('error')}")
        if result["status"] != "completed":
            sys.exit(1)

    print("[3/4] wait for mongot to index the new chunks, then search")
    query: Final = "how long do I have to submit expenses?"
    for attempt in range(60):
        response: Final = await litellm.vector_stores.asearch(
            vector_store_id=INDEX,
            query=query,
            custom_llm_provider="mongodb",
            max_num_results=3,
            _direct_vector_store_embedding_executor=FakeExecutor() if fake else None,
            **STORE,
        )
        if response["data"]:
            break
        time.sleep(1)
    else:
        print("      -> no results after 60s"); sys.exit(1)
    for hit in response["data"]:
        print(f"      -> score={hit['score']:.3f} file={hit['filename']} text={hit['content'][0]['text'][:70]!r}")
    assert all(hit["filename"] and hit["content"][0]["text"] for hit in response["data"])

    print("[3b] filtered search (OpenAI filter schema -> MQL -> $vectorSearch.filter)")
    filtered: Final = await litellm.vector_stores.asearch(
        vector_store_id=INDEX,
        query=query,
        custom_llm_provider="mongodb",
        max_num_results=5,
        filters={"type": "eq", "key": "metadata.department", "value": "finance"},
        _direct_vector_store_embedding_executor=FakeExecutor() if fake else None,
        **STORE,
    )
    print(f"      -> {len(filtered['data'])} hit(s); attributes={[h['attributes'] for h in filtered['data']]}")
    assert filtered["data"] and all(h["attributes"]["department"] == "finance" for h in filtered["data"])

    print("[3c] score threshold 0.99 should return nothing")
    strict: Final = await litellm.vector_stores.asearch(
        vector_store_id=INDEX,
        query=query,
        custom_llm_provider="mongodb",
        ranking_options={"score_threshold": 0.99},
        _direct_vector_store_embedding_executor=FakeExecutor() if fake else None,
        **STORE,
    )
    print(f"      -> {len(strict['data'])} hit(s)")
    assert strict["data"] == [] or all(h["score"] >= 0.99 for h in strict["data"])
    if fake:
        print("[4/4] OK: chain works (ranking not asserted with the hash-based stand-in embedding)")
    else:
        assert response["data"][0]["filename"] == "expenses.md", response["data"][0]
        print("[4/4] OK: top hit came from expenses.md")


if __name__ == "__main__":
    use_fake: Final = ENV.get("FAKE_EMBEDDINGS", "1" if not EMBEDDING_KEY else "0") == "1"
    if not STORE["api_key"]:
        sys.exit("MONGODB_SIDECAR_API_KEY missing from .env")
    if use_fake:
        with patch("litellm.aembedding", fake_aembedding):
            asyncio.run(run(fake=True))
    else:
        if not EMBEDDING_BASE:
            sys.exit("EMBEDDING_API_BASE missing from .env")
        asyncio.run(run(fake=False))
