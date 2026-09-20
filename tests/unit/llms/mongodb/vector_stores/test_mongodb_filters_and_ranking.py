from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.mongodb.vector_stores.filters import translate_filters
from litellm.llms.mongodb.vector_stores.transformation import MongoDBVectorStoreConfig
from litellm.types.utils import EmbeddingResponse

BASE_PARAMS: Final = {
    "api_base": "https://sidecar.example",
    "api_key": "test-sidecar-key",
    "litellm_embedding_model": "embedding-alias",
    "mongodb_database": "knowledge",
    "mongodb_collection": "policies",
}


class RecordingEmbeddingExecutor:
    def __init__(self) -> None:
        self.call: Final = MagicMock(return_value=EmbeddingResponse(data=[{"embedding": [0.1, 0.2, 0.3]}]))

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)


@pytest.mark.parametrize(
    "filters,expected",
    [
        ({"type": "eq", "key": "metadata.department", "value": "hr"}, {"metadata.department": {"$eq": "hr"}}),
        ({"type": "in", "key": "metadata.year", "value": [2025, 2026]}, {"metadata.year": {"$in": [2025, 2026]}}),
        (
            {
                "type": "and",
                "filters": [
                    {"type": "gte", "key": "metadata.year", "value": 2024},
                    {
                        "type": "or",
                        "filters": [
                            {"type": "ne", "key": "a", "value": None},
                            {"type": "lt", "key": "b", "value": 1.5},
                        ],
                    },
                ],
            },
            {"$and": [{"metadata.year": {"$gte": 2024}}, {"$or": [{"a": {"$ne": None}}, {"b": {"$lt": 1.5}}]}]},
        ),
        ({"metadata.department": {"$in": ["hr"]}}, {"metadata.department": {"$in": ["hr"]}}),
        ({"$and": [{"a": 1}, {"$not": {"b": {"$gt": 2}}}]}, {"$and": [{"a": 1}, {"$not": {"b": {"$gt": 2}}}]}),
        (None, None),
    ],
)
def test_openai_and_raw_mql_filters_translate(filters: Mapping[str, object] | None, expected: object) -> None:
    assert translate_filters(filters) == expected


@pytest.mark.parametrize(
    "filters,fragment",
    [
        ({}, "non-empty"),
        ({"type": "regex", "key": "a", "value": "b"}, "unknown filter type"),
        ({"type": "eq", "key": "$where", "value": 1}, r"must not start with \$"),
        ({"type": "eq", "key": "a", "value": {"nested": 1}}, "expects a string"),
        ({"type": "in", "key": "a", "value": "x"}, "expects a list"),
        ({"type": "and", "filters": []}, "non-empty 'filters'"),
        ({"a": {"$regex": ".*"}}, "not allowed"),
        ({"$where": "1"}, "not allowed"),
        ({"a": [1, 2]}, "scalar"),
    ],
)
def test_invalid_filters_fail_with_the_reason(filters: Mapping[str, object], fragment: str) -> None:
    with pytest.raises(litellm.BadRequestError, match=fragment):
        translate_filters(filters)


def test_deeply_nested_openai_filters_are_rejected() -> None:
    node: dict[str, object] = {"type": "eq", "key": "a", "value": 1}
    for _ in range(8):
        node = {"type": "and", "filters": [node]}
    with pytest.raises(litellm.BadRequestError, match="nest"):
        translate_filters(node)


def search_body(params: Mapping[str, object], options: Mapping[str, object]) -> dict[str, object]:
    config: Final = MongoDBVectorStoreConfig(RecordingEmbeddingExecutor())
    _, body = config.transform_search_vector_store_request(
        vector_store_id="policy_index",
        query="travel",
        vector_store_search_optional_params=dict(options),
        api_base="https://sidecar.example",
        litellm_logging_obj=MagicMock(),
        litellm_params={**BASE_PARAMS, **params},
    )
    return body


def test_filters_threshold_and_exact_reach_the_sidecar_body() -> None:
    body: Final = search_body(
        {"mongodb_exact_search": True},
        {
            "filters": {"type": "eq", "key": "metadata.department", "value": "hr"},
            "ranking_options": {"score_threshold": 0.42},
        },
    )
    assert body["filter"] == {"metadata.department": {"$eq": "hr"}}
    assert body["score_threshold"] == 0.42
    assert body["exact"] is True
    assert body["include_metadata"] is True
    assert "hybrid" not in body


def test_store_default_threshold_applies_when_request_has_none() -> None:
    assert search_body({"mongodb_score_threshold": 0.7}, {})["score_threshold"] == 0.7
    assert (
        search_body({"mongodb_score_threshold": 0.7}, {"ranking_options": {"score_threshold": 0.2}})["score_threshold"]
        == 0.2
    )
    assert "score_threshold" not in search_body({}, {})


@pytest.mark.parametrize(
    "params,options",
    [
        ({"mongodb_hybrid_search": True, "mongodb_text_index": "policies_text"}, {}),
        ({"mongodb_text_index": "policies_text"}, {"ranking_options": {"ranker": "hybrid"}}),
    ],
)
def test_hybrid_is_enabled_by_store_config_or_ranker(
    params: Mapping[str, object], options: Mapping[str, object]
) -> None:
    body: Final = search_body({**params, "mongodb_hybrid_weights": {"vector": 0.5, "text": 2}}, options)
    assert body["hybrid"] == {"text_index": "policies_text", "vector_weight": 0.5, "text_weight": 2.0}
    assert "score_threshold" not in body


@pytest.mark.parametrize(
    "params,options,fragment",
    [
        ({"mongodb_hybrid_search": True}, {}, "mongodb_text_index"),
        (
            {"mongodb_text_index": "t", "mongodb_hybrid_search": True, "mongodb_hybrid_weights": {"vector": -1}},
            {},
            "non-negative",
        ),
        (
            {
                "mongodb_text_index": "t",
                "mongodb_hybrid_search": True,
                "mongodb_hybrid_weights": {"vector": 0, "text": 0},
            },
            {},
            "positive weight",
        ),
        (
            {"mongodb_text_index": "t"},
            {"ranking_options": {"ranker": "hybrid", "score_threshold": 0.5}},
            "cannot be combined",
        ),
        ({}, {"ranking_options": {"ranker": "bm25"}}, "not supported"),
        ({"mongodb_score_threshold": 1.5}, {}, "between 0 and 1"),
        ({"mongodb_unknown_param": 1}, {}, "Unrecognised"),
    ],
)
def test_invalid_ranking_configuration_is_rejected(
    params: Mapping[str, object], options: Mapping[str, object], fragment: str
) -> None:
    with pytest.raises(litellm.BadRequestError, match=fragment):
        search_body(params, options)


def test_search_response_keeps_attributes_and_drops_score_details() -> None:
    config: Final = MongoDBVectorStoreConfig(RecordingEmbeddingExecutor())
    payload: Final = {
        "object": "vector_store.search_results.page",
        "search_query": "travel",
        "data": [
            {
                "score": 0.9,
                "file_id": "f1",
                "filename": "travel.md",
                "content": [{"type": "text", "text": "Fly economy."}],
                "attributes": {"department": "hr", "chunk_index": 0, "year": 2026},
                "score_details": {"value": 0.03, "details": []},
            }
        ],
    }
    response: Final = config.transform_search_vector_store_response(httpx.Response(200, json=payload), MagicMock())
    assert response["data"][0]["attributes"] == {"department": "hr", "chunk_index": 0, "year": 2026}
    assert "score_details" not in response["data"][0]


def test_supported_openai_params_advertise_filters_and_ranking() -> None:
    config: Final = MongoDBVectorStoreConfig(RecordingEmbeddingExecutor())
    assert config.get_supported_openai_params("mongodb") == ["filters", "max_num_results", "ranking_options"]


def test_timeouts_are_clamped_to_the_sidecar_ceiling() -> None:
    assert search_body({"timeout": 6000}, {})["timeout_ms"] == 600_000
    assert search_body({"timeout": 0.25}, {})["timeout_ms"] == 250
