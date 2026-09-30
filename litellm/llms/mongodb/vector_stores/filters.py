"""Translate vector store search filters into the MQL pre-filter MongoDB Vector Search accepts.

Two input shapes are supported:

* The OpenAI vector store filter schema: comparison objects ``{"type": "eq", "key": "k", "value": v}``
  (types eq, ne, gt, gte, lt, lte, in, nin) and compound objects ``{"type": "and"|"or", "filters": [...]}``.
* Raw MQL, recognised by the absence of a ``type`` key. It is validated against the same operator
  allowlist the sidecar enforces so mistakes fail here with a precise message.

No MongoDB driver is involved; this is pure data transformation.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from litellm.exceptions import BadRequestError

COMPARISON_TYPES: Final = MappingProxyType(
    {"eq": "$eq", "ne": "$ne", "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte", "in": "$in", "nin": "$nin"}
)
COMPOUND_TYPES: Final = MappingProxyType({"and": "$and", "or": "$or"})
MQL_OPERATORS: Final = frozenset({*COMPARISON_TYPES.values(), *COMPOUND_TYPES.values(), "$not"})
MAX_DEPTH: Final = 6
MAX_LIST_VALUES: Final = 256


def _error(message: str) -> BadRequestError:
    return BadRequestError(message=f"Invalid vector store filter: {message}", model=None, llm_provider="mongodb")


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _check_key(key: object) -> str:
    if not isinstance(key, str) or not key.strip():
        raise _error("filter keys must be nonblank strings")
    if key.startswith("$"):
        raise _error(f"field name {key!r} must not start with $")
    return key


def _check_list(operator: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(f"{operator} expects a list of values")
    if len(value) > MAX_LIST_VALUES:
        raise _error(f"{operator} accepts at most {MAX_LIST_VALUES} values")
    if not all(_is_scalar(item) for item in value):
        raise _error(f"{operator} values must be strings, numbers, booleans, or null")
    return tuple(value)


def _from_openai(node: object, depth: int) -> dict[str, object]:  # mutable-ok: MQL is JSON transport
    if depth > MAX_DEPTH:
        raise _error(f"filters may nest at most {MAX_DEPTH} levels deep")
    if not isinstance(node, Mapping):
        raise _error("each filter must be an object")
    kind: Final = node.get("type")
    if kind in COMPOUND_TYPES:
        children: Final = node.get("filters")
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)) or not children:
            raise _error(f"{kind} filters need a non-empty 'filters' list")
        return {  # mutable-ok: MQL is JSON transport
            COMPOUND_TYPES[str(kind)]: [_from_openai(child, depth + 1) for child in children]  # mutable-ok: JSON list
        }
    if kind in COMPARISON_TYPES:
        key: Final = _check_key(node.get("key"))
        operator: Final = COMPARISON_TYPES[str(kind)]
        value: Final = node.get("value")
        if operator in ("$in", "$nin"):
            return {key: {operator: list(_check_list(operator, value))}}  # mutable-ok: MQL is JSON transport
        if not _is_scalar(value):
            raise _error(f"{kind} on {key!r} expects a string, number, boolean, or null value")
        return {key: {operator: value}}  # mutable-ok: MQL is JSON transport
    raise _error(f"unknown filter type {kind!r}; expected one of {', '.join((*COMPARISON_TYPES, *COMPOUND_TYPES))}")


def _check_mql(node: object, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise _error(f"filters may nest at most {MAX_DEPTH} levels deep")
    if not isinstance(node, Mapping) or not node:
        raise _error("an MQL filter must be a non-empty object")
    for key, value in node.items():
        if key in ("$and", "$or"):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
                raise _error(f"{key} expects a non-empty list of filter objects")
            for child in value:
                _check_mql(child, depth + 1)
        elif key == "$not":
            _check_mql(value, depth + 1)
        elif isinstance(key, str) and key.startswith("$"):
            raise _error(f"operator {key} is not allowed; allowed: {', '.join(sorted(MQL_OPERATORS))}")
        else:
            _check_key(key)
            if isinstance(value, Mapping):
                if not value:
                    raise _error(f"the condition for {key!r} must not be empty")
                for operator, operand in value.items():
                    if operator not in COMPARISON_TYPES.values():
                        raise _error(f"operator {operator!r} on {key!r} is not allowed")
                    if operator in ("$in", "$nin"):
                        _check_list(str(operator), operand)
                    elif not _is_scalar(operand):
                        raise _error(f"{operator} on {key!r} expects a scalar value")
            elif not _is_scalar(value):
                raise _error(f"the value for {key!r} must be a scalar or an operator object")


def translate_filters(filters: Mapping[str, object] | None) -> dict[str, object] | None:  # mutable-ok: JSON transport
    """Return an MQL filter for the sidecar, or None when no filter was given."""
    if filters is None:
        return None
    if not isinstance(filters, Mapping) or not filters:
        raise _error("filters must be a non-empty object")
    if "type" in filters:
        return _from_openai(filters, 1)
    _check_mql(filters, 1)
    return dict(filters)  # mutable-ok: MQL is JSON transport
