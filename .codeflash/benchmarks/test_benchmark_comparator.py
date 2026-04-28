"""Benchmark comparator type dispatch performance.

Exercises the fast-path frozenset lookup vs isinstance MRO traversal
across realistic return value shapes: primitives, nested containers,
and mixed-type structures typical of real optimization verification.
"""

from __future__ import annotations

from collections import OrderedDict
from decimal import Decimal

from codeflash.verification.comparator import comparator

# --- Test data: realistic return value shapes ---

# 1. Flat primitives (int, bool, None, str, float, bytes) — the fast-path sweet spot
_PRIMITIVES_A = [
    42,
    True,
    None,
    3.14,
    "hello",
    b"bytes",
    0,
    False,
    "",
    1.0,
    -1,
    None,
    True,
    99,
    "world",
    b"\x00\x01",
    2**31,
    0.0,
    False,
    None,
]
_PRIMITIVES_B = list(_PRIMITIVES_A)

# 2. Nested dict of lists (common return value shape: API responses, parsed configs)
_NESTED_DICT_A = {
    "users": [{"id": i, "name": f"user_{i}", "active": i % 2 == 0, "score": i * 1.5} for i in range(50)],
    "metadata": {"total": 50, "page": 1, "has_next": True},
    "tags": [f"tag_{i}" for i in range(20)],
    "config": {"timeout": 30, "retries": 3, "debug": False, "threshold": Decimal("0.95")},
}
_NESTED_DICT_B = {
    "users": [{"id": i, "name": f"user_{i}", "active": i % 2 == 0, "score": i * 1.5} for i in range(50)],
    "metadata": {"total": 50, "page": 1, "has_next": True},
    "tags": [f"tag_{i}" for i in range(20)],
    "config": {"timeout": 30, "retries": 3, "debug": False, "threshold": Decimal("0.95")},
}

# 3. List of tuples (common: database rows, CSV data)
_ROWS_A = [(i, f"row_{i}", i * 0.1, i % 3 == 0, None if i % 5 == 0 else i) for i in range(200)]
_ROWS_B = [(i, f"row_{i}", i * 0.1, i % 3 == 0, None if i % 5 == 0 else i) for i in range(200)]


# 4. Deeply nested structure (worst case for recursive comparator)
def _make_deep(depth: int) -> dict:
    if depth == 0:
        return {"leaf": True, "value": 42, "items": [1, 2, 3], "label": "end"}
    return {"level": depth, "child": _make_deep(depth - 1), "siblings": list(range(depth))}


_DEEP_A = _make_deep(15)
_DEEP_B = _make_deep(15)

# 5. Mixed identity types (frozenset, range, slice, OrderedDict, bytes, complex)
_IDENTITY_TYPES_A = [
    frozenset({1, 2, 3}),
    range(100),
    complex(1, 2),
    Decimal("3.14"),
    OrderedDict(a=1, b=2),
    b"binary",
    bytearray(b"mutable"),
    memoryview(b"view"),
    type(None),
    True,
    42,
    None,
] * 10
_IDENTITY_TYPES_B = list(_IDENTITY_TYPES_A)


def _compare_all_primitives() -> None:
    for a, b in zip(_PRIMITIVES_A, _PRIMITIVES_B):
        comparator(a, b)


def _compare_nested_dict() -> None:
    comparator(_NESTED_DICT_A, _NESTED_DICT_B)


def _compare_rows() -> None:
    comparator(_ROWS_A, _ROWS_B)


def _compare_deep() -> None:
    comparator(_DEEP_A, _DEEP_B)


def _compare_identity_types() -> None:
    for a, b in zip(_IDENTITY_TYPES_A, _IDENTITY_TYPES_B):
        comparator(a, b)


def test_benchmark_comparator_primitives(benchmark) -> None:
    """20 flat primitive comparisons (int, bool, None, str, float, bytes)."""
    benchmark(_compare_all_primitives)


def test_benchmark_comparator_nested_dict(benchmark) -> None:
    """Nested dict with 50-element user list, metadata, tags, config."""
    benchmark(_compare_nested_dict)


def test_benchmark_comparator_rows(benchmark) -> None:
    """200 tuples of (int, str, float, bool, Optional[int])."""
    benchmark(_compare_rows)


def test_benchmark_comparator_deep(benchmark) -> None:
    """15-level deep nested dict structure."""
    benchmark(_compare_deep)


def test_benchmark_comparator_identity_types(benchmark) -> None:
    """120 frozenset/range/complex/Decimal/OrderedDict/bytes comparisons."""
    benchmark(_compare_identity_types)
