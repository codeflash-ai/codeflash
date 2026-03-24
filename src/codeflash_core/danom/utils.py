from __future__ import annotations

from functools import reduce
from operator import not_
from typing import TYPE_CHECKING, TypeVar

import attrs

T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Composable = Callable[[T_co], T_co | U_co]
    Filterable = Callable[[T_co], bool]


@attrs.define(frozen=True, hash=True, eq=True)
class _Compose:
    fns: Sequence[Composable]

    def __call__(self, initial: T_co) -> T_co | U_co:
        return reduce(_apply, self.fns, initial)


def _apply(value: T_co, fn: Composable) -> T_co | U_co:
    return fn(value)


def compose(*fns: Composable) -> Composable:
    return _Compose(fns)


@attrs.define(frozen=True, hash=True, eq=True)
class _AllOf:
    fns: Sequence[Filterable]

    def __call__(self, item: T_co) -> bool:
        return all(fn(item) for fn in self.fns)


def all_of(*fns: Filterable) -> Filterable:
    return _AllOf(fns)


@attrs.define(frozen=True, hash=True, eq=True)
class _AnyOf:
    fns: Sequence[Filterable]

    def __call__(self, item: T_co) -> bool:
        return any(fn(item) for fn in self.fns)


def any_of(*fns: Filterable) -> Filterable:
    return _AnyOf(fns)


def none_of(*fns: Filterable) -> Filterable:
    return compose(_AnyOf(fns), not_)


def identity(x: T_co) -> T_co:
    return x


def invert(func: Filterable) -> Filterable:
    return compose(func, not_)
