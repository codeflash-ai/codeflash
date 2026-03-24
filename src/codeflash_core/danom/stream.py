from __future__ import annotations

import asyncio
import itertools
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from enum import Enum, auto
from functools import reduce
from typing import TYPE_CHECKING, TypeVar, Union, cast

try:
    from itertools import batched  # type: ignore[attr-defined]
except ImportError:
    from itertools import islice

    def batched(iterable, n, *, strict=False):  # noqa: ANN201
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            if strict and len(batch) != n:
                raise ValueError("batched(): incomplete batch")
            yield batch


import attrs

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    MapFn = Callable[[T], U]
    FilterFn = Callable[[T], bool]
    TapFn = Callable[[T], None]

    AsyncMapFn = Callable[[T], Awaitable[U]]
    AsyncFilterFn = Callable[[T], Awaitable[bool]]
    AsyncTapFn = Callable[[T], Awaitable[None]]

    StreamFn = Union[MapFn, FilterFn, TapFn]
    AsyncStreamFn = Union[AsyncMapFn, AsyncFilterFn, AsyncTapFn]

    PlannedOps = tuple[str, StreamFn]
    AsyncPlannedOps = tuple[str, AsyncStreamFn]


@attrs.define(frozen=True)
class _BaseStream(ABC):
    seq: tuple = attrs.field(validator=attrs.validators.instance_of(tuple))
    ops: tuple = attrs.field(default=(), validator=attrs.validators.instance_of(tuple), repr=False)

    @classmethod
    @abstractmethod
    def from_iterable(cls, it: Iterable) -> _BaseStream[T]: ...

    @abstractmethod
    def map(self, *fns: MapFn | AsyncMapFn) -> _BaseStream[T]: ...

    @abstractmethod
    def filter(self, *fns: FilterFn | AsyncFilterFn) -> _BaseStream[T]: ...

    @abstractmethod
    def tap(self, *fns: TapFn | AsyncTapFn) -> _BaseStream[T]: ...

    @abstractmethod
    def partition(self, fn: FilterFn) -> tuple[_BaseStream[T], _BaseStream[U]]: ...

    @abstractmethod
    def fold(self, initial: T, fn: Callable[[T, U], T], *, workers: int = 1, use_threads: bool = False) -> T: ...

    @abstractmethod
    def collect(self) -> tuple[U, ...]: ...

    @abstractmethod
    def par_collect(self, workers: int = 4, *, use_threads: bool = False) -> tuple[U, ...]: ...

    @abstractmethod
    async def async_collect(self) -> Awaitable[tuple[U, ...]]: ...

    def __bool__(self) -> bool:
        return bool(self.seq)


@attrs.define(frozen=True)
class Stream(_BaseStream):
    @classmethod
    def from_iterable(cls, it: Iterable) -> Stream[T]:
        if not isinstance(it, Iterable):
            it = [it]
        return cls(seq=tuple(it))

    def map(self, *fns: MapFn | AsyncMapFn) -> Stream[U]:
        plan = (*self.ops, *tuple((_MAP, fn) for fn in fns))
        object.__setattr__(self, "ops", plan)
        return self

    def filter(self, *fns: FilterFn | AsyncFilterFn) -> Stream[T]:
        plan = (*self.ops, *tuple((_FILTER, fn) for fn in fns))
        object.__setattr__(self, "ops", plan)
        return self

    def tap(self, *fns: TapFn | AsyncTapFn) -> Stream[T]:
        plan = (*self.ops, *tuple((_TAP, fn) for fn in fns))
        object.__setattr__(self, "ops", plan)
        return self

    def partition(self, fn: FilterFn, *, workers: int = 1, use_threads: bool = False) -> tuple[Stream[T], Stream[U]]:
        if workers > 1:
            seq_tuple = self.par_collect(workers=workers, use_threads=use_threads)
        else:
            seq_tuple = self.collect()
        return (Stream(seq=tuple(x for x in seq_tuple if fn(x))), Stream(seq=tuple(x for x in seq_tuple if not fn(x))))

    def fold(self, initial: T, fn: Callable[[T, U], T], *, workers: int = 1, use_threads: bool = False) -> T:
        if workers > 1:
            return reduce(fn, self.par_collect(workers=workers, use_threads=use_threads), initial)
        return reduce(fn, self.collect(), initial)

    def collect(self) -> tuple[U, ...]:
        return tuple(_apply_fns(self.seq, self.ops))

    def par_collect(self, workers: int = 4, *, use_threads: bool = False) -> tuple[U, ...]:
        if workers == -1:
            workers = (os.cpu_count() or 5) - 1

        executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        batches = [(list(chunk), self.ops) for chunk in batched(self.seq, n=max(4, len(self.seq) // workers))]

        with executor_cls(max_workers=workers) as ex:
            return cast("tuple[U, ...]", tuple(itertools.chain.from_iterable(ex.map(_apply_fns_worker, batches))))

    async def async_collect(self) -> Awaitable[tuple[U, ...]]:
        if not self.ops:
            return cast("Awaitable[tuple[U, ...]]", self.collect())

        res = await asyncio.gather(*(_async_apply_fns(x, self.ops) for x in self.seq))
        return cast("Awaitable[tuple[U, ...]]", tuple(elem for elem in res if elem != _Nothing.NOTHING))


_MAP = 0
_FILTER = 1
_TAP = 2


class _Nothing(Enum):
    NOTHING = auto()


def _apply_fns_worker(args: tuple[tuple[T], tuple[PlannedOps, ...]]) -> tuple[T]:
    seq, ops = args
    return _par_apply_fns(seq, ops)


def _apply_fns(elements: tuple[T], ops: tuple[PlannedOps, ...]) -> Generator[T, None, None]:
    for elem in elements:
        valid = True
        res = elem
        for op, op_fn in ops:
            if op == _MAP:
                res = op_fn(res)
            elif op == _FILTER and not op_fn(res):
                valid = False
                break
            elif op == _TAP:
                op_fn(deepcopy(res))
        if valid:
            yield res


def _par_apply_fns(elements: tuple[T], ops: tuple[PlannedOps, ...]) -> tuple[T]:
    results = []
    for elem in elements:
        valid = True
        res = elem
        for op, op_fn in ops:
            if op == _MAP:
                res = op_fn(res)
            elif op == _FILTER and not op_fn(res):
                valid = False
                break
            elif op == _TAP:
                op_fn(deepcopy(res))
        if valid:
            results.append(res)
    return tuple(results)


async def _async_apply_fns(elem: T, ops: tuple[AsyncPlannedOps, ...]) -> T | _Nothing:
    res = elem
    for op, op_fn in ops:
        if op == _MAP:
            res = await op_fn(res)
        elif op == _FILTER and not await op_fn(res):
            return _Nothing.NOTHING
        elif op == _TAP:
            await op_fn(deepcopy(res))
    return res
