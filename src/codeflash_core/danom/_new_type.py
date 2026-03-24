from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import attrs

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec, Self

    P = ParamSpec("P")
    C = TypeVar("C", bound=Callable[P, object])

T = TypeVar("T")


def new_type(
    name: str,
    base_type: type,
    validators: Callable | Sequence[Callable] | None = None,
    converters: Callable | Sequence[Callable] | None = None,
    *,
    frozen: bool = True,
):
    kwargs = _callables_to_kwargs(base_type, validators, converters)

    @attrs.define(frozen=frozen, eq=True, hash=frozen)
    class _Wrapper:
        inner: T = attrs.field(**kwargs)  # type: ignore[no-matching-overload]

        def map(self, func: Callable[[T], T]) -> Self:
            return self.__class__(func(self.inner))  # type: ignore[invalid-argument-type]

        locals().update(_create_forward_methods(base_type))

    _Wrapper.__name__ = name
    _Wrapper.__qualname__ = name
    return _Wrapper


def _create_forward_methods(base_type: type) -> dict[str, Callable]:
    methods: dict[str, Callable] = {}
    for attr_name, _ in inspect.getmembers(base_type, inspect.isroutine):
        if attr_name.startswith("_"):
            continue

        def make_forwarder(name: str) -> Callable:
            def method(self, *args: tuple, **kwargs: dict) -> T:
                return getattr(self.inner, name)(*args, **kwargs)

            method.__name__ = name
            method.__doc__ = getattr(base_type, name).__doc__
            return method

        methods[attr_name] = make_forwarder(attr_name)
    return methods


def _callables_to_kwargs(
    base_type: type, validators: Callable | Sequence[Callable] | None, converters: Callable | Sequence[Callable] | None
) -> dict[str, Sequence[Callable]]:
    kwargs = {"validator": [attrs.validators.instance_of(base_type)], "converter": []}
    kwargs["validator"] += [_validate_bool_func(fn) for fn in _to_list(validators)]
    kwargs["converter"] += _to_list(converters)

    return {k: v for k, v in kwargs.items() if v}


def _validate_bool_func(bool_fn: Callable[[T], bool]) -> Callable[[attrs.AttrsInstance, attrs.Attribute, T], None]:
    if not callable(bool_fn):
        raise TypeError("provided boolean function must be callable")

    @wraps(bool_fn)
    def wrapper(_instance: attrs.AttrsInstance, attribute: attrs.Attribute, value: T) -> None:
        if not bool_fn(value):
            msg = f"{attribute.name} does not return True for the given boolean function, received `{value}`."
            raise ValueError(msg)

    return wrapper


def _to_list(value: C | Sequence[C] | None) -> list[C]:
    if value is None:
        return []

    if callable(value):
        return [value]  # type: ignore[invalid-return-type]

    if isinstance(value, Sequence) and not all(callable(fn) for fn in value):
        msg = f"Given items are not all callable: {value = }"
        raise TypeError(msg)

    return list(value)
