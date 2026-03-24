from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

try:
    from typing import Never  # type: ignore[unresolved-import]
except ImportError:
    from typing import NoReturn as Never

import attrs
from attrs.validators import instance_of

T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)
E_co = TypeVar("E_co", bound=object, covariant=True)
F_co = TypeVar("F_co", bound=object, covariant=True)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from typing_extensions import Concatenate, ParamSpec, Self

    P = ParamSpec("P")
    Mappable = Callable[Concatenate[T_co, P], U_co]
    Bindable = Callable[Concatenate[T_co, P], "Result[U_co, E_co]"]


@attrs.define(frozen=True)
class Result(ABC, Generic[T_co, E_co]):
    """`Result` monad. Consists of `Ok` and `Err` for successful and failed operations respectively.

    Each monad is a frozen instance to prevent further mutation.
    """

    @classmethod
    def unit(cls, inner: T_co) -> Ok[T_co]:
        return Ok(inner)

    @abstractmethod
    def is_ok(self) -> bool: ...

    @abstractmethod
    def map(self, func: Mappable, *args: P.args, **kwargs: P.kwargs) -> Result[U_co, E_co]: ...

    @abstractmethod
    def map_err(self, func: Mappable, *args: P.args, **kwargs: P.kwargs) -> Result[U_co, E_co]: ...

    @abstractmethod
    def and_then(self, func: Bindable, *args: P.args, **kwargs: P.kwargs) -> Result[U_co, E_co]: ...

    @abstractmethod
    def or_else(self, func: Bindable, *args: P.args, **kwargs: P.kwargs) -> Result[U_co, E_co]: ...

    @abstractmethod
    def unwrap(self) -> T_co: ...

    @staticmethod
    def result_is_ok(result: Result[T_co, E_co]) -> bool:
        return result.is_ok()

    @staticmethod
    def result_unwrap(result: Result[T_co, E_co]) -> T_co:
        return result.unwrap()


@attrs.define(frozen=True, hash=True)
class Ok(Result[T_co, Never]):
    inner: Any = attrs.field(default=None)

    def is_ok(self) -> Literal[True]:
        return True

    def map(self, func: Mappable, *args: P.args, **kwargs: P.kwargs) -> Ok[U_co]:
        return Ok(func(self.inner, *args, **kwargs))

    def map_err(self, func: Mappable, *args: P.args, **kwargs: P.kwargs) -> Self:
        return self

    def and_then(self, func: Bindable, *args: P.args, **kwargs: P.kwargs) -> Result[U_co, E_co]:
        return func(self.inner, *args, **kwargs)

    def or_else(self, func: Bindable, *args: P.args, **kwargs: P.kwargs) -> Self:
        return self

    def unwrap(self) -> T_co:
        return self.inner


SafeArgs = tuple[tuple[Any, ...], dict[str, Any]]
SafeMethodArgs = tuple[object, tuple[Any, ...], dict[str, Any]]


@attrs.define(frozen=True)
class Err(Result[Never, E_co]):
    error: Any = attrs.field(default=None)
    input_args: tuple[()] | SafeArgs | SafeMethodArgs = attrs.field(
        default=(), validator=instance_of(tuple), repr=False
    )
    traceback: str = attrs.field(default="", validator=instance_of(str))
    details: list[dict[str, Any]] = attrs.field(factory=list, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if isinstance(self.error, Exception):
            object.__setattr__(self, "details", self._extract_details(self.error.__traceback__))

    def _extract_details(self, tb: TracebackType | None) -> list[dict[str, Any]]:
        trace_info = []
        while tb:
            frame = tb.tb_frame
            trace_info.append(
                {
                    "file": frame.f_code.co_filename,
                    "func": frame.f_code.co_name,
                    "line_no": tb.tb_lineno,
                    "locals": frame.f_locals,
                }
            )
            tb = tb.tb_next
        return trace_info

    def is_ok(self) -> Literal[False]:
        return False

    def map(self, func: Mappable, *args: P.args, **kwargs: P.kwargs) -> Self:
        return self

    def map_err(self, func: Mappable, *args: P.args, **kwargs: P.kwargs) -> Err[F_co]:
        return Err(func(self.error, *args, **kwargs))

    def and_then(self, func: Bindable, *args: P.args, **kwargs: P.kwargs) -> Self:
        return self

    def or_else(self, func: Bindable, *args: P.args, **kwargs: P.kwargs) -> Result[U_co, E_co]:
        return func(self.error, *args, **kwargs)

    def unwrap(self) -> T_co:
        if isinstance(self.error, Exception):
            raise self.error
        msg = f"Err does not have a caught error to raise: {self.error = }"
        raise ValueError(msg)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Err):
            return False

        return all(
            (
                type(self.error) is type(other.error),
                str(self.error) == str(other.error),
                self.input_args == other.input_args,
            )
        )

    def __hash__(self) -> int:
        return hash(f"{type(self.error)}{self.error}{self.input_args}")
