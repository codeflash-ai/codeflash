from __future__ import annotations

import functools
import traceback
from typing import TYPE_CHECKING

from codeflash_core.danom.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from typing_extensions import Concatenate, ParamSpec

    from codeflash_core.danom.result import Result

    T = TypeVar("T")
    P = ParamSpec("P")
    U = TypeVar("U")
    E = TypeVar("E")


def safe(func: Callable[P, U]) -> Callable[P, Result[U, Exception]]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[U, Exception]:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            return Err(error=e, input_args=(args, kwargs), traceback=traceback.format_exc())

    return wrapper


def safe_method(func: Callable[Concatenate[T, P], U]) -> Callable[Concatenate[T, P], Result[U, Exception]]:
    @functools.wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> Result[U, Exception]:
        try:
            return Ok(func(self, *args, **kwargs))
        except Exception as e:
            return Err(error=e, input_args=(self, args, kwargs), traceback=traceback.format_exc())

    return wrapper
