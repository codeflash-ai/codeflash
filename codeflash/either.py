from __future__ import annotations

from typing import Generic, TypeVar

L = TypeVar("L")
R = TypeVar("R")


class Result(Generic[L, R]):
    def __init__(self, value: L | R) -> None:
        self.value = value

    def is_failure(self) -> bool:
        return isinstance(self, Failure)

    def is_successful(self) -> bool:
        return isinstance(self, Success)

    def unwrap(self) -> L | R:
        if self.is_failure():
            msg = "Cannot unwrap a failure"
            raise ValueError(msg)
        return self.value

    def failure(self) -> L | R:
        if self.is_successful():
            msg = "Cannot get failure value from a success"
            raise ValueError(msg)
        return self.value


class Failure(Result[L, R]):
    pass


class Success(Result[L, R]):
    pass


def is_successful(result: Result[L, R]) -> bool:
    return result.is_successful()
