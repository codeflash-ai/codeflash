from __future__ import annotations

from typing import Generic, TypeVar

from codeflash.cli_cmds.console import logger

L = TypeVar("L")
R = TypeVar("R")


class CodeflashError:
    def __init__(self, code: str, message_template: str, **formatting_args: str) -> None:
        self.code = code
        self.message_template = message_template
        self.formatting_args = formatting_args

    @property
    def message(self) -> str:
        try:
            formatted = self.message_template.format(**self.formatting_args)
            return f"[{self.code}] {formatted}"  # noqa: TRY300
        except KeyError:
            logger.debug(f"Invalid template: missing {self.formatting_args}")
            return self.message_template

    def __str__(self) -> str:
        return self.message


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

    def failure(self) -> CodeflashError:
        if self.is_successful():
            msg = "Cannot get failure value from a success"
            raise ValueError(msg)
        if isinstance(self, Failure):
            return self.error_code
        raise ValueError("Result is not a failure")


class Failure(Result[L, R]):
    def __init__(self, error_code: CodeflashError) -> None:
        super().__init__(error_code.message)
        self.error_code = error_code


class Success(Result[L, R]):
    pass


def is_successful(result: Result[L, R]) -> bool:
    return result.is_successful()
