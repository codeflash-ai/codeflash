from __future__ import annotations

from typing import Any


class PicklePlaceholderAccessError(Exception):
    """Custom exception raised when attempting to access an unpicklable object."""


class PicklePlaceholder:
    """A placeholder for an object that couldn't be pickled.

    When unpickled, any attempt to access attributes or call methods on this
    placeholder will raise a PicklePlaceholderAccessError.
    """

    def __init__(self, obj_type: str, obj_str: str, error_msg: str, path: list[str] | None = None) -> None:
        # Store these directly in __dict__ to avoid __getattr__ recursion
        self.__dict__["obj_type"] = obj_type
        self.__dict__["obj_str"] = obj_str
        self.__dict__["error_msg"] = error_msg
        self.__dict__["path"] = path if path is not None else []

    def __getattr__(self, name) -> Any:
        path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root object"
        msg = (
            f"Attempt to access unpickleable object: Cannot access attribute '{name}' on unpicklable object at {path_str}. "
            f"Original type: {self.__dict__['obj_type']}. Error: {self.__dict__['error_msg']}"
        )
        raise PicklePlaceholderAccessError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__getattr__(name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root object"
        msg = (
            f"Attempt to access unpickleable object: Cannot call unpicklable object at {path_str}. "
            f"Original type: {self.__dict__['obj_type']}. Error: {self.__dict__['error_msg']}"
        )
        raise PicklePlaceholderAccessError(msg)

    def __repr__(self) -> str:
        try:
            path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root"
            return f"<PicklePlaceholder at {path_str}: {self.__dict__['obj_type']} {self.__dict__['obj_str']}>"
        except:  # noqa: E722
            return "<PicklePlaceholder: (error displaying details)>"

    def __str__(self) -> str:
        return self.__repr__()

    def __reduce__(self) -> tuple:
        return (
            PicklePlaceholder,
            (self.__dict__["obj_type"], self.__dict__["obj_str"], self.__dict__["error_msg"], self.__dict__["path"]),
        )
