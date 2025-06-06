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
        """Initialize a placeholder for an unpicklable object.

        Args:
        ----
            obj_type (str): The type name of the original object
            obj_str (str): String representation of the original object
            error_msg (str): The error message that occurred during pickling
            path (list, optional): Path to this object in the original object graph

        """
        # Store these directly in __dict__ to avoid __getattr__ recursion
        self.__dict__["obj_type"] = obj_type
        self.__dict__["obj_str"] = obj_str
        self.__dict__["error_msg"] = error_msg
        self.__dict__["path"] = path if path is not None else []

    def __getattr__(self, name) -> Any:  # noqa: ANN001, ANN401
        """Raise a custom error when any attribute is accessed."""
        path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root object"
        msg = (
            f"Attempt to access unpickleable object: Cannot access attribute '{name}' on unpicklable object at {path_str}. "
            f"Original type: {self.__dict__['obj_type']}. Error: {self.__dict__['error_msg']}"
        )
        raise PicklePlaceholderAccessError(msg)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Prevent setting attributes."""
        self.__getattr__(name)  # This will raise our custom error

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        """Raise a custom error when the object is called."""
        path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root object"
        msg = (
            f"Attempt to access unpickleable object: Cannot call unpicklable object at {path_str}. "
            f"Original type: {self.__dict__['obj_type']}. Error: {self.__dict__['error_msg']}"
        )
        raise PicklePlaceholderAccessError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the placeholder."""
        try:
            path_str = ".".join(self.__dict__["path"]) if self.__dict__["path"] else "root"
            return f"<PicklePlaceholder at {path_str}: {self.__dict__['obj_type']} {self.__dict__['obj_str']}>"
        except:  # noqa: E722
            return "<PicklePlaceholder: (error displaying details)>"

    def __str__(self) -> str:
        """Return a string representation of the placeholder."""
        return self.__repr__()

    def __reduce__(self) -> tuple:
        """Make sure pickling of the placeholder itself works correctly."""
        return (
            PicklePlaceholder,
            (self.__dict__["obj_type"], self.__dict__["obj_str"], self.__dict__["error_msg"], self.__dict__["path"]),
        )
