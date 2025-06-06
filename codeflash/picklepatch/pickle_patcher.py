"""PicklePatcher - A utility for safely pickling objects with unpicklable components.

This module provides functions to recursively pickle objects, replacing unpicklable
components with placeholders that provide informative errors when accessed.
"""

from __future__ import annotations

import contextlib
import pickle
from typing import Any, ClassVar

import dill

from .pickle_placeholder import PicklePlaceholder


class PicklePatcher:
    """A utility class for safely pickling objects with unpicklable components.

    This class provides methods to recursively pickle objects, replacing any
    components that can't be pickled with placeholder objects.
    """

    # Class-level cache of unpicklable types
    _unpicklable_types: ClassVar[set[type]] = set()

    @staticmethod
    def dumps(obj: object, protocol: int | None = None, max_depth: int = 100, **kwargs) -> bytes:  # noqa: ANN003
        """Safely pickle an object, replacing unpicklable parts with placeholders.

        Args:
        ----
            obj: The object to pickle
            protocol: The pickle protocol version to use
            max_depth: Maximum recursion depth
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
        -------
            bytes: Pickled data with placeholders for unpicklable objects

        """
        return PicklePatcher._recursive_pickle(obj, max_depth, path=[], protocol=protocol, **kwargs)

    @staticmethod
    def loads(pickled_data: bytes) -> object:
        """Unpickle data that may contain placeholders.

        Args:
        ----
            pickled_data: Pickled data with possible placeholders

        Returns:
        -------
            The unpickled object with placeholders for unpicklable parts

        """
        return dill.loads(pickled_data)

    @staticmethod
    def _create_placeholder(obj: object, error_msg: str, path: list[str]) -> PicklePlaceholder:
        """Create a placeholder for an unpicklable object.

        Args:
        ----
            obj: The original unpicklable object
            error_msg: Error message explaining why it couldn't be pickled
            path: Path to this object in the object graph

        Returns:
        -------
            PicklePlaceholder: A placeholder object

        """
        obj_type = type(obj)
        try:
            obj_str = str(obj)[:100] if hasattr(obj, "__str__") else f"<unprintable object of type {obj_type.__name__}>"
        except:  # noqa: E722
            obj_str = f"<unprintable object of type {obj_type.__name__}>"

        print(f"Creating placeholder for {obj_type.__name__} at path {'->'.join(path) or 'root'}: {error_msg}")

        placeholder = PicklePlaceholder(obj_type.__name__, obj_str, error_msg, path)

        # Add this type to our known unpicklable types cache
        PicklePatcher._unpicklable_types.add(obj_type)
        return placeholder

    @staticmethod
    def _pickle(
        obj: object,
        path: list[str] | None = None,  # noqa: ARG004
        protocol: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, bytes | str]:
        """Try to pickle an object using pickle first, then dill. If both fail, create a placeholder.

        Args:
        ----
            obj: The object to pickle
            path: Path to this object in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
        -------
            tuple: (success, result) where success is a boolean and result is either:
                - Pickled bytes if successful
                - Error message if not successful

        """
        # Try standard pickle first
        try:
            return True, pickle.dumps(obj, protocol=protocol, **kwargs)
        except (pickle.PickleError, TypeError, AttributeError, ValueError):
            # Then try dill (which is more powerful)
            try:
                return True, dill.dumps(obj, protocol=protocol, **kwargs)
            except (dill.PicklingError, TypeError, AttributeError, ValueError) as e:
                return False, str(e)

    @staticmethod
    def _recursive_pickle(  # noqa: PLR0911
        obj: object,
        max_depth: int,
        path: list[str] | None = None,
        protocol: int | None = None,
        **kwargs,  # noqa: ANN003
    ) -> bytes:
        """Recursively try to pickle an object, replacing unpicklable parts with placeholders.

        Args:
        ----
            obj: The object to pickle
            max_depth: Maximum recursion depth
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
        -------
            bytes: Pickled data with placeholders for unpicklable objects

        """
        if path is None:
            path = []

        obj_type = type(obj)

        # Check if this type is known to be unpicklable
        if obj_type in PicklePatcher._unpicklable_types:
            placeholder = PicklePatcher._create_placeholder(obj, "Known unpicklable type", path)
            return dill.dumps(placeholder, protocol=protocol, **kwargs)

        # Check for max depth
        if max_depth <= 0:
            placeholder = PicklePatcher._create_placeholder(obj, "Max recursion depth exceeded", path)
            return dill.dumps(placeholder, protocol=protocol, **kwargs)

        # Try standard pickling
        success, result = PicklePatcher._pickle(obj, path, protocol, **kwargs)
        if success:
            return result

        error_msg = result  # Error message from pickling attempt

        # Handle different container types
        if isinstance(obj, dict):
            return PicklePatcher._handle_dict(obj, max_depth, error_msg, path, protocol=protocol, **kwargs)
        if isinstance(obj, (list, tuple, set)):
            return PicklePatcher._handle_sequence(obj, max_depth, error_msg, path, protocol=protocol, **kwargs)
        if hasattr(obj, "__dict__"):
            result = PicklePatcher._handle_object(obj, max_depth, error_msg, path, protocol=protocol, **kwargs)

            # If this was a failure, add the type to the cache
            unpickled = dill.loads(result)
            if isinstance(unpickled, PicklePlaceholder):
                PicklePatcher._unpicklable_types.add(obj_type)
            return result

        # For other unpicklable objects, use a placeholder
        placeholder = PicklePatcher._create_placeholder(obj, error_msg, path)
        return dill.dumps(placeholder, protocol=protocol, **kwargs)

    @staticmethod
    def _handle_dict(
        obj_dict: dict[Any, Any],
        max_depth: int,
        error_msg: str,  # noqa: ARG004
        path: list[str],
        protocol: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> bytes:
        """Handle pickling for dictionary objects.

        Args:
        ----
            obj_dict: The dictionary to pickle
            max_depth: Maximum recursion depth
            error_msg: Error message from the original pickling attempt
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
        -------
            bytes: Pickled data with placeholders for unpicklable objects

        """
        if not isinstance(obj_dict, dict):
            placeholder = PicklePatcher._create_placeholder(
                obj_dict, f"Expected a dictionary, got {type(obj_dict).__name__}", path
            )
            return dill.dumps(placeholder, protocol=protocol, **kwargs)

        result = {}

        for key, value in obj_dict.items():
            # Process the key
            key_success, key_result = PicklePatcher._pickle(key, path, protocol, **kwargs)
            if key_success:
                key_result = key
            else:
                # If the key can't be pickled, use a string representation
                try:
                    key_str = str(key)[:50]
                except:  # noqa: E722
                    key_str = f"<unprintable key of type {type(key).__name__}>"
                key_result = f"<unpicklable_key:{key_str}>"

            # Process the value
            value_path = [*path, f"[{repr(key)[:20]}]"]
            value_success, value_bytes = PicklePatcher._pickle(value, value_path, protocol, **kwargs)

            if value_success:
                value_result = value
            else:
                # Try recursive pickling for the value
                try:
                    value_bytes = PicklePatcher._recursive_pickle(
                        value, max_depth - 1, value_path, protocol=protocol, **kwargs
                    )
                    value_result = dill.loads(value_bytes)
                except Exception as inner_e:
                    value_result = PicklePatcher._create_placeholder(value, str(inner_e), value_path)

            result[key_result] = value_result

        return dill.dumps(result, protocol=protocol, **kwargs)

    @staticmethod
    def _handle_sequence(
        obj_seq: list[Any] | tuple[Any, ...] | set[Any],
        max_depth: int,
        error_msg: str,  # noqa: ARG004
        path: list[str],
        protocol: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> bytes:
        """Handle pickling for sequence types (list, tuple, set).

        Args:
        ----
            obj_seq: The sequence to pickle
            max_depth: Maximum recursion depth
            error_msg: Error message from the original pickling attempt
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
        -------
            bytes: Pickled data with placeholders for unpicklable objects

        """
        result: list[Any] = []

        for i, item in enumerate(obj_seq):
            item_path = [*path, f"[{i}]"]

            # Try to pickle the item directly
            success, _ = PicklePatcher._pickle(item, item_path, protocol, **kwargs)
            if success:
                result.append(item)
                continue

            # If we couldn't pickle directly, try recursively
            try:
                item_bytes = PicklePatcher._recursive_pickle(
                    item, max_depth - 1, item_path, protocol=protocol, **kwargs
                )
                result.append(dill.loads(item_bytes))
            except Exception as inner_e:
                # If recursive pickling fails, use a placeholder
                placeholder = PicklePatcher._create_placeholder(item, str(inner_e), item_path)
                result.append(placeholder)

        # Convert back to the original type
        if isinstance(obj_seq, tuple):
            result = tuple(result)
        elif isinstance(obj_seq, set):
            # Try to create a set from the result

            with contextlib.suppress(Exception):
                result = set(result)

        return dill.dumps(result, protocol=protocol, **kwargs)

    @staticmethod
    def _handle_object(
        obj: object,
        max_depth: int,
        error_msg: str,
        path: list[str],
        protocol: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> bytes:
        """Handle pickling for custom objects with __dict__.

        Args:
        ----
            obj: The object to pickle
            max_depth: Maximum recursion depth
            error_msg: Error message from the original pickling attempt
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
        -------
            bytes: Pickled data with placeholders for unpicklable objects

        """
        # Try to create a new instance of the same class
        try:
            # First try to create an empty instance
            new_obj = object.__new__(type(obj))

            # Handle __dict__ attributes if they exist
            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    attr_path = [*path, attr_name]

                    # Try to pickle directly first
                    success, _ = PicklePatcher._pickle(attr_value, attr_path, protocol, **kwargs)
                    if success:
                        setattr(new_obj, attr_name, attr_value)
                        continue

                    # If direct pickling fails, try recursive pickling
                    try:
                        attr_bytes = PicklePatcher._recursive_pickle(
                            attr_value, max_depth - 1, attr_path, protocol=protocol, **kwargs
                        )
                        setattr(new_obj, attr_name, dill.loads(attr_bytes))
                    except Exception as inner_e:
                        # Use placeholder for unpicklable attribute
                        placeholder = PicklePatcher._create_placeholder(attr_value, str(inner_e), attr_path)
                        setattr(new_obj, attr_name, placeholder)

            # Try to pickle the patched object
            success, result = PicklePatcher._pickle(new_obj, path, protocol, **kwargs)
            if success:
                return result
            # Fall through to placeholder creation
        except Exception:  # noqa: S110
            pass  # Fall through to placeholder creation

        # If we get here, just use a placeholder
        placeholder = PicklePatcher._create_placeholder(obj, error_msg, path)
        return dill.dumps(placeholder, protocol=protocol, **kwargs)
