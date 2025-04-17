"""PicklePatcher - A utility for safely pickling objects with unpicklable components.

This module provides functions to recursively pickle objects, replacing unpicklable
components with placeholders that provide informative errors when accessed.
"""

import pickle
import types

import dill

from .pickle_placeholder import PicklePlaceholder


class PicklePatcher:
    """A utility class for safely pickling objects with unpicklable components.

    This class provides methods to recursively pickle objects, replacing any
    components that can't be pickled with placeholder objects.
    """

    # Class-level cache of unpicklable types
    _unpicklable_types = set()

    @staticmethod
    def dumps(obj, protocol=None, max_depth=100, **kwargs):
        """Safely pickle an object, replacing unpicklable parts with placeholders.

        Args:
            obj: The object to pickle
            protocol: The pickle protocol version to use
            max_depth: Maximum recursion depth
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
            bytes: Pickled data with placeholders for unpicklable objects
        """
        return PicklePatcher._recursive_pickle(obj, max_depth, path=[], protocol=protocol, **kwargs)

    @staticmethod
    def loads(pickled_data):
        """Unpickle data that may contain placeholders.

        Args:
            pickled_data: Pickled data with possible placeholders

        Returns:
            The unpickled object with placeholders for unpicklable parts
        """
        try:
            # We use dill for loading since it can handle everything pickle can
            return dill.loads(pickled_data)
        except Exception as e:
            raise

    @staticmethod
    def _create_placeholder(obj, error_msg, path):
        """Create a placeholder for an unpicklable object.

        Args:
            obj: The original unpicklable object
            error_msg: Error message explaining why it couldn't be pickled
            path: Path to this object in the object graph

        Returns:
            PicklePlaceholder: A placeholder object
        """
        obj_type = type(obj)
        try:
            obj_str = str(obj)[:100] if hasattr(obj, "__str__") else f"<unprintable object of type {obj_type.__name__}>"
        except:
            obj_str = f"<unprintable object of type {obj_type.__name__}>"

        print(f"Creating placeholder for {obj_type.__name__} at path {'->'.join(path) or 'root'}: {error_msg}")

        placeholder = PicklePlaceholder(
            obj_type.__name__,
            obj_str,
            error_msg,
            path
        )

        # Add this type to our known unpicklable types cache
        PicklePatcher._unpicklable_types.add(obj_type)
        return placeholder

    @staticmethod
    def _pickle(obj, path=None, protocol=None, **kwargs):
        """Try to pickle an object using pickle first, then dill. If both fail, create a placeholder.

        Args:
            obj: The object to pickle
            path: Path to this object in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
            tuple: (success, result) where success is a boolean and result is either:
                - Pickled bytes if successful
                - Error message if not successful
        """
        # Try standard pickle first
        try:
            return True, pickle.dumps(obj, protocol=protocol, **kwargs)
        except (pickle.PickleError, TypeError, AttributeError, ValueError) as e:
            # Then try dill (which is more powerful)
            try:
                return True, dill.dumps(obj, protocol=protocol, **kwargs)
            except (dill.PicklingError, TypeError, AttributeError, ValueError) as e:
                return False, str(e)

    @staticmethod
    def _recursive_pickle(obj, max_depth, path=None, protocol=None, **kwargs):
        """Recursively try to pickle an object, replacing unpicklable parts with placeholders.

        Args:
            obj: The object to pickle
            max_depth: Maximum recursion depth
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
            bytes: Pickled data with placeholders for unpicklable objects
        """
        if path is None:
            path = []

        obj_type = type(obj)

        # Check if this type is known to be unpicklable
        if obj_type in PicklePatcher._unpicklable_types:
            placeholder = PicklePatcher._create_placeholder(
                obj,
                "Known unpicklable type",
                path
            )
            return dill.dumps(placeholder, protocol=protocol, **kwargs)

        # Check for max depth
        if max_depth <= 0:
            placeholder = PicklePatcher._create_placeholder(
                obj,
                "Max recursion depth exceeded",
                path
            )
            return dill.dumps(placeholder, protocol=protocol, **kwargs)

        # Try standard pickling
        success, result = PicklePatcher._pickle(obj, path, protocol, **kwargs)
        if success:
            return result

        error_msg = result  # Error message from pickling attempt

        # Handle different container types
        if isinstance(obj, dict):
            return PicklePatcher._handle_dict(obj, max_depth, error_msg, path, protocol=protocol, **kwargs)
        elif isinstance(obj, (list, tuple, set)):
            return PicklePatcher._handle_sequence(obj, max_depth, error_msg, path, protocol=protocol, **kwargs)
        elif hasattr(obj, "__dict__"):
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
    def _handle_dict(obj_dict, max_depth, error_msg, path, protocol=None, **kwargs):
        """Handle pickling for dictionary objects.

        Args:
            obj_dict: The dictionary to pickle
            max_depth: Maximum recursion depth
            error_msg: Error message from the original pickling attempt
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
            bytes: Pickled data with placeholders for unpicklable objects
        """
        if not isinstance(obj_dict, dict):
            placeholder = PicklePatcher._create_placeholder(
                obj_dict,
                f"Expected a dictionary, got {type(obj_dict).__name__}",
                path
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
                except:
                    key_str = f"<unprintable key of type {type(key).__name__}>"
                key_result = f"<unpicklable_key:{key_str}>"

            # Process the value
            value_path = path + [f"[{repr(key)[:20]}]"]
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
                    value_result = PicklePatcher._create_placeholder(
                        value,
                        str(inner_e),
                        value_path
                    )

            result[key_result] = value_result

        return dill.dumps(result, protocol=protocol, **kwargs)

    @staticmethod
    def _handle_sequence(obj_seq, max_depth, error_msg, path, protocol=None, **kwargs):
        """Handle pickling for sequence types (list, tuple, set).

        Args:
            obj_seq: The sequence to pickle
            max_depth: Maximum recursion depth
            error_msg: Error message from the original pickling attempt
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
            bytes: Pickled data with placeholders for unpicklable objects
        """
        result = []

        for i, item in enumerate(obj_seq):
            item_path = path + [f"[{i}]"]

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
                placeholder = PicklePatcher._create_placeholder(
                    item,
                    str(inner_e),
                    item_path
                )
                result.append(placeholder)

        # Convert back to the original type
        if isinstance(obj_seq, tuple):
            result = tuple(result)
        elif isinstance(obj_seq, set):
            # Try to create a set from the result
            try:
                result = set(result)
            except Exception:
                # If we can't create a set (unhashable items), keep it as a list
                pass

        return dill.dumps(result, protocol=protocol, **kwargs)

    @staticmethod
    def _handle_object(obj, max_depth, error_msg, path, protocol=None, **kwargs):
        """Handle pickling for custom objects with __dict__.

        Args:
            obj: The object to pickle
            max_depth: Maximum recursion depth
            error_msg: Error message from the original pickling attempt
            path: Current path in the object graph
            protocol: The pickle protocol version to use
            **kwargs: Additional arguments for pickle/dill.dumps

        Returns:
            bytes: Pickled data with placeholders for unpicklable objects
        """
        # Try to create a new instance of the same class
        try:
            # First try to create an empty instance
            new_obj = object.__new__(type(obj))

            # Handle __dict__ attributes if they exist
            if hasattr(obj, "__dict__"):
                for attr_name, attr_value in obj.__dict__.items():
                    attr_path = path + [attr_name]

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
                        placeholder = PicklePatcher._create_placeholder(
                            attr_value,
                            str(inner_e),
                            attr_path
                        )
                        setattr(new_obj, attr_name, placeholder)

            # Try to pickle the patched object
            success, result = PicklePatcher._pickle(new_obj, path, protocol, **kwargs)
            if success:
                return result
            # Fall through to placeholder creation
        except Exception:
            pass  # Fall through to placeholder creation

        # If we get here, just use a placeholder
        placeholder = PicklePatcher._create_placeholder(obj, error_msg, path)
        return dill.dumps(placeholder, protocol=protocol, **kwargs)