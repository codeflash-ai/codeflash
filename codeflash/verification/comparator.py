import array
import ast
import datetime
import decimal
import enum
import math
import re
import types
from collections import ChainMap, OrderedDict, deque
from importlib.util import find_spec
from typing import Any, Optional

import sentry_sdk

from codeflash.cli_cmds.console import logger
from codeflash.picklepatch.pickle_placeholder import PicklePlaceholderAccessError

HAS_NUMPY = find_spec("numpy") is not None
HAS_SQLALCHEMY = find_spec("sqlalchemy") is not None
HAS_SCIPY = find_spec("scipy") is not None
HAS_PANDAS = find_spec("pandas") is not None
HAS_PYRSISTENT = find_spec("pyrsistent") is not None
HAS_TORCH = find_spec("torch") is not None
HAS_JAX = find_spec("jax") is not None
HAS_XARRAY = find_spec("xarray") is not None
HAS_TENSORFLOW = find_spec("tensorflow") is not None


def _extract_exception_from_message(msg: str) -> Optional[BaseException]:  # noqa: FA100
    """Try to extract a wrapped exception type from an error message.

    Looks for patterns like "got ExceptionType('..." that indicate a wrapped exception.
    Returns a synthetic exception of that type if found in builtins, None otherwise.
    """
    # Pattern: "got ExceptionType('message')" or "got ExceptionType("message")"
    # This pattern is used by torch._dynamo and potentially other libraries
    match = re.search(r"got (\w+)\(['\"]", msg)
    if match:
        exc_name = match.group(1)
        # Try to find this exception type in builtins
        import builtins

        exc_class = getattr(builtins, exc_name, None)
        if exc_class is not None and isinstance(exc_class, type) and issubclass(exc_class, BaseException):
            return exc_class()
    return None


def _get_wrapped_exception(exc: BaseException) -> Optional[BaseException]:  # noqa: FA100
    """Get the wrapped exception if this is a simple wrapper.

    Returns the inner exception if:
    - exc is an ExceptionGroup with exactly one exception
    - exc has a __cause__ (explicit chaining via 'raise X from Y')
    - exc message contains a wrapped exception type pattern (e.g., "got IndexError('...")")

    Returns None if exc is not a wrapper or wraps multiple exceptions.
    """
    # Check for ExceptionGroup with single exception (Python 3.11+)
    if hasattr(exc, "exceptions"):
        exceptions = exc.exceptions
        if len(exceptions) == 1:
            return exceptions[0]
    # Check for explicit exception chaining (__cause__)
    if exc.__cause__ is not None:
        return exc.__cause__
    # Try to extract wrapped exception type from the message (library-agnostic)
    return _extract_exception_from_message(str(exc))


def comparator(orig: Any, new: Any, superset_obj=False) -> bool:  # noqa: ANN001, ANN401, FBT002, PLR0911
    """Compare two objects for equality recursively. If superset_obj is True, the new object is allowed to have more keys than the original object. However, the existing keys/values must be equivalent."""
    try:
        # Handle exceptions specially - before type check to allow wrapper comparison
        if isinstance(orig, BaseException) and isinstance(new, BaseException):
            if isinstance(orig, PicklePlaceholderAccessError) or isinstance(new, PicklePlaceholderAccessError):
                # If this error was raised, there was an attempt to access the PicklePlaceholder, which represents an unpickleable object.
                # The test results should be rejected as the behavior of the unpickleable object is unknown.
                logger.debug("Unable to verify behavior of unpickleable object in replay test")
                return False

            # If types match exactly, compare attributes
            if type(orig) is type(new):
                orig_dict = {k: v for k, v in orig.__dict__.items() if not k.startswith("_")}
                new_dict = {k: v for k, v in new.__dict__.items() if not k.startswith("_")}
                return comparator(orig_dict, new_dict, superset_obj)

            # Types differ - check if one is a wrapper over the other
            # Check if orig wraps something that matches new
            wrapped_orig = _get_wrapped_exception(orig)
            if wrapped_orig is not None and comparator(wrapped_orig, new, superset_obj):
                return True

            # Check if new wraps something that matches orig
            wrapped_new = _get_wrapped_exception(new)
            if wrapped_new is not None and comparator(orig, wrapped_new, superset_obj):  # noqa: SIM103
                return True

            return False

        if type(orig) is not type(new):
            type_obj = type(orig)
            new_type_obj = type(new)
            # distinct type objects are created at runtime, even if the class code is exactly the same, so we can only compare the names
            if type_obj.__name__ != new_type_obj.__name__ or type_obj.__qualname__ != new_type_obj.__qualname__:
                return False
        if isinstance(orig, (list, tuple, deque, ChainMap)):
            if len(orig) != len(new):
                return False
            return all(comparator(elem1, elem2, superset_obj) for elem1, elem2 in zip(orig, new))

        if isinstance(
            orig,
            (
                str,
                int,
                bool,
                complex,
                type(None),
                type(Ellipsis),
                decimal.Decimal,
                set,
                bytes,
                bytearray,
                memoryview,
                frozenset,
                enum.Enum,
                type,
                range,
                slice,
                OrderedDict,
            ),
        ):
            return orig == new
        if isinstance(orig, float):
            if math.isnan(orig) and math.isnan(new):
                return True
            return math.isclose(orig, new)

        if HAS_JAX:
            import jax  # type: ignore  # noqa: PGH003
            import jax.numpy as jnp  # type: ignore  # noqa: PGH003

            # Handle JAX arrays first to avoid boolean context errors in other conditions
            if isinstance(orig, jax.Array):
                if orig.dtype != new.dtype:
                    return False
                if orig.shape != new.shape:
                    return False
                return bool(jnp.allclose(orig, new, equal_nan=True))

        # Handle xarray objects before numpy to avoid boolean context errors
        if HAS_XARRAY:
            import xarray  # type: ignore  # noqa: PGH003

            if isinstance(orig, (xarray.Dataset, xarray.DataArray)):
                return orig.identical(new)

        # Handle TensorFlow objects early to avoid boolean context errors
        if HAS_TENSORFLOW:
            import tensorflow as tf  # type: ignore  # noqa: PGH003

            if isinstance(orig, tf.Tensor):
                if orig.dtype != new.dtype:
                    return False
                if orig.shape != new.shape:
                    return False
                # Use numpy conversion for proper NaN handling
                return comparator(orig.numpy(), new.numpy(), superset_obj)

            if isinstance(orig, tf.Variable):
                if orig.dtype != new.dtype:
                    return False
                if orig.shape != new.shape:
                    return False
                return comparator(orig.numpy(), new.numpy(), superset_obj)

            if isinstance(orig, tf.dtypes.DType):
                return orig == new

            if isinstance(orig, tf.TensorShape):
                return orig == new

            if isinstance(orig, tf.SparseTensor):
                if not comparator(orig.dense_shape.numpy(), new.dense_shape.numpy(), superset_obj):
                    return False
                return comparator(orig.indices.numpy(), new.indices.numpy(), superset_obj) and comparator(
                    orig.values.numpy(), new.values.numpy(), superset_obj
                )

            if isinstance(orig, tf.RaggedTensor):
                if orig.dtype != new.dtype:
                    return False
                if orig.shape.rank != new.shape.rank:
                    return False
                return comparator(orig.to_list(), new.to_list(), superset_obj)

        if HAS_SQLALCHEMY:
            import sqlalchemy  # type: ignore  # noqa: PGH003

            try:
                insp = sqlalchemy.inspection.inspect(orig)
                insp = sqlalchemy.inspection.inspect(new)  # noqa: F841
                orig_keys = orig.__dict__
                new_keys = new.__dict__
                for key in list(orig_keys.keys()):
                    if key.startswith("_"):
                        continue
                    if key not in new_keys or not comparator(orig_keys[key], new_keys[key], superset_obj):
                        return False
                return True  # noqa: TRY300

            except sqlalchemy.exc.NoInspectionAvailable:
                pass

        if HAS_SCIPY:
            import scipy  # type: ignore  # noqa: PGH003
        # scipy condition because dok_matrix type is also a instance of dict, but dict comparison doesn't work for it
        if isinstance(orig, dict) and not (HAS_SCIPY and isinstance(orig, scipy.sparse.spmatrix)):
            if superset_obj:
                return all(k in new and comparator(v, new[k], superset_obj) for k, v in orig.items())
            if len(orig) != len(new):
                return False
            for key in orig:
                if key not in new:
                    return False
                if not comparator(orig[key], new[key], superset_obj):
                    return False
            return True

        # Handle dict view types (dict_keys, dict_values, dict_items)
        # Use type name checking since these are not directly importable types
        type_name = type(orig).__name__
        if type_name == "dict_keys":
            # dict_keys can be compared as sets (order doesn't matter)
            return comparator(set(orig), set(new))
        if type_name == "dict_values":
            # dict_values need element-wise comparison (order matters)
            return comparator(list(orig), list(new))
        if type_name == "dict_items":
            # Convert to dict for order-insensitive comparison (handles unhashable values)
            return comparator(dict(orig), dict(new), superset_obj)

        if HAS_NUMPY:
            import numpy as np  # type: ignore  # noqa: PGH003

            if isinstance(orig, (np.datetime64, np.timedelta64)):
                # Handle NaT (Not a Time) - numpy's equivalent of NaN for datetime
                if np.isnat(orig) and np.isnat(new):
                    return True
                if np.isnat(orig) or np.isnat(new):
                    return False
                return orig == new

            if isinstance(orig, np.ndarray):
                if orig.dtype != new.dtype:
                    return False
                if orig.shape != new.shape:
                    return False
                # Handle 0-d arrays specially to avoid "iteration over a 0-d array" error
                if orig.ndim == 0:
                    try:
                        return np.allclose(orig, new, equal_nan=True)
                    except Exception:
                        return bool(orig == new)
                try:
                    return np.allclose(orig, new, equal_nan=True)
                except Exception:
                    # fails at "ufunc 'isfinite' not supported for the input types"
                    return np.all([comparator(x, y, superset_obj) for x, y in zip(orig, new)])

            if isinstance(orig, (np.floating, np.complexfloating)):
                return np.isclose(orig, new, equal_nan=True)

            if isinstance(orig, (np.integer, np.bool_, np.byte)):
                return orig == new

            if isinstance(orig, np.void):
                if orig.dtype != new.dtype:
                    return False
                return all(comparator(orig[field], new[field], superset_obj) for field in orig.dtype.fields)

            # Handle np.dtype instances (including numpy.dtypes.* classes like Float64DType, Int64DType, etc.)
            if isinstance(orig, np.dtype):
                return orig == new

            # Handle numpy random generators
            if isinstance(orig, np.random.Generator):
                # Compare the underlying BitGenerator state
                orig_state = orig.bit_generator.state
                new_state = new.bit_generator.state
                return comparator(orig_state, new_state, superset_obj)

            if isinstance(orig, np.random.RandomState):
                # Compare the internal state
                orig_state = orig.get_state(legacy=False)
                new_state = new.get_state(legacy=False)
                return comparator(orig_state, new_state, superset_obj)

        if HAS_SCIPY and isinstance(orig, scipy.sparse.spmatrix):
            if orig.dtype != new.dtype:
                return False
            if orig.get_shape() != new.get_shape():
                return False
            return (orig != new).nnz == 0

        if HAS_PANDAS:
            import pandas  # type: ignore  # noqa: ICN001, PGH003

            if isinstance(
                orig, (pandas.DataFrame, pandas.Series, pandas.Index, pandas.Categorical, pandas.arrays.SparseArray)
            ):
                return orig.equals(new)

            if isinstance(orig, (pandas.CategoricalDtype, pandas.Interval, pandas.Period)):
                return orig == new
            if pandas.isna(orig) and pandas.isna(new):
                return True

        if isinstance(orig, array.array):
            if orig.typecode != new.typecode:
                return False
            if len(orig) != len(new):
                return False
            return all(comparator(elem1, elem2, superset_obj) for elem1, elem2 in zip(orig, new))

        # This should be at the end of all numpy checking
        try:
            if HAS_NUMPY and np.isnan(orig):
                return np.isnan(new)
        except Exception:  # noqa: S110
            pass
        try:
            if HAS_NUMPY and np.isinf(orig):
                return np.isinf(new)
        except Exception:  # noqa: S110
            pass

        if HAS_TORCH:
            import torch  # type: ignore  # noqa: PGH003

            if isinstance(orig, torch.Tensor):
                if orig.dtype != new.dtype:
                    return False
                if orig.shape != new.shape:
                    return False
                if orig.requires_grad != new.requires_grad:
                    return False
                if orig.device != new.device:
                    return False
                return torch.allclose(orig, new, equal_nan=True)

            if isinstance(orig, torch.dtype):
                return orig == new

            if isinstance(orig, torch.device):
                return orig == new

        if HAS_PYRSISTENT:
            import pyrsistent  # type: ignore  # noqa: PGH003

            if isinstance(
                orig,
                (
                    pyrsistent.PMap,
                    pyrsistent.PVector,
                    pyrsistent.PSet,
                    pyrsistent.PRecord,
                    pyrsistent.PClass,
                    pyrsistent.PBag,
                    pyrsistent.PList,
                    pyrsistent.PDeque,
                ),
            ):
                return orig == new

        if hasattr(orig, "__attrs_attrs__") and hasattr(new, "__attrs_attrs__"):
            orig_dict = {}
            new_dict = {}

            for attr in orig.__attrs_attrs__:
                if attr.eq:
                    attr_name = attr.name
                    orig_dict[attr_name] = getattr(orig, attr_name, None)
                    new_dict[attr_name] = getattr(new, attr_name, None)

            if superset_obj:
                new_attrs_dict = {}
                for attr in new.__attrs_attrs__:
                    if attr.eq:
                        attr_name = attr.name
                        new_attrs_dict[attr_name] = getattr(new, attr_name, None)
                return all(
                    k in new_attrs_dict and comparator(v, new_attrs_dict[k], superset_obj) for k, v in orig_dict.items()
                )
            return comparator(orig_dict, new_dict, superset_obj)

        # re.Pattern can be made better by DFA Minimization and then comparing
        if isinstance(
            orig, (datetime.datetime, datetime.date, datetime.timedelta, datetime.time, datetime.timezone, re.Pattern)
        ):
            return orig == new

        # If the object passed has a user defined __eq__ method, use that
        # This could fail if the user defined __eq__ is defined with C-extensions
        try:
            if hasattr(orig, "__eq__") and str(type(orig.__eq__)) == "<class 'method'>":
                return orig == new
        except Exception:  # noqa: S110
            pass

        # For class objects
        if hasattr(orig, "__dict__") and hasattr(new, "__dict__"):
            orig_keys = orig.__dict__
            new_keys = new.__dict__
            if type(orig_keys) == types.MappingProxyType and type(new_keys) == types.MappingProxyType:  # noqa: E721
                # meta class objects
                if orig != new:
                    return False
                orig_keys = dict(orig_keys)
                new_keys = dict(new_keys)
                orig_keys = {k: v for k, v in orig_keys.items() if not k.startswith("__")}
                new_keys = {k: v for k, v in new_keys.items() if not k.startswith("__")}

            if superset_obj:
                # allow new object to be a superset of the original object
                return all(k in new_keys and comparator(v, new_keys[k], superset_obj) for k, v in orig_keys.items())

            if isinstance(orig, ast.AST):
                orig_keys = {k: v for k, v in orig.__dict__.items() if k != "parent"}
                new_keys = {k: v for k, v in new.__dict__.items() if k != "parent"}
            return comparator(orig_keys, new_keys, superset_obj)

        if type(orig) in {types.BuiltinFunctionType, types.BuiltinMethodType}:
            return new == orig
        if str(type(orig)) == "<class 'object'>":
            return True
        # TODO : Add other types here
        logger.warning(f"Unknown comparator input type: {type(orig)}")
        sentry_sdk.capture_exception(RuntimeError(f"Unknown comparator input type: {type(orig)}"))
        return False  # noqa: TRY300
    except RecursionError as e:
        logger.error(f"RecursionError while comparing objects: {e}")
        sentry_sdk.capture_exception(e)
        return False
    except Exception as e:
        logger.error(f"Error while comparing objects: {e}")
        sentry_sdk.capture_exception(e)
        return False
