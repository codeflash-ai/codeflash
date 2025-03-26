import ast
import datetime
import decimal
import enum
import math
import re
import types
from typing import Any

import sentry_sdk

from codeflash.cli_cmds.console import logger

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
try:
    import sqlalchemy

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
try:
    import scipy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyrsistent

    HAS_PYRSISTENT = True
except ImportError:
    HAS_PYRSISTENT = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def comparator(orig: Any, new: Any, superset_obj=False) -> bool:
    """Compare two objects for equality recursively. If superset_obj is True, the new object is allowed to have more keys than the original object. However, the existing keys/values must be equivalent."""
    try:
        if type(orig) is not type(new):
            type_obj = type(orig)
            new_type_obj = type(new)
            # distinct type objects are created at runtime, even if the class code is exactly the same, so we can only compare the names
            if type_obj.__name__ != new_type_obj.__name__ or type_obj.__qualname__ != new_type_obj.__qualname__:
                return False
        if isinstance(orig, (list, tuple)):
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
            ),
        ):
            return orig == new
        if isinstance(orig, float):
            if math.isnan(orig) and math.isnan(new):
                return True
            return math.isclose(orig, new)
        if isinstance(orig, BaseException):
            # if str(orig) != str(new):
            #     return False
            # compare the attributes of the two exception objects to determine if they are equivalent.
            orig_dict = {k: v for k, v in orig.__dict__.items() if not k.startswith("_")}
            new_dict = {k: v for k, v in new.__dict__.items() if not k.startswith("_")}
            return comparator(orig_dict, new_dict, superset_obj)

        if HAS_SQLALCHEMY:
            try:
                insp = sqlalchemy.inspection.inspect(orig)
                insp = sqlalchemy.inspection.inspect(new)
                orig_keys = orig.__dict__
                new_keys = new.__dict__
                for key in list(orig_keys.keys()):
                    if key.startswith("_"):
                        continue
                    if key not in new_keys or not comparator(orig_keys[key], new_keys[key], superset_obj):
                        return False
                return True

            except sqlalchemy.exc.NoInspectionAvailable:
                pass
        # scipy condition because dok_matrix type is also a instance of dict, but dict comparison doesn't work for it
        if isinstance(orig, dict) and not (HAS_SCIPY and isinstance(orig, scipy.sparse.spmatrix)):
            if superset_obj:
                return all(k in new.keys() and comparator(v, new[k], superset_obj) for k, v in orig.items())
            if len(orig) != len(new):
                return False
            for key in orig:
                if key not in new:
                    return False
                if not comparator(orig[key], new[key], superset_obj):
                    return False
            return True

        if HAS_NUMPY and isinstance(orig, np.ndarray):
            if orig.dtype != new.dtype:
                return False
            if orig.shape != new.shape:
                return False
            try:
                return np.allclose(orig, new, equal_nan=True)
            except Exception:
                # fails at "ufunc 'isfinite' not supported for the input types"
                return np.all([comparator(x, y, superset_obj) for x, y in zip(orig, new)])

        if HAS_NUMPY and isinstance(orig, (np.floating, np.complex64, np.complex128)):
            return np.isclose(orig, new)

        if HAS_NUMPY and isinstance(orig, (np.integer, np.bool_, np.byte)):
            return orig == new

        if HAS_SCIPY and isinstance(orig, scipy.sparse.spmatrix):
            if orig.dtype != new.dtype:
                return False
            if orig.get_shape() != new.get_shape():
                return False
            return (orig != new).nnz == 0

        if HAS_PANDAS and isinstance(
            orig, (pandas.DataFrame, pandas.Series, pandas.Index, pandas.Categorical, pandas.arrays.SparseArray)
        ):
            return orig.equals(new)

        if HAS_PANDAS and isinstance(orig, (pandas.CategoricalDtype, pandas.Interval, pandas.Period)):
            return orig == new
        if HAS_PANDAS and pandas.isna(orig) and pandas.isna(new):
            return True

        # This should be at the end of all numpy checking
        try:
            if HAS_NUMPY and np.isnan(orig):
                return np.isnan(new)
        except Exception:
            pass
        try:
            if HAS_NUMPY and np.isinf(orig):
                return np.isinf(new)
        except Exception:
            pass

        if HAS_TORCH and isinstance(orig, torch.Tensor):
            if orig.dtype != new.dtype:
                return False
            if orig.shape != new.shape:
                return False
            if orig.requires_grad != new.requires_grad:
                return False
            if orig.device != new.device:
                return False
            return torch.allclose(orig, new, equal_nan=True)

        if HAS_PYRSISTENT and isinstance(
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
        except Exception:
            pass

        # For class objects
        if hasattr(orig, "__dict__") and hasattr(new, "__dict__"):
            orig_keys = orig.__dict__
            new_keys = new.__dict__
            if type(orig_keys) == types.MappingProxyType and type(new_keys) == types.MappingProxyType:
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

        if type(orig) in [types.BuiltinFunctionType, types.BuiltinMethodType]:
            return new == orig
        if str(type(orig)) == "<class 'object'>":
            return True
        # TODO : Add other types here
        logger.warning(f"Unknown comparator input type: {type(orig)}")
        return False
    except RecursionError as e:
        logger.error(f"RecursionError while comparing objects: {e}")
        sentry_sdk.capture_exception(e)
        return False
    except Exception as e:
        logger.error(f"Error while comparing objects: {e}")
        sentry_sdk.capture_exception(e)
        return False
