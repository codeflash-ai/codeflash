import datetime
import decimal
import enum
import logging
import math
import types
from typing import Any

import sentry_sdk

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


def comparator(orig: Any, new: Any) -> bool:
    try:
        if HAS_SQLALCHEMY:
            try:
                insp = sqlalchemy.inspection.inspect(orig)
                insp = sqlalchemy.inspection.inspect(new)
                orig_keys = orig.__dict__
                new_keys = new.__dict__
                for key in list(orig_keys.keys()):
                    if key.startswith("_"):
                        continue
                    if key not in new_keys or not comparator(orig_keys[key], new_keys[key]):
                        return False
                return True

            except sqlalchemy.exc.NoInspectionAvailable:
                pass
        if type(orig) != type(new):
            return False
        if isinstance(orig, (list, tuple)):
            if len(orig) != len(new):
                return False
            for elem1, elem2 in zip(orig, new):
                if not comparator(elem1, elem2):
                    return False
            return True

        if isinstance(
            orig,
            (
                str,
                int,
                bool,
                complex,
                type(None),
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
        # scipy condition because dok_matrix type is also a instance of dict, but dict comparison doesn't work for it
        if isinstance(orig, dict) and not (HAS_SCIPY and isinstance(orig, scipy.sparse.spmatrix)):
            if len(orig) != len(new):
                return False
            for key in orig:
                if key not in new:
                    return False
                if not comparator(orig[key], new[key]):
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
                return np.all([comparator(x, y) for x, y in zip(orig, new)])

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
            orig,
            (
                pandas.DataFrame,
                pandas.Series,
                pandas.Index,
                pandas.Categorical,
                pandas.arrays.SparseArray,
            ),
        ):
            return orig.equals(new)

        if HAS_PANDAS and isinstance(
            orig,
            (pandas.CategoricalDtype, pandas.Interval, pandas.Period),
        ):
            return orig == new

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

        if isinstance(
            orig,
            (datetime.datetime, datetime.date, datetime.timedelta, datetime.time, datetime.timezone),
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

            return comparator(orig_keys, new_keys)

        if type(orig) in [types.BuiltinFunctionType, types.BuiltinMethodType]:
            return new == orig
        if str(type(orig)) == "<class 'object'>":
            return True

        # TODO : Add other types here
        logging.warning(f"Unknown comparator input type: {type(orig)}")
        return False
    except RecursionError as e:
        logging.exception(f"RecursionError while comparing objects: {e}")
        sentry_sdk.capture_exception(e)
        return False
