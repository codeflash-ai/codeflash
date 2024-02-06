import datetime
import logging
import math
from typing import Any

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


def comparator(orig: Any, new: Any) -> bool:
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

    if isinstance(orig, (str, int, bool, complex, type(None))):
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
        except Exception as e:
            # fails at "ufunc 'isfinite' not supported for the input types"
            return np.all([comparator(x, y) for x, y in zip(orig, new)])

    if HAS_NUMPY and isinstance(orig, (np.floating, np.complex64, np.complex128)):
        return np.isclose(orig, new)

    if HAS_NUMPY and isinstance(orig, (np.integer, np.bool_, np.byte)):
        return orig == new

    # This should be at the end of all numpy checking
    try:
        if HAS_NUMPY and np.isnan(orig):
            return np.isnan(new)
    except Exception as e:
        pass
    try:
        if HAS_NUMPY and np.isinf(orig):
            return np.isinf(new)
    except Exception as e:
        pass

    if HAS_SCIPY and isinstance(orig, scipy.sparse.spmatrix):
        return (orig != new).nnz == 0

    if isinstance(orig, (datetime.datetime, datetime.date, datetime.timedelta)):
        return orig == new

    # If the object passed has a user defined __eq__ method, use that
    # This could fail if the user defined __eq__ is defined with cython
    try:
        if hasattr(orig, "__eq__") and str(type(orig.__eq__)) == "<class 'method'>":
            return orig == new
    except Exception as e:
        pass

    # TODO : Add other types here
    logging.warning(f"Unknown comparator input type: {type(orig)}")
    return True
