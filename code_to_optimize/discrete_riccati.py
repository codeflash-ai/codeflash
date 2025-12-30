"""Utility functions used in CompEcon

Based routines found in the CompEcon toolbox by Miranda and Fackler.

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.

"""

from functools import reduce

import numpy as np
import torch
from numba import njit


def ckron(*arrays):
    """Repeatedly applies the np.kron function to an arbitrary number of
    input arrays

    Parameters
    ----------
    *arrays : tuple/list of np.ndarray

    Returns
    -------
    out : np.ndarray
          The result of repeated kronecker products.

    Notes
    -----
    Based of original function `ckron` in CompEcon toolbox by Miranda
    and Fackler.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational
    Economics and Finance, MIT Press, 2002.

    """
    return reduce(np.kron, arrays)


def gridmake(*arrays):
    """Expands one or more vectors (or matrices) into a matrix where rows span the
    cartesian product of combinations of the input arrays. Each column of the
    input arrays will correspond to one column of the output matrix.

    Parameters
    ----------
    *arrays : tuple/list of np.ndarray
              Tuple/list of vectors to be expanded.

    Returns
    -------
    out : np.ndarray
          The cartesian product of combinations of the input arrays.

    Notes
    -----
    Based of original function ``gridmake`` in CompEcon toolbox by
    Miranda and Fackler

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
    and Finance, MIT Press, 2002.

    """
    if all([i.ndim == 1 for i in arrays]):
        d = len(arrays)
        if d == 2:
            out = _gridmake2(*arrays)
        else:
            out = _gridmake2(arrays[0], arrays[1])
            for arr in arrays[2:]:
                out = _gridmake2(out, arr)

        return out
    raise NotImplementedError("Come back here")


def _gridmake2(x1, x2):
    """Expands two vectors (or matrices) into a matrix where rows span the
    cartesian product of combinations of the input arrays. Each column of the
    input arrays will correspond to one column of the output matrix.

    Parameters
    ----------
    x1 : np.ndarray
         First vector to be expanded.

    x2 : np.ndarray
         Second vector to be expanded.

    Returns
    -------
    out : np.ndarray
          The cartesian product of combinations of the input arrays.

    Notes
    -----
    Based of original function ``gridmake2`` in CompEcon toolbox by
    Miranda and Fackler.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
    and Finance, MIT Press, 2002.

    """
    if x1.ndim == 1 and x2.ndim == 1:
        # Determine output dtype using numpy's type promotion
        out_dtype = np.result_type(x1.dtype, x2.dtype)
        return _gridmake2_optimized_1d_1d(x1, x2, out_dtype)
    if x1.ndim > 1 and x2.ndim == 1:
        # For 2D case, use optimized version
        if x1.ndim == 2:
            out_dtype = np.result_type(x1.dtype, x2.dtype)
            return _gridmake2_optimized_2d_1d(x1, x2, out_dtype)
        # For 3D or higher, fall back to original implementation
        # This will raise the appropriate error
        first = np.tile(x1, (x2.shape[0], 1))
        second = np.repeat(x2, x1.shape[0])
        return np.column_stack([first, second])
    raise NotImplementedError("Come back here")


def _gridmake2_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """PyTorch version of _gridmake2.

    Expands two tensors into a matrix where rows span the cartesian product
    of combinations of the input tensors. Each column of the input tensors
    will correspond to one column of the output matrix.

    Parameters
    ----------
    x1 : torch.Tensor
         First tensor to be expanded.

    x2 : torch.Tensor
         Second tensor to be expanded.

    Returns
    -------
    out : torch.Tensor
          The cartesian product of combinations of the input tensors.

    Notes
    -----
    Based on original function ``gridmake2`` in CompEcon toolbox by
    Miranda and Fackler.

    References
    ----------
    Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
    and Finance, MIT Press, 2002.

    """
    if x1.dim() == 1 and x2.dim() == 1:
        # tile x1 by x2.shape[0] times, repeat_interleave x2 by x1.shape[0]
        first = x1.tile(x2.shape[0])
        second = x2.repeat_interleave(x1.shape[0])
        return torch.column_stack([first, second])
    if x1.dim() > 1 and x2.dim() == 1:
        # tile x1 along first dimension
        first = x1.tile(x2.shape[0], 1)
        second = x2.repeat_interleave(x1.shape[0])
        return torch.column_stack([first, second])
    raise NotImplementedError("Come back here")


@njit(cache=True)
def _gridmake2_optimized_1d_1d(x1: np.ndarray, x2: np.ndarray, out_dtype) -> np.ndarray:
    """Optimized version for 1D x 1D case."""
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    total = n1 * n2

    out = np.empty((total, 2), dtype=out_dtype)
    idx = 0
    for j in range(n2):
        for i in range(n1):
            out[idx, 0] = x1[i]
            out[idx, 1] = x2[j]
            idx += 1
    return out


@njit(cache=True)
def _gridmake2_optimized_2d_1d(x1: np.ndarray, x2: np.ndarray, out_dtype) -> np.ndarray:
    """Optimized version for 2D x 1D case."""
    rows = x1.shape[0]
    cols = x1.shape[1]
    n2 = x2.shape[0]
    out_rows = rows * n2

    out = np.empty((out_rows, cols + 1), dtype=out_dtype)
    for j in range(n2):
        for i in range(rows):
            for k in range(cols):
                out[j * rows + i, k] = x1[i, k]
            out[j * rows + i, cols] = x2[j]
    return out
