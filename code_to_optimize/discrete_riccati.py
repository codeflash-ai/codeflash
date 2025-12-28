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


@njit(cache=True)
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
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        out = np.empty((n1 * n2, 2), dtype=x1.dtype)
        for i in range(n2):
            for j in range(n1):
                out[i * n1 + j, 0] = x1[j]
                out[i * n1 + j, 1] = x2[i]
        return out
    if x1.ndim > 1 and x2.ndim == 1:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        n_features = x1.shape[1]
        out = np.empty((n1 * n2, n_features + 1), dtype=x1.dtype)
        for i in range(n2):
            for j in range(n1):
                idx = i * n1 + j
                for k in range(n_features):
                    out[idx, k] = x1[j, k]
                out[idx, n_features] = x2[i]
        return out
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
