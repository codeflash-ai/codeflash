"""Utility functions used in CompEcon

Based routines found in the CompEcon toolbox by Miranda and Fackler.

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.

"""

import numpy as np
import torch


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
        # Preallocate output and fill by blocks to avoid intermediate arrays
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        out = np.empty((n1 * n2, 2), dtype=np.result_type(x1, x2))
        # Fill blockwise: for each element of x2, copy x1 into the next block
        for i in range(n2):
            start = i * n1
            out[start : start + n1, 0] = x1
            out[start : start + n1, 1] = x2[i]
        return out
    if x1.ndim > 1 and x2.ndim == 1:
        # Preallocate output and fill by blocks to avoid intermediate arrays
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        ncols = x1.shape[1]
        out = np.empty((n1 * n2, ncols + 1), dtype=np.result_type(x1, x2))
        for i in range(n2):
            start = i * n1
            out[start : start + n1, :ncols] = x1
            out[start : start + n1, ncols] = x2[i]
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
