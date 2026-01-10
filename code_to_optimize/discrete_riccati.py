"""
Utility functions used in CompEcon

Based routines found in the CompEcon toolbox by Miranda and Fackler.

References
----------
Miranda, Mario J, and Paul L Fackler. Applied Computational Economics
and Finance, MIT Press, 2002.

"""
from functools import reduce
import numpy as np
import torch
import jax
import tensorflow as tf

def _gridmake2(x1, x2):
    """
    Expands two vectors (or matrices) into a matrix where rows span the
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
        return np.column_stack([np.tile(x1, x2.shape[0]),
                               np.repeat(x2, x1.shape[0])])
    elif x1.ndim > 1 and x2.ndim == 1:
        first = np.tile(x1, (x2.shape[0], 1))
        second = np.repeat(x2, x1.shape[0])
        return np.column_stack([first, second])
    else:
        raise NotImplementedError("Come back here")


def _gridmake2_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of _gridmake2.

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
    elif x1.dim() > 1 and x2.dim() == 1:
        # tile x1 along first dimension
        first = x1.tile(x2.shape[0], 1)
        second = x2.repeat_interleave(x1.shape[0])
        return torch.column_stack([first, second])
    else:
        raise NotImplementedError("Come back here")
