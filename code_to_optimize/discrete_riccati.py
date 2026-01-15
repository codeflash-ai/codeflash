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
        return np.column_stack([np.tile(x1, x2.shape[0]), np.repeat(x2, x1.shape[0])])
    if x1.ndim > 1 and x2.ndim == 1:
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
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        # result dtype should follow PyTorch promotion rules (matches column_stack)
        out_dtype = torch.promote_types(x1.dtype, x2.dtype)
        out_device = x1.device
        # preallocate final tensor and fill via shaped views to avoid temporaries
        out = torch.empty((n1 * n2, 2), dtype=out_dtype, device=out_device)
        # cast inputs only if necessary
        x1c = x1 if x1.dtype == out_dtype else x1.to(out_dtype)
        x2c = x2 if x2.dtype == out_dtype else x2.to(out_dtype)
        # view as (n2, n1, 2) so that arr[:, :, 0] holds x1 rows repeated per x2
        arr = out.view(n2, n1, 2)
        arr[:, :, 0] = x1c.unsqueeze(0).expand(n2, n1)
        arr[:, :, 1] = x2c.unsqueeze(1).expand(n2, n1)
        return out
    if x1.dim() > 1 and x2.dim() == 1:
        # tile x1 along first dimension
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        m = x1.shape[1]
        out_dtype = torch.promote_types(x1.dtype, x2.dtype)
        out_device = x1.device
        # final shape (n1 * n2, m + 1)
        out = torch.empty((n1 * n2, m + 1), dtype=out_dtype, device=out_device)
        # cast inputs only if necessary
        x1c = x1 if x1.dtype == out_dtype else x1.to(out_dtype)
        x2c = x2 if x2.dtype == out_dtype else x2.to(out_dtype)
        # view as (n2, n1, m+1); first m columns get repeated x1, last column gets repeated x2
        arr = out.view(n2, n1, m + 1)
        arr[:, :, :m] = x1c.unsqueeze(0).expand(n2, n1, m)
        arr[:, :, m] = x2c.unsqueeze(1).expand(n2, n1)
        return out
    raise NotImplementedError("Come back here")
