"""
Conversion of array-like inputs to NumPy.

The analysis functions in embkit (metrics, latent geometry, spatial and
preprocessing helpers) compute on NumPy arrays, but their inputs often come
straight out of a model as torch tensors, possibly in a reduced-precision dtype
such as ``bfloat16``. NumPy has no ``bfloat16`` type, so ``np.asarray`` on such
a tensor raises. :func:`to_numpy` checks the input type and converts safely.
"""
from typing import Any, Optional

import numpy as np
import torch

_REDUCED_PRECISION = (torch.bfloat16, torch.float16)


def to_numpy(x: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """
    Return ``x`` as a NumPy array.

    Torch tensors are detached and moved to the CPU first. ``bfloat16`` and
    ``float16`` tensors are upcast to ``float32``, since NumPy cannot hold
    ``bfloat16``. Anything else (arrays, lists, DataFrames) goes through
    ``np.asarray``.

    Args:
        x: Array-like input or torch tensor.
        dtype: Optional NumPy dtype for the result.

    Returns:
        A NumPy array.
    """
    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype in _REDUCED_PRECISION:
            x = x.float()
        x = x.cpu().numpy()
    return np.asarray(x, dtype=dtype)
