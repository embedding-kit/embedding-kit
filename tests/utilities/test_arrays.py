"""Unit tests for ``embkit.utilities.arrays``."""

import unittest

import numpy as np
import pandas as pd
import torch

from embkit.utilities.arrays import to_numpy


class TestToNumpy(unittest.TestCase):
    def test_reduced_precision_tensors_upcast(self):
        for dt in (torch.bfloat16, torch.float16):
            out = to_numpy(torch.ones(2, 3, dtype=dt))
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.array_equal(out, np.ones((2, 3))))

    def test_tensor_with_grad_keeps_dtype(self):
        t = torch.ones(2, dtype=torch.float64, requires_grad=True)
        self.assertEqual(to_numpy(t).dtype, np.float64)

    def test_requested_dtype_and_non_tensors(self):
        self.assertEqual(to_numpy(torch.ones(2, dtype=torch.bfloat16), dtype=float).dtype, np.float64)
        self.assertEqual(to_numpy([1, 2]).tolist(), [1, 2])
        df = pd.DataFrame({"a": [1.0, 2.0]})
        self.assertEqual(to_numpy(df).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
