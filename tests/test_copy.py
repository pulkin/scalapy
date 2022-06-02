from common import assert_mpi_env, random_distributed

import numpy as np
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env(size=4)
test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}


@pytest.mark.parametrize("shape,dtype,spy_copy,npy_copy", [
    ((354, 231), np.float64, rt.copy, np.copy),
    ((379, 432), np.complex128, rt.copy, np.copy),
    ((176, 212), np.float64, rt.triu, np.triu),
    ((111, 102), np.float64, rt.tril, np.tril),
])
def test_copy_d(shape, dtype, spy_copy, npy_copy):
    """Test copying a matrix or its part"""
    with core.shape_context(**test_context):

        a_distributed, a = random_distributed(shape, dtype)
        b_distributed = spy_copy(a_distributed)
        b = b_distributed.to_global_array()

        np.testing.assert_equal(npy_copy(a), b)
