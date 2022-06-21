from common import assert_mpi_env, random_distributed

import numpy as np
import scipy.linalg as la
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env(size=4)
test_context = {"block_shape": (16, 16)}


@pytest.mark.parametrize("size,dtype", [
    (353, np.float64),
    (521, np.complex128),
])
def test_inv(size, dtype):
    """Test inverse computation of a distributed matrix"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed((size, size), dtype)
        a_inv_distributed = rt.inv(a_distributed)
        a_inv = a_inv_distributed.numpy()

        np.testing.assert_equal(a_distributed.numpy(), a, err_msg="input matrix changed")
        np.testing.assert_allclose(a_inv @ a, np.eye(size), err_msg="a_inv @ a = I", atol=1e-9)
        np.testing.assert_allclose(a_inv, la.inv(a), err_msg="a_inv_spy = a_inv_npy", atol=1e-9)
