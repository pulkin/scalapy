from common import assert_mpi_env, random_lr_distributed

import numpy as np
import scipy.linalg as la
import pytest

from scalapy import core
import scalapy.routines as rt


assert_mpi_env(size=4)
# TODO: tests fail with the default context
test_context = {"block_shape": (3, 3)}


@pytest.mark.parametrize("shape,dtype,rank,pinv", [
    ((9, 23), np.float64, 9, rt.pinv),
    ((7, 9), np.complex128, 7, rt.pinv),
    ((9, 23), np.float64, 9, rt.pinv2),
    ((7, 9), np.complex128, 7, rt.pinv2),
    ((17, 14), np.complex128, 4, rt.pinv2),
])
def test_pinv(shape, dtype, rank, pinv):
    """Test pseudo-inverse computation of a real double precision distributed matrix"""
    with core.shape_context(**test_context):
        a_distributed, a = random_lr_distributed(shape, dtype, rank=rank)
        a_pinv_distributed = pinv(a_distributed)
        a_pinv = a_pinv_distributed.to_global_array()[:shape[1]]

        np.testing.assert_allclose(a, a @ a_pinv @ a, err_msg="a = a @ p @ a", atol=1e-10)
        np.testing.assert_allclose(a_pinv, a_pinv @ a @ a_pinv, err_msg="p = p @ a @ p", atol=1e-10)
        np.testing.assert_allclose(a_pinv, la.pinv(a))
