from common import mpi_rank, assert_mpi_env, random_distributed

import numpy as np
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env(size=4)
test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}


@pytest.mark.parametrize("shape,dtype", [
    ((235, 326), np.float64),
    ((457, 26), np.complex128),
])
def test_svd(shape, dtype):
    """Test SVD computation of a distributed matrix"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        u_distributed, s, vt_distributed = rt.svd(a_distributed)
        u = u_distributed.to_global_array(rank=0)
        vt = vt_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_allclose(a, u * s[None, :] @ vt, err_msg="u @ s @ vt = a", atol=1e-10)
            np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[1]), err_msg="u not orthonormal", atol=1e-10)
            np.testing.assert_allclose(vt @ vt.conj().T, np.eye(vt.shape[0]), err_msg="v not orthonormal", atol=1e-10)
