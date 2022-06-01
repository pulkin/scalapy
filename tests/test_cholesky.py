from common import mpi_rank, assert_mpi_env, random_pd_distributed

import numpy as np
import scipy.linalg as la
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env()
test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}


@pytest.mark.parametrize("size,dtype,lower", [
    (317, np.float64, False),
    (342, np.complex128, True),
])
def test_cholesky(size, dtype, lower):
    """Test the Cholesky decomposition"""
    with core.shape_context(**test_context):
        a_distributed, a = random_pd_distributed((size, size), dtype)
        u_distributed = rt.cholesky(a_distributed, lower=lower)
        u = u_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            ref = la.cholesky(a, lower=lower)
            np.testing.assert_allclose(u, ref, rtol=1e-4, atol=1e-6)
