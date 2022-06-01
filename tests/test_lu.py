from common import mpi_rank, assert_mpi_env, random_distributed

import numpy as np
import scipy.linalg as la
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env()
test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}


@pytest.mark.parametrize("size,dtype", [
    (357, np.float64),
    (478, np.complex128),
])
def test_lu(size, dtype):
    """Test the LU factorization of a distributed matrix"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed((size, size), dtype)
        a_lu_distributed, a_pivot_distributed = rt.lu(a_distributed)
        a_lu = a_lu_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            p, l, u = la.lu(a)
            np.testing.assert_allclose(a_lu, l + u - np.eye(size), atol=1e-10)
