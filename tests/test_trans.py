from common import mpi_rank, assert_mpi_env, random_distributed

import numpy as np
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env()
test_context = {"gridshape": (2, 2), "block_shape": (16, 16)}


def h_conj(a):
    """Hermitian conjugate"""
    return a.conj().T


@pytest.mark.parametrize("shape,dtype,np_op,spy_op", [
    ((354, 231), np.float64, np.transpose, rt.transpose),
    ((379, 432), np.complex128, np.transpose, rt.transpose),
    ((245, 357), np.float64, np.conj, rt.conj),
    ((630, 62), np.complex128, np.conj, rt.conj),
    ((245, 357), np.float64, h_conj, rt.hconj),
    ((630, 62), np.complex128, h_conj, rt.hconj)
])
def test_trans(shape, dtype, np_op, spy_op):
    """Test transpose of a distributed matrix"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        at_distributed = spy_op(a_distributed)
        at = at_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_equal(np_op(a), at)
