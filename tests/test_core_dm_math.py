from common import mpi_rank, mpi_comm, assert_mpi_env, random_distributed, random

import numpy as np
import pytest

from operator import add, sub, mul, truediv

from scalapy import core

assert_mpi_env()
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}
multiple_shape_parameters = pytest.mark.parametrize("shape,dtype", [
    ((4, 13), np.float32),
    ((5, 14), np.float64),
    ((12, 6), np.complex64),
    ((7, 7), np.complex128)
])
multiple_operators = pytest.mark.parametrize("op", [add, sub, mul, truediv])


@multiple_shape_parameters
@multiple_operators
def test_dm_dm(shape, dtype, op):
    """Test operators on two distributed matrices"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        b_distributed, b = random_distributed(shape, dtype)

        ab_distributed = op(a_distributed, b_distributed)
        ab = ab_distributed.to_global_array(rank=0)
        a_ = a_distributed.to_global_array(rank=0)
        b_ = b_distributed.to_global_array(rank=0)
        if mpi_rank == 0:
            np.testing.assert_equal(a, a_, err_msg="a changed")
            np.testing.assert_equal(b, b_, err_msg="b changed")
            np.testing.assert_equal(op(a, b), ab, err_msg="op(a, b)")

        x_distributed, _ = random_distributed((shape[0], shape[1] + 1), dtype)
        with pytest.raises(ValueError):
            op(a_distributed, x_distributed)


@multiple_shape_parameters
@multiple_operators
def test_dm_scalar(shape, dtype, op):
    """Test operators on a distributed matrix and a scalar"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        scalar = 5.678

        a_alpha_distributed = op(a_distributed, scalar)
        a_alpha = a_alpha_distributed.to_global_array(rank=0)
        a_ = a_distributed.to_global_array(rank=0)
        if mpi_rank == 0:
            np.testing.assert_equal(a, a_, err_msg="a changed")
            np.testing.assert_equal(op(a, scalar), a_alpha, err_msg="op(a, scalar)")

        # reverse with a scalar
        a_alpha_distributed = op(scalar, a_distributed)
        a_alpha = a_alpha_distributed.to_global_array(rank=0)
        a_ = a_distributed.to_global_array(rank=0)
        if mpi_rank == 0:
            np.testing.assert_equal(a, a_, err_msg="a changed")
            np.testing.assert_equal(op(scalar, a), a_alpha, err_msg="op(scalar, a)")


@multiple_shape_parameters
def test_dm_np(shape, dtype):
    """ij,j->ij multiplication"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        v = random(shape[1], dtype)
        mpi_comm.Bcast(v, root=0)

        av_distributed = a_distributed * v
        av = av_distributed.to_global_array(rank=0)
        a_ = a_distributed.to_global_array(rank=0)
        if mpi_rank == 0:
            np.testing.assert_equal(a, a_, err_msg="a changed")
            np.testing.assert_equal(a * v[None, :], av, err_msg="a*v")

        vx = random(shape[1] + 1, dtype)
        mpi_comm.Bcast(vx, root=0)
        with pytest.raises(ValueError):
            a_distributed * vx


@multiple_operators
def test_dm_dm_fail(op):
    """Tests failing gracefully for non-matching block setup"""
    with core.shape_context(**test_context):
        shape = 6, 8
        dtype = np.complex128
        a_distributed, a = random_distributed(shape, dtype)

        with core.shape_context(gridshape=(2, 2), block_shape=(4, 4)):
            b_distributed, b = random_distributed(shape, dtype)

        with pytest.raises(AssertionError):
            op(a_distributed, b_distributed)


@multiple_shape_parameters
def test_neg(shape, dtype):
    """Test operators on two distributed matrices"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        n_distributed = - a_distributed
        n = n_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_equal(n, -a)


@multiple_shape_parameters
def test_dot(shape, dtype):
    """matrix-matrix multiplication (raw version): alpha * A @ B + beta * C"""
    with core.shape_context(**test_context):
        m, n = shape
        k = (m + n) // 2  # does not mean anything particular
        alpha = 1.1
        beta = 8.0

        a_distributed, a = random_distributed((n, k), dtype)
        b_distributed, b = random_distributed((k, m), dtype)
        c_distributed, c = random_distributed((n, m), dtype)

        core.dot_mat_mat(a_distributed, b_distributed, alpha=alpha, beta=beta, out=c_distributed)
        result = c_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_allclose(alpha * a @ b + beta * c, result,
                                       atol=1e-5 if dtype in (np.float32, np.complex64) else 1e-12)


@multiple_shape_parameters
def test_dot_2(shape, dtype):
    """matrix-matrix multiplication (dot version): A @ B"""
    with core.shape_context(**test_context):
        m, n = shape
        k = (m + n) // 2  # does not mean anything particular

        a_distributed, a = random_distributed((n, k), dtype)
        b_distributed, b = random_distributed((k, m), dtype)
        ab_distributed = a_distributed @ b_distributed

        ab = ab_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_allclose(a @ b, ab, atol=1e-5 if dtype in (np.float32, np.complex64) else 1e-12)
