from common import mpi_rank, mpi_comm, assert_mpi_env, random_distributed, random

import numpy as np
import pytest

from scalapy import core

assert_mpi_env(size=4)
test_context = {"block_shape": (3, 3)}


def np_eye_like(a, k=0):
    n, m = a.shape
    return np.eye(n, m, k, dtype=a.dtype)


def np_identity_like(a):
    n, m = a.shape
    assert n == m
    return np.identity(n, dtype=a.dtype)


def test_zeros(shape=(17, 8)):
    with core.shape_context(**test_context):
        np.testing.assert_equal(core.zeros(shape).numpy(), np.zeros(shape))


@pytest.mark.parametrize("n,m,k", [
    (17, 8, 0),
    (14, 15, 1),
    (15, 14, 1),
    (15, 15, -1),
    (17, 18, 19),
])
def test_eye(n, m, k):
    with core.shape_context(**test_context):
        np.testing.assert_equal(core.eye(n, m, k).numpy(), np.eye(n, m, k))


def test_identity(n=12):
    with core.shape_context(**test_context):
        np.testing.assert_equal(core.identity(n).numpy(), np.identity(n))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("np_op,spy_op,shape", [
    (np.zeros_like, core.zeros_like, (17, 8)),
    (np.transpose, core.transpose, (17, 8)),
    (np.conj, core.conj, (17, 8)),
    (np_eye_like, core.eye_like, (17, 8)),
    (np_identity_like, core.identity_like, (10, 10)),
    (np.absolute, core.absolute, (15, 20))
])
def test_mat_op(dtype, np_op, spy_op, shape):
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        np.testing.assert_equal(spy_op(a_distributed).numpy(), np_op(a))
