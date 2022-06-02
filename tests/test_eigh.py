from common import mpi_rank, assert_mpi_env, random_hermitian_distributed, random_hp_distributed

import numpy as np
import pytest

from scalapy import core
import scalapy.routines as rt

assert_mpi_env(size=4)
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}
multiple_shape_parameters = pytest.mark.parametrize("size,dtype,atol", [
    (269, np.float32, 1e-3),
    (270, np.float64, 1e-7),
    (271, np.complex64, 1e-3),
    (272, np.complex128, 1e-7),
])


@multiple_shape_parameters
def test_eigh(size, dtype, atol):
    """Eigenvalue problem"""
    with core.shape_context(**test_context):
        a_distributed, a = random_hermitian_distributed((size, size), dtype)

        vals, vecs_distributed = rt.eigh(a_distributed)
        vecs = vecs_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_allclose(a @ vecs - vecs * vals[None, :], 0, err_msg=f"A @ v - val v = 0", atol=atol)
            np.testing.assert_allclose(vecs.conj().T @ vecs, np.eye(size), err_msg=f"v.T @ v = I", atol=atol)


@multiple_shape_parameters
def test_eigh_generalized(size, dtype, atol):
    """Generalized eigenvalue problem"""
    with core.shape_context(**test_context):
        a_distributed, a = random_hermitian_distributed((size, size), dtype)
        b_distributed, b = random_hp_distributed((size, size), dtype)

        vals, vecs_distributed = rt.eigh(a_distributed, b_distributed)
        vecs = vecs_distributed.to_global_array(rank=0)

        if mpi_rank == 0:
            np.testing.assert_allclose(a @ vecs - b @ vecs * vals[None, :], 0, err_msg=f"A @ v - val B @ v = 0",
                                       atol=atol)
            np.testing.assert_allclose(vecs.conj().T @ b @ vecs, np.eye(size), err_msg=f"v.T @ b @ v = I", atol=atol)


def test_eigh_generalized_fail():
    """Tests failing gracefully for non-matching block setup"""
    with core.shape_context(**test_context):
        size = 270
        dtype = np.complex128
        a_distributed, a = random_hermitian_distributed((size, size), dtype)

        with core.shape_context(gridshape=(2, 2), block_shape=(4, 4)):
            b_distributed, b = random_hermitian_distributed((size, size), dtype)

        with pytest.raises(AssertionError):
            rt.eigh(a_distributed, b_distributed)
