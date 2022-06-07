import pytest

from common import assert_mpi_env, random_distributed

import numpy as np

from scalapy import core
import scalapy.routines as rt

assert_mpi_env(size=4)
test_context = {"block_shape": (16, 16)}


@pytest.mark.parametrize("shape,dtype", [
    ((357, 478), np.float64),
    ((478, 357), np.complex128),
    ((357, 357), np.complex128),
])
def test_qr(shape, dtype):
    """Test the QR factorization"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)
        q_distributed, r_distributed = rt.qr(a_distributed)
        q = q_distributed.to_global_array()
        r = r_distributed.to_global_array()

        _q, _r = np.linalg.qr(a)
        np.testing.assert_allclose(q.conj().T @ q, np.eye(q.shape[1]), err_msg="orthonormal", atol=1e-14)
        np.testing.assert_allclose(r, np.triu(r), err_msg="upper-triangular")
        np.testing.assert_allclose(q @ r, a, err_msg="Q @ R = A", atol=1e-14)
