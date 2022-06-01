from common import mpi_rank, assert_mpi_env, random_distributed

import numpy as np
import pytest

from scalapy import core


assert_mpi_env()
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}


@pytest.mark.parametrize("dtype,fw_rank,bw_rank", [
    (np.float64, 0, 1),
    (np.complex128, 1, 2),
])
def test_np2self(dtype, fw_rank, bw_rank):
    """Test copy a numpy array to a section of the distributed matrix and vice versa"""
    with core.shape_context(**test_context):

        am, an = 13, 5
        host_shape = 39, 23
        srow, scol = 3, 12

        a = np.arange(am * an, dtype=np.float64).reshape(am, an)
        a = np.asfortranarray(a)

        m_distributed, m = random_distributed(host_shape, dtype)
        m_distributed = m_distributed.np2self(a, srow, scol, rank=fw_rank)
        a1 = m_distributed.self2np(srow, am, scol, an, rank=bw_rank)

        if mpi_rank == bw_rank:
            np.testing.assert_equal(a, a1)
