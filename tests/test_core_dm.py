from common import mpi_rank, mpi_comm, assert_mpi_env, random_distributed

import numpy as np
import pytest

from scalapy import core

assert_mpi_env(size=4)
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}


def test_dm_init():
    with core.shape_context(**test_context):
        dm = core.DistributedMatrix([5, 5])

        # Check global shape
        assert dm.shape == (5, 5)

        # Check block size
        assert dm.block_shape == test_context["block_shape"]

        # Check local shape
        shape_list = [(3, 3), (3, 2), (2, 3), (2, 2)]
        assert dm.local_shape == shape_list[mpi_rank]


def test_dm_load_5x5():
    """Test that a 5x5 DistributedMatrix is loaded correctly"""
    with core.shape_context(**test_context):
        distributed_a, a = random_distributed((5, 5), float)
        blocks = [a[:3, :3], a[:3, 3:], a[3:, :3], a[3:, 3:]]
        dm = core.DistributedMatrix.from_global_array(a)

        np.testing.assert_equal(dm.local_array, blocks[mpi_rank])


@pytest.mark.parametrize("g_shape,b_shape", [
    ((3, 3), (5, 5)),
    ((132, 109), (21, 11)),
    ((5631, 5), (3, 2)),
    ((81, 81), (90, 2)),
])
def test_dm_cycle(g_shape, b_shape):
    with core.shape_context(**test_context):
        nr, nc = g_shape
        arr = np.arange(nr*nc, dtype=np.float64).reshape(nr, nc)

        dm = core.DistributedMatrix.from_global_array(arr, block_shape=b_shape)
        np.testing.assert_equal(dm.to_global_array(), arr)


def test_dm_redistribute():
    """Test redistribution of matrices with different blocking and process grids"""
    with core.shape_context(**test_context):

        # Generate matrix
        _, a = random_distributed((5, 5), float)

        # Create DistributedMatrix
        dm3x3 = core.DistributedMatrix.from_global_array(a, block_shape=[3, 3])
        dm2x2 = core.DistributedMatrix.from_global_array(a, block_shape=[2, 2])

        rd2x2 = dm3x3.redistribute(block_shape=[2, 2])

        np.testing.assert_equal(dm2x2.local_array, rd2x2.local_array)

        pc2 = core.GridContext([4, 1], comm=mpi_comm)

        dmpc2 = core.DistributedMatrix.from_global_array(a, block_shape=[1, 1], context=pc2)
        rdpc2 = dm3x3.redistribute(block_shape=[1, 1], context=pc2)

        np.testing.assert_equal(dmpc2.local_array, rdpc2.local_array)


def test_sum(shape=(13, 17), dtype=np.float64):
    """Test sum over elements"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)

        np.testing.assert_allclose(a_distributed.sum(0), a.sum(0))
        np.testing.assert_allclose(a_distributed.sum(1), a.sum(1))
        np.testing.assert_allclose(a_distributed.sum(), a.sum())
