from common import mpi_rank, mpi_comm, assert_mpi_env, random_distributed, random

import numpy as np
from scipy import sparse
import pytest

from scalapy import core

assert_mpi_env(size=4)
test_context = {"block_shape": (3, 3)}
sparse_sample = sparse.csr_array((
    [1., 2., 3.],  # data
    [0, 4, 4],  # indices
    [0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # index pointers
), shape=(14, 15))


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
        dm = core.fromnumpy(a)

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

        with core.shape_context(block_shape=b_shape):
            dm = core.fromnumpy(arr)
        np.testing.assert_equal(dm.numpy(), arr)


def test_dm_redistribute():
    """Test redistribution of matrices with different blocking and process grids"""
    with core.shape_context(**test_context):

        # Generate matrix
        _, a = random_distributed((5, 5), float)

        # Create DistributedMatrix
        with core.shape_context(block_shape=(3, 3)):
            dm3x3 = core.fromnumpy(a)
        with core.shape_context(block_shape=(2, 2)):
            dm2x2 = core.fromnumpy(a)

        rd2x2 = dm3x3.redistribute(block_shape=[2, 2])

        np.testing.assert_equal(dm2x2.local_array, rd2x2.local_array)

        pc2 = core.GridContext([4, 1], comm=mpi_comm)

        with core.shape_context(block_shape=(1, 1), context=pc2):
            dmpc2 = core.fromnumpy(a)
        rdpc2 = dm3x3.redistribute(block_shape=[1, 1], context=pc2)

        np.testing.assert_equal(dmpc2.local_array, rdpc2.local_array)


def test_sum(shape=(13, 17), dtype=np.float64):
    """Test sum over elements"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)

        np.testing.assert_allclose(a_distributed.sum(0), a.sum(0))
        np.testing.assert_allclose(a_distributed.sum(1), a.sum(1))
        np.testing.assert_allclose(a_distributed.sum(), a.sum())


def test_from_sparse():
    """Tests conversion from sparse"""
    with core.shape_context(**test_context):
        a = sparse_sample.toarray()
        a_distributed = core.fromsparse_csr(sparse_sample)
        test = a_distributed.numpy()
        np.testing.assert_equal(test, a)


def test_auto():
    """Tests core.array"""
    with core.shape_context(**test_context):
        # numpy
        a = random((15, 16), float)
        mpi_comm.Bcast(a, 0)
        test = core.array(a)
        np.testing.assert_equal(test.numpy(), a)

        # sparse
        test = core.array(sparse_sample)
        np.testing.assert_equal(test.numpy(), sparse_sample.toarray())

        # distributed
        a_distributed, a = random_distributed((15, 16), float)
        test = core.array(a_distributed)
        np.testing.assert_equal(test.numpy(), a)
        assert test is not a_distributed

        # distributed with a different block shape
        with core.shape_context(block_shape=(4, 4)):
            a_distributed, a = random_distributed((15, 16), float)
        test = core.array(a_distributed)
        np.testing.assert_equal(test.numpy(), a)

        # distributed with an equivalent context
        context = core.GridContext(shape=(2, 2), comm=core.default_grid_context.comm)
        with core.shape_context(context=context):
            a_distributed, a = random_distributed((15, 16), float)
        test = core.array(a_distributed)
        np.testing.assert_equal(test.numpy(), a)

        # distributed with a different context
        context = core.GridContext(shape=(4, 1), comm=core.default_grid_context.comm)
        with core.shape_context(context=context):
            a_distributed, a = random_distributed((15, 16), float)
        test = core.array(a_distributed)
        np.testing.assert_equal(test.numpy(), a)

        # distributed with a different MPI comm
        comm = mpi_comm.Create_group(mpi_comm.group.Incl([3, 2, 0, 1]))
        context = core.GridContext(comm=comm)
        with core.shape_context(context=context):
            a_distributed, a = random_distributed((15, 16), float)
        with pytest.raises(ValueError):
            core.array(a_distributed)
