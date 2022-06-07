from common import mpi_rank, mpi_comm, assert_mpi_env

from scalapy import core

assert_mpi_env(size=4)
pos_list = [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_process_context():
    pc = core.ProcessContext([2, 2], comm=mpi_comm)

    # Test grid shape is correct
    assert pc.shape == (2, 2)

    # Test we have the correct positions
    assert pc.pos == pos_list[mpi_rank]

    # Test the MPI communicator is correct
    assert mpi_comm is pc.comm


def test_initmpi():
    with core.shape_context([2, 2], block_shape=[5, 5]):

        # Test grid shape is correct
        assert core._context.shape == (2, 2)

        # Test we have the correct positions
        assert core._context.pos == pos_list[mpi_rank]

        # Test the block shape is set correctly
        assert core._block_shape == (5, 5)
