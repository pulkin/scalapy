from common import mpi_comm, mpi_rank, assert_mpi_env

assert_mpi_env()


def test_blacs():
    from scalapy import blacs

    ctxt = blacs.sys2blacs_handle(mpi_comm)
    blacs.gridinit(ctxt, 2, 2)
    rank_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

    grid_info = blacs.gridinfo(ctxt)
    grid_shape = grid_info[:2]
    grid_pos = grid_info[2:]

    assert grid_shape == (2, 2)
    assert grid_pos == rank_list[mpi_rank]
