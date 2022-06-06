from common import mpi_comm, mpi_rank, assert_mpi_env

from scalapy import blacs, core

import gc

assert_mpi_env(size=4)


def test_blacs():
    blacs_context = blacs.GridContext((2, 2), comm=mpi_comm)
    rank_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

    grid_info = blacs_context.get_info()
    grid_shape = grid_info[:2]
    grid_pos = grid_info[2:]

    assert grid_shape == (2, 2)
    assert grid_pos == rank_list[mpi_rank]


def test_blacs_cleanup():
    """Tests context cleanup"""
    blacs_context_1 = blacs.GridContext((2, 2), comm=mpi_comm)
    blacs_context_2 = blacs.GridContext((2, 2), comm=mpi_comm)

    assert blacs_context_1.handle != blacs_context_2.handle, f"handles {blacs_context_1.handle} and " \
                                                             f"{blacs_context_2.handle} are not unique"

    handle = blacs_context_2.handle
    del blacs_context_2
    gc.collect()
    blacs_context_3 = blacs.GridContext((2, 2), comm=mpi_comm)

    assert blacs_context_3.handle == handle, f"handle {handle} was not re-used: instead, {blacs_context_3.handle} " \
                                             f"was found"


def test_default():
    """Tests the default context"""
    context = core._context
    assert context.grid_shape == (2, 2)

    core.DistributedMatrix((100, 100))
