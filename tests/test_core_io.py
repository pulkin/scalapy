from common import mpi_comm, mpi_rank, assert_mpi_env, random_distributed

import numpy as np

from tempfile import NamedTemporaryFile

from scalapy import core

assert_mpi_env(size=4)
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}


def test_save_load(shape=(5, 5), dtype=np.float64):
    with core.shape_context(**test_context):
        from common import non_random
        a_distributed, a = random_distributed(shape, dtype, f_random=non_random)

        if mpi_rank == 0:
            f_temp = NamedTemporaryFile()
            fn_temp = f_temp.name
        else:
            fn_temp = None
        fn_temp = mpi_comm.bcast(fn_temp, 0)

        a_distributed.to_file(fn_temp)
        test_distributed = core.DistributedMatrix.from_file(fn_temp, shape, dtype)

        np.testing.assert_equal(test_distributed.local_array, a_distributed.local_array)

        # Test convention
        test = np.fromfile(fn_temp)
        np.testing.assert_equal(test.reshape(shape).T, a)  # TODO: fix convention?
