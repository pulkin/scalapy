from common import mpi_comm, mpi_rank, assert_mpi_env, random_distributed

import numpy as np

from tempfile import NamedTemporaryFile

from scalapy import core

assert_mpi_env(size=4)
test_context = {"gridshape": (2, 2), "block_shape": (3, 3)}


def test_basic_io(shape=(5, 5), dtype=np.float64):
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed(shape, dtype)

        with NamedTemporaryFile() as f:
            if mpi_rank == 0:
                a.T.tofile(f)  # TODO: saved in Fortran order here: fix?
                f.flush()
                f.seek(0)

            mpi_comm.Barrier()

            test = core.DistributedMatrix.from_file(f.name, shape, dtype)

        np.testing.assert_equal(test.local_array, a_distributed.local_array)

        with NamedTemporaryFile() as f:
            a_distributed.to_file(f.name)

            if mpi_rank == 0:
                test = np.asfortranarray(np.fromfile(f.name, dtype=dtype).reshape(shape).T)  # TODO: and here
            else:
                test = np.empty(shape, dtype, order='F')
            mpi_comm.Bcast(test, 0)
            np.testing.assert_equal(test, a)
