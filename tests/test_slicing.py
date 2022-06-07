from common import assert_mpi_env, random_distributed

import numpy as np

from scalapy import core

assert_mpi_env(size=4)
test_context = {"block_shape": (2, 2)}


def test_dm_slicing():
    """Test redistribution of matrices with different blocking and process grids"""
    with core.shape_context(**test_context):
        a_distributed, a = random_distributed((5, 5), np.float64)
        a_slice_distributed = a_distributed[1:4, -3:]
        a_slice = a[1:4, -3:]
        test = core.fromnumpy(a_slice)

        np.testing.assert_equal(a_slice_distributed.local_array, test.local_array)
