import numpy as np

from scalapy.core import DistributedMatrix


def random_distributed(shape, dtype):
    """
    Generates a random distributed matrix with a dense copy on rank zero.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    dtype
        Matrix data type.

    Returns
    -------
    a_distributed : DistributedMatrix
        The distributed matrix.
    a : np.ndarray
        The dense copy on rank zero or None otherwise.
    """
    a = np.random.standard_normal(shape)
    if np.issubdtype(dtype, np.complex):
        a = a + 1j * np.random.standard_normal(shape)
    a = np.asfortranarray(a.astype(dtype))
    a_distributed = DistributedMatrix.from_global_array(a, rank=0)
    return a_distributed, a
