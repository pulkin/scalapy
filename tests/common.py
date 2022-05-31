from functools import partial

import numpy as np
from scipy.stats import ortho_group, unitary_group

from scalapy.core import DistributedMatrix


def random(shape, dtype=float):
    """
    Simply a random matrix of a given type and shape.

    Parameters
    ----------
    shape : tuple
        The requested output shape.
    dtype
        Output matrix type.

    Returns
    -------
    a : np.ndarray
        The resulting array.
    """
    a = np.random.standard_normal(shape)
    if np.issubdtype(dtype, complex):
        a = a + 1j * np.random.standard_normal(shape)
    return a.astype(dtype)


def random_eig(size, lower=0, upper=1, dtype=float):
    """
    Random Hermitian matrix with eigenvalues
    distributed uniformly within a window.

    Parameters
    ----------
    size : int
        Matrix size.
    lower : float
        Lower eigenvalue bound.
    upper : float
        Upper eigenvalue bound.
    dtype
        The output data type.

    Returns
    -------
    h : np.ndarray
        The resulting matrix.
    """
    if isinstance(size, tuple):
        a, b = size
        assert a == b, f"non-square matrix requested: shape={size}"
        size = a
    if np.issubdtype(dtype, complex):
        g = unitary_group
    else:
        g = ortho_group
    vecs = g.rvs(size)
    vals = np.random.rand(size) * (upper - lower) + lower
    return ((vecs.conj().T * vals[None, :]) @ vecs).astype(dtype)


def random_distributed(shape, dtype, intermediate=None, f_random=random):
    """
    Generates a random distributed matrix with a dense copy on rank zero.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    dtype
        Matrix data type.
    intermediate
        A callable intermediate function before the random matrix is distributed.
    f_random
        Random generator.

    Returns
    -------
    a_distributed : DistributedMatrix
        The distributed matrix.
    a : np.ndarray
        The dense copy on rank zero or None otherwise.
    """
    a = f_random(shape, dtype=dtype)
    if intermediate is not None:
        a = intermediate(a)
    a = np.asfortranarray(a.astype(dtype))
    a_distributed = DistributedMatrix.from_global_array(a, rank=0)
    return a_distributed, a


def random_hermitian_distributed(shape, dtype):
    """
    Generates a random Hermitian distributed matrix with a dense copy on rank zero.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    dtype
        Matrix data type.

    Returns
    -------
    a_distributed : DistributedMatrix
        The distributed Hermitian matrix.
    a : np.ndarray
        The dense copy on rank zero or None otherwise.
    """
    s = shape[0]
    return random_distributed(
        shape, dtype,
        f_random=partial(random_eig, lower=-s/2, upper=s/2),
    )


def random_hp_distributed(shape, dtype, lower=0.5, upper=1.5):
    """
    Generates a random Hermitian positive distributed matrix with a dense copy on rank zero.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    dtype
        Matrix data type.
    lower : float
        Lower bound for matrix eigenvalues.
    upper : float
        Upper bound for matrix eigenvalues.

    Returns
    -------
    a_distributed : DistributedMatrix
        The distributed Hermitian positive matrix.
    a : np.ndarray
        The dense copy on rank zero or None otherwise.
    """
    s = shape[0]
    return random_distributed(
        shape, dtype,
        intermediate=lambda a: a * s,
        f_random=partial(random_eig, lower=lower, upper=upper),
    )
