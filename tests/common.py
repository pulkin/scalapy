from functools import partial

import numpy as np
from scipy.stats import ortho_group, unitary_group
from mpi4py import MPI

from scalapy.core import DistributedMatrix, fromnumpy


mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.size


def assert_mpi_env(size):
    """
    Asserts MPI environment size.

    Parameters
    ----------
    size : int
        MPI process count.
    """
    assert mpi_size == size, f"MPI environment mismatch: size={mpi_size} vs requested {size}"


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
        The resulting matrix.
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


def random_low_rank(shape, dtype, rank=None):
    """
    A random matrix whose rank is lower than its dimensions.

    Parameters
    ----------
    shape : tuple
        The requested output shape.
    dtype
        Output matrix type.
    rank : int
        Target rank.

    Returns
    -------
    a : np.ndarray
        The resulting matrix.
    """
    if rank is None:
        rank = min(shape) // 2
    a = random((shape[0], rank), dtype)
    b = random((rank, shape[1]), dtype)
    return a @ b


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
    f_random : Callable
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
    mpi_comm.Bcast(a, 0)
    a_distributed = fromnumpy(a, rank=0)
    return a_distributed, a


def random_pd_distributed(shape, dtype):
    """
    Generates a random positive-definite distributed matrix
    with a dense copy on rank zero.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    dtype
        Matrix data type.

    Returns
    -------
    a_distributed : DistributedMatrix
        The distributed positive-definite matrix.
    a : np.ndarray
        The dense copy on rank zero or None otherwise.
    """
    return random_distributed(shape, dtype, intermediate=lambda a: a @ a.conj().T)


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


def random_lr_distributed(shape, dtype, rank=None):
    """
    Generates a random low-rank distributed matrix
    with a dense copy on MPI rank zero.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    dtype
        Matrix data type.
    rank : int
        The desired matrix rank.

    Returns
    -------
    a_distributed : DistributedMatrix
        The distributed low-rank matrix.
    a : np.ndarray
        The dense copy on MPI rank zero or None otherwise.
    """
    return random_distributed(
        shape, dtype,
        f_random=partial(random_low_rank, rank=rank),
    )
