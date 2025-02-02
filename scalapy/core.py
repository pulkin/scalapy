"""
===========================================
Core (:mod:`scalapy.core`)
===========================================

.. currentmodule:: scalapy.core

This module contains the core of `scalapy`: a set of routines and classes to
describe the distribution of MPI processes involved in the computation, and
interface with ``BLACS``; and a class which holds a block cyclic distributed
matrix for computation.


Routines
========

.. autosummary::
    :toctree: generated/

    initmpi


Classes
=======

.. autosummary::
    :toctree: generated/

    ProcessContext
    DistributedMatrix
    ScalapyException
    ScalapackException

"""
from contextlib import contextmanager

from numbers import Number
import numpy as np
from scipy import sparse
from mpi4py import MPI

from . import blockcyclic
from .blacs import GridContext
from . import mpi3util
from . import lowlevel as ll
from .util import real_equiv


class ScalapyException(Exception):
    """Error in scalapy."""
    pass


class ScalapackException(Exception):
    """Error in calling Scalapack."""
    pass


default_grid_context = GridContext()
default_block_shape = (32, 32)


# Map numpy type into MPI type
typemap = { np.float32: MPI.FLOAT,
            np.float64: MPI.DOUBLE,
            np.complex64: MPI.COMPLEX,
            np.complex128: MPI.COMPLEX16 }


def _chk_2d_size(shape, positive=True):
    # Check that the shape describes a valid 2D grid. Zero shape not allowed when positive = True.

    if len(shape) != 2:
        return False

    if positive:
        if shape[0] <= 0 or shape[1] <= 0:
            return False
    else:
        if shape[0] < 0 or shape[1] < 0:
            return False

    return True


@contextmanager
def shape_context(context=None, block_shape=None):
    """
    Sets a temporary context for Scalapack matrix distribution.

    Parameters
    ----------
    context : GridContext
        Process grid context to work with.
    block_shape : tuple
        Matrix block size ad a pair of integers.
    """
    global default_grid_context, default_block_shape

    prev_context = default_grid_context
    prev_bs = default_block_shape
    if context is not None:
        default_grid_context = context
    if block_shape is not None:
        default_block_shape = tuple(block_shape)
    yield None
    default_grid_context = prev_context
    default_block_shape = prev_bs


def create_1d_comm_group(context, dim):
    """
    Assembles all MPI processes along the specified
    dimension into a separate MPI communicator.

    Parameters
    ----------
    context : GridContext
        The context specifying the process grid.
    dim : int
        Dimension to assemble along.

    Returns
    -------
    result : MPI.Comm
        The resulting communicator.
    """
    pos = context.pos
    comm = context.comm
    if dim == 0:
        ix = pos[0], slice(None)
    else:
        ix = slice(None), pos[1]
    return comm.Create_group(comm.group.Incl(context.rank_grid[ix]))


class MatrixLikeAlgebra:
    """Defines commuting and such"""
    def copy(self):
        """
        A copy of this matrix.

        Returns
        -------
        result : MatrixLikeAlgebra
            A copy of this matrix.
        """
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        result = self.copy()
        result.__iadd__(other)
        return result
    __radd__ = __add__

    def __isub__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        result = self.copy()
        result.__isub__(other)
        return result

    def __rsub__(self, other):
        result = self.__neg__()
        result.__iadd__(other)
        return result

    def __imul__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        result = self.copy()
        result.__imul__(other)
        return result
    __rmul__ = __mul__

    def __itruediv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        result = self.copy()
        result.__itruediv__(other)
        return result

    def __rtruediv__(self, other):
        raise NotImplementedError


def array(source, rank=None):
    """
    Converts input to a distributed matrix.

    Parameters
    ----------
    source
        The source object.
    rank : int
        Indicates that the input is only available
        in the process specified by this rank.

    Returns
    -------
    out : DistributedMatrix
        The resulting distributed matrix.
    """
    comm = default_grid_context.comm
    # broadcast input type but not the input itself
    if rank is None:
        source_type = type(source)
    else:
        source_type = comm.bcast(type(source), rank)

    if issubclass(source_type, DistributedMatrix):
        if rank is not None:
            raise ValueError(f"input is distributed and rank is not None: rank={rank}")
        if source.context == default_grid_context and source.block_shape == default_block_shape:  # contexts are fully compatible
            return source.copy()
        elif source.context.comm is comm:  # context needs an update
            return source.redistribute(block_shape=default_block_shape, context=default_grid_context)
        else:
            raise ValueError(f"the input (distributed matrix) is associated with a different comm {source.context.comm}"
                             f" vs default (assumed) comm {comm}")

    elif issubclass(source_type, sparse.csr_matrix):
        return fromsparse_csr(source, rank=rank)

    else:  # try converting to numpy and assembling a distributed matrix
        if rank is None or default_grid_context.comm.rank == rank:
            source = np.asanyarray(source)
        return fromnumpy(source, rank=rank)


def empty(shape, **kwargs):
    """
    Empty distributed matrix.

    Parameters
    ----------
    shape : tuple
        Matrix shape.
    kwargs
        Other arguments to pass to the constructor.

    Returns
    -------
    result : DistributedMatrix
        The resulting distributed matrix.
    """
    return DistributedMatrix(shape, **kwargs)


def empty_like(mat, **kwargs):
    """
    Empty distributed matrix derived from the input.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to derive from.
    kwargs
        Matrix parameters to update.

    Returns
    -------
    result : DistributedMatrix
        The derived matrix.
    """
    derived = {"shape": mat.shape, "dtype": mat.dtype, "block_shape": mat.block_shape, "context": mat.context}
    derived.update(kwargs)
    return DistributedMatrix(**derived)


def zeros(shape, **kwargs):
    """
    Distributed matrix filled with zeros.

    Parameters
    ----------
    shape : tuple
        Global matrix shape.
    kwargs
        Other arguments to pass to the constructor.

    Returns
    -------
    result : DistributedMatrix
        The resulting zero-filled distributed matrix.
    """
    result = empty(shape, **kwargs)
    result.local_array[:] = 0
    return result


def zeros_like(mat, **kwargs):
    """
    Zero-filled distributed matrix derived from the input.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to derive from.
    kwargs
        Matrix parameters to update.

    Returns
    -------
    result : DistributedMatrix
        The derived matrix filled with zeros.
    """
    result = empty_like(mat, **kwargs)
    result.local_array[:] = 0
    return result


def transpose(mat, hconj=False):
    """
    Transpose the matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to transpose.
    hconj : bool
        If True, performs Hermitian conjugation
        (i.e. transpose of complex-conjugate).

    Returns
    -------
    result : DistributedMatrix
        The resulting transposed matrix.
    """
    result = empty_like(mat, shape=mat.shape[::-1])
    args = [*result.shape, 1.0, mat, 0.0, result]
    call_table = {('S', False): (ll.pstran, args),
                  ('D', False): (ll.pdtran, args),
                  ('S', True): (ll.pstran, args),
                  ('D', True): (ll.pdtran, args),
                  ('C', False): (ll.pctranu, args),
                  ('Z', False): (ll.pztranu, args),
                  ('C', True): (ll.pctranc, args),
                  ('Z', True): (ll.pztranc, args)}
    func, args = call_table[mat.sc_dtype, hconj]
    func(*args)
    return result


def conj(mat):
    """
    Complex conjugate of the matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to conjugate.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix.
    """
    result = empty_like(mat)
    result.local_array[:] = mat.local_array.conj()
    return result


def hconj(mat):
    """
    Hermitian conjugate of the matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to conjugate.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix.
    """
    return transpose(mat, hconj=True)


def eye(n, m=None, k=0, **kwargs):
    """
    A matrix with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Row count.
    m : int
        Column count.
    k : int
        Index of the diagonal.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix.
    """
    if m is None:
        m = n
    result = zeros((n, m), **kwargs)
    _, r, _, c = result.local_diagonal_indices(diagonal=k)
    result.local_array[r, c] = 1
    return result


def eye_like(mat, k=0, **kwargs):
    """
    A matrix with ones on the diagonal and zeros elsewhere
    similar to the input matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The base matrix.
    k : int
        Index of the diagonal.
    kwargs
        Other arguments to the DistributedMatrix constructor.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix with ones on the diagonal.
    """
    result = zeros_like(mat, **kwargs)
    _, r, _, c = result.local_diagonal_indices(diagonal=k)
    result.local_array[r, c] = 1
    return result


def identity(n, **kwargs):
    """
    Identity matrix.

    Parameters
    ----------
    n : int
        Matrix size.
    kwargs
        Other arguments to the DistributedMatrix constructor.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix.
    """
    return eye(n, n, 0, **kwargs)


def identity_like(mat, **kwargs):
    """
    Identity matrix similar to the input matrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The base matrix.
    kwargs
        Other arguments to the DistributedMatrix constructor.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix.
    """
    return eye_like(mat, 0, **kwargs)


def absolute(mat, out=None):
    """
    Absolute of a DistributedMatrix.

    Parameters
    ----------
    mat : DistributedMatrix
        The matrix to take the absolute value of.
    out : DistributedMatrix
        The output matrix.

    Returns
    -------
    result : DistributedMatrix
        The resulting matrix.
    """
    if out is None:
        out = empty_like(mat, dtype=real_equiv(mat.dtype))
    else:
        mat.assert_same_distribution(out)
        if mat.shape != out.shape:
            raise ValueError(f"out.shape = {out.shape} != mat.shape = {mat.shape}")
    out.local_array[:] = np.absolute(mat.local_array)
    return out


class DistributedMatrix(MatrixLikeAlgebra):
    r"""A matrix distributed over multiple MPI processes.

    Parameters
    ----------
    shape : list of integers
        The size of the global matrix eg. ``[Nr, Nc]``.
    dtype : np.dtype, optional
        The datatype of the array. See `Notes`_ for the supported types.
    block_shape: list of integers, optional
        The blocking size, packed as ``[Br, Bc]``. If ``None`` uses the default blocking
        (set via :func:`initmpi`).
    context : ProcessContext, optional
        The process context. If not set uses the default (recommended).

    Attributes
    ----------
    local_array
    desc
    context
    dtype
    mpi_dtype
    sc_dtype
    shape
    local_shape
    block_shape

    Methods
    -------
    empty_like
    indices
    numpy
    from_file
    to_file
    redistribute


    .. _notes:

    Notes
    -----
    The type of the array must be specified with the standard numpy types. A
    :class:`DistributedMatrix` has properties for fetching the equivalent
    ``MPI`` (with :attr:`mpi_dtype`) and ``Scalapack`` types (which is a
    character given by :attr:`sc_dtype`).

    =================  =================  ==============  ===============================
    Numpy type         MPI type           Scalapack type  Description
    =================  =================  ==============  ===============================
    ``np.float32``     ``MPI.FLOAT``      ``S``           Single precision float
    ``np.float64``     ``MPI.DOUBLE``     ``D``           Double precision float
    ``np.complex64``   ``MPI.COMPLEX``    ``C``           Single precision complex number
    ``np.complex128``  ``MPI.COMPLEX16``  ``Z``           Double precision complex number
    =================  =================  ==============  ===============================

    Slicing
    -------

    Basic slicing is implemented for :class:`DistributedMatrix` objects allowing
    us to cut out sections of the global matrix, giving a new
    :class:`DistributedMatrix` containing the section and distributed over the
    same :class:`ProcessContext`. Note that this creates a *copy* of the
    original data, *not* a view of it. To use this simply do::

        dm = DistributedMatrix((10, 10), dtype=np.complex128)

        # Copy the first five columns
        five_cols = dm[:, :5]

        # Copy out the third and second to last rows
        two_rows = dm[-3:-1]

        # Copy every other row
        alternating_rows = dm[::2]

    """

    @property
    def local_array(self):
        """The local, block-cyclic packed segment of the matrix.

        This is an ndarray and is readonly. However, only the
        reference is readonly, the array itself can be modified in
        place.
        """
        if self._loccal_empty:
            return np.zeros(self.local_shape, order='F', dtype=self.dtype)
        else:
            return self._local_array


    @property
    def desc(self):
        """The Scalapack array descriptor. See [1]_. Returned as an integer
        ndarray and is readonly.

        .. [1] http://www.netlib.org/scalapack/slug/node77.html
        """
        return self._desc.copy()


    @property
    def context(self):
        """The ProcessContext of this matrix."""
        return self._context


    @property
    def dtype(self):
        """The numpy datatype of this matrix."""
        return self._dtype


    @property
    def mpi_dtype(self):
        """The base MPI Datatype."""
        return typemap[self.dtype]


    @property
    def sc_dtype(self):
        """The Scalapack type as a character."""
        _sc_type = {np.float32: 'S',
                    np.float64: 'D',
                    np.complex64: 'C',
                    np.complex128: 'Z'}

        return _sc_type[self.dtype]


    @property
    def shape(self):
        """The shape of the global matrix."""
        return self._global_shape


    @property
    def local_shape(self):
        """The shape of the local matrix."""

        lshape = tuple(map(blockcyclic.numrc, self.shape,
                           self.block_shape, self.context.pos,
                           self.context.shape))

        return tuple(lshape)


    @property
    def block_shape(self):
        """The blocksize for the matrix."""
        return self._block_shape

    def __init__(self, shape, dtype=np.float64, block_shape=None, context=None):
        r"""Initialise an empty DistributedMatrix.

        """

        ## Check and set data type
        if dtype not in list(typemap.keys()):
            raise Exception(f"dtype={dtype} not supported by Scalapack.")

        self._dtype = dtype

        ## Check and set global_shape
        if not _chk_2d_size(shape, positive=False):
            raise ScalapyException("Array global shape invalid.")

        self._global_shape = tuple(shape)

        ## Check and set default block_shape
        if not default_block_shape and not block_shape:
            raise ScalapyException("No supplied or default blocksize.")

        block_shape = block_shape if block_shape else default_block_shape

        # Validate block_shape.
        if not _chk_2d_size(block_shape):
            raise ScalapyException("Block shape invalid.")

        self._block_shape = block_shape

        ## Check and set context.
        if not context and not default_grid_context:
            raise ScalapyException("No supplied or default context.")
        self._context = context if context else default_grid_context

        # Allocate the local array.
        self._loccal_empty = True if self.local_shape[0] == 0 or self.local_shape[1] == 0 else False
        if self._loccal_empty:
            # as f2py can not handle zero sized array, we have to create an non-empty local array
            self._local_array = np.empty(1, dtype=dtype)
        else:
            self._local_array = np.empty(self.local_shape, order='F', dtype=dtype)

        # Create the descriptor
        self._mkdesc()

        # Create the MPI distributed datatypes
        self._mk_mpi_dtype()

    def __del__(self):
        try:  # TODO: not careful enough in case some attributes exist while others not
            self._free_mpi_dtype()
        except AttributeError:
            pass

    def _mkdesc(self):
        # Make the Scalapack array descriptor
        self._desc = np.zeros(9, dtype=np.int32)

        self._desc[0] = 1  # Dense matrix
        self._desc[1] = self.context
        self._desc[2] = self.shape[0]
        self._desc[3] = self.shape[1]
        self._desc[4] = self.block_shape[0]
        self._desc[5] = self.block_shape[1]
        self._desc[6] = 0
        self._desc[7] = 0
        self._desc[8] = self.local_shape[0]

    def _mk_mpi_dtype(self):
        ## Construct the MPI datatypes (both Fortran and C ordered)
        ##   These are required for reading in and out of arrays and files.

        # Get MPI process info
        if self.shape[0] == 0 or self.shape[1] == 0:
            self._darr_f = None
            self._darr_c = None
            self._darr_list = []
        else:
            rank = self.context.comm.rank
            size = self.context.comm.size

            # Create distributed array view (F-ordered)
            self._darr_f = self.mpi_dtype.Create_darray(size, rank,
                                                        self.shape,
                                                        [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                                        self.block_shape, self.context.shape,
                                                        MPI.ORDER_F)
            self._darr_f.Commit()

            # Create distributed array view (F-ordered)
            self._darr_c = self.mpi_dtype.Create_darray(size, rank,
                                                        self.shape,
                                                        [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                                        self.block_shape, self.context.shape,
                                                        MPI.ORDER_C).Commit()

            # Create list of types for all ranks (useful for passing to global array)
            self._darr_list = [ self.mpi_dtype.Create_darray(size, ri,
                                                             self.shape,
                                                             [MPI.DISTRIBUTE_CYCLIC, MPI.DISTRIBUTE_CYCLIC],
                                                             self.block_shape, self.context.shape,
                                                             MPI.ORDER_F).Commit() for ri in range(size) ]

    def _free_mpi_dtype(self):
        for i in (self._darr_f, self._darr_c, *self._darr_list):
            if i:
                i.Free()

    def copy(self):
        """Create a copy of this DistributedMatrix.

        This includes a full copy of the local data. However, the
        :attr:`context` is a reference to the original :class:`ProcessContext`.

        Returns
        -------
        copy : DistributedMatrix
        """
        cp = empty_like(self)
        cp.local_array[:] = self.local_array
        return cp


    def row_indices(self):
        """The row indices of the global array local to the process.
        """
        return blockcyclic.indices_rc(self.shape[0],
                                      self.block_shape[0],
                                      self.context.pos[0],
                                      self.context.shape[0])


    def col_indices(self):
        """The column indices of the global array local to the process.
        """
        return blockcyclic.indices_rc(self.shape[1],
                                      self.block_shape[1],
                                      self.context.pos[1],
                                      self.context.shape[1])


    def indices(self, full=True):
        r"""The indices of the elements stored in the local matrix.

        This can be used to easily build up distributed matrices that
        depend on their co-ordinates.

        Parameters
        ----------
        full : boolean, optional
            If False the matrices of indices are not fleshed out, if True the
            full matrices are returned. This is like the difference between
            np.ogrid and np.mgrid.

        Returns
        -------
        im : tuple of ndarrays
            The first element contains the matrix of row indices and
            the second of column indices.

        Notes
        -----

        As an example a DistributedMatrix defined globally as
        :math:`M_{ij} = i + j` can be created by::

            dm = DistributedMatrix(100, 100)
            rows, cols = dm.indices()
            dm.local_array[:] = rows + cols
        """

        ri, ci = tuple(map(blockcyclic.indices_rc,
                           self.shape,
                           self.block_shape,
                           self.context.pos,
                           self.context.shape))

        ri = ri.reshape((-1, 1), order='F')
        ci = ci.reshape((1, -1), order='F')

        if full:
            ri, ci = np.broadcast_arrays(ri, ci)
            ri = np.asfortranarray(ri)
            ci = np.asfortranarray(ci)

        return (ri, ci)


    def local_diagonal_indices(self, diagonal=0):
        """Returns triple of 1D arrays (global_index, local_row_index, local_column_index).

        Each of these arrays has length equal to the number of elements on the global diagonal
        which are stored in the local matrix.  For each such element, global_index[i] is its
        position in the global diagonal, and (local_row_index[i], local_column_index[i]) gives
        its position in the local array.

        As an example of the use of these arrays, the global operation A_{ij} += i^2 delta_{ij}
        could be implemented with::

           (global_index, local_row_index, local_column_index) = A.local_diagonal_indices()
           A.local_array[local_row_index, local_column_index] += global_index**2
        """

        ri, ci = tuple(map(blockcyclic.indices_rc,
                           self.shape,
                           self.block_shape,
                           self.context.pos,
                           self.context.shape))

        global_row_index = np.intersect1d(ri, ci - diagonal)
        global_col_index = global_row_index + diagonal

        (rank, local_row_index) = blockcyclic.localize_indices(global_row_index, self.block_shape[0], self.context.shape[0])
        assert np.all(rank == self.context.pos[0])

        (rank, local_col_index) = blockcyclic.localize_indices(global_col_index, self.block_shape[1], self.context.shape[1])
        assert np.all(rank == self.context.pos[1])

        return global_row_index, local_row_index, global_col_index, local_col_index


    def trace(self):
        """Returns global matrix trace (the trace is returned on all ranks)."""
        # TODO: not tested

        _, r, _, c = self.local_diagonal_indices()

        # Note: np.sum() returns 0 for length-zero array
        ret = np.array(np.sum(self.local_array[r,c]))
        self.context.comm.Allreduce(ret.copy(), ret, MPI.SUM)

        return ret

    def _load_array(self, mat):
        ## Copy the local data out of the global mat.

        if (self.shape[0] == 0) or (self.shape[1] == 0):
            return

        self._darr_f.Pack(mat, self.local_array[:], 0, self.context.comm)

    def numpy(self, rank=None):
        """Copy distributed data into a global array.

        This is mainly intended for testing. Would be a bad idea for larger problems.

        Parameters
        ----------
        rank : integer, optional
            If rank is None (default) then gather onto all nodes. If rank is
            set, then gather only onto one node.

        Returns
        -------
        matrix : np.ndarray
            The global matrix.
        """

        if (self.shape[0] == 0) or (self.shape[1] == 0):
            return np.zeros(self.shape, dtype=self.dtype)

        comm = self.context.comm

        bcast = False
        if rank is None:
            rank = 0
            bcast = True

        # Double check that rank is valid.
        if rank < 0 or rank >= comm.size:
            raise ScalapyException("Invalid rank.")

        global_array = None
        if comm.rank == rank or bcast:
            global_array = np.zeros(self.shape, dtype=self.dtype, order='F')

        # Each process should send its local sections.
        sreq = comm.Isend([self.local_array, self.mpi_dtype], dest=rank, tag=0)

        if comm.rank == rank:
            # Post each receive
            reqs = [ comm.Irecv([global_array, self._darr_list[sr]], source=sr, tag=0)
                        for sr in range(comm.size) ]

            # Wait for requests to complete
            MPI.Prequest.Waitall(reqs)

        # Wait on send request. Important, as can get weird synchronisation
        # bugs otherwise as processes exit before completing their send.
        sreq.Wait()

        # Distribute to all processes if requested
        if bcast:
            comm.Bcast([global_array, self.mpi_dtype], root=rank)

        # Barrier to synchronise all processes
        comm.Barrier()

        return global_array

    def assert_same_distribution(self, *args):
        """
        Assert same distribution and parallel blocking scheme.

        Parameters
        ----------
        args
            Other distributed matrices to compare to.
        """
        for x in args:
            assert self.block_shape == x.block_shape, f"block_shape mismatch: {self.block_shape} vs {x.block_shape}"
            assert self.context == x.context, f"context mismatch: {self.context} vs {x.context}"

    def __str__(self):
        return f"{self.__class__.__name__} with shape={self.shape} and dtype={self.dtype} distributed over {self.context})"

    def __pos__(self):
        return self

    def __neg__(self):
        result = self.copy()
        np.negative(result.local_array, out=result.local_array)
        return result

    def __abs__(self):
        return absolute(self)

    def __iadd__(self, x, np_op=np.ndarray.__iadd__, op_inplace=True):
        if isinstance(x, DistributedMatrix):
            if self.shape != x.shape:
                raise ValueError(f"operands have different shapes: {self.shape}, {x.shape}")
            self.assert_same_distribution(x)
            op_result = np_op(self.local_array, x.local_array)

        elif isinstance(x, Number):
            op_result = np_op(self.local_array, x)

        elif isinstance(x, np.ndarray):
            if x.ndim != 2:
                raise ValueError(f"the numpy operand is not a matrix: v.shape={x.shape}")
            nr, nc = self.shape
            if x.shape == (nr, nc):
                op_result = np_op(self.local_array, x[self.row_indices(), :][:, self.col_indices()])
            elif x.shape == (nr, 1):
                op_result = np_op(self.local_array, x[self.row_indices(), :])
            elif x.shape == (1, nc):
                op_result = np_op(self.local_array, x[:, self.col_indices()])
            else:
                raise ValueError(f"operands have incompatible shapes: {self.shape}, {x.shape}")

        else:
            raise NotImplementedError(f"cannot perform '{np_op}' on {self} and {x}")

        if not op_inplace:
            self.local_array[:] = op_result
        return self

    def __isub__(self, other):
        return self.__iadd__(other, np_op=np.ndarray.__isub__)

    def __imul__(self, x):
        return self.__iadd__(x, np_op=np.ndarray.__imul__)

    def __itruediv__(self, other):
        return self.__iadd__(other, np_op=np.ndarray.__itruediv__)

    def __rtruediv__(self, other):
        def _rev_op(a, b):
            return b / a
        result = self.copy()
        result.__iadd__(other, np_op=_rev_op, op_inplace=False)
        return result

    def dot(self, other):
        """
        Dot product.

        Parameters
        ----------
        other
            Another matrix to take the product with.

        Returns
        -------
        result
            The result of the product.
        """
        if isinstance(other, DistributedMatrix):
            return dot_mat_mat(self, other)
        elif isinstance(other, np.ndarray):
            return dot_mat_vec(self, other)
        else:
            raise NotImplementedError(f"cannot matmul {other}")
    __matmul__ = dot

    def sum(self, axis=None):
        """
        Sum of matrix elements.

        Parameters
        ----------
        axis
            Axes to sum over.

        Returns
        -------
        result
            The sum of matrix elements: either along the specified axis or
            along all axes.
        """
        if axis is None:
            axis = 0, 1
        if isinstance(axis, int):
            axis = axis,
        axis = set(axis)
        if axis == {0, 1}:
            local_sum = self.local_array.sum()
            local_sum = np.array([local_sum], dtype=self.dtype)
            self.context.comm.Allreduce(MPI.IN_PLACE, local_sum, MPI.SUM)
            return local_sum[0]

        elif axis == {0} or axis == {1}:
            sum_axis = axis.pop()
            free_axis = not sum_axis
            rc_contexts = create_1d_comm_group(self.context, 1), create_1d_comm_group(self.context, 0)
            sum_context = rc_contexts[sum_axis]
            distribution_context = rc_contexts[free_axis]

            # Sum local blocks
            local_sum = self.local_array.sum(axis=sum_axis)
            sum_context.Allreduce(MPI.IN_PLACE, local_sum, MPI.SUM)

            # Distribute sum results
            result = np.empty(self.shape[free_axis], dtype=self.dtype)
            chunk_dtypes = []
            for remote_rank in range(distribution_context.size):
                dtype = self.mpi_dtype.Create_darray(
                    distribution_context.size,
                    remote_rank,
                    result.shape,
                    [MPI.DISTRIBUTE_CYCLIC],
                    [self.block_shape[free_axis]],
                    [distribution_context.size],
                    MPI.ORDER_F,
                )
                dtype.Commit()
                chunk_dtypes.append(dtype)
            send_request = distribution_context.Isend([local_sum, self.mpi_dtype], dest=0)
            if distribution_context.rank == 0:
                MPI.Prequest.Waitall([
                    distribution_context.Irecv([result, dtype], source=src)
                    for src, dtype in enumerate(chunk_dtypes)
                ])
            send_request.Wait()
            distribution_context.Bcast(result, root=0)
            for i in chunk_dtypes:
                i.Free()
            for i in rc_contexts:
                i.Free()
            return result

        else:
            raise ValueError(f"unknown axis to sum over: {tuple(axis)}")

    def _section(self, srow=0, nrow=None, scol=0, ncol=None):
        ## return a section [srow:srow+nrow, scol:scol+ncol] of the global array as a new distributed array
        nrow = self.shape[0] - srow if nrow is None else nrow
        ncol = self.shape[1] - scol if ncol is None else ncol
        assert nrow > 0 and ncol > 0, 'Invalid number of rows/columns: %d/%d' % (nrow, ncol)

        B = DistributedMatrix([nrow, ncol], dtype=self.dtype, block_shape=self.block_shape, context=self.context)

        args = [nrow, ncol, self._local_array , srow+1, scol+1, self.desc, B._local_array, 1, 1, B.desc, self.context]

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        func(*args)

        return B

    def _sec2sec(self, B, srowb=0, scolb=0, srow=0, nrow=None, scol=0, ncol=None):
        # copy a section [srow:srow+nrow, scol:scol+ncol] of the global array to
        # another distributed array B starting at (srowb, scolb)

        # Copy to the end of the row/column if the numbers are not set.
        nrow = self.shape[0] - srow if nrow is None else nrow
        ncol = self.shape[1] - scol if ncol is None else ncol

        # Check the number of rows and columns
        if nrow <= 0 or ncol <= 0:
            raise ScalapyException('Invalid number of rows/columns: %d/%d' % (nrow, ncol))

        # Set up the argument list
        args = [nrow, ncol,
                self._local_array, srow+1, scol+1, self.desc,
                B._local_array, srowb+1, scolb+1, B.desc,
                self.context]

        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        func, args = call_table[self.sc_dtype]
        func(*args)

    def __getitem__(self, items):
        # numpy-like global slicing operation, but returns a distributed array.
        #
        # Supports basic numpy slicing with start and stop, and positive step

        def swap(a, b):
            return b, a

        def regularize_idx(idx, N, axis):
            # Regularize an index to check it is valid
            idx1 = idx if idx >= 0 else idx + N
            if idx1 < 0 or idx1 >= N:
                raise IndexError('Index %d is out of bounds for axis %d with size %d' % (idx, axis, N))

            return 1, [(idx1, 1)]

        def regularize_slice(slc, N):
            # Regularize a slice object
            #
            # Takes a slice object and the axis length, and returns the total
            # number of elements in the slice and an array of (start, length)
            # tuples describing the blocks making up the slice

            start, stop, step = slc.start, slc.stop, slc.step
            step = step if step is not None else 1

            # Check the step
            if step == 0:
                raise ValueError('slice step cannot be zero')
            if step > 0:
                start = start if start is not None else 0
                stop = stop if stop is not None else N
            else:
                start = start if start is not None else N-1
                if stop is None:
                    stop_is_none = True
                else:
                    stop_is_none = False
                stop = stop if stop is not None else 0
            start = start if start >= 0 else start + N
            stop = stop if stop >= 0 else stop + N
            start = max(0, start)
            start = min(N, start)
            stop = max(0, stop)
            stop = min(N, stop)

            # If the step is 1 things are simple...
            if step == 1:
                m = stop - start
                if m > 0:
                    return m, [(start, m)]
                else:
                    return 0, []

            # ... if it is greater than one divide up into a series of blocks
            else:
                m = 0
                lst = []
                if step > 0:
                    while(start < stop):
                        lst.append((start, 1))
                        m += 1
                        start += step
                if step < 0:
                    while(start > stop):
                        lst.append((start, 1))
                        m += 1
                        start += step
                    if stop_is_none and start == stop:
                        m += 1
                        lst.append((0, 1))

                return m, lst


        nrow, ncol = self.shape

        # First replace any Ellipsis with a full slice(None, None, None) object, this
        # is fine because the matrix is always 2D and it vastly simplifies the logic
        if items is Ellipsis:
            items = slice(None, None, None)
        if items is tuple:
            items = tuple([slice(None, None, None) if items is Ellipsis else item for item in items])

        # First case deal with just a single slice (either an int or slice object)
        if isinstance(items, int):
            m, rows = regularize_idx(items, nrow, 0)
            n = ncol  # number of columns
            cols = [(0, ncol)]
        elif isinstance(items, slice):
            if items == slice(None, None, None):
                return self.copy()

            m, rows = regularize_slice(items, nrow)
            n = ncol
            cols = [(0, ncol)]

        # Then deal with the case of a tuple (i.e. slicing both dimensions)
        elif isinstance(items, tuple):

            # Check we have two indexed dimensions
            if len(items) != 2:
                raise ValueError('Two many indices for 2D matrix: %s' % repr(items))

            # Check the types in the slicing are correct
            if not ((isinstance(items[0], (int, slice)) or items[0] is Ellipsis) and
                    (isinstance(items[1], (int, slice)) or items[1] is Ellipsis)):
                raise ValueError('Invalid indices %s' % items)

            # Process case of wanting a specific row
            if isinstance(items[0], int):
                m, rows = regularize_idx(items[0], nrow, 0)

                if isinstance(items[1], int):
                    n, cols = regularize_idx(items[1], ncol, 1)
                elif isinstance(items[1], slice):
                    n, cols = regularize_slice(items[1], ncol)
                else:
                    raise ValueError('Invalid indices %s' % items)

            # Case of wanting a slice of rows
            elif isinstance(items[0], slice):
                m, rows = regularize_slice(items[0], nrow)

                if isinstance(items[1], int):
                    n, cols = regularize_idx(items[1], ncol, 1)
                elif isinstance(items[1], slice):
                    if items[0] == slice(None, None, None) and items[1] == slice(None, None, None):
                        return self.copy()

                    n, cols = regularize_slice(items[1], ncol)
                else:
                    raise ValueError('Invalid indices %s' % items)

            else:
                raise ValueError('Invalid indices %s' % items)
        else:
            raise ValueError('Invalid indices %s' % items)

        # Create output DistributedMatrix
        B = DistributedMatrix([m, n], dtype=self.dtype, block_shape=self.block_shape, context=self.context)
        srowb = 0

        # Iterate over blocks to copy from self to new output matrix
        for (srow, nrow) in rows:
            scolb = 0
            for (scol, ncol) in cols:
                self._sec2sec(B, srowb, scolb, srow, nrow, scol, ncol)
                scolb += ncol
            srowb += nrow

        return B


    def _copy_from_np(self, a, asrow=0, anrow=None, ascol=0, ancol=None, srow=0, scol=0, block_shape=None, rank=0):
        ## copy a section of a numpy array a[asrow:asrow+anrow, ascol:ascol+ancol] to self[srow:srow+anrow, scol:scol+ancol], once per block_shape

        Nrow, Ncol = self.shape
        srow = srow if srow >= 0 else srow + Nrow
        srow = max(0, srow)
        srow = min(srow, Nrow)
        scol = scol if scol >= 0 else scol + Ncol
        scol = max(0, scol)
        scol = min(scol, Ncol)
        if self.context.comm.rank == rank:
            if not (a.ndim == 1 or a.ndim == 2):
                raise ScalapyException('Unsupported high dimensional array.')

            a = np.asfortranarray(a.astype(self.dtype)) # type conversion
            a = a.reshape(-1, a.shape[-1]) # reshape to two dimensional
            am, an = a.shape
            asrow = asrow if asrow >= 0 else asrow + am
            asrow = max(0, asrow)
            asrow = min(asrow, am)
            ascol = ascol if ascol >= 0 else ascol + an
            ascol = max(0, ascol)
            ascol = min(ascol, an)
            m = am - asrow if anrow is None else anrow
            m = max(0, m)
            m = min(m, am - asrow, Nrow - srow)
            n = an - ascol if ancol is None else ancol
            n = max(0, n)
            n = min(n, an - ascol, Ncol - scol)
        else:
            m, n = 1, 1

        asrow = self.context.comm.bcast(asrow, root=rank)
        ascol = self.context.comm.bcast(ascol, root=rank)
        m = self.context.comm.bcast(m, root=rank) # number of rows to copy
        n = self.context.comm.bcast(n, root=rank) # number of columes to copy

        if m == 0 or n == 0:
            return self

        block_shape = self.block_shape if block_shape is None else block_shape
        if not _chk_2d_size(block_shape):
            raise ScalapyException("Invalid block_shape")

        bm, bn = block_shape
        br = blockcyclic.num_blocks(m, bm) # number of blocks for row
        bc = blockcyclic.num_blocks(n, bn) # number of blocks for column
        rm = m - (br - 1) * bm # remained number of rows of the last block
        rn = n - (bc - 1) * bn # remained number of columes of the last block

        # due to bugs in scalapy, it is needed to first init an process context here
        GridContext([1, self.context.comm.size], comm=self.context.comm) # process context

        for bri in range(br):
            M = bm if bri != br - 1 else rm
            for bci in range(bc):
                N = bn if bci != bc - 1 else rn
                if self.context.comm.rank == rank:
                    pc = GridContext([1, 1], comm=MPI.COMM_SELF) # process context
                    desc = self.desc
                    desc[1] = pc
                    desc[2], desc[3] = a.shape
                    desc[4], desc[5] = a.shape
                    desc[8] = a.shape[0]
                    args = [M, N, a, asrow+1+bm*bri, ascol+1+bn*bci, desc, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, self.context]
                else:
                    desc = np.zeros(9, dtype=np.int32)
                    desc[1] = -1
                    args = [M, N, np.zeros(1, dtype=self.dtype) , asrow+1+bm*bri, ascol+1+bn*bci, desc, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, self.context]

                call_table = {'S': (ll.psgemr2d, args),
                              'D': (ll.pdgemr2d, args),
                              'C': (ll.pcgemr2d, args),
                              'Z': (ll.pzgemr2d, args)}

                func, args = call_table[self.sc_dtype]
                func(*args)

        return self


    def np2self(self, a, srow=0, scol=0, block_shape=None, rank=0):
        """Copy a one or two dimensional numpy array `a` owned by
        rank `rank` to the section of the distributed matrix starting
        at row `srow` and column `scol`. Once copy a section equal
        or less than `block_shape` if `a` is large.
        """
        return self._copy_from_np(a, asrow=0, anrow=None, ascol=0, ancol=None,srow=srow, scol=scol, block_shape=block_shape, rank=rank)


    def self2np(self, srow=0, nrow=None, scol=0, ncol=None, block_shape=None, rank=0):
        """Copy a section of the distributed matrix
        self[srow:srow+nrow, scol:scol+ncol] to a two dimensional numpy
        array owned by rank `rank`. Once copy a section equal or less
        than `block_shape` if the copied section is large.
        """
        Nrow, Ncol = self.shape
        srow = srow if srow >= 0 else srow + Nrow
        srow = max(0, srow)
        srow = min(srow, Nrow)
        scol = scol if scol >= 0 else scol + Ncol
        scol = max(0, scol)
        scol = min(scol, Ncol)
        m = Nrow - srow if nrow is None else nrow
        m = max(0, m)
        m = min(m, Nrow - srow)
        n = Ncol - scol if ncol is None else ncol
        n = max(0, n)
        n = min(n, Ncol - scol)

        if self.context.comm.rank == rank:
            a = np.empty((m, n), dtype=self.dtype, order='F')
        else:
            a = None

        if m == 0 or n == 0:
            return a

        block_shape = self.block_shape if block_shape is None else block_shape
        if not _chk_2d_size(block_shape):
            raise ScalapyException("Invalid block_shape")

        bm, bn = block_shape
        br = blockcyclic.num_blocks(m, bm) # number of blocks for row
        bc = blockcyclic.num_blocks(n, bn) # number of blocks for column
        rm = m - (br - 1) * bm # remained number of rows of the last block
        rn = n - (bc - 1) * bn # remained number of columes of the last block

        # due to bugs in scalapy, it is needed to first init an process context here
        GridContext([1, self.context.comm.size], comm=self.context.comm) # process context

        for bri in range(br):
            M = bm if bri != br - 1 else rm
            for bci in range(bc):
                N = bn if bci != bc - 1 else rn
                if self.context.comm.rank == rank:
                    pc = GridContext([1, 1], comm=MPI.COMM_SELF) # process context
                    desc = self.desc
                    desc[1] = pc
                    desc[2], desc[3] = a.shape
                    desc[4], desc[5] = a.shape
                    desc[8] = a.shape[0]
                    args = [M, N, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, a , 1+bm*bri, 1+bn*bci, desc, self.context]
                else:
                    desc = np.zeros(9, dtype=np.int32)
                    desc[1] = -1
                    args = [M, N, self._local_array, srow+1+bm*bri, scol+1+bn*bci, self.desc, np.zeros(1, dtype=self.dtype) , 1+bm*bri, 1+bn*bci, desc, self.context]

                call_table = {'S': (ll.psgemr2d, args),
                              'D': (ll.pdgemr2d, args),
                              'C': (ll.pcgemr2d, args),
                              'Z': (ll.pzgemr2d, args)}

                func, args = call_table[self.sc_dtype]
                func(*args)

        return a

    def is_tiny(self):
        """True if some MPI processes may contain empty blocks"""
        shape = np.array(self.local_shape, dtype=int)
        out = np.empty_like(shape)
        self.context.comm.Allreduce(shape, out, op=MPI.MIN)
        return np.any(out == 0)

    def assert_not_tiny(self, intro="DistributedMatrix"):
        """Checks if this matrix does not contain empty blocks on any of the processes"""
        if self.is_tiny():
            raise ValueError(f"{intro}: the matrix with shape {self.shape} "
                             f"contains empty blocks; block_shape={self.block_shape} "
                             f"mpi_grid_shape={self.context.shape}")

    @classmethod
    def from_file(cls, filename, global_shape, dtype, block_shape=None, context=None, order='F', displacement=0):
        """Read in a global array from a file.

        Parameters
        ----------
        filename : string
            Name of file to read in.
        global_shape : [nrows, ncols]
            Shape of global array.
        dtype : numpy datatype
            Datatype of array.
        block_shape : [brows, bcols]
            Shape of block, if None, try to use default size.
        context : ProcessContext
            Description of process distribution.
        order : 'F' or 'C'
            Storage order on disk.
        displacement : integer
            Displacement from the start of file (in bytes)

        Returns
        -------
        dm : DistributedMatrix
        """

        if (global_shape[0] == 0) or (global_shape[1] == 0):
            return None

        # Initialise DistributedMatrix
        dm = cls(global_shape, dtype=dtype, block_shape=block_shape, context=context)

        # Open the file, and read out the segments
        f = MPI.File.Open(dm.context.comm, filename, MPI.MODE_RDONLY)
        f.Set_view(displacement, dm.mpi_dtype, dm._darr_f, "native")
        f.Read_all(dm.local_array)
        f.Close()

        return dm


    def to_file(self, filename, order='F', displacement=0):
        """Write a DistributedMatrix out to a file.

        Parameters
        ----------
        filename : string
            Name of file to write to.
        """

        if (self.shape[0] == 0) or (self.shape[1] == 0):
            return

        # Open the file, and read out the segments
        f = MPI.File.Open(self.context.comm, filename, MPI.MODE_RDWR | MPI.MODE_CREATE)

        filelength = displacement + mpi3util.type_get_extent(self._darr_f)[1]  # Extent is index 1

        # Preallocate to ensure file is long enough for writing.
        f.Preallocate(filelength)

        # Set view and write out.
        f.Set_view(displacement, self.mpi_dtype, self._darr_f, "native")
        f.Write_all(self.local_array)
        f.Close()

    def redistribute(self, block_shape=None, context=None):
        """Redistribute a matrix with another grid or block shape.

        Parameters
        ----------
        block_shape : [brows, bcols], optional
            New block shape. If `None` use the current block shape.
        context : ProcessContext, optional
            New process context. Must be over the same MPI communicator. If
            `None` use the current communicator.

        Returns
        -------
        dm : DistributedMatrix
            Newly distributed matrix.
        """

        # Check that we are actually redistributing across something
        if (block_shape is None) and (context is None):
            import warnings
            warnings.warn("Neither block_shape or context is set.")

        # Fix up default parameters
        if block_shape is None:
            block_shape = self.block_shape
        if context is None:
            context = self.context

        # Check that we are redistributing over the same communicator
        if context.comm != self.context.comm:
            raise ScalapyException("Can only redsitribute over the same MPI communicator.")

        dm = DistributedMatrix(self.shape, dtype=self.dtype, block_shape=block_shape, context=context)

        args = [self.shape[0], self.shape[1], self, dm, self.context]

        # Prepare call table
        call_table = {'S': (ll.psgemr2d, args),
                      'D': (ll.pdgemr2d, args),
                      'C': (ll.pcgemr2d, args),
                      'Z': (ll.pzgemr2d, args)}

        # Call routine
        func, args = call_table[self.sc_dtype]
        func(*args)

        return dm

    def min(self, op=np.ndarray.min, mpi_op=MPI.MIN):
        """Minimum of a matrix."""
        buffer = np.empty(1, dtype=real_equiv(self.dtype))
        buffer[:] = op(self.local_array)
        self.context.comm.Allreduce(MPI.IN_PLACE, buffer, op=mpi_op)
        return buffer[0]

    def max(self):
        return self.min(op=np.ndarray.max, mpi_op=MPI.MAX)

    transpose = transpose
    conj = conj
    hconj = hconj

    @property
    def T(self):
        return self.transpose()

    @property
    def C(self):
        return self.conj()

    @property
    def H(self):
        return self.hconj()


def dot_mat_mat(a, b, trans_a='N', trans_b='N', alpha=1., beta=0., out=None):
    """
    Matrix-matrix dot product.

    Parameters
    ----------
    a : DistributedMatrix
    b : DistributedMatrix
        Matrices to multiply.
    trans_a : str
        An optional operation on self.
    trans_b : str
        An optional operation on other.
    alpha : float
        Pre-factor for the product.
    beta : float
        In case out is specified, the result will be
        appended to ``out * beta``.
    out : DistributedMatrix
        The output array.

    Returns
    -------
    result : DistributedMatrix
        The resulting product.
    """
    # TODO: ScalapyExceptions should be just ValueErrors
    if trans_a not in 'NTC':
        raise ScalapyException(f"trans_a={trans_a} not in 'NTC'")
    if trans_b not in 'NTC':
        raise ScalapyException(f"trans_b={trans_b} not in 'NTC'")
    a.assert_same_distribution(b)
    if a.dtype != b.dtype:
        raise ScalapyException(f"a.dtype={a.dtype} != b.dtype={b.dtype}")

    m = a.shape[0] if trans_a == 'N' else a.shape[1]
    n = b.shape[1] if trans_b == 'N' else b.shape[0]
    k = a.shape[1] if trans_a == 'N' else a.shape[0]
    l = b.shape[0] if trans_b == 'N' else b.shape[1]

    if l != k:
        raise ScalapyException(f"dimension mismatch a.shape={a.shape} trans_a={trans_a} and "
                               f"b.shape={b.shape} trans_b={trans_b}")

    # TODO: fix small matrices?
    a.assert_not_tiny("A")
    b.assert_not_tiny("B")

    if out is not None:
        a.assert_same_distribution(out)
        if out.shape != (m, n):
            raise ScalapyException(f"out.shape={out.shape} != {(m, n)}")
        if a.dtype != out.dtype:
            raise ScalapyException(f"a.dtype={a.dtype} != out.dtype={out.dtype}")
    else:
        out = DistributedMatrix([m, n], dtype=a.dtype, block_shape=a.block_shape, context=a.context)
    args = [trans_a, trans_b, m, n, k, alpha, a, b, beta, out]

    call_table = {'S': (ll.psgemm, args),
                  'C': (ll.pcgemm, args),
                  'D': (ll.pdgemm, args),
                  'Z': (ll.pzgemm, args)}

    func, args = call_table[a.sc_dtype]
    func(*args)
    return out


def dot_mat_vec(a, v, left=False):
    """
    An matrix-vector product with a numpy array.

    ij, j -> i

    or

    ij, i -> j

    Parameters
    ----------
    a : DistributedMatrix
        The matrix to multiply.
    v : np.ndarray
        The vector to multiply.
    left : bool
        If True, performs the `v @ A` product.
        Otherwise, perform the `A @ v` product.

    Returns
    -------
    result : np.ndarray
        The resulting vector.
    """
    if v.ndim != 1:
        raise ValueError(f"the input v is not a vector: v.shape={v.shape}")
    # TODO: optimize memory
    if left:
        return (a * v[:, None]).sum(axis=0)
    else:
        return (a * v[None, :]).sum(axis=1)


def fromnumpy(mat, rank=None, out=None):
    """
    Distributes numpy matrix.

    Parameters
    ----------
    mat : np.ndarray
        The matrix to distribute.
    rank : int
        Indicates that the matrix `mat` is only available
        in the process specified by this rank.
    out : DistributedMatrix
        The output to write to.

    Returns
    -------
    out : DistributedMatrix
        The resulting distributed matrix.
    """
    # Broadcast if rank is not set.
    if rank is not None:
        comm = default_grid_context.comm

        # Double check that rank is valid.
        if rank < 0 or rank >= comm.size:
            raise ScalapyException("Invalid rank.")

        if comm.rank == rank:
            if mat.ndim != 2:
                raise ScalapyException("Array must be 2d.")

            mat = np.asfortranarray(mat)
            mat_shape = mat.shape
            mat_dtype = mat.dtype.type
        else:
            mat_shape = None
            mat_dtype = None

        mat_shape = comm.bcast(mat_shape, root=rank)
        mat_dtype = comm.bcast(mat_dtype, root=rank)

        if out is not None:
            if out.shape != mat_shape:
                raise ValueError(f"out.shape mismatch: expected {mat_shape}, found {out.shape}")
            if out.dtype != mat_dtype:
                raise ValueError(f"out.dtype mismatch: expected {mat_dtype}, found {out.dtype}")
        else:
            out = DistributedMatrix(mat_shape, dtype=mat_dtype)

        if mat_shape[0] != 0 and mat_shape[1] != 0:
            # Each process should receive its local sections.
            rreq = comm.Irecv([out.local_array, out.mpi_dtype], source=rank, tag=0)

            if comm.rank == rank:
                # Post each send
                reqs = [comm.Isend([mat, out._darr_list[dt]], dest=dt, tag=0)
                        for dt in range(comm.size)]

                # Wait for requests to complete
                MPI.Prequest.Waitall(reqs)

            rreq.Wait()

    else:
        if mat.ndim != 2:
            raise ScalapyException("Array must be 2d.")

        out = DistributedMatrix(mat.shape, dtype=mat.dtype.type)

        mat = np.asfortranarray(mat)
        out._load_array(mat)

    return out


def fromsparse_csr(source, rank=None, out=None):
    """
    Distributes a sparse CSR matrix.

    Parameters
    ----------
    source : sparse.csr_matrix
        The sparse matrix to distribute.
    rank : int
        Indicates that the source object is only available
        in the process specified by this rank.
    out : DistributedMatrix
        The output to write to.

    Returns
    -------
    out : DistributedMatrix
        The resulting distributed matrix.
    """
    comm = default_grid_context.comm
    # just broadcast the entire sparse matrix which is, presumably, not too large
    if rank is not None:
        source = source if rank == comm.rank else None
        source = comm.bcast(source, rank)

    if out is None:
        out = DistributedMatrix(source.shape, source.dtype.type)
    else:
        if out.shape != source.shape:
            raise ValueError(f"out.shape mismatch: expected {source.shape}, found {out.shape}")
        if out.dtype != source.dtype:
            raise ValueError(f"out.dtype mismatch: expected {source.dtype}, found {out.dtype}")

    source[out.row_indices(), :][:, out.col_indices()].toarray(out=out.local_array)
    return out
