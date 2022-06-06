from mpi4py import MPI

from mpi4py cimport MPI
from blacs cimport *
from libc.math cimport sqrt, ceil


cdef int int_sqrt(int x):
    cdef int i = <int>ceil(sqrt(x))
    while x % i:
        i -= 1
    return i


class BLACSException(Exception):
    pass


class BlacsContext:
    def __init__(self, comm=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.handle = Csys2blacs_handle(<MPI_Comm>(<MPI.Comm>comm).ob_mpi)

    def __int__(self):
        return self.handle

    def __del__(self):
        try:
            Cfree_blacs_system_handle(<int>self.handle)
        except AttributeError:
            pass

    def __str__(self):
        return f"BlacsContext-{self.handle}"


class GridContext:
    def __init__(self, shape, order="Row", blacs_context=None, comm=None):
        cdef int n_rows, n_cols

        if blacs_context is None:
            blacs_context = BlacsContext(comm)
        self.blacs_context = blacs_context
        self.order = order
        if shape is None:
            size = blacs_context.comm.size
            n_cols = int_sqrt(size)
            n_rows = size // n_cols
        else:
            n_rows, n_cols = shape

        cdef int handle = <int>blacs_context.handle
        order = order.encode()
        assert order == b"Row", "only row order is supported"
        Cblacs_gridinit(&handle, order, n_rows, n_cols)
        self.handle = handle
        # test
        self.get_info()

    def get_info(self):
        """
        Fetch the grid info.

        Returns
        -------
        n_rows : int
        n_cols : int
            Grid size.
        row : int
        col : int
            Grid position.

        Raises
        ------
        BLACSException
            If process grid undefined.
        """
        cdef int handle = <int>self.handle, n_rows, n_cols, row, col
        Cblacs_gridinfo(handle, &n_rows, &n_cols, &row, &col)
        if n_rows == -1:
            raise BLACSException("grid context does not exist")
        return (n_rows, n_cols, row, col)

    def __int__(self):
        return self.handle

    def __del__(self):
        try:
            Cblacs_gridexit(<int>self.handle)
        except AttributeError:
            pass

    def __str__(self):
        return f"GridContext-{self.handle}"