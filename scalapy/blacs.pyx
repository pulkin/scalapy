from mpi4py import MPI

from mpi4py cimport MPI
from blacs cimport *


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
        Cfree_blacs_system_handle(<int>self.handle)


class GridContext:
    def __init__(self, int n_rows, int n_cols, order="Row", blacs_context=None, comm=None):
        if blacs_context is None:
            blacs_context = BlacsContext(comm)
        self.blacs_context = blacs_context
        self.order = order

        cdef int handle = <int>blacs_context.handle
        order = order.encode()
        assert order == b"Row", "only row order is supported"
        Cblacs_gridinit(&handle, order, n_rows, n_cols)
        self.handle = handle
        # test
        self.get_info()

    def get_info(self):
        """
        Fetch the process grid info.

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
        Cblacs_gridexit(<int>self.handle)
