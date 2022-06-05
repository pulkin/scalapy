include "mpi4py/mpi.pxi"

## BLACS routines
cdef extern:
    void Cblacs_gridinit( int * icontxt, char * order, int nprow, int npcol )
    void Cblacs_gridexit(int icontxt)
    void Cblacs_gridinfo( int icontxt, int * nprow, int * npcol, int * myprow, int * mypcol )
    int Csys2blacs_handle(MPI_Comm SysCtxt)
    void Cfree_blacs_system_handle(int icontxt)
