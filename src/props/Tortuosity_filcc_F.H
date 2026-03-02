#include <AMReX_BLFort.H>
#ifdef __cplusplus
extern "C"
{
#endif


    void tortuosity_filct(amrex_real*, const int*, const int*, const int*, // cell-type fab
                          const int*, const int*, const int*, const int*, // phase fab
                          const int*, const int*, // domain dimensions
                          const int*); // phase index

    void tortuosity_filbc(amrex_real*, const int*, const int*, const int*, // fab
                          const int*, const int*, const int*, const int*, // phase fab
                          const int*, const int*, // domain dimensions
                          const amrex_real*, const amrex_real*, // boundary values
                          const int*); // boundary condition types

    void tortuosity_filic(amrex_real*, const int*, const int*, const int*, // fab
                          const int*, const int*, const int*, const int*, // phase fab
                          const int*, const int*, // valid box w/o ghosts
                          const int*, const int*, // domain dimensions
                          const amrex_real*, const amrex_real*, // boundary values
                          const int*, // phase
                          const int*); // direction

    void tortuosity_remspot(const int*, const int*, const int*, const int*, // phase fab
                            const int*, const int*,  // valid box dimensions w/o ghosts
                            const int*, const int*); // domain dimensions
#ifdef __cplusplus
}
#endif