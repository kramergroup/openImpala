#ifndef TORTUOSITY_HYPRE_FILL_F_H_
#define TORTUOSITY_HYPRE_FILL_F_H_

#include <AMReX_BLFort.H>

#ifdef __cplusplus
extern "C" {
#endif

  void tortuosity_fillmtx (amrex_real* a, amrex_real* rhs, amrex_real* xinit, const size_t* n_a, // matrix, xinit, and rhs coefficients
                           const int* p, const int* plo, const int* phi, // phase fab
                           const int* bxlo, const int* bxhi, // validbox w/o ghosts
                           const int* domlo, const int* domhi, // domain 
                           const amrex_real* vlo, const amrex_real* vhi, // values at domain boundary
                           const int* phase, // fluid phase index
                           const Direction* dir); // flow direction
#ifdef __cplusplus
}
#endif

#endif
