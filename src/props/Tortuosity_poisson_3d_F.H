#ifndef TORTUOSITY_POISSON_F_H_
#define TORTUOSITY_POISSON_F_H_

#include <AMReX_BLFort.H>

#ifdef __cplusplus
extern "C" {
#endif

  void tortuosity_poisson_flux (const int* lo, const int* hi,
                               amrex_real* fx, const int* fxlo, const int* fxhi,
                               amrex_real* fy, const int* fylo, const int* fyhi,
                               amrex_real* fz, const int* fzlo, const int* fzhi,
                               const amrex_real* sol, const int* slo, const int* shi, const int* scomp,
                               const amrex_real* dxinv, const int face_only);

  void tortuosity_poisson_fio (const int* lo, const int* hi,
                               const amrex_real* fx, const int* fxlo, const int* fxhi,
                               const amrex_real* fy, const int* fylo, const int* fyhi,
                               const amrex_real* fz, const int* fzlo, const int* fzhi,
                               const int* dir, amrex_real* flux_in, amrex_real* flux_out);

  void tortuosity_poisson_update (const int* lo, const int* hi,
                                  const amrex_real* p, const int* plo, const int* phi, const int* pcomp,
                                  const amrex_real* n, const int* nlo, const int* nhi, const int* ncomp,
                                  amrex_real* fx, const int* fxlo, const int* fxhi,
                                  amrex_real* fy, const int* fylo, const int* fyhi,
                                  amrex_real* fz, const int* fzlo, const int* fzhi,
                                  const amrex_real* dxinv, const amrex_real* dt);

#ifdef __cplusplus
}
#endif

#endif
