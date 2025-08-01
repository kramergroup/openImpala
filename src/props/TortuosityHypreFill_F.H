#ifndef OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_
#define OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_

#include <AMReX_BLFort.H> // Provides amrex_real

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C interface for the multi-phase Fortran matrix fill subroutine.
 *
 * Fills the HYPRE matrix ('a'), Right-Hand Side ('rhs'), and initial guess ('xinit')
 * for a single box. This routine discretizes the variable-coefficient Poisson equation,
 * nabla(D(x) * nabla(phi)) = 0, using a 7-point finite-volume stencil.
 *
 * It operates on cells marked as active (value=1) in the 'active_mask'. The spatially
 * varying coefficient D(x) is provided by the 'D' array. At interfaces between active
 * cells, the conductivity is calculated using a robust harmonic mean.
 *
 * Dirichlet boundary conditions are applied at domain faces perpendicular to the flow
 * direction ('dir'). A zero Neumann condition is implicitly applied at boundaries
 * between active and inactive cells. Inactive cells are decoupled from the system
 * (Aii=1, Aij=0, bi=0).
 *
 * @param[out]    a             Pointer to matrix coefficient array (size nval*7).
 * @param[out]    rhs           Pointer to RHS array (size nval).
 * @param[out]    xinit         Pointer to initial guess array (size nval).
 * @param[in]     nval          Pointer to the number of points in the current box.
 * @param[in]     D             Pointer to the spatially varying coefficient data (amrex_real).
 * @param[in]     d_lo          Pointer to the lower bound of the D array.
 * @param[in]     d_hi          Pointer to the upper bound of the D array (including ghosts).
 * @param[in]     mask          Pointer to the activity mask array (INTEGER, 1=active, 0=inactive).
 * @param[in]     mask_lo       Pointer to the lower bound of the mask array.
 * @param[in]     mask_hi       Pointer to the upper bound of the mask array (including ghosts).
 * @param[in]     bxlo          Pointer to the lower bound of the current box.
 * @param[in]     bxhi          Pointer to the upper bound of the current box.
 * @param[in]     domlo         Pointer to the lower bound of the full domain.
 * @param[in]     domhi         Pointer to the upper bound of the full domain.
 * @param[in]     dx            Pointer to an array[3] of grid spacings [dx, dy, dz].
 * @param[in]     vlo           Pointer to the low boundary value (Dirichlet BC).
 * @param[in]     vhi           Pointer to the high boundary value (Dirichlet BC).
 * @param[in]     dir           Pointer to the direction index (0=X, 1=Y, 2=Z).
 * @param[in]     verbose       Pointer to the verbosity level for debug printing.
 */
void tortuosity_fillmtx_mp(
    amrex_real* a,
    amrex_real* rhs,
    amrex_real* xinit,
    const int* nval,
    const amrex_real* D,
    const int* d_lo,
    const int* d_hi,
    const int* mask,
    const int* mask_lo,
    const int* mask_hi,
    const int* bxlo,
    const int* bxhi,
    const int* domlo,
    const int* domhi,
    const amrex_real* dx,
    const amrex_real* vlo,
    const amrex_real* vhi,
    const int* dir,
    const int* verbose);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_
