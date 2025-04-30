#ifndef OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_
#define OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_

#include <AMReX_BLFort.H> // Provides amrex_real

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C interface for the Fortran tortuosity_fillmtx subroutine.
 *
 * Fills the HYPRE matrix coefficients ('a'), Right-Hand Side ('rhs'),
 * and initial guess ('xinit') for a single box ('bxlo':'bxhi') based
 * on an activity mask, phase data ('p'), domain boundaries ('domlo':'domhi'),
 * flow direction ('dir'), and boundary values ('vlo', 'vhi').
 * Implements a 7-point stencil discretization of the Poisson/Laplace equation
 * primarily on cells marked active by 'active_mask', handling boundaries between
 * active and inactive cells (zero Neumann) and domain boundaries perpendicular
 * to flow (Dirichlet). Uses grid spacing provided via 'dxinv'. Inactive cells
 * are decoupled (Aii=1, Aij=0, bi=0, xinit=0).
 *
 * @param[out]  a           Pointer to matrix coefficient array (size nval*7, flattened). Stencil: C,W,E,S,N,B,T assumed.
 * @param[out]  rhs         Pointer to RHS array (size nval).
 * @param[out]  xinit       Pointer to initial guess array (size nval).
 * @param[in]   nval        Pointer to number of points in the box (*bxlo to *bxhi). Passed by reference.
 * @param[in]   p           Pointer to phase data array (INTEGER). Accessed as 4D in Fortran. (May be optional if mask is sufficient).
 * @param[in]   p_lo        Pointer to lower bound of p array (Fortran indexing).
 * @param[in]   p_hi        Pointer to upper bound of p array (Fortran indexing, incl. ghosts).
 * @param[in]   active_mask Pointer to activity mask array (INTEGER, 1=active, 0=inactive). <<< NEW
 * @param[in]   mask_lo     Pointer to lower bound of active_mask array (Fortran indexing). <<< NEW
 * @param[in]   mask_hi     Pointer to upper bound of active_mask array (Fortran indexing, incl. ghosts). <<< NEW
 * @param[in]   bxlo        Pointer to lower bound of current box (Fortran indexing).
 * @param[in]   bxhi        Pointer to upper bound of current box (Fortran indexing).
 * @param[in]   domlo       Pointer to lower bound of domain (Fortran indexing).
 * @param[in]   domhi       Pointer to upper bound of domain (Fortran indexing).
 * @param[in]   dxinv       Pointer to array[3] of inverse grid spacing squared [1/dx^2, 1/dy^2, 1/dz^2].
 * @param[in]   vlo         Pointer to low boundary value (Dirichlet BC). Passed by reference.
 * @param[in]   vhi         Pointer to high boundary value (Dirichlet BC). Passed by reference.
 * @param[in]   phase_unused Pointer to phase ID considered conductive (Now potentially unused by Fortran, as mask dictates activity). Passed by reference.
 * @param[in]   dir         Pointer to direction index (0=X, 1=Y, 2=Z). Passed by reference.
 *
 * @warning Assumes Fortran uses 1-based indexing for components when accessing 'p' and 'active_mask'.
 * @warning Ensure 'nval' matches the number of points in the box bxlo:bxhi.
 * @warning Assumed HYPRE stencil convention: Center,W,E,S,N,B,T matches Fortran indices 1..7 in calculation. Verify this!
 */
void tortuosity_fillmtx (
    amrex_real* a,
    amrex_real* rhs,
    amrex_real* xinit,
    const int* nval,
    const int* p,
    const int* p_lo,
    const int* p_hi,
    const int* active_mask, // <<< NEW
    const int* mask_lo,     // <<< NEW
    const int* mask_hi,     // <<< NEW
    const int* bxlo,
    const int* bxhi,
    const int* domlo,
    const int* domhi,
    const amrex_real* dxinv,
    const amrex_real* vlo,
    const amrex_real* vhi,
    const int* phase_unused, // <<< Renamed
    const int* dir,
    const int* debug_print_level // <<< ADDED THIS LINE
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_
