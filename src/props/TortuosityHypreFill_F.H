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
 * on phase data ('p'), domain boundaries ('domlo':'domhi'), flow direction ('dir'),
 * and boundary values ('vlo', 'vhi'). Implements a 7-point stencil discretization
 * of the Poisson/Laplace equation, handling internal phase boundaries (zero Neumann)
 * and domain boundaries perpendicular to flow (Dirichlet). Uses grid spacing
 * provided via 'dxinv'.
 *
 * @param[out]   a       Pointer to matrix coefficient array (size nval*7, flattened). Stencil: C,W,E,S,N,B,T assumed.
 * @param[out]   rhs     Pointer to RHS array (size nval).
 * @param[out]   xinit   Pointer to initial guess array (size nval).
 * @param[in]    nval    Pointer to number of points in the box (*bxlo to *bxhi). Passed by reference.
 * @param[in]    p       Pointer to phase data array (INTEGER). Accessed as 4D in Fortran (comp_phase=1).
 * @param[in]    p_lo    Pointer to lower bound of p array (Fortran indexing).
 * @param[in]    p_hi    Pointer to upper bound of p array (Fortran indexing, incl. ghosts).
 * @param[in]    bxlo    Pointer to lower bound of current box (Fortran indexing).
 * @param[in]    bxhi    Pointer to upper bound of current box (Fortran indexing).
 * @param[in]    domlo   Pointer to lower bound of domain (Fortran indexing).
 * @param[in]    domhi   Pointer to upper bound of domain (Fortran indexing).
 * @param[in]    dxinv   Pointer to array[3] of inverse grid spacing squared [1/dx^2, 1/dy^2, 1/dz^2].
 * @param[in]    vlo     Pointer to low boundary value (Dirichlet BC). Passed by reference.
 * @param[in]    vhi     Pointer to high boundary value (Dirichlet BC). Passed by reference.
 * @param[in]    phase   Pointer to phase ID considered conductive. Passed by reference.
 * @param[in]    dir     Pointer to direction index (0=X, 1=Y, 2=Z). Passed by reference.
 *
 * @warning Assumes Fortran uses 1-based indexing for components when accessing 'p'.
 * @warning Ensure 'nval' matches the number of points in the box bxlo:bxhi.
 * @warning Assumed HYPRE stencil convention: 1=Center, 2=-X(W), 3=+X(E), 4=-Y(S), 5=+Y(N), 6=-Z(B), 7=+Z(T). Verify this!
 */
void tortuosity_fillmtx (
    amrex_real* a,
    amrex_real* rhs,
    amrex_real* xinit,
    const int* nval,
    const int* p,
    const int* p_lo,
    const int* p_hi,
    const int* bxlo,
    const int* bxhi,
    const int* domlo,
    const int* domhi,
    const amrex_real* dxinv,
    const amrex_real* vlo,
    const amrex_real* vhi,
    const int* phase,
    const int* dir
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // OPENIMPALA_TORTUOSITY_HYPRE_FILL_F_H_
