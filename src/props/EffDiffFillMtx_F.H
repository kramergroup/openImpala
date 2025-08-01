#ifndef OPENIMPALA_EFFDIFF_FILL_MTX_F_H_
#define OPENIMPALA_EFFDIFF_FILL_MTX_F_H_

#include <AMReX_BLFort.H> // Provides amrex_real and other Fortran interoperability features

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C interface for the Fortran `effdiff_fillmtx_mp` subroutine (multi-phase).
 *
 * This subroutine fills the HYPRE matrix coefficients ('a_out'),
 * Right-Hand Side ('rhs_out'), and initial guess ('xinit_out') for a
 * single box. It discretizes the multi-phase homogenization cell problem:
 * ∇ ⋅ (D(x) ∇χ_k) = -∇ ⋅ (D(x) ê_k)
 *
 * The spatially varying material property D(x) is provided by the `D_ptr` array.
 * The solver operates on cells marked as active by 'active_mask_ptr'.
 * The direction 'k' of the unit vector ê_k is specified by 'dir_k_in'.
 *
 * The discretization includes:
 * - The LHS operator ∇ ⋅ (D(x)∇χ_k), using a harmonic mean for face conductivity.
 * - The RHS source term -∇ ⋅ (D(x)ê_k), using a central difference of D(x).
 * - Internal Neumann boundary conditions n̂ ⋅ ∇χ_k = -n̂ ⋅ ê_k on
 * interfaces between active and inactive cells.
 * - Decoupling of inactive cells (A_ii=1, A_ij=0, b_i=0).
 *
 * @param[out]    a_out             Pointer to matrix stencil coefficient array (size npts_valid*7).
 * @param[out]    rhs_out           Pointer to RHS array (size npts_valid).
 * @param[out]    xinit_out         Pointer to initial guess array for χ_k (size npts_valid).
 * @param[in]     npts_valid        Pointer to the number of cells in the 'valid_bx' region.
 * @param[in]     D_ptr             Pointer to the spatially varying coefficient data array (amrex_real).
 * @param[in]     d_lo              Pointer to the lower bound of the 'D_ptr' array.
 * @param[in]     d_hi              Pointer to the upper bound of the 'D_ptr' array (including ghosts).
 * @param[in]     active_mask_ptr   Pointer to the active_mask data array (INTEGER, 1=active, 0=inactive).
 * @param[in]     mask_lo           Pointer to the lower bound of the 'active_mask_ptr' array.
 * @param[in]     mask_hi           Pointer to the upper bound of the 'active_mask_ptr' array (including ghosts).
 * @param[in]     valid_bx_lo       Pointer to the lower bound of the current valid box to fill.
 * @param[in]     valid_bx_hi       Pointer to the upper bound of the current valid box to fill.
 * @param[in]     domain_lo         Pointer to the lower bound of the overall simulation domain.
 * @param[in]     domain_hi         Pointer to the upper bound of the overall simulation domain.
 * @param[in]     cell_sizes_in     Pointer to an array[3] of cell sizes [dx, dy, dz].
 * @param[in]     dir_k_in          Pointer to an integer representing the direction 'k' (0=X, 1=Y, 2=Z).
 * @param[in]     verbose_level_in  Pointer to an integer for the verbosity/debug print level.
 */
void effdiff_fillmtx_mp (
    amrex_real* a_out,
    amrex_real* rhs_out,
    amrex_real* xinit_out,
    const int* npts_valid,
    const amrex_real* D_ptr,
    const int* d_lo,
    const int* d_hi,
    const int* active_mask_ptr,
    const int* mask_lo,
    const int* mask_hi,
    const int* valid_bx_lo,
    const int* valid_bx_hi,
    const int* domain_lo,
    const int* domain_hi,
    const amrex_real* cell_sizes_in,
    const int* dir_k_in,
    const int* verbose_level_in
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // OPENIMPALA_EFFDIFF_FILL_MTX_F_H_
