#ifndef OPENIMPALA_EFFDIFF_FILL_MTX_F_H_
#define OPENIMPALA_EFFDIFF_FILL_MTX_F_H_

#include <AMReX_BLFort.H> // Provides amrex_real and other Fortran interoperability features

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C interface for the Fortran `effdiff_fillmtx` subroutine.
 *
 * This subroutine fills the HYPRE matrix coefficients ('a_out'),
 * Right-Hand Side ('rhs_out'), and initial guess ('xinit_out') for a
 * single box defined by 'valid_bx_lo':'valid_bx_hi'. It discretizes
 * the homogenization cell problem:
 *   ∇_ξ ⋅ (D ∇_ξ χ_k) = -∇_ξ ⋅ (D ê_k)
 *
 * The material property D is implicitly 1 in cells marked active by
 * 'active_mask_ptr' (pore phase) and 0 otherwise (solid phase).
 * The direction 'k' of the unit vector ê_k is specified by 'dir_k_in'.
 *
 * The discretization includes:
 * - The RHS term -∇_ξ ⋅ (D ê_k).
 * - The LHS Laplacian term ∇_ξ ⋅ (∇_ξ χ_k) for active cells.
 * - Internal Neumann boundary conditions n̂ ⋅ ∇_ξ χ_k = -n̂ ⋅ ê_k on
 *   interfaces between active and inactive cells.
 * - Decoupling of inactive cells (A_ii=1, A_ij=0, b_i=0).
 *
 * @param[out]  a_out             Pointer to matrix stencil coefficient array (size npts_valid*7, flattened).
 *                                  Assumed HYPRE 7-point stencil order: Center, -X, +X, -Y, +Y, -Z, +Z.
 * @param[out]  rhs_out           Pointer to RHS array (size npts_valid).
 * @param[out]  xinit_out         Pointer to initial guess array for χ_k (size npts_valid).
 * @param[in]   npts_valid        Pointer to the number of points/cells in the 'valid_bx' region.
 * @param[in]   active_mask_ptr   Pointer to the active_mask data array (INTEGER, 1=active/pore, 0=inactive/solid).
 *                                  Accessed as 3D in Fortran.
 * @param[in]   mask_lo           Pointer to the lower bound of the 'active_mask_ptr' array (Fortran indexing, including ghosts).
 * @param[in]   mask_hi           Pointer to the upper bound of the 'active_mask_ptr' array (Fortran indexing, including ghosts).
 * @param[in]   valid_bx_lo       Pointer to the lower bound of the current valid box/region to fill (Fortran indexing).
 * @param[in]   valid_bx_hi       Pointer to the upper bound of the current valid box/region to fill (Fortran indexing).
 * @param[in]   domain_lo         Pointer to the lower bound of the overall simulation domain (Fortran indexing).
 * @param[in]   domain_hi         Pointer to the upper bound of the overall simulation domain (Fortran indexing).
 * @param[in]   cell_sizes_in     Pointer to an array[3] of cell sizes [dx, dy, dz].
 * @param[in]   dir_k_in          Pointer to an integer representing the direction 'k' of ê_k (0 for X, 1 for Y, 2 for Z).
 * @param[in]   verbose_level_in  Pointer to an integer for the verbosity/debug print level in the Fortran kernel.
 *
 * @warning Ensure 'npts_valid' matches the number of points in the box defined by 'valid_bx_lo':'valid_bx_hi'.
 * @warning The 'active_mask_ptr' array must have its ghost cells filled appropriately before calling,
 *          as the kernel will access neighbor information.
 */
void effdiff_fillmtx (
    amrex_real* a_out,
    amrex_real* rhs_out,
    amrex_real* xinit_out,
    const int* npts_valid,
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
