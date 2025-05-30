#ifndef TORTUOSITY_POISSON_F_H_
#define TORTUOSITY_POISSON_F_H_

#include <AMReX_BLFort.H> // Provides amrex_real and Fortran interface conventions

#ifdef __cplusplus
extern "C" {
#endif

// Note on Fortran vs C Indexing:
// Fortran arrays in the module use 1-based indexing (e.g., components 1 and 2).
// The C++ code calling these functions uses 0-based indexing.
// Ensure correct mapping is handled in the C++ wrapper code that calls these functions.

//-----------------------------------------------------------------------
// Flux Calculation Kernel
//-----------------------------------------------------------------------

/**
 * @brief Fortran kernel to calculate face fluxes based on a cell-centered solution field.
 *
 * Calculates fluxes using finite differences: Flux ~ (sol(i) - sol(i-1))/dx.
 * Sets flux to zero if either adjacent cell is 'blocked' (cell type 0).
 * Assumes unit diffusivity (D=1) or diffusivity is handled elsewhere.
 *
 * @param[in] lo     Pointer to lower corner of the valid region (i,j,k) in the C++ indexing space.
 * @param[in] hi     Pointer to upper corner of the valid region (i,j,k) in the C++ indexing space.
 * @param[out] fx    Pointer to the X-face flux array data (output).
 * @param[in] fxlo   Pointer to lower corner of the fx array Fortran bounds.
 * @param[in] fxhi   Pointer to upper corner of the fx array Fortran bounds.
 * @param[out] fy    Pointer to the Y-face flux array data (output).
 * @param[in] fylo   Pointer to lower corner of the fy array Fortran bounds.
 * @param[in] fyhi   Pointer to upper corner of the fy array Fortran bounds.
 * @param[out] fz    Pointer to the Z-face flux array data (output).
 * @param[in] fzlo   Pointer to lower corner of the fz array Fortran bounds.
 * @param[in] fzhi   Pointer to upper corner of the fz array Fortran bounds.
 * @param[in] sol    Pointer to the input solution array data (contains phi at Fortran comp 1, cell type at Fortran comp 2).
 * @param[in] slo    Pointer to lower corner of the sol array Fortran bounds (including ghost cells).
 * @param[in] shi    Pointer to upper corner of the sol array Fortran bounds (including ghost cells).
 * @param[in] dxinv  Pointer to array[3] of inverse cell sizes (1/dx, 1/dy, 1/dz).
 */
void tortuosity_poisson_flux (const int* lo, const int* hi,
                              amrex_real* fx, const int* fxlo, const int* fxhi,
                              amrex_real* fy, const int* fylo, const int* fyhi,
                              amrex_real* fz, const int* fzlo, const int* fzhi,
                              const amrex_real* sol, const int* slo, const int* shi,
                              const amrex_real* dxinv);


//-----------------------------------------------------------------------
// Solution Update Kernel
//-----------------------------------------------------------------------

/**
 * @brief Fortran kernel to update the solution field using Forward Euler.
 *
 * Performs one step: n = p + dt * Div(Flux), approximating d(phi)/dt = Div(Flux).
 * Assumes fluxes were computed by tortuosity_poisson_flux.
 * Assumes p and n are 4D arrays in Fortran including component dimension.
 *
 * @param[in] lo     Pointer to lower corner of the valid region (i,j,k) to update.
 * @param[in] hi     Pointer to upper corner of the valid region (i,j,k) to update.
 * @param[in] p      Pointer to the old solution array data (phi at Fortran comp 1).
 * @param[in] plo    Pointer to lower corner of the p array Fortran bounds.
 * @param[in] phi    Pointer to upper corner of the p array Fortran bounds.
 * @param[out] n     Pointer to the new solution array data (phi at Fortran comp 1) (output).
 * @param[in] nlo    Pointer to lower corner of the n array Fortran bounds.
 * @param[in] nhi    Pointer to upper corner of the n array Fortran bounds.
 * @param[in] fx     Pointer to the X-face flux array data (input).
 * @param[in] fxlo   Pointer to lower corner of the fx array Fortran bounds.
 * @param[in] fxhi   Pointer to upper corner of the fx array Fortran bounds.
 * @param[in] fy     Pointer to the Y-face flux array data (input).
 * @param[in] fylo   Pointer to lower corner of the fy array Fortran bounds.
 * @param[in] fyhi   Pointer to upper corner of the fy array Fortran bounds.
 * @param[in] fz     Pointer to the Z-face flux array data (input).
 * @param[in] fzlo   Pointer to lower corner of the fz array Fortran bounds.
 * @param[in] fzhi   Pointer to upper corner of the fz array Fortran bounds.
 * @param[in] ncomp  Pointer to the number of components in p and n arrays. Passed by reference. <<< ADDED >>>
 * @param[in] dxinv  Pointer to array[3] of inverse cell sizes (1/dx, 1/dy, 1/dz).
 * @param[in] dt     Pointer to the time step (or relaxation factor). Passed by reference from Fortran default bind(c).
 */
void tortuosity_poisson_update (const int* lo, const int* hi,
                                const amrex_real* p, const int* plo, const int* phi,
                                amrex_real* n, const int* nlo, const int* nhi,
                                const amrex_real* fx, const int* fxlo, const int* fxhi,
                                const amrex_real* fy, const int* fylo, const int* fyhi,
                                const amrex_real* fz, const int* fzlo, const int* fzhi,
                                const int* ncomp, // <<< ADDED ncomp argument >>>
                                const amrex_real* dxinv, const amrex_real* dt);


//-----------------------------------------------------------------------
// Boundary Flux Summation Kernel
//-----------------------------------------------------------------------

/**
 * @brief Fortran kernel to sum fluxes on the low and high boundaries of a box.
 *
 * Calculates the total flux entering (flux_in) through the face at lo(dir)
 * and exiting (flux_out) through the face at hi(dir)+1 for the specified direction,
 * considering only the portion relevant to the box defined by lo/hi.
 * Checks ownership of boundary face data using flux array bounds (fxlo/hi etc.).
 *
 * @param[in] lo       Pointer to lower corner of the valid region (i,j,k) for this box.
 * @param[in] hi       Pointer to upper corner of the valid region (i,j,k) for this box.
 * @param[in] fx       Pointer to the X-face flux array data (input).
 * @param[in] fxlo     Pointer to lower corner of the fx array Fortran bounds.
 * @param[in] fxhi     Pointer to upper corner of the fx array Fortran bounds.
 * @param[in] fy       Pointer to the Y-face flux array data (input).
 * @param[in] fylo     Pointer to lower corner of the fy array Fortran bounds.
 * @param[in] fyhi     Pointer to upper corner of the fy array Fortran bounds.
 * @param[in] fz       Pointer to the Z-face flux array data (input).
 * @param[in] fzlo     Pointer to lower corner of the fz array Fortran bounds.
 * @param[in] fzhi     Pointer to upper corner of the fz array Fortran bounds.
 * @param[in] dir      Pointer to the direction index (0=X, 1=Y, 2=Z). Passed by reference from Fortran default bind(c).
 * @param[out] flux_in Pointer to store the total flux entering on the low face (output).
 * @param[out] flux_out Pointer to store the total flux exiting on the high face (output).
 */
void tortuosity_poisson_fio (const int* lo, const int* hi,
                             const amrex_real* fx, const int* fxlo, const int* fxhi,
                             const amrex_real* fy, const int* fylo, const int* fyhi,
                             const amrex_real* fz, const int* fzlo, const int* fzhi,
                             const int* dir, amrex_real* flux_in, amrex_real* flux_out);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // TORTUOSITY_POISSON_F_H_
