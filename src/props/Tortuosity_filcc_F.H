#ifndef TORTUOSITY_FILCC_F_H_
#define TORTUOSITY_FILCC_F_H_

#include <AMReX_BLFort.H> // Provides amrex_real and Fortran interface conventions

#ifdef __cplusplus
extern "C" {
#endif

// Note on Fortran vs C/C++ Indexing:
// ===================================
// Fortran arrays in the corresponding module (tortuosity_filcc_module) use
// 1-based indexing (e.g., components 1 and 2). The C++ code calling these
// functions typically uses 0-based indexing. Ensure correct mapping is handled
// in the C++ wrapper code that calls these functions, especially for component indices.
// ===================================

//-----------------------------------------------------------------------
// Cell Type and Phase Data Manipulation Kernels
//-----------------------------------------------------------------------

/**
 * @brief Fills the cell type component (comp_ct) of array 'q' based on phase data in 'p'.
 *
 * Iterates over the domain defined by domlo/domhi. For each cell (i,j,k):
 * If p(i,j,k,comp_phase) matches the input 'phase', sets q(i,j,k,comp_ct) = cell_type_free (1).
 * Otherwise, sets q(i,j,k,comp_ct) = cell_type_blocked (0).
 * Assumes Fortran 1-based component indexing (comp_phase=1, comp_ct=2).
 *
 * @param[in,out] q       Pointer to the output array data (REAL). Component comp_ct is modified.
 * @param[in]    q_lo   Pointer to lower corner of the q array Fortran bounds (incl. ghosts).
 * @param[in]    q_hi   Pointer to upper corner of the q array Fortran bounds (incl. ghosts).
 * @param[in]    q_ncomp Pointer to the number of components in array q.
 * @param[in]    p      Pointer to the input phase array data (INTEGER). Component comp_phase is read.
 * @param[in]    p_lo   Pointer to lower corner of the p array Fortran bounds (incl. ghosts).
 * @param[in]    p_hi   Pointer to upper corner of the p array Fortran bounds (incl. ghosts).
 * @param[in]    p_ncomp Pointer to the number of components in array p.
 * @param[in]    domlo  Pointer to the domain lower corner indices (Fortran indexing).
 * @param[in]    domhi  Pointer to the domain upper corner indices (Fortran indexing).
 * @param[in]    phase  Pointer to the integer ID of the phase considered 'free'. (Passed by reference).
 */
void tortuosity_filct(amrex_real* q, const int* q_lo, const int* q_hi, const int* q_ncomp,
                      const int* p, const int* p_lo, const int* p_hi, const int* p_ncomp,
                      const int* domlo, const int* domhi,
                      const int* phase);

/**
 * @brief Removes isolated single cells (islands) from a phase field.
 *
 * Iterates over the valid box defined by bxlo/bxhi. For each cell, it checks
 * its phase (comp_phase) against its 6 neighbors. If no neighbor has the same phase,
 * the cell's phase is flipped (0->1 or 1->0). Modifies array 'q' in place.
 * Uses -1 to denote neighbors outside the physical domain defined by domlo/domhi.
 * Assumes Fortran 1-based component indexing (comp_phase=1).
 *
 * @param[in,out] q      Pointer to the phase array data (INTEGER). Component comp_phase is modified.
 * @param[in]    q_lo   Pointer to lower corner of the q array Fortran bounds (incl. ghosts).
 * @param[in]    q_hi   Pointer to upper corner of the q array Fortran bounds (incl. ghosts).
 * @param[in]    ncomp  Pointer to the number of components in array q.
 * @param[in]    bxlo   Pointer to lower corner of the valid box indices to iterate over.
 * @param[in]    bxhi   Pointer to upper corner of the valid box indices to iterate over.
 * @param[in]    domlo  Pointer to the physical domain lower corner indices.
 * @param[in]    domhi  Pointer to the physical domain upper corner indices.
 */
void tortuosity_remspot(int* q, const int* q_lo, const int* q_hi, const int* ncomp,
                        const int* bxlo, const int* bxhi,
                        const int* domlo, const int* domhi);


//-----------------------------------------------------------------------
// Boundary Condition and Initial Condition Kernels
//-----------------------------------------------------------------------

/**
 * @brief Fills ghost cells of array 'q' with external Dirichlet boundary values.
 *
 * Checks boundary flags in 'bc' for each face (low/high in X, Y, Z). If a face
 * has the `amrex_bc_ext_dir` flag, the corresponding ghost cells in 'q'
 * are filled with 'vlo' (for low faces) or 'vhi' (for high faces). Operates
 * on all components of 'q'.
 *
 * @param[in,out] q      Pointer to the array data (REAL) whose ghost cells are to be filled.
 * @param[in]    q_lo   Pointer to lower corner of the q array Fortran bounds (incl. ghosts).
 * @param[in]    q_hi   Pointer to upper corner of the q array Fortran bounds (incl. ghosts).
 * @param[in]    ncomp  Pointer to the number of components in array q.
 * @param[in]    domlo  Pointer to the domain lower corner indices.
 * @param[in]    domhi  Pointer to the domain upper corner indices.
 * @param[in]    vlo    Pointer to the boundary value for low faces. (Passed by reference).
 * @param[in]    vhi    Pointer to the boundary value for high faces. (Passed by reference).
 * @param[in]    bc     Pointer to the boundary condition flags array [Fortran: bc(amrex_spacedim, 2)].
 */
void tortuosity_filbc(amrex_real* q, const int* q_lo, const int* q_hi, const int* ncomp,
                      const int* domlo, const int* domhi,
                      const amrex_real* vlo, const amrex_real* vhi,
                      const int* bc);

/**
 * @brief Fills the initial condition for the potential field in array 'q'.
 *
 * Iterates over the valid box defined by lo/hi. For cells where the phase
 * in array 'p' matches the input 'phase', sets 'q' to a linear interpolation
 * between 'vlo' and 'vhi' along the specified 'dir'. Sets 'q' to 0.0 otherwise.
 * Operates on all components of 'q'.
 * Assumes Fortran 1-based component indexing (comp_phase=1 for reading p).
 *
 * @param[in,out] q      Pointer to the output array data (REAL) for initial condition.
 * @param[in]    q_lo   Pointer to lower corner of the q array Fortran bounds.
 * @param[in]    q_hi   Pointer to upper corner of the q array Fortran bounds.
 * @param[in]    ncomp  Pointer to the number of components in array q.
 * @param[in]    p      Pointer to the input phase array data (INTEGER). Component comp_phase is read.
 * @param[in]    p_lo   Pointer to lower corner of the p array Fortran bounds.
 * @param[in]    p_hi   Pointer to upper corner of the p array Fortran bounds.
 * @param[in]    p_ncomp Pointer to the number of components in array p.
 * @param[in]    lo     Pointer to lower corner of the valid box indices to fill.
 * @param[in]    hi     Pointer to upper corner of the valid box indices to fill.
 * @param[in]    domlo  Pointer to the domain lower corner indices.
 * @param[in]    domhi  Pointer to the domain upper corner indices.
 * @param[in]    vlo    Pointer to the boundary value at the low end for interpolation. (Passed by reference).
 * @param[in]    vhi    Pointer to the boundary value at the high end for interpolation. (Passed by reference).
 * @param[in]    phase  Pointer to the integer ID of the phase to initialize. (Passed by reference).
 * @param[in]    dir    Pointer to the direction index (0=X, 1=Y, 2=Z) for interpolation. (Passed by reference).
 */
void tortuosity_filic(amrex_real* q, const int* q_lo, const int* q_hi, const int* ncomp,
                      const int* p, const int* p_lo, const int* p_hi, const int* p_ncomp,
                      const int* lo, const int* hi,
                      const int* domlo, const int* domhi,
                      const amrex_real* vlo, const amrex_real* vhi,
                      const int* phase,
                      const int* dir);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // TORTUOSITY_FILCC_F_H_
