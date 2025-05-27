#ifndef EFFECTIVEDIFFUSIVITYHYPRE_H
#define EFFECTIVEDIFFUSIVITYHYPRE_H

#include <string>
#include <vector>
#include <limits>

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H> // For m_dx

#include <HYPRE.h>
#include <HYPRE_struct_ls.h>

// Assuming OpenImpala::Direction is defined in a shared header like "Tortuosity.H" or a new "Common.H"
#include "Tortuosity.H" // Or your common enum definitions header

/** @file EffectiveDiffusivityHypre.H
 * @brief Defines a class to solve the cell problem for effective diffusivity calculation using HYPRE.
 */

namespace OpenImpala {

/** @brief Solves the cell problem ∇_ξ ⋅ (D ∇_ξ χ_k) = -∇_ξ ⋅ (D ê_k) for a corrector function χ_k.
 *
 * This class uses HYPRE's structured-grid solvers to find the corrector function χ_k
 * for a given direction k (X, Y, or Z). The material phase where D=D_material is
 * identified from the input phase data. The class sets up and solves the PDE
 * with periodic boundary conditions for χ_k and specific Neumann conditions
 * on internal pore-solid interfaces.
 *
 * The full effective diffusivity tensor D_eff calculation, which requires
 * all three χ_k fields, is intended to be performed outside this class.
 */
class EffectiveDiffusivityHypre
{

public:

    /** @brief Specifies the HYPRE structured solver algorithm to use.
     *  (Identical to TortuosityHypre::SolverType, could be a shared enum)
     */
    enum class SolverType {
        Jacobi,
        GMRES,
        FlexGMRES,
        PCG,
        BiCGSTAB,
        SMG,
        PFMG
    };

    /** @brief Construct new solver for the effective diffusivity cell problem.
     *
     * @param geom AMReX Geometry object defining the domain (REV).
     * @param ba AMReX BoxArray defining the distribution of grid patches.
     * @param dm AMReX DistributionMapping defining processor assignments.
     * @param mf_phase_input Input iMultiFab containing phase IDs (expects 1 component, >=1 ghost cell).
     * @param phase_id The ID of the phase considered to be the conductive material (where D=D_material, e.g., pores).
     * @param dir_of_chi_k The direction 'k' (X, Y, or Z) for which the corrector function χ_k is being solved.
     *                     This determines the ê_k vector in the RHS of the cell problem.
     * @param solver_type Solver type to use (from SolverType enum).
     * @param resultspath Path to directory for writing output files (e.g., plotfiles for χ_k).
     * @param verbose_level Verbosity level (0=silent, 1=basic, 2+=detailed HYPRE output). Default 1.
     * @param write_plotfile_flag If true, write a plotfile of the solved χ_k field. Default false.
     */
    EffectiveDiffusivityHypre(
        const amrex::Geometry& geom,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        const amrex::iMultiFab& mf_phase_input,
        const int phase_id,
        const OpenImpala::Direction dir_of_chi_k,
        const SolverType solver_type,
        const std::string& resultspath,
        int verbose_level = 1,
        bool write_plotfile_flag = false);

    /** @brief Destructor. Cleans up allocated HYPRE resources. */
    ~EffectiveDiffusivityHypre();

    // Deleted Copy Operations
    EffectiveDiffusivityHypre(const EffectiveDiffusivityHypre&) = delete;
    EffectiveDiffusivityHypre& operator=(const EffectiveDiffusivityHypre&) = delete;

    /** @brief Solves the cell problem for the configured χ_k.
     *
     *  Sets up and calls the HYPRE solver. The solved χ_k field can be retrieved
     *  using `getChiSolution()`.
     *
     * @return True if the solver converged to the specified tolerance, false otherwise.
     */
    bool solve();

    /** @brief Retrieves the solved corrector function χ_k.
     *
     *  This method should be called after `solve()` has returned true.
     *  It copies the solution from the internal HYPRE vector to the provided
     *  AMReX MultiFab. The `chi_field` should be pre-allocated with the same
     *  BoxArray, DistributionMapping, 1 component, and desired number of ghost cells
     *  (typically 1 for subsequent gradient calculations).
     *
     * @param chi_field Output MultiFab to store the solved χ_k.
     */
    void getChiSolution(amrex::MultiFab& chi_field);


    // --- Public Getters for Status and Solver Information ---
    bool getSolverConverged() const { return m_converged; }
    amrex::Real getFinalRelativeResidualNorm() const { return m_final_res_norm; }
    int getSolverIterations() const { return m_num_iterations; }
    const amrex::iMultiFab& getActiveMask() const { return m_mf_active_mask; } // Useful for D_eff calc

    // Static Helper Functions for HYPRE box conversion (can be shared if moved to a common utility)
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> loV (const amrex::Box& b);
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hiV (const amrex::Box& b);


private:

    // --- Private Methods ---
    void setupGrids();
    void setupStencil();
    void setupMatrixEquation();
    void generateActiveMask(); // Simpler version for D=0/1 based on phase_id

    // --- Member Variables ---
    // Configuration
    const SolverType m_solvertype;
    std::string m_resultspath;
    const int m_phase_id;                 // Phase ID where D=D_material
    const OpenImpala::Direction m_dir_solve; // Direction k for current chi_k
    amrex::Real m_eps;                    // Solver tolerance
    int m_maxiter;                  // Max solver iterations
    int m_verbose;
    bool m_write_plotfile;
    amrex::RealVect m_dx;                 // Cell sizes [dx, dy, dz]

    // AMReX Data
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    amrex::iMultiFab m_mf_phase_original; // Stores the input phase data
    amrex::iMultiFab m_mf_active_mask;    // Binary mask (1 for active phase, 0 otherwise)
    amrex::MultiFab m_mf_chi;             // Stores the solved chi_k for internal use/plotting

    // Solver Statistics
    HYPRE_Int m_num_iterations;
    amrex::Real m_final_res_norm;
    bool m_converged;

    // HYPRE Data Structures
    HYPRE_StructGrid m_grid;
    HYPRE_StructStencil m_stencil;
    HYPRE_StructMatrix m_A;
    HYPRE_StructVector m_b;
    HYPRE_StructVector m_x; // Stores the solution chi_k
};


// --- Fortran Interface Declaration ---
// This will be for your new Fortran kernel for the cell problem
extern "C" {
    /**
     * @brief Fills HYPRE matrix (A) and RHS (b) for the cell problem:
     *        ∇_ξ ⋅ (D ∇_ξ χ_k) = -∇_ξ ⋅ (D ê_k)
     * @param a Output buffer for matrix stencil coefficients.
     * @param rhs Output buffer for RHS values.
     * @param xinit Output buffer for initial guess of chi_k (typically zero).
     * @param npts_valid Number of points in the valid_bx region to fill.
     * @param active_mask_ptr Pointer to the active_mask data (1 for D=D_material, 0 otherwise).
     * @param mask_lo Lower bounds of the active_mask FAB.
     * @param mask_hi Upper bounds of the active_mask FAB.
     * @param valid_bx_lo Lower bounds of the current valid region to fill.
     * @param valid_bx_hi Upper bounds of the current valid region to fill.
     * @param domain_lo Lower bounds of the overall simulation domain.
     * @param domain_hi Upper bounds of the overall simulation domain.
     * @param cell_sizes Array of cell sizes [dx, dy, dz].
     * @param dir_k Integer representing the direction of ê_k (0 for X, 1 for Y, 2 for Z).
     * @param verbose_level Verbosity level for debugging in Fortran.
     */
    void effdiff_fillmtx( // Name matches the one used in the .cpp
        amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit,
        const int* npts_valid,
        const int* active_mask_ptr, const int* mask_lo, const int* mask_hi,
        const int* valid_bx_lo, const int* valid_bx_hi,
        const int* domain_lo, const int* domain_hi,
        const amrex::Real* cell_sizes, // Pass dx, dy, dz
        const int* dir_k,
        const int* verbose_level
    );
}


} // namespace OpenImpala

#endif // EFFECTIVEDIFFUSIVITYHYPRE_H
