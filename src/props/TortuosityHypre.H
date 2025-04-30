#ifndef TortuosityHypre_H
#define TortuosityHypre_H

#include <string> // Needed for std::string
#include <vector> // Needed for Vector<IntVect> in helper functions
#include <limits> // Needed for std::numeric_limits

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H> // Include Geometry explicitly
#include <AMReX_Array.H>    // Needed for return type of loV/hiV
#include <AMReX_IntVect.H>  // Needed for Vector<IntVect>

#include <HYPRE.h>           // Main HYPRE header
#include <HYPRE_struct_ls.h> // Struct interface header (should provide HYPRE_Int)

#include "Tortuosity.H" // Assumed to define OpenImpala::Tortuosity and OpenImpala::Direction

/** @file TortuosityHypre.H
 * @brief Defines a class to compute tortuosity using HYPRE solvers.
 */

namespace OpenImpala {

/** Computes the tortuosity of a porous structure using HYPRE.
 *
 * Solves a diffusion equation (-Div(Grad(phi)) = 0 with Dirichlet BCs)
 * on a masked subset of the domain derived from phase data, using HYPRE's
 * structured-grid solvers (PFMG, SMG, Krylov methods) to find the potential field.
 * Tortuosity is calculated from the resulting flux and **active** volume fraction, // <-- Doc update
 * including checks for solver convergence and flux conservation.
 */
class TortuosityHypre : public OpenImpala::Tortuosity
{

public:

    /** @brief Specifies the HYPRE structured solver algorithm to use. */
    enum class SolverType {
        Jacobi,     /**< HYPRE_StructJacobi */
        GMRES,      /**< HYPRE_StructGMRES */
        FlexGMRES,  /**< HYPRE_StructFlexGMRES */
        PCG,        /**< HYPRE_StructPCG */
        BiCGSTAB,   /**< HYPRE_StructBiCGSTAB */
        SMG,        /**< HYPRE_StructSMG */
        PFMG        /**< HYPRE_StructPFMG */
    };

    /** Construct new Tortuosity solver using HYPRE.
     *
     * @param geom AMReX Geometry object defining the domain.
     * @param ba AMReX BoxArray defining the distribution of grid patches.
     * @param dm AMReX DistributionMapping defining processor assignments.
     * @param mf_phase_input Input iMultiFab containing phase IDs (expects 1 component, >=1 ghost cell).
     * @param vf Volume fraction of the phase of interest (*total* VF, passed for potential reference). // <-- Doc update
     * @param phase Phase ID to calculate tortuosity for.
     * @param dir Direction across which to apply potential difference (X, Y, or Z).
     * @param solvertype Solver type to use (from SolverType enum).
     * @param resultspath Path to directory for writing output files (e.g., plotfiles).
     * @param vlo Potential value applied at the low boundary in the specified direction. Default 0.0.
     * @param vhi Potential value applied at the high boundary in the specified direction. Default 1.0.
     * @param verbose Verbosity level (0=silent, 1=basic, 2=detailed HYPRE output). Default 0.
     * @param write_plotfile If true, write a plotfile of the solution potential field. Default false.
     */
    TortuosityHypre(const amrex::Geometry& geom,
                    const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm,
                    const amrex::iMultiFab& mf_phase_input,
                    const amrex::Real vf,
                    const int phase,
                    const OpenImpala::Direction dir,
                    const SolverType solvertype,
                    const std::string& resultspath,
                    const amrex::Real vlo = 0.0,
                    const amrex::Real vhi = 1.0,
                    int verbose = 0,
                    bool write_plotfile = false);

    /** Destructor. Cleans up allocated HYPRE resources. */
    virtual ~TortuosityHypre() override;

    // Deleted Copy Operations
    TortuosityHypre(const TortuosityHypre&) = delete;
    TortuosityHypre& operator=(const TortuosityHypre&) = delete;

    /** Calculate or retrieve the tortuosity value.
     *
     * Solves the underlying diffusion equation using HYPRE if not already solved
     * or if refresh is requested. Checks solver convergence and flux conservation.
     * Calculates flux and computes tortuosity using the average flux magnitude
     * and the *active* volume fraction if checks pass.                    // <-- Doc update
     * Returns NaN if the solver fails to converge, if flux is not conserved,
     * or if calculation leads to invalid states (e.g., zero active VF or Deff).
     *
     * @param refresh If true, forces recalculation by running the solver again. Default false.
     * @return Tortuosity value (ActiveVF / (Deff/D0)), or NaN on failure. // <-- Doc update
     */
    amrex::Real value(const bool refresh = false) override;

    /**
     * @brief Performs mathematical checks on the assembled HYPRE matrix and RHS vector.
     * @details Iterates over local cells, retrieves matrix/vector values using HYPRE
     * functions, and verifies properties like NaN/Inf absence, expected diagonal
     * values based on cell activity/BCs, and row sums for interior nodes.
     * Intended for use in testing/debugging *after* object construction but *before* calling value() or solve().
     * @return True if all checks pass on all ranks, False otherwise.
     */
    bool checkMatrixProperties();

    // --- Public Getters for Status and Results ---
    bool        getSolverConverged() const { return m_converged; }
    amrex::Real getFinalRelativeResidualNorm() const { return m_final_res_norm; }
    int         getSolverIterations() const { return m_num_iterations; }
    amrex::Real getFluxIn() const { return m_flux_in; }
    amrex::Real getFluxOut() const { return m_flux_out; }
    // --- ADDED Getter for Active VF ---
    amrex::Real getActiveVolumeFraction() const { return m_active_vf; }
    // --- END ADD ---


    // Static Helper Functions
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> loV (const amrex::Box& b);
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hiV (const amrex::Box& b);


private:

    // --- Private Methods ---
    bool solve();
    void setupGrids();
    void setupStencil();
    void setupMatrixEquation();
    void preconditionPhaseFab();
    void generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID, OpenImpala::Direction dir);
    void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab, int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints);
    void getSolution (amrex::MultiFab& soln, int ncomp=0); // Not implemented
    void getCellTypes(amrex::MultiFab& phi, int ncomp=1); // Not implemented
    void global_fluxes();

    // --- Member Variables ---
    // Configuration
    const SolverType m_solvertype;
    std::string m_resultspath;
    const int m_phase;
    const OpenImpala::Direction m_dir;
    const amrex::Real m_vlo;
    const amrex::Real m_vhi;
    amrex::Real m_eps;
    int m_maxiter;
    int m_verbose;
    const amrex::Real m_vf; // Original total VF (kept for reference if needed)
    bool m_write_plotfile;

    // AMReX Data
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    amrex::iMultiFab m_mf_phase;
    amrex::MultiFab m_mf_phi;
    amrex::iMultiFab m_mf_active_mask;

    // State
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_first_call = true;

    // Solver Statistics and Results
    HYPRE_Int m_num_iterations = -1;
    amrex::Real m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool        m_converged = false;
    amrex::Real m_flux_in = 0.0;
    amrex::Real m_flux_out = 0.0;
    // --- ADDED: Active Volume Fraction ---
    amrex::Real m_active_vf = 0.0; // Volume fraction of the percolating phase
    // --- END ADD ---


    // HYPRE Data Structures
    HYPRE_StructGrid m_grid = NULL;
    HYPRE_StructStencil m_stencil = NULL;
    HYPRE_StructMatrix m_A = NULL;
    HYPRE_StructVector m_b = NULL;
    HYPRE_StructVector m_x = NULL;
};


// --- Fortran Interface Declarations ---
// (Remain the same as in your provided file)
extern "C" {
    void tortuosity_fillmtx(
        amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit,
        const int* nval,
        const int* p, const int* p_lo, const int* p_hi,
        const int* active_mask, const int* mask_lo, const int* mask_hi,
        const int* bxlo, const int* bxhi,
        const int* domlo, const int* domhi,
        const amrex::Real* dxinv,
        const amrex::Real* vlo, const amrex::Real* vhi,
        const int* phase_unused, const int* dir,
        const int* debug_print_level
    );

    void tortuosity_remspot(
        int* q, const int* q_lo, const int* q_hi,
        const int* ncomp,
        const int* bxlo, const int* bxhi,
        const int* domlo, const int* domhi
    );

    // Add others if needed...
}


} // namespace OpenImpala

#endif // TortuosityHypre_H
