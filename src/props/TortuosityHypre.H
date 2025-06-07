#ifndef TortuosityHypre_H
#define TortuosityHypre_H

#include <string>
#include <vector>
#include <limits>
#include <map> // <-- ADDED for multi-phase coefficients

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array.H>
#include <AMReX_IntVect.H>

#include <HYPRE.h>
#include <HYPRE_struct_ls.h>

#include "Tortuosity.H"

/** @file TortuosityHypre.H
 * @brief Defines a class to compute tortuosity using HYPRE solvers.
 */

namespace OpenImpala {

/** Computes the tortuosity of a porous structure using HYPRE.
 *
 * Solves a steady-state diffusion equation (-∇·(D∇ϕ) = 0) with Dirichlet BCs.
 * It can operate in single-phase mode (D=1 in active phase, D=0 elsewhere)
 * or multi-phase mode where each phase can have a distinct, non-zero transport
 * coefficient D. Tortuosity is calculated from the resulting flux, active volume
 * fraction, and a reference bulk coefficient.
 */
class TortuosityHypre : public OpenImpala::Tortuosity
{

public:

    /** @brief Specifies the HYPRE structured solver algorithm to use. */
    enum class SolverType {
        Jacobi,
        GMRES,
        FlexGMRES,
        PCG,
        BiCGSTAB,
        SMG,
        PFMG
    };

    /**
     * @brief Construct new Tortuosity solver for **single-phase** calculations (legacy).
     * @param geom AMReX Geometry object defining the domain.
     * @param ba AMReX BoxArray defining the distribution of grid patches.
     * @param dm AMReX DistributionMapping defining processor assignments.
     * @param mf_phase_input Input iMultiFab containing phase IDs.
     * @param vf Volume fraction of the phase of interest.
     * @param phase The single active phase ID.
     * @param dir Direction across which to apply potential difference.
     * @param solvertype Solver type to use.
     * @param resultspath Path to directory for writing output files.
     * @param vlo Potential value applied at the low boundary.
     * @param vhi Potential value applied at the high boundary.
     * @param verbose Verbosity level.
     * @param write_plotfile If true, write a plotfile of the solution.
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

    /**
     * @brief Construct new Tortuosity solver for **multi-phase** calculations.
     * @param geom AMReX Geometry object defining the domain.
     * @param ba AMReX BoxArray defining the distribution of grid patches.
     * @param dm AMReX DistributionMapping defining processor assignments.
     * @param mf_phase_input Input iMultiFab containing phase IDs.
     * @param phase_coeff_map A map from phase ID to its transport coefficient (D).
     * @param reference_coeff The bulk coefficient D₀ used for tortuosity normalization.
     * @param dir Direction across which to apply potential difference.
     * @param solvertype Solver type to use.
     * @param resultspath Path to directory for writing output files.
     * @param vlo Potential value applied at the low boundary.
     * @param vhi Potential value applied at the high boundary.
     * @param verbose Verbosity level.
     * @param write_plotfile If true, write a plotfile of the solution.
     */
    TortuosityHypre(const amrex::Geometry& geom,
                    const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm,
                    const amrex::iMultiFab& mf_phase_input,
                    const std::map<int, amrex::Real>& phase_coeff_map,
                    const amrex::Real reference_coeff,
                    const OpenImpala::Direction dir,
                    const SolverType solvertype,
                    const std::string& resultspath,
                    const amrex::Real vlo = 0.0,
                    const amrex::Real vhi = 1.0,
                    int verbose = 0,
                    bool write_plotfile = false);

    /** @brief Destructor. Cleans up allocated HYPRE resources. */
    virtual ~TortuosityHypre() override;

    // Deleted Copy Operations
    TortuosityHypre(const TortuosityHypre&) = delete;
    TortuosityHypre& operator=(const TortuosityHypre&) = delete;

    /** Calculate or retrieve the tortuosity value.
     *
     * Solves the underlying diffusion equation using HYPRE if not already solved.
     * Checks solver convergence and flux conservation. Calculates tortuosity using
     * the average flux, active volume fraction, and reference coefficient.
     * Returns NaN on failure.
     *
     * @param refresh If true, forces recalculation.
     * @return Tortuosity value (ActiveVF * D₀ / Deff), or NaN on failure.
     */
    amrex::Real value(const bool refresh = false) override;

    /** @brief Performs mathematical checks on the assembled HYPRE matrix and RHS vector. */
    bool checkMatrixProperties();

    // --- Public Getters for Status and Results ---
    bool        getSolverConverged() const { return m_converged; }
    amrex::Real getFinalRelativeResidualNorm() const { return m_final_res_norm; }
    int         getSolverIterations() const { return m_num_iterations; }
    amrex::Real getFluxIn() const { return m_flux_in; }
    amrex::Real getFluxOut() const { return m_flux_out; }
    amrex::Real getActiveVolumeFraction() const { return m_active_vf; }

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
    // Overloaded for single-phase and multi-phase initialization
    void generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID, OpenImpala::Direction dir);
    void generateActivityMask(const amrex::iMultiFab& phaseFab, const std::map<int, amrex::Real>& pcm, OpenImpala::Direction dir);
    void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab, int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints);
    void global_fluxes();

    // --- Member Variables ---
    // Configuration
    const SolverType m_solvertype;
    std::string m_resultspath;
    const OpenImpala::Direction m_dir;
    const amrex::Real m_vlo;
    const amrex::Real m_vhi;
    amrex::Real m_eps;
    int m_maxiter;
    int m_verbose;
    bool m_write_plotfile;

    // Legacy single-phase members
    const int m_phase;
    const amrex::Real m_vf;

    // <<< ADDED for multi-phase >>>
    amrex::Real m_reference_coeff; // D₀ for tortuosity normalization
    // <<< END ADDED >>>

    // AMReX Data
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    amrex::iMultiFab m_mf_phase;
    amrex::MultiFab m_mf_phi;
    amrex::iMultiFab m_mf_active_mask;
    // <<< ADDED for multi-phase >>>
    amrex::MultiFab m_mf_diff_coeff; // Spatially varying coefficient D(x)
    // <<< END ADDED >>>

    // State
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_first_call = true;

    // Solver Statistics and Results
    HYPRE_Int m_num_iterations = -1;
    amrex::Real m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool        m_converged = false;
    amrex::Real m_flux_in = 0.0;
    amrex::Real m_flux_out = 0.0;
    amrex::Real m_active_vf = 0.0; // Volume fraction of the percolating phase(s)

    // HYPRE Data Structures
    HYPRE_StructGrid m_grid = NULL;
    HYPRE_StructStencil m_stencil = NULL;
    HYPRE_StructMatrix m_A = NULL;
    HYPRE_StructVector m_b = NULL;
    HYPRE_StructVector m_x = NULL;
};


// --- Fortran Interface Declarations ---
extern "C" {
    // <<< MODIFIED for multi-phase: pass diff_coeff instead of phase/mask >>>
    void tortuosity_fillmtx(
        amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit,
        const int* nval,
        const amrex::Real* diff_coeff, const int* coeff_lo, const int* coeff_hi,
        const int* bxlo, const int* bxhi,
        const int* domlo, const int* domhi,
        const amrex::Real* dx, // Pass dx array directly
        const amrex::Real* vlo, const amrex::Real* vhi,
        const int* dir,
        const int* debug_print_level
    );

    void tortuosity_remspot(
        int* q, const int* q_lo, const int* q_hi,
        const int* ncomp,
        const int* bxlo, const int* bxhi,
        const int* domlo, const int* domhi
    );
}


} // namespace OpenImpala

#endif // TortuosityHypre_H
