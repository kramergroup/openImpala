#ifndef TortuosityHypre_H
#define TortuosityHypre_H

#include <string> // Needed for std::string
#include <vector> // Needed for Vector<IntVect> in helper functions

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
 * (Rest of Doxygen comments remain the same...)
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
        BiCGSTAB    /**< HYPRE_StructBiCGSTAB */
    };

    /** Construct new Tortuosity solver using HYPRE.
     *
     * (Constructor Doxygen comments remain the same...)
     */
    TortuosityHypre(const amrex::Geometry& geom,
                    const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm,
                    const amrex::iMultiFab& mf_phase_input, // Renamed arg for clarity
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
     * (value() Doxygen comments remain the same...)
     */
    amrex::Real value(const bool refresh = false) override;

    // Static Helper Functions
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> loV (const amrex::Box& b);
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hiV (const amrex::Box& b);


private:

    // --- Private Methods ---
    bool solve();
    void setupGrids();
    void setupStencil();
    void setupMatrixEquation(); // Calls modified Fortran routine
    void preconditionPhaseFab();
    void generateActivityMask(const amrex::iMultiFab& phaseFab, int phaseID, OpenImpala::Direction dir); // <<< NEW
    void parallelFloodFill(amrex::iMultiFab& reachabilityMask, const amrex::iMultiFab& phaseFab, int phaseID, const amrex::Vector<amrex::IntVect>& seedPoints); // <<< NEW
    void getSolution (amrex::MultiFab& soln, int ncomp=0);
    void getCellTypes(amrex::MultiFab& phi, int ncomp=1);
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout);

    // --- Member Variables ---
    // Configuration
    const SolverType m_solvertype;
    std::string m_resultspath;
    const int m_phase;
    const OpenImpala::Direction m_dir;
    const amrex::Real m_vlo;
    const amrex::Real m_vhi;
    amrex::Real m_eps = 1e-7; // Consider using param parse default from cpp
    int m_maxiter = 50000; // Consider using param parse default from cpp
    int m_verbose = 0;
    amrex::Real m_vf;
    bool m_write_plotfile;

    // AMReX Data
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    amrex::iMultiFab m_mf_phase;       // Original phase data (aliased from input)
    amrex::MultiFab m_mf_phi;         // Solution field (potential, etc.)
    amrex::iMultiFab m_mf_active_mask; // Mask for active/percolating cells <<< NEW

    // State
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
    bool m_first_call = true;

    // Solver Statistics
    HYPRE_Int m_num_iterations = -1;
    amrex::Real m_final_res_norm = -1.0;

    // HYPRE Data Structures
    HYPRE_StructGrid m_grid = NULL;
    HYPRE_StructStencil m_stencil = NULL;
    HYPRE_StructMatrix m_A = NULL;
    HYPRE_StructVector m_b = NULL;
    HYPRE_StructVector m_x = NULL;
};


// --- Updated Fortran Interface Declarations ---
// Place this outside the class, potentially within the namespace or globally
// Ensure these match the declarations used for binding in your build system
extern "C" {
    void tortuosity_fillmtx(
        amrex::Real* a, amrex::Real* rhs, amrex::Real* xinit,
        const int* nval,
        const int* p, const int* p_lo, const int* p_hi,             // Phase data
        const int* active_mask, const int* mask_lo, const int* mask_hi, // <<< NEW: Mask data
        const int* bxlo, const int* bxhi,
        const int* domlo, const int* domhi,
        const amrex::Real* dxinv,
        const amrex::Real* vlo, const amrex::Real* vhi,
        const int* phase_unused, const int* dir // phase arg now unused in F90 logic
    );

    void tortuosity_remspot(int* q, const int* q_lo, const int* q_hi, const int* ncomp,
                            const int* bxlo, const int* bxhi,
                            const int* domlo, const int* domhi);

    // Add declarations for other Fortran routines called from C++ if needed
    // void tortuosity_filct(...);
    // void tortuosity_filbc(...);
    // void tortuosity_filic(...);
}


} // namespace OpenImpala

#endif // TortuosityHypre_H
