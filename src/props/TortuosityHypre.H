#ifndef TortuosityHypre_H
#define TortuosityHypre_H

#include <string> // Needed for std::string

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H> // Include Geometry explicitly

#include <HYPRE.h>
#include <HYPRE_struct_ls.h>

#include "Tortuosity.H" // Assumed to define OpenImpala::Tortuosity and OpenImpala::Direction

/** @file TortuosityHypre.H
 * @brief Defines a class to compute tortuosity using HYPRE solvers.
 */

// Forward declaration (if needed, though likely defined in Tortuosity.H)
// namespace OpenImpala { enum class Direction; }

/** Computes the tortuosity of a porous structure using HYPRE.
 *
 * This class solves the steady-state Poisson (Laplace) equation:
 * ∇ ⋅ ( -∇φ ) = 0
 * within a specified phase of a porous medium, subject to Dirichlet boundary
 * conditions in a given direction, to determine the effective transport
 * properties and calculate the tortuosity.
 *
 * It leverages the HYPRE library's structured grid solvers (Struct interface).
 *
 * @warning The caller must ensure that the Geometry, BoxArray, and
 * DistributionMapping objects passed to the constructor remain valid for
 * the lifetime of the TortuosityHypre object, as they are stored by reference.
 *
 * @warning The implementation (.cpp file) must rigorously check HYPRE function
 * return codes for error handling.
 */
class TortuosityHypre : public OpenImpala::Tortuosity
{

public:

    /** @brief Specifies the HYPRE structured solver algorithm to use. */
    enum class SolverType {
        Jacobi,     /**< HYPRE_StructJacobi */
        GMRES,      /**< HYPRE_StructGMRES */
        FlexGMRES,  /**< HYPRE_StructFlexGMRES */
        PCG         /**< HYPRE_StructPCG (Added as a common alternative) */
        // Add others like BiCGSTAB if needed
    };

    /** Construct new Tortuosity solver using HYPRE.
     *
     * Calculates tortuosity by solving Laplace's equation within the specified phase.
     *
     * @param geom The geometry defining the computational domain. Must remain valid.
     * @param ba The distribution of boxes across processes. Must remain valid.
     * @param dm The distribution mapping of boxes to processes. Must remain valid.
     * @param mf The amrex::iMultiFab describing the phase space. A copy will be stored.
     * @param vf The volume fraction of the phase of interest (used in tortuosity calculation).
     * @param phase The integer index of the phase for which tortuosity is calculated.
     * @param dir The principal direction (X, Y, or Z) of the imposed potential gradient.
     * @param solvertype The HYPRE solver algorithm to use (e.g., SolverType::GMRES).
     * @param resultspath The directory path for output files (e.g., plotfiles).
     * Path expansion (like '~') is NOT handled automatically; provide
     * an absolute path or path relative to the execution directory.
     * @param vlo The potential value applied at the low boundary in the specified direction.
     * @param vhi The potential value applied at the high boundary in the specified direction.
     * @param verbose Verbosity level (0 = minimal, higher values for more output).
     */
    TortuosityHypre(const amrex::Geometry& geom,
                    const amrex::BoxArray& ba,
                    const amrex::DistributionMapping& dm,
                    const amrex::iMultiFab& mf, // Pass const&, store copy
                    const amrex::Real vf,       // Pass by value
                    const int phase,
                    const OpenImpala::Direction dir,
                    const SolverType solvertype,
                    const std::string& resultspath,
                    const amrex::Real vlo = 0.0, // Default BC values if often used
                    const amrex::Real vhi = 1.0,
                    int verbose = 0);

    /** Destructor. Cleans up allocated HYPRE resources. */
    virtual ~TortuosityHypre() override;

    // --- Deleted Copy Operations ---
    // Prevent copying due to raw HYPRE handles and reference members
    TortuosityHypre(const TortuosityHypre&) = delete;
    TortuosityHypre& operator=(const TortuosityHypre&) = delete;
    // --- Move Operations could be implemented if needed, but require care ---
    // TortuosityHypre(TortuosityHypre&&) = default; // Or implement carefully
    // TortuosityHypre& operator=(TortuosityHypre&&) = default; // Or implement carefully

    /** Calculate or retrieve the tortuosity value.
     *
     * Solves the underlying Poisson equation using HYPRE on the first call
     * or if refresh is true. Caches the result otherwise.
     * Returns NaN if the solver fails or if calculation is invalid (e.g., zero flux).
     *
     * @param refresh Force recalculation even if previously computed.
     * @return The calculated tortuosity value, or NaN on failure.
     */
    amrex::Real value(const bool refresh = false) override;

    // --- Static Helper Functions ---

    /** Extract lower bound from amrex::Box in HYPRE_Int array format. */
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> loV (const amrex::Box& b);

    /** Extract upper bound from amrex::Box in HYPRE_Int array format. */
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hiV (const amrex::Box& b);


private:

    /** Set up and execute the HYPRE solver.
     *
     * Coordinates the setup of HYPRE grids, stencil, matrix, vectors,
     * and then calls the chosen HYPRE solver.
     * Implementation must check HYPRE return codes.
     *
     * @return `true` if solver converges successfully, `false` otherwise.
     */
    bool solve();

    /** Setup HYPRE grid structure (`m_grid`).
     *
     * Defines the distributed grid based on the BoxArray `m_ba`.
     * Implementation must check HYPRE return codes.
     */
    void setupGrids();

    /** Setup HYPRE stencil object (`m_stencil`).
     *
     * Defines the finite difference stencil (typically 7-point for 3D Laplace).
     * Implementation must check HYPRE return codes.
     */
    void setupStencil();

    /** Setup HYPRE matrix (`m_A`) and RHS vector (`m_b`).
     *
     * Initializes the matrix coefficients based on the stencil, phase data,
     * and boundary conditions. Sets up the right-hand side vector (often zero
     * for internal nodes in Laplace eqn, modified by BCs).
     * Implementation must check HYPRE return codes.
     */
    void setupMatrixEquation();

    /** Precondition the phase MultiFab (`m_mf_phase`).
     *
     * Modifies the internal phase data (`m_mf_phase`) to handle potentially
     * ill-posed problems. Typically involves identifying phase cells disconnected
     * from the main flow path (e.g., not connected to the inlet boundary)
     * and treating them as solid/inactive (value 0).
     * @warning This method modifies the `m_mf_phase` member.
     */
    void preconditionPhaseFab();

    /** Retrieve solution from HYPRE vector (`m_x`) into an AMReX MultiFab.
     *
     * Copies the computed potential field from the HYPRE solution vector
     * into the specified component of the `soln` MultiFab.
     *
     * @param soln The amrex::MultiFab to store the solution.
     * @param ncomp The component index within `soln` to store the result (default 0).
     */
    void getSolution (amrex::MultiFab& soln, int ncomp=0);

    /** Fill a MultiFab with cell type information (1=fluid, 0=solid).
     *
     * Creates a representation of the phase geometry used for solving,
     * where cells belonging to the phase of interest (`m_phase`) are marked 1.0
     * and others 0.0. Useful for visualization or diagnostics.
     * Based on the potentially preconditioned `m_mf_phase`.
     *
     * @param phi The amrex::MultiFab to fill with cell types.
     * @param ncomp The component index within `phi` to store the types (default 1).
     */
    void getCellTypes(amrex::MultiFab& phi, int ncomp=1);

    /** Compute global fluxes across domain boundaries.
     *
     * Calculates the integrated flux entering the domain at the low boundary
     * and exiting at the high boundary in the specified direction (`m_dir`).
     * Assumes the solution (`m_mf_phi`) has been computed.
     * Requires careful implementation considering finite differences at boundaries.
     *
     * @param[out] fxin Total flux entering the domain.
     * @param[out] fxout Total flux exiting the domain.
     */
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const;

    // --- Member Variables ---

    // Configuration (set at construction)
    const SolverType m_solvertype;        /**< HYPRE solver algorithm to use. */
    std::string m_resultspath;            /**< Base path for results output. */
    const int m_phase;                    /**< Index of the phase of interest. */
    const OpenImpala::Direction m_dir;    /**< Principal direction of flow/gradient. */
    const amrex::Real m_vlo;              /**< Potential value at low boundary in m_dir. */
    const amrex::Real m_vhi;              /**< Potential value at high boundary in m_dir. */
    amrex::Real m_eps = 1e-7;             /**< Solver relative convergence tolerance. */
    int m_maxiter = 50000;                /**< Solver maximum iterations. */
    int m_verbose = 0;                    /**< Verbosity level. */
    amrex::Real m_vf;                     /**< Volume Fraction (stored by value). */

    // AMReX Geometry & Data (references + owned copies/results)
    const amrex::Geometry& m_geom;        /**< Domain geometry (reference, lifetime managed externally). */
    const amrex::BoxArray& m_ba;          /**< Box layout (reference, lifetime managed externally). */
    const amrex::DistributionMapping& m_dm; /**< Box distribution (reference, lifetime managed externally). */
    amrex::iMultiFab m_mf_phase;          /**< Phase data (owned copy, potentially preconditioned). */
    amrex::MultiFab m_mf_phi;             /**< Potential field solution (owned). */

    // State Variables
    amrex::Real m_value = std::numeric_limits<amrex::Real>::quiet_NaN(); /**< Cached tortuosity value. */
    bool m_first_call = true;             /**< Flag to trigger solve() on first call to value(). */

    // HYPRE Data Structures (managed internally)
    HYPRE_StructGrid m_grid = NULL;       /**< HYPRE grid object. */
    HYPRE_StructStencil m_stencil = NULL; /**< HYPRE stencil object. */
    HYPRE_StructMatrix m_A = NULL;        /**< HYPRE matrix object (Laplacian). */
    HYPRE_StructVector m_b = NULL;        /**< HYPRE RHS vector object. */
    HYPRE_StructVector m_x = NULL;        /**< HYPRE solution vector object. */
};

#endif // TortuosityHypre_H
