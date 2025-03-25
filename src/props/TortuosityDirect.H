#ifndef TORTUOSITYDIRECT_H
#define TORTUOSITYDIRECT_H

#include <AMReX.H> // Includes REAL, IntVect, Box, etc.
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H> // Include Geometry explicitly
#include <AMReX_DistributionMapping.H>
#include <AMReX_Array.H>    // For amrex::Array
#include "Tortuosity.H"   // Includes base class and enums under OpenImpala namespace

#include <string>       // For std::string if needed (result paths etc.)
#include <limits>       // For numeric_limits if using error codes like infinity

// Forward declaration if needed, although includes seem sufficient
// namespace amrex { class MultiFab; class iMultiFab; ... }

namespace OpenImpala {

/**
 * @brief Computes tortuosity using a simple iterative Finite Volume solver for the Poisson equation.
 *
 * This class implements the Tortuosity interface by solving the Laplace/Poisson equation
 * on the specified phase within the domain using a basic, parallel iterative method.
 * Boundary conditions are applied to drive a flux in the specified direction.
 *
 * Note: The implementation aims for correctness but prioritizes simplicity over
 * high performance compared to specialized solvers (like HYPRE). It serves
 * well as a baseline or for smaller problems.
 *
 * Note: This class stores references to geometry and input data. The caller must
 * ensure these referenced objects remain valid for the lifetime of any
 * TortuosityDirect instance.
 *
 * Note: This class manages internal state (cached value, solver residuals). It is
 * not designed for concurrent access from multiple threads within the same
 * MPI rank. Ensure each instance is used sequentially.
 */
class TortuosityDirect : public OpenImpala::Tortuosity
{

public:

    /**
     * @brief Construct a new TortuosityDirect solver instance.
     *
     * @param geom Reference to the problem domain geometry. Must remain valid.
     * @param ba Reference to the BoxArray defining the domain decomposition. Must remain valid.
     * @param dm Reference to the DistributionMapping for parallel distribution. Must remain valid.
     * @param mf_phase Reference to the iMultiFab containing phase identifiers. Requires at least 1 ghost cell. Must remain valid.
     * @param phase The integer identifier of the phase for which tortuosity is calculated.
     * @param dir The principal direction (X, Y, or Z) for applying boundary conditions to drive flux.
     * @param eps Convergence criterion (e.g., relative residual norm) for the iterative solver.
     * @param max_steps Maximum number of iterations allowed for the solver.
     * @param vlo Potential value applied at the low boundary in the specified direction.
     * @param vhi Potential value applied at the high boundary in the specified direction.
     */
    TortuosityDirect(const amrex::Geometry& geom,
                     const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm,
                     const amrex::iMultiFab& mf_phase, // Phase data input requires >= 1 ghost cell (verify!)
                     const int phase,
                     const OpenImpala::Direction dir,
                     const amrex::Real eps,
                     const size_t max_steps,
                     const amrex::Real vlo,
                     const amrex::Real vhi);

    // Override the virtual destructor from the base class
    ~TortuosityDirect() override = default; // Default is sufficient if no raw resources owned

    // Delete copy constructor and assignment operator (managing state/references makes copying complex/unsafe)
    TortuosityDirect(const TortuosityDirect&) = delete;
    TortuosityDirect& operator=(const TortuosityDirect&) = delete;

    /**
     * @brief Calculate and return the tortuosity value. Overrides base class method.
     *
     * Computes or retrieves the cached tortuosity. Calls the internal solver if needed.
     *
     * @param refresh If true, forces a recalculation. If false (default), returns cached value if available.
     * @return The calculated tortuosity. Returns cached value if refresh=false and calculation was already done.
     */
    amrex::Real value(const bool refresh = false) override;

    // --- Solver Diagnostics ---

    /**
     * @brief Returns the number of iterations performed during the last call to solve().
     * @return Number of iterations, or 0 if solve() hasn't been called.
     */
    int getNumIterations() const { return m_last_iterations; }

    /**
     * @brief Returns the final residual norm achieved during the last call to solve().
     * @return Final residual norm, or a negative value if solve() hasn't been called.
     */
    amrex::Real getFinalResidual() const { return m_last_residual; }


private:

    /** @brief Executes the iterative solver loop until convergence or max iterations. */
    bool solve();

    /**
     * @brief Computes the global fluxes entering and exiting the domain boundaries.
     * Expected to be called after solve() has successfully completed.
     * @param fxin Output: Total flux entering the domain through the low boundary face.
     * @param fxout Output: Total flux exiting the domain through the high boundary face.
     */
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const;

    /**
     * @brief Computes a residual norm (e.g., L2 norm of change) between two solution states.
     * Used as a convergence check in the iterative solver.
     * @param phiold MultiFab containing the solution from the previous iteration.
     * @param phinew MultiFab containing the solution from the current iteration.
     * @return The calculated residual norm.
     */
    amrex::Real residual(const amrex::MultiFab& phiold, const amrex::MultiFab& phinew) const;

    /** @brief Sets up the m_bc member variable describing boundary condition types (e.g., Dirichlet, Neumann). */
    void initializeBoundaryConditions() ;

    /** @brief Creates and initializes the m_flux MultiFab array for storing face fluxes. */
    void initializeFluxMultiFabs();

    /**
     * @brief Applies boundary conditions to the ghost cells of the potential MultiFab.
     * Uses the geometry and m_bc settings.
     * @param phi The MultiFab whose ghost cells need filling based on BCs.
     */
    void fillDomainBoundary(amrex::MultiFab& phi);

    /**
     * @brief Fills a component of a MultiFab with CellType information.
     * This data can be used by kernels (like in advance()) to apply different logic
     * based on cell location (e.g., near boundary, blocked, free).
     * @param phi The MultiFab to fill (typically fills component PhaseComp).
     * Requires at least numComponents components.
     */
    void fillCellTypes(amrex::MultiFab& phi); // Typically fills component PhaseComp

    /**
     * @brief Sets the initial guess for the potential field (phi).
     * Often a linear interpolation between boundary values (m_vlo, m_vhi).
     * @param phi The MultiFab to fill with the initial state.
     */
    void fillInitialState(amrex::MultiFab& phi);


    /**
     * @brief Performs one iteration of the Finite Volume solver (e.g., Jacobi, SOR).
     * Calculates face fluxes and updates the potential field based on conservation laws.
     * @param phi_old Input: Potential field from the previous iteration (requires ghost cells).
     * @param phi_new Output: Updated potential field for the current iteration.
     */
    void advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new);

    // --- Member Variables ---

    // References to external data (caller must ensure lifetime)
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    const amrex::iMultiFab& m_mf_phase; // Input phase data (needs ghost cells)

    // Internal data structures
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_flux; // Stores face fluxes
    amrex::BCRec m_bc; // Describes boundary condition types

    // Phase and Direction info
    const int m_phase;                      // Phase ID to compute tortuosity for
    const OpenImpala::Direction m_dir; // Principal direction for flux

    // Solver Control Parameters (set via constructor)
    const size_t m_n_steps;             // Max iterations
    const amrex::Real m_eps;            // Convergence tolerance

    // Boundary Condition Values (set via constructor)
    const amrex::Real m_vlo; // Potential at low boundary face in m_dir
    const amrex::Real m_vhi; // Potential at high boundary face in m_dir

    // Caching and Diagnostics
    amrex::Real m_value;            // Cached tortuosity value
    bool m_first_call;              // Flag for caching logic
    int m_last_iterations;          // Iterations used in last solve
    amrex::Real m_last_residual;    // Residual achieved in last solve
};

} // namespace OpenImpala

#endif // TORTUOSITYDIRECT_H
