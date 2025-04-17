#ifndef TORTUOSITYDIRECT_H
#define TORTUOSITYDIRECT_H

#include <AMReX.H> // Includes REAL, IntVect, Box, etc.
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H> // Include Geometry explicitly
#include <AMReX_DistributionMapping.H>
#include <AMReX_Array.H>     // For amrex::Array
#include <AMReX_GpuContainers.H> // For amrex::GpuArray (used for m_dxinv)
#include "Tortuosity.H"      // Includes base class and enums under OpenImpala namespace

#include <string>            // For std::string if needed (result paths etc.)
#include <limits>            // For numeric_limits if using error codes like infinity

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
     * @param n_steps Maximum number of iterations allowed for the solver. <<< Changed from size_t to int for consistency >>>
     * @param plot_interval Frequency for checking residual and optionally plotting (0 disables). <<< ADDED Argument >>>
     * @param plot_basename Base filename for diagnostic plotfiles. <<< ADDED Argument >>>
     * @param vlo Potential value applied at the low boundary in the specified direction.
     * @param vhi Potential value applied at the high boundary in the specified direction.
     */
    TortuosityDirect(const amrex::Geometry& geom,
                       const amrex::BoxArray& ba,
                       const amrex::DistributionMapping& dm,
                       const amrex::iMultiFab& mf_phase, // Phase data input
                       const int phase,
                       const OpenImpala::Direction dir,
                       const amrex::Real eps,
                       // const size_t max_steps, // Original was size_t
                       const int n_steps,         // <<< Changed to int >>>
                       const int plot_interval,         // <<< ADDED >>>
                       const std::string& plot_basename, // <<< ADDED >>>
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

    /** @brief Executes the iterative solver loop until convergence or max iterations. Returns true if converged. */
    bool solve();

    /**
     * @brief Computes the global fluxes entering and exiting the domain boundaries.
     * Expected to be called after solve() has successfully completed.
     * @param fxin Output: Total flux entering the domain through the low boundary face.
     * @param fxout Output: Total flux exiting the domain through the high boundary face.
     */
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const;

    /**
     * @brief Computes a residual norm (e.g., L1 norm of change) between two solution states.
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
     * @brief Applies boundary conditions to the ghost cells of the potential MultiFab for a specific component.
     * Uses the geometry and m_bc settings. Primarily handles external Dirichlet.
     * @param phi The MultiFab whose ghost cells need filling based on BCs.
     * @param comp The component index to fill boundary conditions for.
     */
    void fillDomainBoundary (amrex::MultiFab& phi, int comp);


    /**
     * @brief Fills a component of a MultiFab with CellType information (e.g., 0=blocked, 1=free).
     * Based on the input phase data (m_mf_phase).
     * @param phi The MultiFab to fill (typically fills component comp_ct).
     * Requires at least 2 components.
     */
    void fillCellTypes(amrex::MultiFab& phi); // Fills component comp_ct

    /**
     * @brief Sets the initial guess for the potential field (phi component).
     * Often a linear interpolation between boundary values (m_vlo, m_vhi) in the specified direction.
     * @param phi The MultiFab to fill with the initial state (modifies comp_phi).
     */
    void fillInitialState(amrex::MultiFab& phi); // Fills component comp_phi


    /**
     * @brief Performs one iteration of the Finite Volume solver (e.g., Jacobi, SOR).
     * Calculates face fluxes (updates m_flux) based on phi_old.
     * Updates the potential field phi_new based on conservation laws using calculated fluxes.
     * @param phi_old Input: Potential field from the previous iteration (requires filled ghost cells).
     * @param phi_new Output: Updated potential field for the current iteration.
     * @param dt Timestep or relaxation parameter used in the update.
     */
    void advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new, const amrex::Real& dt); // Added dt param


    // --- Member Variables ---

    // References to external data (caller must ensure lifetime)
    const amrex::Geometry& m_geom;
    const amrex::BoxArray& m_ba;
    const amrex::DistributionMapping& m_dm;
    const amrex::iMultiFab& m_mf_phase; // Input phase data (needs ghost cells)

    // Internal data structures
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_flux; // Stores face fluxes
    amrex::BCRec m_bc; // Describes boundary condition types for phi (comp 0)

    // Phase and Direction info
    const int m_phase;                       // Phase ID to compute tortuosity for
    const OpenImpala::Direction m_dir;       // Principal direction for flux

    // Solver Control Parameters (set via constructor)
    // const size_t m_n_steps; // Original was size_t
    const int m_n_steps;           // <<< Changed to int >>> Max iterations
    const amrex::Real m_eps;       // Convergence tolerance

    // Boundary Condition Values (set via constructor)
    const amrex::Real m_vlo; // Potential at low boundary face in m_dir
    const amrex::Real m_vhi; // Potential at high boundary face in m_dir

    // Caching and Diagnostics
    amrex::Real m_value;            // Cached tortuosity value
    bool m_first_call;              // Flag for caching logic
    int m_last_iterations;          // Iterations used in last solve
    amrex::Real m_last_residual;    // Residual achieved in last solve

    // <<< ADDED Missing Member Declarations >>>
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxinv; // Inverse cell sizes (calculated in constructor)
    int m_plot_interval;                            // Plotting frequency (from constructor)
    std::string m_plot_basename;                    // Base name for plotfiles (from constructor)

};

} // namespace OpenImpala

#endif // TORTUOSITYDIRECT_H
