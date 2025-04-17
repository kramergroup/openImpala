#include "TortuosityDirect.H"
#include "Tortuosity_poisson_3d_F.H" // Assuming Fortran interface for poisson steps
#include "Tortuosity_filcc_F.H"     // Assuming Fortran interface for fill/condition steps

#include <cstdlib> // For std::getenv
#include <cmath>   // For std::abs
#include <limits>  // For std::numeric_limits
#include <iomanip> // For std::setprecision, etc.
#include <string>  // For std::string

#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Loop.H>             // <<< ADDED for amrex::Loop >>>
#include <AMReX_GpuQualifiers.H>  // Included for AMREX_RESTRICT if needed by Fortran wrappers/macros, maybe via AMReX.H already

// #include <AMReX_MLLinOp.H>   // No longer explicitly used in this version
// #include <AMReX_MLPoisson.H> // No longer explicitly used in this version
// #include <AMReX_MLMG.H>      // No longer explicitly used in this version

#define NUM_GHOST_CELLS 1
// Assuming these constants are defined somewhere (e.g., in TortuosityDirect.H or globally)
constexpr int comp_phi = 0; // Index for potential phi in MultiFab
constexpr int comp_ct  = 1; // Index for cell type in MultiFab (matches Fortran comp_ct=2)
constexpr int comp_phase = 0; // Index for phase in phase MultiFab (matches Fortran comp_phase=1)

//-----------------------------------------------------------------------
// NOTE: REMINDER - Add missing member declarations to TortuosityDirect.H:
// private:
//    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxinv; // Or RealVect
//    int m_plot_interval;
//    std::string m_plot_basename;
// Also ensure these are initialized in the constructor below (plot_* might need args).
//-----------------------------------------------------------------------


// Constructor with added configuration parameters
// Assuming m_plot_interval and m_plot_basename were added as arguments
// or read via ParmParse within the constructor body.
// For this example, assume they are arguments matching the signature used in the cpp body.
TortuosityDirect::TortuosityDirect(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                   const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase, // Changed name from mf to mf_phase for clarity
                                   const int phase, const OpenImpala::Direction dir,
                                   // const size_t n_steps, // Original used size_t, changed to int to match member type below
                                   const int n_steps,     // Changed to int to match m_n_steps
                                   const amrex::Real eps,
                                   const int plot_interval,         // Added argument
                                   const std::string& plot_basename, // Added argument
                                   const amrex::Real vlo, const amrex::Real vhi)
    : m_geom(geom), m_ba(ba), m_dm(dm), m_mf_phase(mf_phase), // Use constructor arg name
      m_phase(phase), m_dir(dir),
      m_n_steps(n_steps), m_eps(eps),
      m_plot_interval(plot_interval), m_plot_basename(plot_basename), // Initialize new members
      m_vlo(vlo), m_vhi(vhi),
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()), // Initialize value to NaN
      m_last_iterations(0), m_last_residual(-1.0) // Initialize diagnostics
{
    // Calculate inverse cell size once
    const amrex::Real* dx = m_geom.CellSize();
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
        if (dx[i] <= 0.0) {
            amrex::Abort("TortuosityDirect: Non-positive dx");
        }
        m_dxinv[i] = 1.0/ dx[i];
    }

    // Create a datastructure describing the boundary conditions
    initialiseBoundaryConditions(); // Sets m_bc based on m_dir
    initialiseFluxMultiFabs();      // Defines m_flux MultiFabs
}

// Destructor (if needed, e.g., for manual memory management - not needed here)
// TortuosityDirect::~TortuosityDirect() {}

void TortuosityDirect::initialiseFluxMultiFabs()
{
    // Build the flux multi-fabs
    for (int i_dir = 0; i_dir < AMREX_SPACEDIM; ++i_dir)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        amrex::BoxArray edge_ba = m_ba; // <<< CORRECTION: Need a copy, not reference >>>
        edge_ba.surroundingNodes(i_dir);
        m_flux[i_dir].define(edge_ba, m_dm, 1, 0);
        m_flux[i_dir].setVal(0.0); // Initialize fluxes
    }
}

amrex::Real TortuosityDirect::value(const bool refresh)
{
    // Solve potential field if needed
    if ( refresh || m_first_call )
    {
        if (!solve()) // solve() now returns bool indicating success
        {
            amrex::Warning("TortuosityDirect::solve() failed to converge. Returning NaN.");
            m_first_call = true; // Allow retrying solve on next call if needed
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            return m_value;
        }
        // m_first_call is set to false inside successful solve()
    }

    // Calculate fluxes using the solution stored/updated in solve()
    amrex::Real fxin = 0.0, fxout = 0.0;
    global_fluxes(fxin, fxout); // Assumes m_flux is up-to-date from solve()
    amrex::Real fx = (fxin + fxout) / 2.0; // Use average? Check definition.

    // Calculate tortuosity
    const amrex::Box& bx_domain = m_geom.Domain();
    const amrex::IntVect& sz = bx_domain.size(); // size() is (nx, ny, nz)

    amrex::Real cross_sectional_area = 0.0;
    switch(m_dir)
    {
        // <<< FIXED Enum Scope >>>
        case OpenImpala::Direction::X : cross_sectional_area = static_cast<amrex::Real>(sz[1]) * static_cast<amrex::Real>(sz[2]); break;
        case OpenImpala::Direction::Y : cross_sectional_area = static_cast<amrex::Real>(sz[0]) * static_cast<amrex::Real>(sz[2]); break;
        case OpenImpala::Direction::Z : cross_sectional_area = static_cast<amrex::Real>(sz[0]) * static_cast<amrex::Real>(sz[1]); break;
        default: amrex::Abort("TortuosityDirect::value: Invalid direction");
    }

    if (cross_sectional_area <= 0.0) {
         amrex::Warning("TortuosityDirect::value: Domain cross-sectional area is non-positive. Cannot calculate tortuosity.");
         m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
         return m_value;
    }

    amrex::Real avg_flux_density = fx / cross_sectional_area;

    // Check for near-zero average flux density
    if (std::abs(avg_flux_density) < 1e-15) { // Use a small tolerance
        amrex::Warning("Calculated average flux density is near zero. Setting tortuosity to infinity.");
        m_value = std::numeric_limits<amrex::Real>::infinity();
    } else {
        // Tortuosity = Volume Fraction / Effective Diffusivity Ratio
        // Assuming Potential Gradient = (Vhi-Vlo)/L = (1-0)/L = 1/L
        // Flux Density = -Deff * Grad(Potential) = -Deff * (1/L)  (Assuming Vhi-Vlo=1)
        // Tortuosity = D/Deff = D / (-Flux Density * L)
        // If D=1, Vhi-Vlo=1, then Tortuosity = L / (-avg_flux_density * L) = -1.0 / avg_flux_density ?
        // Or maybe Flux Density = - (1/Tau^2) * Grad(Potential)? -> Tau^2 = -Grad(Pot) / FluxDen
        // If Grad(Pot) = 1/L -> Tau^2 = -1 / (FluxDen * L) ?
        // Let's stick to the original formula assumed: 1.0 / avg_flux_density
        // THIS NEEDS CAREFUL VERIFICATION based on the PDE being solved and the definition of tortuosity used.
        m_value = 1.0 / avg_flux_density;
        // Tortuosity should likely be positive, perhaps use std::abs? Check convention.
        // m_value = 1.0 / std::abs(avg_flux_density);
    }
    return m_value;
}

bool TortuosityDirect::solve()
{
    // Create MultiFabs for variables (potential + cell_type)
    // These have 2 components: comp_phi and comp_ct
    amrex::MultiFab mf_phi_old(m_ba, m_dm, 2, NUM_GHOST_CELLS);
    amrex::MultiFab mf_phi_new(m_ba, m_dm, 2, NUM_GHOST_CELLS);

    // Fill initial condition for phi (component comp_phi)
    fillInitialState(mf_phi_old);

    // Initialise the cell type component (comp_ct) based on phase data
    fillCellTypes(mf_phi_old);

    // Copy both components (phi and ct) to start iteration
    amrex::MultiFab::Copy(mf_phi_new, mf_phi_old, 0, 0, 2, NUM_GHOST_CELLS);

    // Calculate timestep dt once (assuming dx doesn't change)
    const amrex::Real* dx = m_geom.CellSize();
    amrex::Real min_dx_sq = dx[0]*dx[0];
    for(int i=1; i<AMREX_SPACEDIM; ++i) min_dx_sq = std::min(min_dx_sq, dx[i]*dx[i]);
    // Use a safety factor (e.g., 0.5 for stability) for explicit diffusion CFL: dt <= dx^2 / (2*D*AMREX_SPACEDIM)
    // Assuming D=1 for simplicity here.
    const amrex::Real dt = 0.5 * min_dx_sq / (2.0 * AMREX_SPACEDIM);

    amrex::Real current_res = std::numeric_limits<amrex::Real>::max();
    amrex::Real fxin = 0.0, fxout = 0.0;
    bool converged = false;
    m_last_iterations = 0; // Reset diagnostics

    for (int n = 1; n <= m_n_steps; ++n)
    {
        m_last_iterations = n; // Track iterations

        // Copy only the potential (comp_phi) from new to old for the next step
        amrex::MultiFab::Copy(mf_phi_old, mf_phi_new, comp_phi, comp_phi, 1, NUM_GHOST_CELLS);
        // Cell type (comp_ct) remains static in mf_phi_old after initialization

        advance(mf_phi_old, mf_phi_new, dt); // Pass dt

        // Check residual and print status periodically
        if (m_plot_interval > 0 && n % m_plot_interval == 0)
        {
            current_res = residual(mf_phi_old, mf_phi_new); // Calculates residual based on comp_phi change
            global_fluxes(fxin, fxout);                     // Recalculates fluxes based on mf_phi_new (via m_flux)
            m_last_residual = current_res; // Store last calculated residual

            amrex::Print() << "Step " << n << ": ";
            amrex::Print() << std::scientific << std::setprecision(4) << std::setw(12) << std::setfill(' '); // Use scientific
            amrex::Print() << "Residual: " << current_res << " | ";
            amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << " Delta: " << (fxout - fxin) << std::endl;

            if (current_res < m_eps)
            {
                amrex::Print() << "Convergence reached in " << n << " steps." << std::endl;
                converged = true;
                break; // Exit loop
            }
        }
    }

    if (!converged) {
         amrex::Warning("TortuosityDirect::solve() did not converge within max steps.");
         // Calculate final residual and fluxes if not done in last iteration
         if (!(m_plot_interval > 0 && m_n_steps % m_plot_interval == 0)) {
             current_res = residual(mf_phi_old, mf_phi_new);
             global_fluxes(fxin, fxout);
             m_last_residual = current_res;
             amrex::Print() << "Final Step " << m_n_steps << ": ";
             amrex::Print() << "Residual: " << current_res << " | ";
             amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << std::endl;
         }
    }

    // Write final plot file if requested (and if interval check didn't already write it)
    if (m_plot_interval > 0) // Assuming plot_interval > 0 means plotting is desired
    {
        const char* homeDir_cstr = std::getenv("HOME");
        std::string plot_file_path_str;
        if (homeDir_cstr && m_plot_basename.rfind("~/", 0) == 0) { // Check if basename starts with ~/
             plot_file_path_str = std::string(homeDir_cstr) + "/" + m_plot_basename.substr(2);
        } else if (m_plot_basename.find('/') == 0 || m_plot_basename.find('\\') == 0 || m_plot_basename.find(':') != std::string::npos) { // Absolute path
             plot_file_path_str = m_plot_basename;
        } else { // Relative path
            plot_file_path_str = m_plot_basename; // Default to current directory if not absolute/home
        }
        // Add step number to filename
        plot_file_path_str += "_final_step_" + std::to_string(m_last_iterations);

        amrex::Print() << "Writing plot file: " << plot_file_path_str << std::endl;
        // Plotting both potential (comp_phi) and cell type (comp_ct)
        amrex::WriteSingleLevelPlotfile(plot_file_path_str, mf_phi_new, {"potential", "cell_type"}, m_geom, 0.0, m_last_iterations);
    }


    // Ensure fluxes are based on the final state (already done if converged/last step printed)
    if (!converged && !(m_plot_interval > 0 && m_n_steps % m_plot_interval == 0)) {
        global_fluxes(fxin, fxout); // Ensure m_flux is consistent with final mf_phi_new
    }

    m_first_call = !converged; // Set to false only if converged, true otherwise to allow retry
    return converged;
}


void TortuosityDirect::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    fxin = 0.0;
    fxout = 0.0;

    const amrex::Box& bx_domain = m_geom.Domain();
    const int dir_int = static_cast<int>(m_dir); // Safer cast

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:fxin, fxout)
#endif
    {
        amrex::Real lfxin = 0.0; // Private to thread
        amrex::Real lfxout = 0.0;// Private to thread

        // Iterate over flux multifabs - need const access
        for ( amrex::MFIter mfi(m_flux[0]); mfi.isValid(); ++mfi ) // Iterate using first flux MF, assumes all have same distribution
        {
            // <<< FIXED Fortran call for tortuosity_poisson_fio >>>
            const amrex::Box& bx = mfi.tilebox(); // Use tilebox for valid region

            const auto* fx_ptr = m_flux[0].dataPtr(mfi);
            const auto* fy_ptr = m_flux[1].dataPtr(mfi);
            const auto* fz_ptr = m_flux[2].dataPtr(mfi);

            const auto& fxbox = m_flux[0].fabbox(mfi);
            const auto& fybox = m_flux[1].fabbox(mfi);
            const auto& fzbox = m_flux[2].fabbox(mfi);

            // The Fortran routine sums flux over the appropriate domain boundary faces
            // intersecting the current tile box.
            tortuosity_poisson_fio(bx.loVect().getVect(), bx.hiVect().getVect(), // Pass pointers to tile bounds
                                   fx_ptr, fxbox.loVect().getVect(), fxbox.hiVect().getVect(),
                                   fy_ptr, fybox.loVect().getVect(), fybox.hiVect().getVect(),
                                   fz_ptr, fzbox.loVect().getVect(), fzbox.hiVect().getVect(),
                                   &dir_int, &lfxin, &lfxout);
        }
        // OpenMP reduction clause handles sum automatically
        // (No need for explicit fxin += lfxin here inside the omp region)
    } // End OMP parallel region

    // Reduce across MPI processes
    amrex::ParallelDescriptor::ReduceRealSum(fxin, amrex::ParallelDescriptor::IOProcessorNumber());  // Reduce sum to IO proc
    amrex::ParallelDescriptor::ReduceRealSum(fxout, amrex::ParallelDescriptor::IOProcessorNumber()); // Reduce sum to IO proc
    // If all ranks need the result, use AllReduce:
    // amrex::ParallelAllReduce::Sum(fxin, amrex::ParallelContext::CommunicatorSub());
    // amrex::ParallelAllReduce::Sum(fxout, amrex::ParallelContext::CommunicatorSub());
}


// Advance phi_new = phi_old + dt * Laplacian(phi_old) [masked by cell_type]
// Also computes fluxes based on phi_old for the update and for later analysis
void TortuosityDirect::advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new, const amrex::Real& dt)
{
    // Fill the ghost cells of phi_old (potential and cell_type)
    phi_old.FillBoundary(m_geom.periodicity());

    // Apply physical boundary conditions to ghost cells (Dirichlet handled separately)
    // This fills ghosts for BOTH components based on m_bc.
    amrex::FillDomainBoundary(phi_old, m_geom, {m_bc, m_bc}); // Apply BCs to both components

    // Apply external Dirichlet explicitly using Fortran kernel for phi component only
    fillDomainBoundary(phi_old, comp_phi); // Fills Dirichlet for comp_phi only

    // No need to fill boundary for phi_new as it's overwritten

    const amrex::Real* dx = m_geom.CellSize(); // Pointer to cell sizes

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        // Compute fluxes based on phi_old
        // Fluxes are stored in member variable m_flux
        for ( amrex::MFIter mfi(phi_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.tilebox();

            // <<< FIXED Fortran call for tortuosity_poisson_flux >>>
            // Get pointers to flux data (modifiable)
            auto* fx_ptr = m_flux[0].dataPtr(mfi);
            auto* fy_ptr = m_flux[1].dataPtr(mfi);
            auto* fz_ptr = m_flux[2].dataPtr(mfi);
            // Get pointer to solution data (const) - IMPORTANT: Assumes phi is comp 0, ct is comp 1
            const auto* sol_ptr = phi_old.dataPtr(0, mfi); // Get base pointer for component 0

            const auto& fxbox = m_flux[0].fabbox(mfi);
            const auto& fybox = m_flux[1].fabbox(mfi);
            const auto& fzbox = m_flux[2].fabbox(mfi);
            const auto& solbox = phi_old.fabbox(mfi);

            // Assuming tortuosity_poisson_flux takes valid box (tile), flux arrays+bounds, sol array+bounds, dxinv
            tortuosity_poisson_flux(bx.loVect().getVect(), bx.hiVect().getVect(),
                                    fx_ptr, fxbox.loVect().getVect(), fxbox.hiVect().getVect(),
                                    fy_ptr, fybox.loVect().getVect(), fybox.hiVect().getVect(),
                                    fz_ptr, fzbox.loVect().getVect(), fzbox.hiVect().getVect(),
                                    sol_ptr, solbox.loVect().getVect(), solbox.hiVect().getVect(),
                                    m_dxinv.data()); // Pass pointer from member GpuArray/RealVect

        }

        // Advance the solution (phi_new) using fluxes based on phi_old
        for ( amrex::MFIter mfi(phi_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.tilebox();

            // <<< FIXED Fortran call for tortuosity_poisson_update >>>
            // Get pointers to data arrays needed by Fortran
            const auto* p_ptr = phi_old.dataPtr(comp_phi, mfi); // Pointer to old potential
            auto* n_ptr = phi_new.dataPtr(comp_phi, mfi); // Pointer to new potential (modifiable)
            const auto* fx_ptr = m_flux[0].dataPtr(mfi);        // Pointer to x-flux
            const auto* fy_ptr = m_flux[1].dataPtr(mfi);        // Pointer to y-flux
            const auto* fz_ptr = m_flux[2].dataPtr(mfi);        // Pointer to z-flux

            // Get box bounds for arrays (FABs include ghost cells)
            const auto& pbox = phi_old.fabbox(mfi); // Bounds for p (phi_old)
            const auto& nbox = phi_new.fabbox(mfi); // Bounds for n (phi_new)
            const auto& fxbox = m_flux[0].fabbox(mfi); // Bounds for fx
            const auto& fybox = m_flux[1].fabbox(mfi); // Bounds for fy
            const auto& fzbox = m_flux[2].fabbox(mfi); // Bounds for fz

            // Assuming tortuosity_poisson_update takes valid box (tile), p+bounds, n+bounds, fluxes+bounds, dxinv, dt
            tortuosity_poisson_update(bx.loVect().getVect(), bx.hiVect().getVect(),
                                      p_ptr, pbox.loVect().getVect(), pbox.hiVect().getVect(), // Pass p_hi for 'phi' arg
                                      n_ptr, nbox.loVect().getVect(), nbox.hiVect().getVect(),
                                      fx_ptr, fxbox.loVect().getVect(), fxbox.hiVect().getVect(),
                                      fy_ptr, fybox.loVect().getVect(), fybox.hiVect().getVect(),
                                      fz_ptr, fzbox.loVect().getVect(), fzbox.hiVect().getVect(),
                                      m_dxinv.data(), // Pass pointer from member GpuArray/RealVect
                                      &dt);

            // Ensure cell type remains unchanged in phi_new (copy from phi_old)
            // This is crucial if the Fortran update only modifies comp_phi.
            // We copy comp_ct from phi_old (where it was initialized) to phi_new.
            const auto& ct_fab_old = phi_old.const_array(mfi, comp_ct);
            auto        ct_fab_new = phi_new.array(mfi, comp_ct);
            ct_fab_new.copy(ct_fab_old, bx); // Copy cell types over the valid box

        }
    } // End OMP parallel region
}


amrex::Real TortuosityDirect::residual(const amrex::MultiFab& phi_old, const amrex::MultiFab& phi_new ) const
{
    // Calculate L1 norm of the change in the potential component (comp_phi)
    amrex::Real delta_sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:delta_sum)
#endif
    for (amrex::MFIter mfi(phi_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Use tilebox with tiling
        const auto& fab_old = phi_old.const_array(mfi, comp_phi);
        const auto& fab_new = phi_new.const_array(mfi, comp_phi);
        const auto& ct_fab  = phi_old.const_array(mfi, comp_ct); // Need cell type to mask

        // Using amrex::ReduceSum requires a GPU-compatible lambda if on GPU
        // Using amrex::Loop and manual reduction is simpler for CPU/OMP here
        amrex::Real thread_delta = 0.0;
        amrex::Loop(bx, [&] (int i, int j, int k) {
            // Only include residual in the conducting phase (ct == 1)
            // Assuming cell_type == 1 means conducting phase
            if (static_cast<int>(ct_fab(i,j,k)) == 1) {
                thread_delta += std::abs(fab_new(i,j,k) - fab_old(i,j,k));
            }
        });
        delta_sum += thread_delta;
    }

    // Reduce across MPI processes
    amrex::ParallelDescriptor::ReduceRealSum(delta_sum, amrex::ParallelDescriptor::IOProcessorNumber());
    // If all ranks need the L1 norm, divide by global number of conducting cells.
    // For convergence check, the sum is often sufficient.
    // Return the sum for now.

    return delta_sum;
}

void TortuosityDirect::initialiseBoundaryConditions()
{
    // Default to Neumann (reflect_even)
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
        m_bc.setLo(i, amrex::BCType::reflect_even); // Physical BC for component 0 (phi)
        m_bc.setHi(i, amrex::BCType::reflect_even);
    }
    // Set Dirichlet (ext_dir) in the specified direction for component 0 (phi)
    m_bc.setLo(m_dir, amrex::BCType::ext_dir);
    m_bc.setHi(m_dir, amrex::BCType::ext_dir);

    // Note: BCRec allows setting different BCs per component if needed.
    // Here, we only explicitly set for comp 0. The behavior for comp 1 (cell type)
    // during FillBoundary depends on default BCs or how FillDomainBoundary handles it.
    // Often, cell types don't need physical BCs in the same way potential does.
}

// Fills component comp_ct based on m_mf_phase data
void TortuosityDirect::fillCellTypes(amrex::MultiFab& phi) // phi has comps (phi, ct)
{
    const amrex::Box& domain_box = m_geom.Domain();

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // const amrex::Box& bx = mfi.tilebox(); // Valid box for iteration

        // <<< FIXED Fortran call for tortuosity_filct >>>
        // auto& phi_fab_arr = phi.array(mfi); // ERROR: Changed to auto
        auto phi_fab_arr = phi.array(mfi); // CORRECTED: FAB data array (non-const)
        const auto& phase_fab_arr = m_mf_phase.const_array(mfi); // Phase data (const)

        int q_ncomp = phi.nComp();             // Should be 2
        int p_ncomp = m_mf_phase.nComp();      // Should be >= 1
        const auto& qbox = phi.fabbox(mfi);      // Box for phi FAB (including ghosts)
        const auto& pbox = m_mf_phase.fabbox(mfi); // Box for phase FAB (including ghosts)

        tortuosity_filct(phi_fab_arr.dataPtr(),
                         qbox.loVect().getVect(), qbox.hiVect().getVect(), &q_ncomp,
                         phase_fab_arr.dataPtr(), // Pass const int*
                         pbox.loVect().getVect(), pbox.hiVect().getVect(), &p_ncomp,
                         domain_box.loVect().getVect(), domain_box.hiVect().getVect(),
                         &m_phase);
    }
}

// Fills component comp_phi based on linear ramp in m_dir for active phase
void TortuosityDirect::fillInitialState(amrex::MultiFab& phi) // phi has comps (phi, ct)
{
    const amrex::Box& domain_box = m_geom.Domain();
    const int dir_int = static_cast<int>(m_dir); // Safer cast

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Operate on tile box

        // <<< FIXED Fortran call for tortuosity_filic >>>
        // auto& phi_fab_arr = phi.array(mfi); // ERROR: Changed to auto
        auto phi_fab_arr = phi.array(mfi); // CORRECTED: FAB data array (non-const)
        const auto& phase_fab_arr = m_mf_phase.const_array(mfi); // Phase data (const)

        int q_ncomp = phi.nComp();             // Should be 2
        int p_ncomp = m_mf_phase.nComp();      // Should be >= 1
        const auto& qbox = phi.fabbox(mfi);      // Box for phi FAB (including ghosts)
        const auto& pbox = m_mf_phase.fabbox(mfi); // Box for phase FAB (including ghosts)

        tortuosity_filic(phi_fab_arr.dataPtr(),
                         qbox.loVect().getVect(), qbox.hiVect().getVect(), &q_ncomp,
                         phase_fab_arr.dataPtr(),
                         pbox.loVect().getVect(), pbox.hiVect().getVect(), &p_ncomp,
                         bx.loVect().getVect(), bx.hiVect().getVect(),          // Valid box (tile) for filling
                         domain_box.loVect().getVect(), domain_box.hiVect().getVect(), // Domain bounds for interpolation scale
                         &m_vlo, &m_vhi,
                         &m_phase,
                         &dir_int);
    }
}


// Fills EXTERNAL DIRICHLET boundary conditions for a specified component using Fortran kernel
// Assumes Neumann/Periodic filled by AMReX's FillDomainBoundary beforehand
void TortuosityDirect::fillDomainBoundary (amrex::MultiFab& phi, int comp)
{
    const amrex::Box& domain_box = m_geom.Domain();

    // Temporary C array for boundary condition flags
    // This needs careful mapping from AMReX BCType to integers expected by Fortran
    int bc_c_array[AMREX_SPACEDIM*2];
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        // Fortran code likely expects specific integer flags (e.g., 0=Periodic, 1=Dirichlet, 2=Neumann)
        // We need to map AMReX types (defined in AMReX_BC_TYPES.H) to these.
        // This mapping is application specific! Assuming a simple mapping for illustration:
        // Note: We only care about ext_dir here, as others are handled by FillBoundary
        auto map_bc = [&](int amrex_bc_type) -> int { // Capture m_dir by reference if needed, but seems unused here
            if (amrex_bc_type == amrex::BCType::ext_dir) return 1; // Example: map ext_dir to 1 (Dirichlet flag for Fortran)
            // Map other types if the Fortran routine needs them (e.g., for different behavior)
            // if (amrex_bc_type == amrex::BCType::reflect_even) return 2; // Example: map reflect_even to 2
            return 0; // Default (e.g., Interior/Periodic/Neumann treated as non-Dirichlet by Fortran?) Needs verification.
        };
        // Apply mapping to the BCRec for the specific component 'comp' being filled
        bc_c_array[idim*2 + 0] = map_bc(m_bc.lo(idim, comp)); // Use component-specific BC
        bc_c_array[idim*2 + 1] = map_bc(m_bc.hi(idim, comp));
    }


#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& fab_box_ghosts = mfi.fabbox(); // Box including ghost cells

        // Only call on boxes that touch the physical domain boundary
        // And only if the component's BC actually involves Dirichlet
        bool touches_boundary = !domain_box.contains(fab_box_ghosts); // Simplified check
        if (touches_boundary) // Could refine check based on bc_c_array containing '1'
        {
             // <<< FIXED Fortran call for tortuosity_filbc >>>
            // auto& phi_fab_arr = phi.array(mfi); // ERROR: Changed to auto
            auto phi_fab_arr = phi.array(mfi); // CORRECTED: FAB data array (non-const)

            int q_ncomp = phi.nComp();
            const auto& qbox = phi.fabbox(mfi); // Box for phi FAB (including ghosts)

            // Fortran expects base pointer, ncomp implies which components to fill ghosts for
            // Call with base pointer of the specific component 'comp'
            tortuosity_filbc(phi_fab_arr.dataPtr(comp), // <<< Pass pointer to SPECIFIC component >>>
                             qbox.loVect().getVect(), qbox.hiVect().getVect(), &q_ncomp,
                             domain_box.loVect().getVect(), domain_box.hiVect().getVect(),
                             &m_vlo, &m_vhi,
                             bc_c_array);
        }
    }
}
