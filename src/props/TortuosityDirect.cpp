#include "TortuosityDirect.H"
#include "Tortuosity_poisson_3d_F.H" // Assuming Fortran interface for poisson steps
#include "Tortuosity_filcc_F.H"     // Assuming Fortran interface for fill/condition steps

#include <cstdlib> // For std::getenv
#include <cmath>   // For std::abs
#include <limits>  // For std::numeric_limits
#include <iomanip> // For std::setprecision, etc.
#include <string>  // For std::string
#include <vector>  // Used in setupMatrixEquation

#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Loop.H>             // <<< For amrex::Loop >>>
#include <AMReX_ParallelFor.H>      // <<< For amrex::ParallelFor >>>
#include <AMReX_GpuQualifiers.H>  // For AMREX_GPU_DEVICE etc.
#include <AMReX_ParallelDescriptor.H> // For reductions etc.
#include <AMReX_Array4.H>           // Explicit include for Array4 if needed

// Define constants for clarity and maintainability
namespace {
    constexpr int comp_phi = 0; // Index for potential phi in MultiFab
    constexpr int comp_ct  = 1; // Index for cell type in MultiFab (matches Fortran comp_ct=2)
    // constexpr int comp_phase = 0; // Index for phase in phase MultiFab (matches Fortran comp_phase=1) - Seems unused directly by name now
    constexpr int NUM_GHOST_CELLS = 1; // Required ghost cells for MultiFabs
}

//-----------------------------------------------------------------------
// NOTE: REMINDER - Ensure these members are declared in TortuosityDirect.H:
// private:
//    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxinv;
//    int m_plot_interval;
//    std::string m_plot_basename;
// Also ensure the constructor signature in TortuosityDirect.H matches below.
//-----------------------------------------------------------------------

// Constructor implementation
TortuosityDirect::TortuosityDirect(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                   const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf_phase,
                                   const int phase, const OpenImpala::Direction dir,
                                   const amrex::Real eps,
                                   const int n_steps,
                                   const int plot_interval,
                                   const std::string& plot_basename,
                                   const amrex::Real vlo, const amrex::Real vhi)
    : m_geom(geom), m_ba(ba), m_dm(dm), m_mf_phase(mf_phase, amrex::make_alias, 0, mf_phase.nComp()), // Use alias constructor if mf_phase lifetime guaranteed externally
      m_phase(phase), m_dir(dir),
      m_n_steps(n_steps), m_eps(eps),
      m_plot_interval(plot_interval), m_plot_basename(plot_basename),
      m_vlo(vlo), m_vhi(vhi),
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_last_iterations(0), m_last_residual(-1.0)
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


void TortuosityDirect::initialiseFluxMultiFabs()
{
    // Build the flux multi-fabs
    for (int i_dir = 0; i_dir < AMREX_SPACEDIM; ++i_dir)
    {
        amrex::BoxArray edge_ba = m_ba; // Make a copy to modify
        edge_ba.surroundingNodes(i_dir);
        m_flux[i_dir].define(edge_ba, m_dm, 1, 0); // 1 component, 0 ghost cells
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
    } else if (std::isnan(m_value)) {
         // If previous solve failed (cached NaN) and refresh=false, return NaN immediately
         return m_value;
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
    constexpr amrex::Real flux_dens_tol = 1e-15; // Tolerance for near-zero check
    if (std::abs(avg_flux_density) < flux_dens_tol) {
        amrex::Warning("Calculated average flux density is near zero. Setting tortuosity to infinity.");
        m_value = std::numeric_limits<amrex::Real>::infinity();
    } else {
        // Assuming definition: Tortuosity = Volume Fraction / RelativeDiffusivity
        // Where RelativeDiffusivity = D_eff / D_bulk = - FluxDensity * Length / (Vhi - Vlo) (for D_bulk=1)
        // => Tortuosity = VF * (Vhi - Vlo) / (-FluxDensity * Length)
        // Need Volume Fraction (VF) - Assuming it should be passed or calculated? Using 1.0 for now.
        // VF should ideally be calculated from m_mf_phase for the phase m_phase.
        // Example: OpenImpala::VolumeFraction vf_calc(m_mf_phase, m_phase, 0); amrex::Real vf = vf_calc.value();
        amrex::Real vf = 1.0; // Placeholder - Needs actual Volume Fraction calculation!
        amrex::Real length = m_geom.ProbLength(static_cast<int>(m_dir));
        amrex::Real delta_V = m_vhi - m_vlo;

        if (std::abs(delta_V) < flux_dens_tol || length <= 0.0 ) {
             amrex::Warning("TortuosityDirect::value: Cannot calculate tortuosity due to zero potential difference or non-positive length.");
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else {
             amrex::Real rel_diff = -avg_flux_density * length / delta_V;
             if (std::abs(rel_diff) < flux_dens_tol) {
                  amrex::Warning("Calculated relative diffusivity is near zero. Setting tortuosity to infinity.");
                  m_value = std::numeric_limits<amrex::Real>::infinity();
             } else {
                 m_value = vf / rel_diff;
                 // Tortuosity should generally be >= VF. Check sign based on conventions.
                  if (m_value < 0.0 && vf > 0.0) {
                      amrex::Warning("Calculated negative tortuosity, check flux direction, BCs, or definition.");
                  }
             }
        }
    }
    return m_value;
}

bool TortuosityDirect::solve()
{
    // Create MultiFabs for variables (potential + cell_type)
    // Need 1 ghost cell for finite difference stencils
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
    m_last_residual = -1.0;

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

            if(amrex::ParallelDescriptor::IOProcessor()) { // Print only on IO rank
                amrex::Print() << "Step " << n << ": ";
                amrex::Print() << std::scientific << std::setprecision(4) << std::setw(12) << std::setfill(' '); // Use scientific
                amrex::Print() << "Residual: " << current_res << " | ";
                amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << " Delta: " << std::abs(fxout + fxin) << std::endl; // Flux conservation: in = -out ideally
            }

            if (current_res < m_eps)
            {
                if(amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "Convergence reached in " << n << " steps." << std::endl;
                }
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
             if(amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Final Step " << m_n_steps << ": ";
                 amrex::Print() << "Residual: " << current_res << " | ";
                 amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << std::endl;
             }
         }
    }

    // Write final plot file if requested
    if (m_plot_interval > 0) // Assuming plot_interval > 0 implies plotting desired
    {
        std::string plot_file_path_str = m_plot_basename + "_step_" + std::to_string(m_last_iterations);
        if(amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Writing plot file: " << plot_file_path_str << std::endl;
        }
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

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:fxin, fxout)
#endif
    {
        amrex::Real lfxin = 0.0; // Private to thread
        amrex::Real lfxout = 0.0;// Private to thread

        // Iterate over flux multifabs
        for ( amrex::MFIter mfi(m_flux[idir]); mfi.isValid(); ++mfi ) // Iterate over the relevant flux direction
        {
            // <<< FIXED Fortran call for tortuosity_poisson_fio >>>
            const amrex::Box& bx = mfi.tilebox(); // Use tilebox for valid region

            // Pointers need to be const for Fortran routine expecting const pointers
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
        fxin += lfxin; // Combine thread-local sums into shared variables
        fxout += lfxout;
    } // End OMP parallel region

    // Reduce across MPI processes - Sum results from all ranks
    amrex::ParallelDescriptor::ReduceRealSum(fxin);  // Reduce sum across all ranks
    amrex::ParallelDescriptor::ReduceRealSum(fxout); // Reduce sum across all ranks
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

#ifdef AMREX_USE_OMP
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
            // Get pointer to solution data (const) - Pass base pointer of phi_old FAB
            const auto* sol_ptr = phi_old.dataPtr(0, mfi);

            const auto& fxbox = m_flux[0].fabbox(mfi);
            const auto& fybox = m_flux[1].fabbox(mfi);
            const auto& fzbox = m_flux[2].fabbox(mfi);
            const auto& solbox = phi_old.fabbox(mfi);

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
            const auto* p_ptr = phi_old.dataPtr(comp_phi, mfi); // Pointer to old potential (comp_phi)
            auto* n_ptr = phi_new.dataPtr(comp_phi, mfi); // Pointer to new potential (comp_phi, modifiable)
            const auto* fx_ptr = m_flux[0].dataPtr(mfi);      // Pointer to x-flux
            const auto* fy_ptr = m_flux[1].dataPtr(mfi);      // Pointer to y-flux
            const auto* fz_ptr = m_flux[2].dataPtr(mfi);      // Pointer to z-flux

            // Get box bounds for arrays (FABs include ghost cells)
            const auto& pbox = phi_old.fabbox(mfi); // Bounds for p (phi_old)
            const auto& nbox = phi_new.fabbox(mfi); // Bounds for n (phi_new)
            const auto& fxbox = m_flux[0].fabbox(mfi); // Bounds for fx
            const auto& fybox = m_flux[1].fabbox(mfi); // Bounds for fy
            const auto& fzbox = m_flux[2].fabbox(mfi); // Bounds for fz

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
            const auto& ct_fab_old = phi_old.const_array(mfi, comp_ct);
            auto        ct_fab_new = phi_new.array(mfi, comp_ct);

            // <<< FIXED Array4 copy >>>
            // ct_fab_new.copy(ct_fab_old, bx); // ERROR: Array4 has no .copy() member
            // Replace with a loop:
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                ct_fab_new(i,j,k) = ct_fab_old(i,j,k);
            });
        }
    } // End OMP parallel region
}


amrex::Real TortuosityDirect::residual(const amrex::MultiFab& phi_old, const amrex::MultiFab& phi_new ) const
{
    // Calculate L1 norm of the change in the potential component (comp_phi)
    amrex::Real delta_sum = 0.0;

#ifdef AMREX_USE_OMP
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
        amrex::Loop(bx, [&] (int i, int j, int k) { // Using amrex::Loop now
            // Only include residual in the conducting phase (ct == 1)
            // Assuming cell_type == 1 means conducting phase
            if (static_cast<int>(ct_fab(i,j,k)) == 1) {
                thread_delta += std::abs(fab_new(i,j,k) - fab_old(i,j,k));
            }
        });
        delta_sum += thread_delta;
    }

    // Reduce across MPI processes - Use AllReduce to get sum on all ranks
    amrex::ParallelAllReduce::Sum(delta_sum, amrex::ParallelContext::CommunicatorSub());

    // Optional: Normalize the residual? e.g., divide by number of active cells or initial norm.
    // For now, return the raw L1 sum of differences.
    return delta_sum;
}

void TortuosityDirect::initialiseBoundaryConditions()
{
    // Initialize BCRec for both components (phi, ct) - use number of comps = 2
    // Default to Neumann (reflect_even) for both initially
    m_bc = amrex::BCRec(amrex::IntVect(0,0), 2); // Initialize for 2 components

    for (int comp=0; comp < 2; ++comp) { // Loop over components
        for (int i=0; i<AMREX_SPACEDIM; ++i)
        {
            m_bc.setLo(i, comp, amrex::BCType::reflect_even); // Default Neumann
            m_bc.setHi(i, comp, amrex::BCType::reflect_even);
        }
    }

    // Set Dirichlet (ext_dir) in the specified direction for component comp_phi (0) only
    m_bc.setLo(m_dir, comp_phi, amrex::BCType::ext_dir);
    m_bc.setHi(m_dir, comp_phi, amrex::BCType::ext_dir);

    // Cell type (comp_ct = 1) retains Neumann BCs, which is usually fine
    // as its value should not depend on ghost cells.
}

// Fills component comp_ct based on m_mf_phase data
void TortuosityDirect::fillCellTypes(amrex::MultiFab& phi) // phi has comps (phi, ct)
{
    const amrex::Box& domain_box = m_geom.Domain();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // <<< FIXED Fortran call for tortuosity_filct >>>
        // auto& phi_fab_arr = phi.array(mfi); // ERROR: Changed to auto
        auto phi_fab_arr = phi.array(mfi); // CORRECTED: FAB data array (non-const)
        const auto& phase_fab_arr = m_mf_phase.const_array(mfi); // Phase data (const)

        int q_ncomp = phi.nComp();             // Should be 2
        int p_ncomp = m_mf_phase.nComp();      // Should be >= 1
        const auto& qbox = phi.fabbox(mfi);      // Box for phi FAB (including ghosts)
        const auto& pbox = m_mf_phase.fabbox(mfi); // Box for phase FAB (including ghosts)

        // Fortran routine modifies component comp_ct (Fortran index 2) based on component comp_phase (Fortran index 1) from p
        tortuosity_filct(phi_fab_arr.dataPtr(),           // Base pointer to phi FAB (Real)
                         qbox.loVect().getVect(), qbox.hiVect().getVect(), &q_ncomp,
                         phase_fab_arr.dataPtr(),         // Base pointer to phase FAB (Int)
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

#ifdef AMREX_USE_OMP
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

        // Fortran routine fills component comp_phi (Fortran index 1) based on component comp_phase (Fortran index 1) from p
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
    int bc_c_array[AMREX_SPACEDIM*2];
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        auto map_bc = [&](int amrex_bc_type) -> int {
            if (amrex_bc_type == amrex::BCType::ext_dir) return 1; // Map ext_dir to 1
            // Map others if needed by Fortran, otherwise map to non-Dirichlet (e.g., 0)
            return 0;
        };
        bc_c_array[idim*2 + 0] = map_bc(m_bc.lo(idim, comp)); // Use component-specific BC
        bc_c_array[idim*2 + 1] = map_bc(m_bc.hi(idim, comp));
    }


#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& fab_box_ghosts = mfi.fabbox(); // Box including ghost cells

        // Only call on boxes that touch the physical domain boundary
        bool touches_boundary = !domain_box.contains(fab_box_ghosts);
        if (touches_boundary)
        {
             // <<< FIXED Fortran call for tortuosity_filbc >>>
            // auto& phi_fab_arr = phi.array(mfi); // ERROR: Changed to auto
            auto phi_fab_arr = phi.array(mfi); // CORRECTED: Array4 view (non-const)

            int q_ncomp = phi.nComp();
            const auto& qbox = phi.fabbox(mfi); // Box for phi FAB (including ghosts)

            // Assuming Fortran tortuosity_filbc handles applying BCs only to the
            // component specified by the base pointer `phi_fab_arr.dataPtr(comp)`
            // if ncomp > 1, or applies to all components if it ignores comp implicitly.
            // Passing base pointer for the specific component 'comp'.
            tortuosity_filbc(phi_fab_arr.dataPtr(comp), // <<< Pass pointer to SPECIFIC component >>>
                             qbox.loVect().getVect(), qbox.hiVect().getVect(), &q_ncomp,
                             domain_box.loVect().getVect(), domain_box.hiVect().getVect(),
                             &m_vlo, &m_vhi,
                             bc_c_array);
        }
    }
}
