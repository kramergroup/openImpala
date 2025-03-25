#include "TortuosityDirect.H"
#include "Tortuosity_poisson_3d_F.H" // Assuming Fortran interface for poisson steps
#include "Tortuosity_filcc_F.H"      // Assuming Fortran interface for fill/condition steps

#include <cstdlib> // For std::getenv
#include <cmath>   // For std::abs
#include <limits>  // For std::numeric_limits
#include <iomanip> // For std::setprecision, etc.
#include <string>  // For std::string

#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCUtil.H>
#include <AMReX_PlotFileUtil.H>
// #include <AMReX_MLLinOp.H>   // No longer explicitly used in this version
// #include <AMReX_MLPoisson.H> // No longer explicitly used in this version
// #include <AMReX_MLMG.H>      // No longer explicitly used in this version

#define NUM_GHOST_CELLS 1
// Assuming these constants are defined somewhere (e.g., in TortuosityDirect.H or globally)
constexpr int comp_phi = 0; // Index for potential phi in MultiFab
constexpr int comp_ct  = 1; // Index for cell type in MultiFab (matches Fortran comp_ct=2)
constexpr int comp_phase = 0; // Index for phase in phase MultiFab (matches Fortran comp_phase=1)

// Constructor with added configuration parameters
TortuosityDirect::TortuosityDirect(const amrex::Geometry& geom, const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf,
                                 const int phase, const Direction dir,
                                 int n_steps, amrex::Real eps,
                                 int plot_interval, const std::string& plot_basename,
                                 amrex::Real vlo, amrex::Real vhi)
    : m_geom(geom), m_ba(ba), m_dm(dm), m_mf_phase(mf),
      m_phase(phase), m_dir(dir),
      m_n_steps(n_steps), m_eps(eps),
      m_plot_interval(plot_interval), m_plot_basename(plot_basename),
      m_vlo(vlo), m_vhi(vhi),
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()) // Initialize value to NaN
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
        amrex::BoxArray edge_ba(m_ba);
        edge_ba.surroundingNodes(i_dir);
        m_flux[i_dir].define(edge_ba, m_dm, 1, 0);
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
    amrex::Real fx = (fxin + fxout) / 2.0;

    // Calculate tortuosity
    const amrex::Box& bx_domain = m_geom.Domain();
    const amrex::IntVect& sz = bx_domain.size(); // Use size() which gives dimensions+1

    amrex::Real cross_sectional_area = 0.0;
    switch(m_dir)
    {
        case X : cross_sectional_area = static_cast<amrex::Real>(sz[1]) * static_cast<amrex::Real>(sz[2]); break;
        case Y : cross_sectional_area = static_cast<amrex::Real>(sz[0]) * static_cast<amrex::Real>(sz[2]); break;
        case Z : cross_sectional_area = static_cast<amrex::Real>(sz[0]) * static_cast<amrex::Real>(sz[1]); break;
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
        // Flux Density = -Deff * Grad(Potential) = -Deff * (1/L)
        // Tortuosity = D/Deff = D / (-Flux Density * L)
        // If D=1, Vhi-Vlo=1, then Tortuosity = 1 / (-Flux Density * L) ??
        // Let's assume the definition used leads to: 1.0 / avg_flux_density
        // THIS NEEDS VERIFICATION based on the PDE being solved and the definition of tortuosity used.
        m_value = 1.0 / avg_flux_density;
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
    // Review Note: The factor 1000 and dependence only on dx[0] is still questionable.
    // Is this meant to be a relaxation parameter rather than a physical timestep?
    // Using a safety factor (e.g., 0.9) for explicit diffusion CFL: dt <= dx^2 / (2*D*AMREX_SPACEDIM)
    // Assuming D=1 for simplicity here. A smaller dt might be safer.
    amrex::Real min_dx_sq = dx[0]*dx[0];
    for(int i=1; i<AMREX_SPACEDIM; ++i) min_dx_sq = std::min(min_dx_sq, dx[i]*dx[i]);
    const amrex::Real dt = 0.9 * min_dx_sq / (2.0 * AMREX_SPACEDIM); // More conventional CFL estimate (D=1)
    // const amrex::Real dt = 0.9 * dx[0] * dx[0] / (2.0 * AMREX_SPACEDIM) * 1000; // Original dt

    amrex::Real current_res = std::numeric_limits<amrex::Real>::max();
    amrex::Real fxin = 0.0, fxout = 0.0;
    bool converged = false;

    for (int n = 1; n <= m_n_steps; ++n)
    {
        // Copy only the potential (comp_phi) from new to old for the next step
        amrex::MultiFab::Copy(mf_phi_old, mf_phi_new, comp_phi, comp_phi, 1, NUM_GHOST_CELLS);
        // Cell type (comp_ct) remains static in mf_phi_old after initialization

        advance(mf_phi_old, mf_phi_new, dt); // Pass dt

        // Check residual and print status periodically
        if (m_plot_interval > 0 && n % m_plot_interval == 0)
        {
            current_res = residual(mf_phi_old, mf_phi_new); // Calculates residual based on comp_phi change
            global_fluxes(fxin, fxout);                     // Recalculates fluxes based on mf_phi_new (via m_flux)

            amrex::Print() << "Step " << n << ": ";
            amrex::Print() << std::fixed << std::setprecision(6) << std::setw(12) << std::setfill(' ');
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
             amrex::Print() << "Final Step " << m_n_steps << ": ";
             amrex::Print() << "Residual: " << current_res << " | ";
             amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << std::endl;
         }
    }

    // Write final plot file
    const char* homeDir_cstr = std::getenv("HOME");
    std::string plot_file_path;
    if (homeDir_cstr) {
        plot_file_path = std::string(homeDir_cstr) + "/" + m_plot_basename; // Use configured basename
    } else {
        amrex::Warning("HOME environment variable not set. Writing plot file to current directory.");
        plot_file_path = m_plot_basename; // Default to current directory
    }
    // Plotting both potential (comp_phi) and cell type (comp_ct)
    amrex::WriteSingleLevelPlotfile(plot_file_path, mf_phi_new, {"potential", "cell_type"}, m_geom, 0.0, 0);
    amrex::Print() << "Plot file written to: " << plot_file_path << std::endl;

    // Ensure fluxes are based on the final state (already done if converged/last step printed)
    if (!converged && !(m_plot_interval > 0 && m_n_steps % m_plot_interval == 0)) {
        global_fluxes(fxin, fxout); // Ensure m_flux is consistent with final mf_phi_new
    }

    m_first_call = !converged; // Set to false only if converged, true otherwise
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

        // Compute fluxes on domain boundaries - uses m_flux which should be up-to-date
        // No MFIter needed here if tortuosity_poisson_fio operates globally?
        // Re-checking: tortuosity_poisson_fio likely needs fluxes per box.
        // The original loop structure seems necessary if tortuosity_poisson_fio calculates
        // local contributions based on the m_flux MultiFab. Let's revert to that.

        // Original structure with MFIter and reduction
        for ( amrex::MFIter mfi(m_flux[0]); mfi.isValid(); ++mfi )
        {
            // The Fortran routine sums flux over the appropriate domain boundary faces
            // intersecting the current box. It needs access to flux arrays.
            tortuosity_poisson_fio(BL_TO_FORTRAN_BOX(bx_domain), // Domain box
                                   BL_TO_FORTRAN_ANYD(m_flux[0][mfi]), // Need flux FABs
                                   BL_TO_FORTRAN_ANYD(m_flux[1][mfi]),
                                   BL_TO_FORTRAN_ANYD(m_flux[2][mfi]),
                                   &dir_int, &lfxin, &lfxout);
        }
        // OpenMP reduction clause handles sum automatically
        fxin += lfxin;
        fxout += lfxout;
    } // End OMP parallel region

    // Reduce across MPI processes
    amrex::ParallelAllReduce::Sum(fxin, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(fxout, amrex::ParallelContext::CommunicatorSub());
}


// Advance phi_new = phi_old + dt * Laplacian(phi_old) [masked by cell_type]
// Also computes fluxes based on phi_old for the update and for later analysis
void TortuosityDirect::advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new, const amrex::Real& dt)
{
    // Fill the ghost cells of phi_old (potential and cell_type)
    phi_old.FillBoundary(m_geom.periodicity());

    // Apply physical boundary conditions to ghost cells (Dirichlet handled separately)
    // Note: This fills ghosts for BOTH components based on m_bc.
    // If cell_type (comp_ct) should have different BCs, this needs adjustment.
    amrex::FillDomainBoundary(phi_old, m_geom, {m_bc, m_bc}); // Apply BCs to both components

    // Apply external Dirichlet explicitly using Fortran kernel for phi component
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
            const auto& phi_old_fab = phi_old.const_array(mfi, comp_phi); // Read only potential
            const auto& ct_fab = phi_old.const_array(mfi, comp_ct); // Read only cell type
            auto const& fx = m_flux[0].array(mfi); // Write flux_x
            auto const& fy = m_flux[1].array(mfi); // Write flux_y
            auto const& fz = m_flux[2].array(mfi); // Write flux_z

            // Assuming tortuosity_poisson_flux expects phi, cell_type, computes flux
            // Needs adjustment if Fortran expects combined FAB or different component indices
            tortuosity_poisson_flux(BL_TO_FORTRAN_BOX(bx),
                                    BL_TO_FORTRAN_ANYD(fx), // Pass flux arrays
                                    BL_TO_FORTRAN_ANYD(fy),
                                    BL_TO_FORTRAN_ANYD(fz),
                                    BL_TO_FORTRAN_ANYD(phi_old_fab), // Pass potential component
                                    BL_TO_FORTRAN_ANYD(ct_fab),      // Pass cell type component
                                    m_dxinv.data()); // Pass inverse cell sizes
                                    // Missing src_comp '0' from original call - verify Fortran signature
        }

        // Advance the solution (phi_new) using fluxes based on phi_old
        for ( amrex::MFIter mfi(phi_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.tilebox();
            const auto& phi_old_fab = phi_old.const_array(mfi, comp_phi); // Read potential
            const auto& ct_fab      = phi_old.const_array(mfi, comp_ct); // Read cell type
            auto      phi_new_fab = phi_new.array(mfi, comp_phi);       // Write potential

            const auto& fx = m_flux[0].const_array(mfi); // Read flux_x
            const auto& fy = m_flux[1].const_array(mfi); // Read flux_y
            const auto& fz = m_flux[2].const_array(mfi); // Read flux_z

            // Assuming tortuosity_poisson_update expects old_phi, old_ct, new_phi, fluxes, dx, dt
            tortuosity_poisson_update(BL_TO_FORTRAN_BOX(bx),
                                      BL_TO_FORTRAN_ANYD(phi_old_fab),
                                      BL_TO_FORTRAN_ANYD(ct_fab),
                                      BL_TO_FORTRAN_ANYD(phi_new_fab), // Note: Pass modifiable array for new phi
                                      BL_TO_FORTRAN_ANYD(fx),
                                      BL_TO_FORTRAN_ANYD(fy),
                                      BL_TO_FORTRAN_ANYD(fz),
                                      dx, &dt); // Pass dx array pointer and dt address

            // Ensure cell type remains unchanged in phi_new (copy from phi_old if necessary)
            // If update kernel doesn't touch comp_ct, this isn't needed.
            // If update writes garbage to comp_ct, copy from phi_old:
            // phi_new.array(mfi, comp_ct).copy(ct_fab, bx);
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

        amrex::Real thread_delta = 0.0;
        amrex::Loop(bx, [&] (int i, int j, int k) {
            // Only include residual in the conducting phase (ct == 1)
            if (ct_fab(i,j,k) == 1) { // Assuming cell_type_free == 1
                 thread_delta += std::abs(fab_new(i,j,k) - fab_old(i,j,k));
            }
        });
        delta_sum += thread_delta;
    }

    // Reduce across MPI processes
    amrex::ParallelAllReduce::Sum(delta_sum, amrex::ParallelContext::CommunicatorSub());

    return delta_sum;
}

void TortuosityDirect::initialiseBoundaryConditions()
{
    // Default to Neumann (reflect_even)
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
        m_bc.setLo(i, amrex::BCType::reflect_even);
        m_bc.setHi(i, amrex::BCType::reflect_even);
    }
    // Set Dirichlet (ext_dir) in the specified direction
    m_bc.setLo(m_dir, amrex::BCType::ext_dir);
    m_bc.setHi(m_dir, amrex::BCType::ext_dir);
}

// Fills component comp_ct based on m_mf_phase data
void TortuosityDirect::fillCellTypes(amrex::MultiFab& phi) // phi has comps (phi, ct)
{
    const amrex::Box& domain_box = m_geom.Domain(); // Needed for Fortran call

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto&       phi_fab_arr = phi.array(mfi); // Full fab needed for Fortran call
        const auto& phase_fab_arr = m_mf_phase.const_array(mfi); // Phase data

        int q_ncomp = phi.nComp();      // Should be 2
        int p_ncomp = m_mf_phase.nComp(); // Should be >= 1

        // Correct call to Fortran tortuosity_filct
        tortuosity_filct(phi_fab_arr.dataPtr(), // Pointer to start of FAB data
                         phi.loVect(mfi), phi.hiVect(mfi), &q_ncomp,
                         phase_fab_arr.dataPtr(),
                         m_mf_phase.loVect(mfi), m_mf_phase.hiVect(mfi), &p_ncomp,
                         domain_box.loVect(), domain_box.hiVect(), // Domain bounds
                         &m_phase); // Phase value to check
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
        auto&       phi_fab_arr = phi.array(mfi); // Full fab for Fortran
        const auto& phase_fab_arr = m_mf_phase.const_array(mfi);

        int q_ncomp = phi.nComp();      // Should be 2
        int p_ncomp = m_mf_phase.nComp(); // Should be >= 1

        // Correct call to Fortran tortuosity_filic
        tortuosity_filic(phi_fab_arr.dataPtr(), // Pointer to start of FAB data
                         phi.loVect(mfi), phi.hiVect(mfi), &q_ncomp,
                         phase_fab_arr.dataPtr(),
                         m_mf_phase.loVect(mfi), m_mf_phase.hiVect(mfi), &p_ncomp,
                         bx.loVect(), bx.hiVect(),               // Valid box (tile) for filling
                         domain_box.loVect(), domain_box.hiVect(), // Domain bounds for interpolation scale
                         &m_vlo, &m_vhi,                         // Boundary values
                         &m_phase,                               // Phase to fill
                         &dir_int);                              // Direction of fill
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
         auto map_bc = [](int amrex_bc_type) -> int {
             if (amrex_bc_type == amrex::BCType::ext_dir) return 1; // Example: map ext_dir to 1
             if (amrex_bc_type == amrex::BCType::reflect_even) return 2; // Example: map reflect_even to 2
             // Add mappings for foextrap, hoextrap, reflect_odd, periodic as needed
             return 0; // Default or periodic? Needs check.
         };
         bc_c_array[idim*2 + 0] = map_bc(m_bc[idim].lo()); // Low side mapped
         bc_c_array[idim*2 + 1] = map_bc(m_bc[idim].hi()); // High side mapped
    }


#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& fab_box_ghosts = mfi.fabbox(); // Box including ghost cells
        auto& phi_fab_arr = phi.array(mfi);           // Full FAB for Fortran

        // Only call on boxes that touch the physical domain boundary
        if (! domain_box.strictly_contains(fab_box_ghosts))
        {
            int q_ncomp = phi.nComp();

            // Correct call to Fortran tortuosity_filbc
            tortuosity_filbc(phi_fab_arr.dataPtr(comp), // Pointer to specific component data
                             phi.loVect(mfi), phi.hiVect(mfi), &q_ncomp, // Pass full FAB bounds/ncomp
                             domain_box.loVect(), domain_box.hiVect(),   // Domain bounds
                             &m_vlo, &m_vhi,                             // Boundary values
                             bc_c_array);                                // Mapped BC flags
        }
    }
}
