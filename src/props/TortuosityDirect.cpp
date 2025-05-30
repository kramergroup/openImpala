#include "TortuosityDirect.H"
#include "Tortuosity_poisson_3d_F.H" // Assuming Fortran interface for poisson steps
#include "Tortuosity_filcc_F.H"      // Assuming Fortran interface for fill/condition steps

#include <cstdlib> // For std::getenv
#include <cmath>   // For std::abs
#include <limits>  // For std::numeric_limits
#include <iomanip> // For std::setprecision, etc.
#include <string>  // For std::string
#include <vector>  // Used in setupMatrixEquation

#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H> // Defines BCType enum
#include <AMReX_BCRec.H>    // Include BCRec header
#include <AMReX_BCUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Loop.H>            // For amrex::Loop
#include <AMReX_GpuLaunch.H>       // For amrex::ParallelFor (check if needed for CPU build)
#include <AMReX_GpuQualifiers.H> // For AMREX_GPU_DEVICE etc. (check if needed for CPU build)
#include <AMReX_ParallelDescriptor.H> // For reductions etc.
#include <AMReX_Array4.H>          // Explicit include for Array4
#include <AMReX_Vector.H>          // For amrex::Vector
#include <AMReX_Array.H>           // For amrex::Array (m_dxinv type)


namespace OpenImpala {

// Define constants for clarity and maintainability
namespace {
    constexpr int comp_phi = 0; // Index for potential phi in MultiFab
    constexpr int comp_ct  = 1; // Index for cell type in MultiFab (matches Fortran comp_ct=2)
    constexpr int NUM_GHOST_CELLS = 1; // Required ghost cells for MultiFabs
}

//-----------------------------------------------------------------------
// REMINDER: Ensure m_dxinv is declared correctly in TortuosityDirect.H
// It should likely be: amrex::Array<amrex::Real, AMREX_SPACEDIM> m_dxinv;
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
    : m_geom(geom), m_ba(ba), m_dm(dm), m_mf_phase(mf_phase),
      m_phase(phase), m_dir(dir),
      m_n_steps(n_steps), m_eps(eps),
      m_plot_interval(plot_interval), m_plot_basename(plot_basename),
      m_vlo(vlo), m_vhi(vhi),
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_last_iterations(0), m_last_residual(-1.0)
{
    // Ensure m_dxinv type is amrex::Array<amrex::Real, AMREX_SPACEDIM> in header
    const amrex::Real* dx = m_geom.CellSize();
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
        if (dx[i] <= 0.0) {
            amrex::Abort("TortuosityDirect: Non-positive dx");
        }
        m_dxinv[i] = 1.0/ dx[i];
    }

    // Create a datastructure describing the boundary conditions
    initializeBoundaryConditions();
    initializeFluxMultiFabs();
}


void TortuosityDirect::initializeFluxMultiFabs()
{
    // Build the flux multi-fabs
    for (int i_dir = 0; i_dir < AMREX_SPACEDIM; ++i_dir)
    {
        amrex::BoxArray edge_ba = m_ba;
        edge_ba.surroundingNodes(i_dir);
        m_flux[i_dir].define(edge_ba, m_dm, 1, 0);
        m_flux[i_dir].setVal(0.0);
    }
}

amrex::Real TortuosityDirect::value(const bool refresh)
{
    if ( refresh || m_first_call )
    {
        if (!solve())
        {
            amrex::Warning("TortuosityDirect::solve() failed to converge. Returning NaN.");
            m_first_call = true;
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            return m_value;
        }
    } else if (std::isnan(m_value)) {
         return m_value;
    }

    amrex::Real fxin = 0.0, fxout = 0.0;
    global_fluxes(fxin, fxout);
    amrex::Real fx = (fxin + fxout) / 2.0;

    const amrex::Box& bx_domain = m_geom.Domain();
    const amrex::IntVect& sz = bx_domain.size();

    amrex::Real cross_sectional_area = 0.0;
    switch(m_dir)
    {
        case Direction::X : cross_sectional_area = static_cast<amrex::Real>(sz[1]) * static_cast<amrex::Real>(sz[2]); break;
        case Direction::Y : cross_sectional_area = static_cast<amrex::Real>(sz[0]) * static_cast<amrex::Real>(sz[2]); break;
        case Direction::Z : cross_sectional_area = static_cast<amrex::Real>(sz[0]) * static_cast<amrex::Real>(sz[1]); break;
        default: amrex::Abort("TortuosityDirect::value: Invalid direction");
    }

    if (cross_sectional_area <= 0.0) {
         amrex::Warning("TortuosityDirect::value: Domain cross-sectional area is non-positive. Cannot calculate tortuosity.");
         m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
         return m_value;
    }

    amrex::Real avg_flux_density = fx / cross_sectional_area;

    constexpr amrex::Real flux_dens_tol = 1e-15;
    if (std::abs(avg_flux_density) < flux_dens_tol) {
        amrex::Warning("Calculated average flux density is near zero. Setting tortuosity to infinity.");
        m_value = std::numeric_limits<amrex::Real>::infinity();
    } else {
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
    amrex::MultiFab mf_phi_old(m_ba, m_dm, 2, NUM_GHOST_CELLS);
    amrex::MultiFab mf_phi_new(m_ba, m_dm, 2, NUM_GHOST_CELLS);

    fillInitialState(mf_phi_old);
    fillCellTypes(mf_phi_old);
    amrex::MultiFab::Copy(mf_phi_new, mf_phi_old, 0, 0, 2, NUM_GHOST_CELLS);

    const amrex::Real* dx = m_geom.CellSize();
    amrex::Real min_dx_sq = dx[0]*dx[0];
    for(int i=1; i<AMREX_SPACEDIM; ++i) min_dx_sq = std::min(min_dx_sq, dx[i]*dx[i]);
    const amrex::Real dt = 0.5 * min_dx_sq / (2.0 * AMREX_SPACEDIM);

    amrex::Real current_res = std::numeric_limits<amrex::Real>::max();
    amrex::Real fxin = 0.0, fxout = 0.0;
    bool converged = false;
    m_last_iterations = 0;
    m_last_residual = -1.0;

    for (int n = 1; n <= m_n_steps; ++n)
    {
        m_last_iterations = n;
        amrex::MultiFab::Copy(mf_phi_old, mf_phi_new, comp_phi, comp_phi, 1, NUM_GHOST_CELLS);
        advance(mf_phi_old, mf_phi_new, dt);

        if (m_plot_interval > 0 && n % m_plot_interval == 0)
        {
            current_res = residual(mf_phi_old, mf_phi_new);
            global_fluxes(fxin, fxout);
            m_last_residual = current_res;

            if(amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "Step " << n << ": ";
                amrex::Print() << std::scientific << std::setprecision(4) << std::setw(12) << std::setfill(' ');
                amrex::Print() << "Residual: " << current_res << " | ";
                amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << " Delta: " << std::abs(fxout + fxin) << std::endl;
            }

            if (current_res < m_eps)
            {
                if(amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "Convergence reached in " << n << " steps." << std::endl;
                }
                converged = true;
                break;
            }
        }
    }

    if (!converged) {
         amrex::Warning("TortuosityDirect::solve() did not converge within max steps.");
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

    if (m_plot_interval > 0)
    {
        std::string plot_file_path_str = m_plot_basename + "_step_" + std::to_string(m_last_iterations);
        if(amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Writing plot file: " << plot_file_path_str << std::endl;
        }
        amrex::Vector<std::string> plot_varnames = {"potential", "cell_type"};
        amrex::WriteSingleLevelPlotfile(plot_file_path_str, mf_phi_new, plot_varnames, m_geom, 0.0, m_last_iterations);
    }

    if (!converged && !(m_plot_interval > 0 && m_n_steps % m_plot_interval == 0)) {
        global_fluxes(fxin, fxout);
    }

    m_first_call = !converged;
    return converged;
}


void TortuosityDirect::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    fxin = 0.0;
    fxout = 0.0;

    const amrex::Box& bx_domain = m_geom.Domain();
    const int dir_int = static_cast<int>(m_dir);

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:fxin, fxout)
#endif
    {
        amrex::Real lfxin = 0.0;
        amrex::Real lfxout = 0.0;
        int idir = dir_int;
        if (idir < 0 || idir >= AMREX_SPACEDIM) {
            amrex::Abort("Invalid direction specified for global_fluxes");
        }

        for ( amrex::MFIter mfi(m_flux[idir]); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.tilebox();

            const auto& fx_arr = m_flux[0].const_array(mfi);
            const auto& fy_arr = m_flux[1].const_array(mfi);
            const auto& fz_arr = m_flux[2].const_array(mfi);
            const auto* fx_ptr = fx_arr.dataPtr();
            const auto* fy_ptr = fy_arr.dataPtr();
            const auto* fz_ptr = fz_arr.dataPtr();

            const auto& fxbox = m_flux[0].box(mfi.index());
            const auto& fybox = m_flux[1].box(mfi.index());
            const auto& fzbox = m_flux[2].box(mfi.index());

            tortuosity_poisson_fio(bx.loVect(), bx.hiVect(),
                                   fx_ptr, fxbox.loVect(), fxbox.hiVect(),
                                   fy_ptr, fybox.loVect(), fybox.hiVect(),
                                   fz_ptr, fzbox.loVect(), fzbox.hiVect(),
                                   &dir_int, &lfxin, &lfxout);
        }
    } // End OMP parallel region

    amrex::ParallelDescriptor::ReduceRealSum(fxin);
    amrex::ParallelDescriptor::ReduceRealSum(fxout);
}


void TortuosityDirect::advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new, const amrex::Real& dt)
{
    phi_old.FillBoundary(m_geom.periodicity());
    // Removed incorrect amrex::FillDomainBoundary call
    fillDomainBoundary(phi_old, comp_phi); // Apply explicit Dirichlet for comp_phi

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        for ( amrex::MFIter mfi(phi_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.tilebox();

            auto fx_arr = m_flux[0].array(mfi);
            auto fy_arr = m_flux[1].array(mfi);
            auto fz_arr = m_flux[2].array(mfi);
            auto* fx_ptr = fx_arr.dataPtr();
            auto* fy_ptr = fy_arr.dataPtr();
            auto* fz_ptr = fz_arr.dataPtr();

            const auto& sol_arr = phi_old.const_array(mfi);
            const auto* sol_ptr = sol_arr.dataPtr();

            const auto& fxbox = m_flux[0].box(mfi.index());
            const auto& fybox = m_flux[1].box(mfi.index());
            const auto& fzbox = m_flux[2].box(mfi.index());
            const auto& solbox = phi_old.box(mfi.index());

            const amrex::Real* dxinv_ptr = m_dxinv.data(); // Assumes m_dxinv is amrex::Array

            tortuosity_poisson_flux(bx.loVect(), bx.hiVect(),
                                    fx_ptr, fxbox.loVect(), fxbox.hiVect(),
                                    fy_ptr, fybox.loVect(), fybox.hiVect(),
                                    fz_ptr, fzbox.loVect(), fzbox.hiVect(),
                                    sol_ptr, solbox.loVect(), solbox.hiVect(),
                                    dxinv_ptr);
        }

        for ( amrex::MFIter mfi(phi_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.tilebox();

            const auto& p_arr = phi_old.const_array(mfi);
            auto        n_arr = phi_new.array(mfi);
            const auto& fx_arr = m_flux[0].const_array(mfi);
            const auto& fy_arr = m_flux[1].const_array(mfi);
            const auto& fz_arr = m_flux[2].const_array(mfi);

            const auto* p_ptr = p_arr.dataPtr();
            auto* n_ptr = n_arr.dataPtr();
            const auto* fx_ptr = fx_arr.dataPtr();
            const auto* fy_ptr = fy_arr.dataPtr();
            const auto* fz_ptr = fz_arr.dataPtr();

            const auto& pbox = phi_old.box(mfi.index());
            const auto& nbox = phi_new.box(mfi.index());
            const auto& fxbox = m_flux[0].box(mfi.index());
            const auto& fybox = m_flux[1].box(mfi.index());
            const auto& fzbox = m_flux[2].box(mfi.index());

            const amrex::Real* dxinv_ptr = m_dxinv.data(); // Assumes m_dxinv is amrex::Array

            // *** FIX: Add ncomp argument to the call ***
            const int ncomp_val = phi_new.nComp(); // Get number of components

            tortuosity_poisson_update(bx.loVect(), bx.hiVect(),
                                      p_ptr, pbox.loVect(), pbox.hiVect(),
                                      n_ptr, nbox.loVect(), nbox.hiVect(),
                                      fx_ptr, fxbox.loVect(), fxbox.hiVect(),
                                      fy_ptr, fybox.loVect(), fybox.hiVect(),
                                      fz_ptr, fzbox.loVect(), fzbox.hiVect(),
                                      &ncomp_val, // Pass address of ncomp
                                      dxinv_ptr,
                                      &dt);

            // Ensure cell type remains unchanged in phi_new (copy from phi_old)
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                n_arr(i,j,k, comp_ct) = p_arr(i,j,k, comp_ct);
            });
        }
    } // End OMP parallel region
}


amrex::Real TortuosityDirect::residual(const amrex::MultiFab& phi_old, const amrex::MultiFab& phi_new ) const
{
    amrex::Real delta_sum = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:delta_sum)
#endif
    for (amrex::MFIter mfi(phi_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        const auto& arr_old = phi_old.const_array(mfi, comp_phi);
        const auto& arr_new = phi_new.const_array(mfi, comp_phi);
        const auto& ct_arr  = phi_old.const_array(mfi, comp_ct);

        amrex::Real thread_delta = 0.0;
        amrex::Loop(bx, [&] (int i, int j, int k) {
            if (static_cast<int>(ct_arr(i,j,k)) == 1) {
                thread_delta += std::abs(arr_new(i,j,k) - arr_old(i,j,k));
            }
        });
        delta_sum += thread_delta;
    }

    amrex::ParallelAllReduce::Sum(delta_sum, amrex::ParallelContext::CommunicatorSub());
    return delta_sum;
}


void TortuosityDirect::initializeBoundaryConditions()
{
    m_bc = amrex::BCRec(); // Use default constructor

    // Set default Neumann conditions for component 0
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
        m_bc.setLo(i, static_cast<int>(amrex::BCType::reflect_even));
        m_bc.setHi(i, static_cast<int>(amrex::BCType::reflect_even));
    }

    // Set Dirichlet for component comp_phi (0) in the specified direction
    m_bc.setLo(static_cast<int>(m_dir), static_cast<int>(amrex::BCType::ext_dir));
    m_bc.setHi(static_cast<int>(m_dir), static_cast<int>(amrex::BCType::ext_dir));
}

void TortuosityDirect::fillCellTypes(amrex::MultiFab& phi)
{
    const amrex::Box& domain_box = m_geom.Domain();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto phi_arr = phi.array(mfi);
        const auto& phase_arr = m_mf_phase.const_array(mfi);

        int q_ncomp = phi.nComp();
        int p_ncomp = m_mf_phase.nComp();
        const auto& qbox = phi.box(mfi.LocalTileIndex());
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex());

        tortuosity_filct(phi_arr.dataPtr(),
                         qbox.loVect(), qbox.hiVect(), &q_ncomp,
                         phase_arr.dataPtr(),
                         pbox.loVect(), pbox.hiVect(), &p_ncomp,
                         domain_box.loVect(), domain_box.hiVect(),
                         &m_phase);
    }
}

void TortuosityDirect::fillInitialState(amrex::MultiFab& phi)
{
    const amrex::Box& domain_box = m_geom.Domain();
    const int dir_int = static_cast<int>(m_dir);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();

        auto phi_arr = phi.array(mfi);
        const auto& phase_arr = m_mf_phase.const_array(mfi);

        int q_ncomp = phi.nComp();
        int p_ncomp = m_mf_phase.nComp();
        const auto& qbox = phi.box(mfi.LocalTileIndex());
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex());

        tortuosity_filic(phi_arr.dataPtr(),
                         qbox.loVect(), qbox.hiVect(), &q_ncomp,
                         phase_arr.dataPtr(),
                         pbox.loVect(), pbox.hiVect(), &p_ncomp,
                         bx.loVect(), bx.hiVect(),
                         domain_box.loVect(), domain_box.hiVect(),
                         &m_vlo, &m_vhi,
                         &m_phase,
                         &dir_int);
    }
}


void TortuosityDirect::fillDomainBoundary (amrex::MultiFab& phi, int comp)
{
    if (comp != comp_phi) return;

    const amrex::Box& domain_box = m_geom.Domain();

    int bc_c_array[AMREX_SPACEDIM*2];
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        auto map_bc = [&](int amrex_bc_type_int) -> int {
            if (amrex_bc_type_int == static_cast<int>(amrex::BCType::ext_dir)) return 1;
            return 0;
        };
        bc_c_array[idim*2 + 0] = map_bc(m_bc.lo(idim));
        bc_c_array[idim*2 + 1] = map_bc(m_bc.hi(idim));
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& fab_box_ghosts = mfi.fabbox();

        bool touches_boundary = !domain_box.contains(fab_box_ghosts);
        if (touches_boundary)
        {
             auto phi_arr = phi.array(mfi);

             int q_ncomp = phi.nComp();
             const auto& qbox = phi.box(mfi.LocalTileIndex());

             tortuosity_filbc(phi_arr.dataPtr(),
                              qbox.loVect(), qbox.hiVect(), &q_ncomp,
                              domain_box.loVect(), domain_box.hiVect(),
                              &m_vlo, &m_vhi,
                              bc_c_array);
        }
    }
}

} // namespace OpenImpala
