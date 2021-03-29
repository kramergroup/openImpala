#include "TortuosityDirect.H"
#include "Tortuosity_poisson_3d_F.H"
#include "Tortuosity_filcc_F.H"
#include <stdlib.h>
#include <ctime>

#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCUtil.H>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>


#define NUM_GHOST_CELLS 1


TortuosityDirect::TortuosityDirect(const amrex::Geometry& geom, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const amrex::iMultiFab& mf, const int phase, const Direction dir) 
                      : m_geom(geom), m_ba(ba), m_dm(dm), m_mf_phase(mf), m_phase(phase), m_dir(dir), m_first_call(true)
{
    // Create a datastructure describing the boundary conditions
    initialiseBoundaryConditions();
    initialiseFluxMultiFabs();
}


void TortuosityDirect::initialiseFluxMultiFabs() 
{
    // Build the flux multi-fabs
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        // flux(dir) has one component, zero ghost cells, and is nodal in direction dir
        amrex::BoxArray edge_ba(m_ba);
        edge_ba.surroundingNodes(dir);
        m_flux[dir].define(edge_ba, m_dm, 1, 0);
    }
}

amrex::Real TortuosityDirect::value(const bool refresh) 
{
  // Solve potential field
  if ( refresh || m_first_call ) 
  {
    if (!solve())
    {
        m_first_call = true;
        return 0.0;
    }
  }

  // Calculate fluxes
  amrex::Real fx, fxin, fxout;
  global_fluxes(fxin,fxout);
  fx = (fxin+fxout) / 2.0;

  // Calculate tortuosity
  const amrex::Box bx = m_geom.Domain();
  const amrex::IntVect sz = bx.size();

  switch(m_dir)
  {
    case X :
        m_value =  1.0 / (fx / sz[1] / sz[2]);
        break;
    case Y :
        m_value =  1.0 / (fx / sz[0] / sz[2]);
        break;
    case Z :
        m_value =  1.0 / (fx / sz[0] / sz[1]);
        break;
  }
  return m_value;
}

bool TortuosityDirect::solve()
{

    // Flag invocation of the solver
    m_first_call = false;

    // Create MultiFabs for variables
    amrex::MultiFab mf_phi_old(m_ba,m_dm,2,NUM_GHOST_CELLS);
    amrex::MultiFab mf_phi_new(m_ba,m_dm,2,NUM_GHOST_CELLS);

    // Fill initial condition
    fillInitialState(mf_phi_old);

    // Initialise the cell type fab
    fillCellTypes(mf_phi_old);
    amrex::MultiFab::Copy(mf_phi_new, mf_phi_old, 0, 0, 2, NUM_GHOST_CELLS);

    // plot interval
    int plot_int = 1000;
    amrex::Real res(1e6);
    amrex::Real fxin(0.0);
    amrex::Real fxout(0.0);

    for (size_t n = 1; n <= m_n_steps; ++n)
    {
        amrex::MultiFab::Copy(mf_phi_old, mf_phi_new, 0, 0, 1, NUM_GHOST_CELLS);
        advance(mf_phi_old, mf_phi_new);
        
        // Write a plotfile of the current data 
        if (plot_int > 0 && n%plot_int == 0)
        {
            res = residual(mf_phi_old, mf_phi_new);
            global_fluxes(fxin,fxout);
             // Tell the I/O Processor to write out which step we're doing
            amrex::Print() << "Step " << n << ": ";
            amrex::Print() << std::fixed << std::setprecision(6) << std::setw(12) << std::setfill(' ');
            amrex::Print() << "Residual: " << res << " | ";
            amrex::Print() << "flux[in]: " << fxin << " flux[out]: " << fxout << " Delta: " << (fxout-fxin) << std::endl;

            if (res < m_eps) 
            {
                amrex::Print() << "Convergence reached" << std::endl;
                break;
            }
        }
    }
    // Get the users home directory to write plot file to right place
    const char* homeDir = getenv("HOME");
    // Get the current time to append to the filename
    const int MAXLEN = 80;
    char plot_time[MAXLEN];
    time_t t = time(0);
    strftime(plot_time, MAXLEN, "%Y/%m/%d", localtime(&t));
    // Write plot file
    amrex::WriteSingleLevelPlotfile(homeDir + std::string(amrex::Concatenate("/openimpalaresults/diffusionplot",std::string(plot_time))), mf_phi_new, {"concentration","phase"}, m_geom, 0.0, 0);

    // calculate fluxes again to make sure we have full consistency
    global_fluxes(fxin,fxout);

    return (res < m_eps);
}


void TortuosityDirect::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    fxin = 0.0;
    fxout = 0.0;

    const amrex::Box& bx = m_geom.Domain();

#ifdef _OPENMP
#pragma omp parallel
#endif
    // Compute fluxes on domain boundaries
    for ( amrex::MFIter mfi(m_flux[0]); mfi.isValid(); ++mfi )
    {

        amrex::Real lfxin(0.0);
        amrex::Real lfxout(0.0);
        tortuosity_poisson_fio(BL_TO_FORTRAN_BOX(bx),
                               BL_TO_FORTRAN_ANYD(m_flux[0][mfi]),
                               BL_TO_FORTRAN_ANYD(m_flux[1][mfi]),
                               BL_TO_FORTRAN_ANYD(m_flux[2][mfi]),
                               (int*)&m_dir, &lfxin, &lfxout);
#ifdef _OPENMP
#pragma omp critical 
#endif
        fxin += lfxin;
        fxout += lfxout;
    }

    // Reduce parallel processes
    amrex::ParallelAllReduce::Sum(fxin, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(fxout, amrex::ParallelContext::CommunicatorSub());
}

void TortuosityDirect::advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new) 
{
    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    phi_old.FillBoundary(m_geom.periodicity());

    // Fill in the boundary conditions
    fillDomainBoundary(phi_old);
    fillDomainBoundary(phi_new);

    
    const amrex::Real* dx = m_geom.CellSize();
    amrex::Real dt = 0.9*dx[0]*dx[0] / (2.0*AMREX_SPACEDIM) * 1000;

    amrex::RealArray dxinv;
    for (int i=0; i<AMREX_SPACEDIM; ++i) 
    {
        dxinv[i] = 1.0/ dx[i];
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    // Compute fluxes one grid at a time
    for ( amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const amrex::Box& bx = mfi.validbox();
   
        tortuosity_poisson_flux(BL_TO_FORTRAN_BOX(bx),
                                BL_TO_FORTRAN_ANYD(m_flux[0][mfi]),
                                BL_TO_FORTRAN_ANYD(m_flux[1][mfi]), 
                                BL_TO_FORTRAN_ANYD(m_flux[2][mfi]),
                                BL_TO_FORTRAN_FAB(phi_old[mfi]),
                                dxinv.data(), 0);
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    // Advance the solution one grid at a time
    for ( amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const amrex::Box& bx = mfi.validbox();
        
        tortuosity_poisson_update(BL_TO_FORTRAN_BOX(bx),
                                  BL_TO_FORTRAN_FAB(phi_old[mfi]),
                                  BL_TO_FORTRAN_FAB(phi_new[mfi]),
                                  BL_TO_FORTRAN_ANYD(m_flux[0][mfi]),
                                  BL_TO_FORTRAN_ANYD(m_flux[1][mfi]),
                                  BL_TO_FORTRAN_ANYD(m_flux[2][mfi]),
                                  dx, &dt);
    }
}

amrex::Real TortuosityDirect::residual(const amrex::MultiFab& phiold, const amrex::MultiFab& phinew ) const
{

  amrex::Real delta(0.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  // Iterate over all boxes and count cells with value=m_phase
  for (amrex::MFIter mfi(phiold); mfi.isValid(); ++mfi) // Loop over grids
  {
    const amrex::Box& bx = mfi.validbox();
    const amrex::FArrayBox& fab_old = phiold[mfi];
    const amrex::FArrayBox& fab_new = phinew[mfi];
    
    // Iterate over all cells in Box and threshold
    for (amrex::BoxIterator bit(bx); bit.ok(); ++bit) 
    {
      delta += fabs(fab_new(bit(),0) - fab_old(bit(),0));
    } 
  }

  // Reduce parallel processes
  amrex::ParallelAllReduce::Sum(delta, amrex::ParallelContext::CommunicatorSub());
  
  return delta;

}

void TortuosityDirect::initialiseBoundaryConditions() 
{
  for (int i=0;i<3;++i)
  {
      m_bc.setLo(i,amrex::BCType::reflect_even);
      m_bc.setHi(i,amrex::BCType::reflect_even);
  }

  m_bc.setLo(m_dir,amrex::BCType::ext_dir);
  m_bc.setHi(m_dir,amrex::BCType::ext_dir);
}

void TortuosityDirect::fillCellTypes(amrex::MultiFab& phi)
{
    
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab = phi[mfi];
        const amrex::IArrayBox& phase_fab = m_mf_phase[mfi];

        const amrex::Box& fab_box = mfi.validbox(); 
        
        tortuosity_filct(BL_TO_FORTRAN_FAB(fab),
                         BL_TO_FORTRAN_FAB(phase_fab),
                         BL_TO_FORTRAN_BOX(fab_box),
                         &m_phase);
    
    }
}

void TortuosityDirect::fillInitialState(amrex::MultiFab& phi)
{
    // Fill Dirichlet conditions manually
    const amrex::Box& domain_box = m_geom.Domain();

    // Fill the Dirichlet conditions
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab = phi[mfi];
        const amrex::IArrayBox& phase_fab = m_mf_phase[mfi];

        const amrex::Box& fab_box = mfi.validbox(); 
        
        tortuosity_filic(BL_TO_FORTRAN_FAB(fab),
                         BL_TO_FORTRAN_FAB(phase_fab),
                         BL_TO_FORTRAN_BOX(fab_box),
                         BL_TO_FORTRAN_BOX(domain_box),
                         &m_vlo,&m_vhi, 
                         &m_phase,
                         (const int*)&m_dir);
    
    }
}

void TortuosityDirect::fillDomainBoundary (amrex::MultiFab& phi)
{

    // This fills all boundary conditions apart from Dirichlet conditions
    amrex::FillDomainBoundary(phi,m_geom,{m_bc});

    // Fill Dirichlet conditions manually
    const amrex::Box& domain_box = m_geom.Domain();

    // Fill the Dirichlet conditions
    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab = phi[mfi];
        const amrex::IArrayBox& phase_fab = m_mf_phase[mfi];

        const amrex::Box& fab_box = fab.box(); // including ghost cells
        
        if (! domain_box.strictly_contains(fab_box))
        {
            tortuosity_filbc(BL_TO_FORTRAN_FAB(fab),
                             BL_TO_FORTRAN_FAB(phase_fab),
                             BL_TO_FORTRAN_BOX(domain_box),
                             &m_vlo,&m_vhi, 
                             m_bc.data());
        }
    }
}
