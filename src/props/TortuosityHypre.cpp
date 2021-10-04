#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"
#include "TortuosityHypreFill_F.H"
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

/**
 *
 * Constructor
 *
 */
TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm,
                                 amrex::iMultiFab& mf,
                                 const amrex::Real& vf,
                                 const int phase,
                                 const Direction dir,
                                 const SolverType st,
                                 std::string const& resultspath) : m_geom(geom), m_ba(ba), m_dm(dm),
                                                        m_mf_phase(mf), m_phase(phase), m_vf(vf),
                                                        m_dir(dir), m_solvertype(st), 
                                                        m_resultspath(resultspath),
                                                        m_mf_phi(ba,dm,2,0),
                                                        m_first_call(true)
{
    preconditionPhaseFab();
    setupGrids();
    setupStencil();
    setupMatrixEquation();
}


/**
 *
 * Setup Grids
 *
 */
void TortuosityHypre::setupGrids()
{

  // 1 - Initialise the grid owned by this MPI rank
  HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);

  // 2 - Configure the dimensions of each box owned by this MPI rank
  int ilower[2], iupper[2];
  for ( amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi )
  {
    const amrex::Box bx = mfi.validbox();
    auto lo = TortuosityHypre::loV(bx);
    auto hi = TortuosityHypre::hiV(bx);

    HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());

  }

  // 3 - Finish setup
   HYPRE_StructGridAssemble(m_grid);
}


/**
 *
 * Setup stencil
 *
 */
void TortuosityHypre::setupStencil()
{
  int size         = 7;
  int offsets[][3] = {{0,0,0},
                      {-1,0,0}, {1,0,0},
                      {0,-1,0}, {0,1,0},
                      {0,0,-1}, {0,0,1}};

  HYPRE_StructStencilCreate(AMREX_SPACEDIM, size, &m_stencil);

  /* Set stencil entries */
  for (int i = 0; i < size; i++)
  {
    HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]);
  }

}

void TortuosityHypre::preconditionPhaseFab()
{

  AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() > 0, "Phase fab should have ghost cells");

  const amrex::Box& domain_box = m_geom.Domain();

  for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi)
  {
      amrex::IArrayBox& fab = m_mf_phase[mfi];
      const amrex::Box& fab_box = mfi.validbox();

      tortuosity_remspot(BL_TO_FORTRAN_FAB(fab),
                         BL_TO_FORTRAN_BOX(fab_box),
                         BL_TO_FORTRAN_BOX(domain_box));
  }
}

/**
 *
 * Setup Matrix
 *
 */
void TortuosityHypre::setupMatrixEquation()
{
  // 1 - Initialise the data structures for matrix and RHS
  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
  HYPRE_StructMatrixInitialize(m_A);

  HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b);
  HYPRE_StructVectorInitialize(m_b);

   // Initialise solution vector
  HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
  HYPRE_StructVectorInitialize(m_x);

  // 2 - Configure interior of domain and boundary conditions
  const amrex::Box domain = m_geom.Domain();

  // Iterate over all boxes
  for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi)
  {
    const amrex::Box bx = mfi.validbox();

    // Determine if box is touching domain edge
    int lo[AMREX_SPACEDIM];
    int hi[AMREX_SPACEDIM];
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
      lo[i] = bx.loVect()[i];
      hi[i] = bx.hiVect()[i];
    }

    // Fill 7 point stencil through full domain
    int stencil_indices[7] = {0,1,2,3,4,5,6};
    size_t nvalues(bx.numPts());
    amrex::Real values[nvalues*7];
    amrex::Real rhs[nvalues];
    amrex::Real xinit[nvalues];

    tortuosity_fillmtx(values, rhs, xinit, &nvalues,
                       BL_TO_FORTRAN_ANYD(m_mf_phase[mfi]),
                       BL_TO_FORTRAN_BOX(bx),
                       BL_TO_FORTRAN_BOX(domain),
                       &m_vlo, &m_vhi,
                       &m_phase, &m_dir);

    HYPRE_StructMatrixSetBoxValues(m_A, lo, hi, 7,stencil_indices, values);
    HYPRE_StructVectorSetBoxValues(m_b, lo, hi, rhs);
    HYPRE_StructVectorSetBoxValues(m_x, lo, hi, xinit);
  }

  HYPRE_StructMatrixAssemble(m_A);
}

bool TortuosityHypre::solve()
{

  // What time is it now?
  std::time_t strt_time;
  std::tm* timeinfo;
  char datetime [80];
  
  std::time(&strt_time);
  timeinfo = std::localtime(&strt_time);
  
  std::strftime(datetime,80,"%Y%m%d%H%M",timeinfo);


  /* Create Solver */
  HYPRE_StructSolver solver;
  HYPRE_Int ierr;
  HYPRE_Int num_iterations;
  amrex::Real res;

  switch (m_solvertype)
  {
    case Jacobi:
      HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructJacobiSetTol(solver, m_eps);
      HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
      HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
      ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
      HYPRE_StructJacobiDestroy(solver);
      HYPRE_StructJacobiGetNumIterations(solver, &num_iterations);
      HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &res);
      break;

    case FlexGMRES:
      HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructFlexGMRESSetTol(solver, m_eps);
      HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
      HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x);
      ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
      HYPRE_StructFlexGMRESDestroy(solver);
      HYPRE_StructFlexGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &res);
      break;

    default:
      HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
      HYPRE_StructGMRESSetTol(solver, m_eps);
      HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
      HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
      ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
      HYPRE_StructGMRESDestroy(solver);
      HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &res);
      break;
  }

  amrex::Print() << std::endl << num_iterations
                  << " Iterations, Relative Residual "
                  << res << std::endl;

  if (ierr)
  {
    if (HYPRE_CheckError(ierr,HYPRE_ERROR_CONV))
    {
      amrex::Print() << "ERROR: Solver did not converge" << std::endl;
      // Write plot file to user's home dir with datetime appended in YYmmDDHHMM format
      amrex::WriteSingleLevelPlotfile(m_resultspath + std::string("/failedplot") += datetime, m_mf_phi, {"concentration","phase"}, m_geom, 0.0, 0);
    } else if (HYPRE_CheckError(ierr,HYPRE_ERROR_MEMORY)) {
      amrex::Print() << "ERROR: Solver was unable to allocate memory" << std::endl;
    } else if (HYPRE_CheckError(ierr,HYPRE_ERROR_ARG)) {
      amrex::Print() << "ERROR: Solver returned argument error" << std::endl;
    } else {
      amrex::Print() << "ERROR: Solver returned error code:" << ierr << std::endl;
    }
    return false;
  }

  getSolution(m_mf_phi,0);
  getCellTypes(m_mf_phi,1);

  // Write plot file to user's home dir with datetime appended in YYmmDDHHMM format
  amrex::WriteSingleLevelPlotfile(m_resultspath + std::string("/diffusionplot") += datetime, m_mf_phi, {"concentration","phase"}, m_geom, 0.0, 0);

  return true;
}

amrex::Real TortuosityHypre::value(const bool refresh)
{
    if (refresh || m_first_call)
    {
      m_first_call = false;
      solve();
    }

    amrex::Real fluxlo = 0.0;
    amrex::Real fluxhi = 0.0;
    amrex::Real phisumlo = 0.0;
    amrex::Real phisumhi = 0.0;
    int num_phase_cells_0 = 0;
    int num_phase_cells_1 = 0;

    // Iterate over all boxes and count cells with value=m_phase
    if ( m_dir==0)
    {
    for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi) // Loop over grids
    {
      const amrex::Box& box = mfi.validbox();
      const amrex::IArrayBox& phase_fab = m_mf_phase[mfi];
      const amrex::FArrayBox& phi_fab = m_mf_phi[mfi];

      // Obtain Array4 from FArrayBox.  We can also do
      //     Array4<Real> const& a = mf.array(mfi);
      amrex::Array4<int const> const& phase_fab_4 = phase_fab.array();
      amrex::Array4<amrex::Real const> const& phi_fab_4 = phi_fab.array();

      size_t idx;
      // Iterate over all cells in Box and threshold
      const auto lo = lbound(box);
      const auto hi = ubound(box);


      // Sum all concentration values for each slice in x direction
      const auto domain_min_x = m_geom.Domain().loVect()[0];      
      for (int x = (lo.x+1); x <= (hi.x-1); ++x) {
            for (int y = lo.y; y <= hi.y; ++y) {
              for (int z = lo.z; z <= hi.z; ++z) {
                if ( phase_fab_4(x,y,z) != phase_fab_4(x-1,y,z) && phase_fab_4(x,y,z) == m_phase ) {
                  phisumhi += phi_fab_4(x,y,z) - phi_fab_4(x+1,y,z);
                  num_phase_cells_0 += 1;
              }
            }
        }
      }

      const auto domain_max_x = m_geom.Domain().hiVect()[0];;
      for (int x = (lo.x+1); x <= (hi.x-1); ++x) {
            for (int y = lo.y; y <= hi.y; ++y) {
              for (int z = lo.z; z <= hi.z; ++z) {
                if ( phase_fab_4(x,y,z) != phase_fab_4(x+1,y,z) && phase_fab_4(x,y,z) == m_phase ) {
                  phisumlo += phi_fab_4(x-1,y,z) - phi_fab_4(x,y,z);
                  num_phase_cells_1 += 1;
              }
            }
        }
      }

    }
  }


  else if ( m_dir==1)
  {
  for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi) // Loop over grids
  {
    const amrex::Box& box = mfi.validbox();
    const amrex::IArrayBox& phase_fab = m_mf_phase[mfi];
    const amrex::FArrayBox& phi_fab = m_mf_phi[mfi];

    // Obtain Array4 from FArrayBox.  We can also do
    //     Array4<Real> const& a = mf.array(mfi);
    amrex::Array4<int const> const& phase_fab_4 = phase_fab.array();
    amrex::Array4<amrex::Real const> const& phi_fab_4 = phi_fab.array();

    size_t idx;
    // Iterate over all cells in Box and threshold
    const auto lo = lbound(box);
    const auto hi = ubound(box);


    // Sum all concentration values for each slice in y direction
    const auto domain_min_y = m_geom.Domain().loVect()[1];
    for (int y = (lo.y+1); y <= hi.y; ++y) {
          for (int x = lo.x; x <= hi.x; ++x) {
            for (int z = lo.z; z <= hi.z; ++z) {
              if ( phase_fab_4(x,y,z) != phase_fab_4(x,y-1,z) && phase_fab_4(x,y,z) == m_phase ) {
                phisumhi += phi_fab_4(x,y,z) - phi_fab_4(x,y+1,z);
                num_phase_cells_0 += 1;
            }
          }
      }
    }

    const auto domain_max_y = m_geom.Domain().hiVect()[1];;
    for (int y = lo.y; y <= (hi.y-1); ++y) {
          for (int x = lo.x; x <= hi.x; ++x) {
            for (int z = lo.z; z <= hi.z; ++z) {
              if ( phase_fab_4(x,y,z) != phase_fab_4(x,y+1,z) && phase_fab_4(x,y,z) == m_phase ) {
                phisumlo += - phi_fab_4(x,y,z) + phi_fab_4(x,y-1,z);
                num_phase_cells_1 += 1;
            }
          }
      }
    }

  }
}

else if ( m_dir==2)
{
for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi) // Loop over grids
{
  const amrex::Box& box = mfi.validbox();
  const amrex::IArrayBox& phase_fab = m_mf_phase[mfi];
  const amrex::FArrayBox& phi_fab = m_mf_phi[mfi];

  // Obtain Array4 from FArrayBox.  We can also do
  //     Array4<Real> const& a = mf.array(mfi);
  amrex::Array4<int const> const& phase_fab_4 = phase_fab.array();
  amrex::Array4<amrex::Real const> const& phi_fab_4 = phi_fab.array();

  size_t idx;
  // Iterate over all cells in Box and threshold
  const auto lo = lbound(box);
  const auto hi = ubound(box);


  // Sum all concentration values for each slice in x direction
  const auto domain_min_z = m_geom.Domain().loVect()[2];
    for (int z = (lo.z+1); z <= hi.z; ++z) {
        for (int x = lo.x; x <= hi.x; ++x) {
          for (int y = lo.y; y <= hi.y; ++y) {
              if ( phase_fab_4(x,y,z) != phase_fab_4(x,y,z-1) && phase_fab_4(x,y,z) == m_phase ) {
                phisumhi += phi_fab_4(x,y,z) - phi_fab_4(x,y,z+1);
                num_phase_cells_0 += 1;
          }
        }
    }
  }

  const auto domain_max_z = m_geom.Domain().hiVect()[2];;
    for (int z = lo.z; z <= (hi.z-1); ++z) {
        for (int x = lo.x; x <= hi.x; ++x) {
          for (int y = lo.y; y <= hi.y; ++y) {
              if ( phase_fab_4(x,y,z) != phase_fab_4(x,y,z+1) && phase_fab_4(x,y,z) == m_phase ) {
                phisumlo += phi_fab_4(x,y,z) - phi_fab_4(x,y,z-1);
                num_phase_cells_1 += 1;
          }
        }
    }
  }

}
}

    // Reduce parallel processes
    if (!refresh) {
      amrex::ParallelAllReduce::Sum(phisumlo, amrex::ParallelContext::CommunicatorSub());
      }

    if (!refresh) {
      amrex::ParallelAllReduce::Sum(phisumhi, amrex::ParallelContext::CommunicatorSub());
      }

    // Total problem length each direction
    auto length_x = m_geom.ProbLength(0);
    auto length_y = m_geom.ProbLength(1);
    auto length_z = m_geom.ProbLength(2);

    // Cell size in each direction
    amrex::Real dx = m_geom.CellSize(0);
    amrex::Real dy = m_geom.CellSize(1);
    amrex::Real dz = m_geom.CellSize(2);

    // Number of cells in each direction
    auto num_cell_x = length_x/dx;
    auto num_cell_y = length_y/dy;
    auto num_cell_z = length_z/dz;

    // Compute flux between adjacent slices
    fluxlo = phisumlo * (num_cell_x*num_cell_x) / dx * num_phase_cells_1 * (dy*dz);
    fluxhi = phisumhi * (num_cell_x*num_cell_x) / dx * num_phase_cells_0 * (dy*dz);

    // Compute maximum flux as max_flux = (phi(left) - phi(right))*(b*c)/a
    amrex::Real flux_max=0.0;

    if ( m_dir==0) {
      flux_max = (m_vhi-m_vlo) / length_x * (length_y*length_z);
    }

    else if ( m_dir==1) {
      flux_max = ((m_vhi-m_vlo) / length_x * (length_y*length_z)) * ((num_cell_x*num_cell_x) / (num_cell_y*num_cell_y));
    }

    else if ( m_dir==2) {
      flux_max = ((m_vhi-m_vlo) / length_x * (length_y*length_z)) * ((num_cell_x*num_cell_x) / (num_cell_z*num_cell_z));
    }
  
    // Print all of fluxvect values
    amrex::Print() << std::endl << " Number phase cells 0: "
                    << num_phase_cells_0 << std::endl << " Number phase cells 1: "
                    << num_phase_cells_1 << std::endl ;

    amrex::Print() << std::endl << " Phi Sum High: "
                    << phisumhi << std::endl << " Phi Sum Low: "
                    << phisumlo << std::endl ;  

    amrex::Print() << std::endl << " Flux Sum High: "
                    << fluxhi << std::endl << " Flux Sum Low: "
                    << fluxlo << std::endl << " Flux Max:" 
                    << flux_max << std::endl ;  
  
    // Compute Volume Fractions

    amrex::Real rel_diffusivity = (fluxlo+fluxhi)/2.0/flux_max;

    amrex::Real tau = m_vf / rel_diffusivity;

    // Print all of fluxvect values
    amrex::Print() << std::endl << " Relative Effective Diffusivity (D_eff / D): "
                    << rel_diffusivity << std::endl ;

    amrex::Print() << " Check difference between top and bottom fluxes is nil: " << abs(fluxlo - fluxhi) << std::endl;

    return tau;


}

void TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp)
{
  amrex::FArrayBox rfab;
  for (amrex::MFIter mfi(soln); mfi.isValid(); ++mfi)
  {
    const amrex::Box &reg = mfi.validbox();

    amrex::FArrayBox *xfab;
    if (soln.nGrow() == 0) { // need a temporary if soln is the wrong size
        xfab = &soln[mfi];
    }
    else {
        xfab = &rfab;
        xfab->resize(reg);
    }

    auto reglo = TortuosityHypre::loV(reg);
    auto reghi = TortuosityHypre::hiV(reg);
    HYPRE_StructVectorGetBoxValues(m_x, reglo.data(), reghi.data(), xfab->dataPtr());

    if (soln.nGrow() != 0) {
        soln[mfi].copy(*xfab, 0, ncomp, 1);
    }
  }
}


void TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp)
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

//**************** HELPER FUNCTIONS *********************************

/**
 *
 * Generate lower bounds array in HYPRE format for box
 *
 */
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::loV (const amrex::Box& b) {
  const auto& v = b.loVect();
  return {AMREX_D_DECL(static_cast<HYPRE_Int>(v[0]),
                       static_cast<HYPRE_Int>(v[1]),
                       static_cast<HYPRE_Int>(v[2]))};
}


/**
 *
 * Generate upper bounds array in HYPRE format for box
 *
 */
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::hiV (const amrex::Box& b) {
  const auto& v = b.hiVect();
  return {AMREX_D_DECL(static_cast<HYPRE_Int>(v[0]),
                       static_cast<HYPRE_Int>(v[1]),
                       static_cast<HYPRE_Int>(v[2]))};
}
