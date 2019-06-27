#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"
#include "TortuosityHypreFill_F.H"
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
                                 const int phase, 
                                 const Direction dir,
                                 const SolverType st) : m_geom(geom), m_ba(ba), m_dm(dm), 
                                                        m_mf_phase(mf), m_phase(phase), 
                                                        m_dir(dir), m_solvertype(st), 
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

  amrex::WriteSingleLevelPlotfile("phi", m_mf_phi, {"phi","celltypes"}, m_geom, 0.0, 0);

  return true;
}

amrex::Real TortuosityHypre::value(const bool refresh)
{
    if (refresh || m_first_call)
    {
      m_first_call = false;
      solve();
    }
    return 0.0;
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