#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H" // Assuming this provides tortuosity_remspot
#include "TortuosityHypreFill_F.H" // Assuming this provides tortuosity_fillmtx and tortuosity_filct
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H> // Needed for ParallelAllReduce

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
                                                                   m_mf_phi(ba,dm,2,0), // 2 components: [0]=concentration, [1]=phase for plotfile
                                                                   m_first_call(true)
{
    // NOTE: Assuming preconditionPhaseFab, setupGrids, setupStencil, setupMatrixEquation
    // are implemented correctly as shown previously and compatible with this value() method.
    preconditionPhaseFab();
    setupGrids();
    setupStencil();
    setupMatrixEquation();
}


/**
 *
 * Setup Grids (Assuming implementation as provided previously)
 *
 */
void TortuosityHypre::setupGrids()
{
 // 1 - Initialise the grid owned by this MPI rank
 HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);

 // 2 - Configure the dimensions of each box owned by this MPI rank
 // int ilower[AMREX_SPACEDIM], iupper[AMREX_SPACEDIM]; // Use AMREX_SPACEDIM
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
 * Setup stencil (Assuming implementation as provided previously)
 *
 */
void TortuosityHypre::setupStencil()
{
 int size       = 7; // 2*AMREX_SPACEDIM + 1
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

/**
 *
 * Precondition PhaseFab (Assuming implementation as provided previously)
 * Calls Fortran routine tortuosity_remspot
 *
 */
void TortuosityHypre::preconditionPhaseFab()
{
 AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() > 0, "Phase fab should have ghost cells");

 const amrex::Box& domain_box = m_geom.Domain();

 for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi)
 {
      amrex::IArrayBox& fab = m_mf_phase[mfi];
      const amrex::Box& fab_box = mfi.validbox();

      // Assuming tortuosity_remspot is declared correctly in Tortuosity_filcc_F.H
      tortuosity_remspot(BL_TO_FORTRAN_FAB(fab),
                         BL_TO_FORTRAN_BOX(fab_box),
                         BL_TO_FORTRAN_BOX(domain_box));
 }
 // Optional: Refill boundary ghost cells after modification if necessary
 // m_mf_phase.FillBoundary(m_geom.periodicity());
}

/**
 *
 * Setup Matrix (Assuming implementation as provided previously)
 * Calls Fortran routine tortuosity_fillmtx
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
   // Use dynamic allocation if nvalues can be very large
   std::vector<amrex::Real> values(nvalues*7);
   std::vector<amrex::Real> rhs(nvalues);
   std::vector<amrex::Real> xinit(nvalues);

   // Assuming tortuosity_fillmtx is declared correctly in TortuosityHypreFill_F.H
   // Check if m_vlo and m_vhi are member variables (declared in .H file)
   tortuosity_fillmtx(values.data(), rhs.data(), xinit.data(), &nvalues,
                      BL_TO_FORTRAN_ANYD(m_mf_phase[mfi]),
                      BL_TO_FORTRAN_BOX(bx),
                      BL_TO_FORTRAN_BOX(domain),
                      &m_vlo, &m_vhi, // Assuming m_vlo, m_vhi defined in header (-1.0, +1.0)
                      &m_phase, &m_dir);

   HYPRE_StructMatrixSetBoxValues(m_A, lo, hi, 7, stencil_indices, values.data());
   HYPRE_StructVectorSetBoxValues(m_b, lo, hi, rhs.data());
   HYPRE_StructVectorSetBoxValues(m_x, lo, hi, xinit.data());
 }

 HYPRE_StructMatrixAssemble(m_A);
}

/**
 *
 * Solve linear system (Assuming implementation as provided previously)
 *
 */
bool TortuosityHypre::solve()
{
 // What time is it now? For plotfile naming
 std::time_t current_time;
 std::tm* timeinfo;
 char datetime [80];

 std::time(&current_time); // Use different variable name
 timeinfo = std::localtime(&current_time);

 std::strftime(datetime,80,"%Y%m%d%H%M",timeinfo);


 /* Create Solver */
 HYPRE_StructSolver solver;
 HYPRE_Int ierr = 0; // Initialize ierr
 HYPRE_Int num_iterations = 0; // Initialize
 amrex::Real res = -1.0; // Initialize

 switch (m_solvertype)
 {
   case Jacobi:
     HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
     HYPRE_StructJacobiSetTol(solver, m_eps); // Assuming m_eps is member var
     HYPRE_StructJacobiSetMaxIter(solver, m_maxiter); // Assuming m_maxiter is member var
     HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
     ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
     // Get diagnostics before destroying solver
     HYPRE_StructJacobiGetNumIterations(solver, &num_iterations);
     HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &res);
     HYPRE_StructJacobiDestroy(solver); // Destroy after getting info
     break;

   case FlexGMRES:
     HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver);
     HYPRE_StructFlexGMRESSetTol(solver, m_eps);
     HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
     // Add Preconditioner? Example: SMG
     // HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
     // HYPRE_StructSMGSetMaxIter(precond, 1); // Use as preconditioner
     // HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
     HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x);
     ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
     // Get diagnostics before destroying solver
     HYPRE_StructFlexGMRESGetNumIterations(solver, &num_iterations);
     HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &res);
     HYPRE_StructFlexGMRESDestroy(solver);
     // HYPRE_StructSMGDestroy(precond); // Destroy preconditioner if used
     break;

   // Default case GMRES (original code had this as default)
   // case GMRES: // Explicitly add case if needed
   default:
     HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
     HYPRE_StructGMRESSetTol(solver, m_eps);
     HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
     // Add Preconditioner? Example: SMG
     // HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
     // HYPRE_StructSMGSetMaxIter(precond, 1); // Use as preconditioner
     // HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
     HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
     ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
     // Get diagnostics before destroying solver
     HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
     HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &res);
     HYPRE_StructGMRESDestroy(solver);
     // HYPRE_StructSMGDestroy(precond); // Destroy preconditioner if used
     break;
 }

 amrex::Print() << std::endl << num_iterations
                << " Iterations, Relative Residual "
                << res << std::endl;

 if (ierr)
 {
   // Use HYPRE_GetErrorDescription if available for more details
   if (HYPRE_CheckError(ierr, HYPRE_ERROR_CONV))
   {
     amrex::Print() << "ERROR: Solver did not converge (HYPRE error code: " << ierr << ")" << std::endl;
   } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_MEMORY)) {
     amrex::Print() << "ERROR: Solver memory allocation error (HYPRE error code: " << ierr << ")" << std::endl;
   } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_ARG)) {
     amrex::Print() << "ERROR: Solver argument error (HYPRE error code: " << ierr << ")" << std::endl;
   } else {
     amrex::Print() << "ERROR: Solver returned unknown error code: " << ierr << std::endl;
   }
   return false; // Indicate solver failure
 }

 // Check residual against tolerance manually if needed
 if (res > m_eps) {
    amrex::Print() << "WARNING: Solver finished but residual (" << res << ") exceeds tolerance (" << m_eps << ")" << std::endl;
    // Decide whether to treat this as failure based on application needs
    // return false;
 }


 getSolution(m_mf_phi, 0); // Fill component 0 with concentration
 getCellTypes(m_mf_phi, 1); // Fill component 1 with phase info

 // Construct plotfile name
 std::string plotfilename = m_resultspath + "/diffusionplot" + std::string(datetime);
 // Write plot file to user's specified path with datetime appended
 amrex::WriteSingleLevelPlotfile(plotfilename, m_mf_phi, {"concentration","phase"}, m_geom, 0.0, 0);
 amrex::Print() << "Plotfile written to: " << plotfilename << std::endl;

 return true; // Indicate solver success
}


/**
 *
 * Calculate Tortuosity using Boundary Flux Method (Method 1)
 *
 */
amrex::Real TortuosityHypre::value(const bool refresh)
{
    if (refresh || m_first_call)
    {
      m_first_call = false;
      bool solve_success = solve();
      if (!solve_success) {
          // Handle solver failure, e.g., return NaN or throw exception
          amrex::Abort("Solver failed to converge in TortuosityHypre::value");
          return std::numeric_limits<amrex::Real>::quiet_NaN(); // Or appropriate error value
      }
    }

    amrex::Real fluxlo = 0.0; // Flux across low boundary face in m_dir
    amrex::Real fluxhi = 0.0; // Flux across high boundary face in m_dir
    amrex::Real phisumlo = 0.0; // Sum of Delta(phi) on low boundary
    amrex::Real phisumhi = 0.0; // Sum of Delta(phi) on high boundary
    // These counts seem unused in this version, can be removed
    // int num_phase_cells_0 = 0;
    // int num_phase_cells_1 = 0;
    // int num_phase_cells_2 = 0;
    // int num_phase_cells_3 = 0;

    const int direction_index = static_cast<int>(m_dir); // Cast Direction enum/Real to int index (0, 1, or 2)
    const amrex::Box& domain_box = m_geom.Domain();
    const amrex::Real dx = m_geom.CellSize(direction_index); // Cell size in the flux direction

    // Iterate over all boxes and calculate flux at domain boundaries
    for (amrex::MFIter mfi(m_mf_phi); mfi.isValid(); ++mfi) // Loop over grids, use m_mf_phi which holds solution
    {
      const amrex::Box& box = mfi.validbox();
      const amrex::IArrayBox& phase_fab = m_mf_phase[mfi]; // Phase info needed to check if cells are in correct phase
      const amrex::FArrayBox& phi_fab = m_mf_phi[mfi]; // Concentration field

      amrex::Array4<int const> const& phase = phase_fab.array();
      amrex::Array4<amrex::Real const> const& phi = phi_fab.array(); // Component 0 should be concentration

      const auto lo = amrex::lbound(box);
      const auto hi = amrex::ubound(box);

      // Calculate flux contribution at the low boundary face in direction 'direction_index'
      if (box.smallEnd(direction_index) == domain_box.smallEnd(direction_index)) {
          amrex::Box face_box = amrex::bdryLo(box, direction_index);
          // Loop over face connecting cell 'iv' and 'iv+e_dir'
          amrex::IntVect iv_lo = face_box.smallEnd();
          amrex::IntVect iv_hi = face_box.bigEnd();
          // Adjust loop bounds to match Fortran/lower-level access if needed, AMReX loops handle this
          for (amrex::IntVect iv = iv_lo; iv <= iv_hi; amrex::incCycle(iv, direction_index+1)) // Cycle through dimensions != direction_index
          {
              amrex::IntVect iv_neighbor = iv;
              iv_neighbor[direction_index] += 1; // Cell just inside the boundary

              // Check if both cells defining the face are in the phase of interest
              if (phase(iv) == m_phase && phase(iv_neighbor) == m_phase) {
                  // Finite difference across the face: phi(inside) - phi(at boundary)
                  phisumlo += phi(iv_neighbor) - phi(iv);
              }
          }
      }

      // Calculate flux contribution at the high boundary face in direction 'direction_index'
      if (box.bigEnd(direction_index) == domain_box.bigEnd(direction_index)) {
          // Face connects cell 'iv' (at boundary) and 'iv-e_dir' (just inside)
          amrex::Box face_box = amrex::bdryHi(box, direction_index); // This box includes the boundary cells
          amrex::IntVect iv_lo = face_box.smallEnd();
          amrex::IntVect iv_hi = face_box.bigEnd();

          for (amrex::IntVect iv = iv_lo; iv <= iv_hi; amrex::incCycle(iv, direction_index+1))
          {
              amrex::IntVect iv_neighbor = iv;
              iv_neighbor[direction_index] -= 1; // Cell just inside the boundary

              if (phase(iv) == m_phase && phase(iv_neighbor) == m_phase) {
                  // Finite difference across the face: phi(at boundary) - phi(inside)
                  phisumhi += phi(iv) - phi(iv_neighbor);
              }
          }
      }
    } // End MFIter loop

    // Reduce parallel processes for the boundary sums
    // Use the appropriate communicator, assume default MPI_COMM_WORLD context if needed
    // amrex::ParallelAllReduce::Sum({phisumlo, phisumhi}, amrex::ParallelContext::CommunicatorSub()); // Reduce both sums at once
    amrex::ParallelDescriptor::ReduceRealSum(phisumlo);
    amrex::ParallelDescriptor::ReduceRealSum(phisumhi);


    // Calculate face area perpendicular to the flux direction
    amrex::Real face_area = 1.0;
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        if (i != direction_index) {
            face_area *= m_geom.ProbLength(i); // Multiply by physical lengths
        }
    }
    // If using unit cube, face_area can be calculated from cell sizes and counts too

    // Calculate flux = -D * grad(phi) * Area => Flux ~ - (DeltaPhi / dx) * Area
    // Our phisum represents Sum(DeltaPhi), so Flux ~ - (phisum / dx) * Area_cell * Num_cells
    // Or more simply: Average_Grad = phisum / Num_cells / dx. Flux = D * Average_Grad * Total_Area
    // Let's use the implementation's logic: Flux = Sum(DeltaPhi) / dx * Area_cell
    // Assuming Area_cell is implicitly handled or dx includes length scaling. Check units.
    // Original scaling was: phisumlo / dx * (dy*dz) where dx, dy, dz were CELL SIZES.

    amrex::Real area_factor = 1.0;
     for (int i=0; i<AMREX_SPACEDIM; ++i) {
        if (i != direction_index) {
            area_factor *= m_geom.CellSize(i); // Product of cell sizes perp to direction
        }
    }

    fluxlo = phisumlo / dx * area_factor; // Flux = Sum(DeltaPhi) / CellSize_dir * Area_perp_cell
    fluxhi = phisumhi / dx * area_factor;

    // Compute maximum possible flux (Bulk flux: D0 * Area * DeltaC / Length)
    // DeltaC = m_vhi - m_vlo
    // Length = m_geom.ProbLength(direction_index)
    // Area = face_area calculated above
    // Assume D0 = 1 (relative diffusivity)
    amrex::Real delta_C = m_vhi - m_vlo; // Total concentration difference applied
    amrex::Real length = m_geom.ProbLength(direction_index);
    amrex::Real flux_max = 1.0 * face_area * delta_C / length; // Assuming D0=1


    // Compute Relative Effective Diffusivity (D_eff / D0)
    // Use average flux, check for zero flux_max
    amrex::Real avg_flux = 0.5 * (fluxlo + fluxhi);
    amrex::Real rel_diffusivity = (flux_max != 0.0) ? (avg_flux / flux_max) : 0.0;

    // Sanity check: Flux conservation (should be small for steady state)
    amrex::Real flux_diff = std::abs(fluxlo - fluxhi);
    if (flux_diff > 1e-6 * std::abs(avg_flux)) { // Relative tolerance check
         amrex::Print() << "WARNING: Flux difference check failed! |Flux_in - Flux_out| = " << flux_diff
                        << ", Avg Flux = " << avg_flux << std::endl;
    }

    // Calculate Tortuosity = VolumeFraction / RelativeDiffusivity
    // Handle potential division by zero or negative diffusivity
    amrex::Real tau = 0.0;
    if (rel_diffusivity > 1e-12) { // Check for small positive diffusivity
        tau = m_vf / rel_diffusivity;
    } else {
        amrex::Print() << "WARNING: Relative diffusivity is non-positive (" << rel_diffusivity
                       << "). Tortuosity set to large value or NaN." << std::endl;
        // Set to a large value or NaN depending on desired behavior for non-percolating paths
        tau = std::numeric_limits<amrex::Real>::infinity();
    }

    // Print summary values
    amrex::Print() << " Direction: " << direction_index << std::endl;
    amrex::Print() << "  Volume Fraction (Input): " << m_vf << std::endl;
    amrex::Print() << "  Calculated Avg Flux: " << avg_flux << std::endl;
    amrex::Print() << "  Max Theoretical Flux: " << flux_max << std::endl;
    amrex::Print() << "  Relative Diffusivity (Deff/D0): " << rel_diffusivity << std::endl;
    amrex::Print() << "  Calculated Tortuosity (VF / RelDiff): " << tau << std::endl;
    // amrex::Print() << " Check difference between top and bottom fluxes is nil: " << flux_diff << std::endl;


    return tau;
}


/**
 *
 * Retrieve solution field from HYPRE vector (Assuming implementation as provided previously)
 *
 */
void TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp)
{
 amrex::FArrayBox rfab; // Temporary fab if soln has ghost cells
 for (amrex::MFIter mfi(soln); mfi.isValid(); ++mfi)
 {
   const amrex::Box &reg = mfi.validbox();

   amrex::FArrayBox *xfab_ptr;
   // Use soln[mfi] directly if it has no ghost cells and matches the box 'reg'
   // Otherwise, use a temporary buffer 'rfab'
   if (soln.nGrow() == 0 && soln[mfi].box() == reg) {
       xfab_ptr = &soln[mfi];
   }
   else {
       xfab_ptr = &rfab;
       xfab_ptr->resize(reg, soln.nComp()); // Resize with correct number of components
   }

   auto reglo = TortuosityHypre::loV(reg);
   auto reghi = TortuosityHypre::hiV(reg);
   // Get values for all components if needed, or just the first one (ncomp=0)
   HYPRE_StructVectorGetBoxValues(m_x, reglo.data(), reghi.data(), xfab_ptr->dataPtr(ncomp)); // Get data for component 'ncomp'

   // If we used the temporary buffer, copy the relevant component back to soln
   if (xfab_ptr == &rfab) {
       soln[mfi].copy(*xfab_ptr, ncomp, ncomp, 1); // Copy only the component we filled
   }
 }
}


/**
 *
 * Fill MultiFab component with phase data (Assuming implementation as provided previously)
 * Calls Fortran routine tortuosity_filct
 *
 */
void TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp)
{
    // Ensure ncomp is within bounds for phi
    AMREX_ASSERT(ncomp >= 0 && ncomp < phi.nComp());

    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab = phi[mfi]; // Target FArrayBox in MultiFab 'phi'
        const amrex::IArrayBox& phase_fab = m_mf_phase[mfi]; // Source IArrayBox from m_mf_phase

        const amrex::Box& fab_box = mfi.validbox();

        // Assuming tortuosity_filct takes FArrayBox for output and IArrayBox for input
        // and modifies the specified component 'ncomp' of the FArrayBox
        // Also assuming it requires BL_TO_FORTRAN_FAB which expects contiguous data.
        // Need to ensure tortuosity_filct handles the correct component.
        // If it only works on component 0, we might need a temporary fab.
        // Assuming it can work on component 'ncomp' directly:
        tortuosity_filct(BL_TO_FORTRAN_ANYD(fab), // Pass FArrayBox
                         BL_TO_FORTRAN_ANYD(phase_fab), // Pass IArrayBox
                         BL_TO_FORTRAN_BOX(fab_box),
                         &m_phase,
                         &ncomp); // Pass component index if Fortran needs it
        // If tortuosity_filct modifies component 0 regardless, adjust call or copy data after.
    }
}

//**************** HELPER FUNCTIONS *********************************

/**
 *
 * Generate lower bounds array in HYPRE format for box (Assuming implementation as provided previously)
 *
 */
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::loV (const amrex::Box& b) {
  const auto& v = b.loVect();
  // Ensure HYPRE_Int can hold the values
  return {AMREX_D_DECL(static_cast<HYPRE_Int>(v[0]),
                       static_cast<HYPRE_Int>(v[1]),
                       static_cast<HYPRE_Int>(v[2]))};
}


/**
 *
 * Generate upper bounds array in HYPRE format for box (Assuming implementation as provided previously)
 *
 */
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::hiV (const amrex::Box& b) {
  const auto& v = b.hiVect();
  // Ensure HYPRE_Int can hold the values
  return {AMREX_D_DECL(static_cast<HYPRE_Int>(v[0]),
                       static_cast<HYPRE_Int>(v[1]),
                       static_cast<HYPRE_Int>(v[2]))};
}
