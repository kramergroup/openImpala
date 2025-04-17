#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"     // For tortuosity_remspot
#include "TortuosityHypreFill_F.H"  // For tortuosity_fillmtx

#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <cmath>
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For potential error throwing (optional)

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>     // For amrex::average_down (potentially needed elsewhere, good include)
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H> // <<< CORRECTED HEADER for ParallelAllReduce >>>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>          // For amrex::UtilCreateDirectory
#include <AMReX_Assert.H>           // For AMREX_ASSERT, AMREX_ALWAYS_ASSERT
#include <AMReX_GpuContainers.H>    // For GpuArray used in m_dxinv

// HYPRE includes (already in TortuosityHypre.H but good practice here too)
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
// Required for HYPRE_StructVectorSetConstantValues and GetBoxValues if used elsewhere
// #include <HYPRE_struct_mv.h> // This might be needed indirectly or for other functions

// Define HYPRE error checking macro for convenience
#define HYPRE_CHECK(ierr) do { \
    if (ierr != 0) { \
        char hypre_error_msg[256]; \
        HYPRE_DescribeError(ierr, hypre_error_msg); \
        amrex::Abort("HYPRE Error: " + std::string(hypre_error_msg) + \
                     " - Error Code: " + std::to_string(ierr) + \
                     " File: " + __FILE__ + " Line: " + std::to_string(__LINE__)); \
    } \
} while (0)


// Define constants for clarity and maintainability
namespace {
    constexpr int SolnComp = 0; // Component index for the potential/concentration solution
    constexpr int PhaseComp = 1; // Component index for the cell type/phase visualization
    constexpr int numComponents = 2; // Total components in m_mf_phi
    constexpr amrex::Real tiny_flux_threshold = 1.e-15; // Tolerance for checking near-zero values
}

//-----------------------------------------------------------------------------
// Helper Functions are now static members defined in the header
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// TortuosityHypre Class Implementation
//-----------------------------------------------------------------------------

/**
 * @brief Constructor: Sets up geometry, solver params, and HYPRE structures.
 */
TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm,
                                 const amrex::iMultiFab& mf_phase_input, // Renamed for clarity
                                 const amrex::Real vf,
                                 const int phase,
                                 const OpenImpala::Direction dir,
                                 const SolverType st,
                                 // These should match the H file declaration now
                                 const std::string& resultspath,
                                 const amrex::Real vlo, // Default args from H
                                 const amrex::Real vhi, // Default args from H
                                 int verbose)         // Default args from H
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase(mf_phase_input, amrex::make_alias, 0, mf_phase_input.nComp()), // Create owned copy/alias
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      // <<< Initialize eps and maxiter from header defaults or add as args >>>
      // <<< Assuming they might be read from ParmParse or have fixed defaults later >>>
      // <<< For now, using placeholder values - ADJUST AS NEEDED >>>
      m_eps(1e-9), m_maxiter(200),
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose), // Initialize added members
      m_mf_phi(ba, dm, numComponents, 1), // Allocate solution multifab (SolnComp + PhaseComp), 1 ghost cell
      m_first_call(true),
      // Initialize HYPRE handles to NULL
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL)
{
    // --- Read potential missing params from ParmParse (Example) ---
    amrex::ParmParse pp("hypre"); // Look for params under "hypre." prefix
    pp.query("eps", m_eps);
    pp.query("maxiter", m_maxiter);
    // Ensure plot file writing control exists if needed by TortuosityHypre::solve logic
    // pp.query("write_plotfile", m_write_plotfile); // Need to add m_write_plotfile member if used

    // --- Validate Inputs ---
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0, "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf_phase_input.nGrow() >= 1, "Input phase iMultiFab needs at least 1 ghost cell");

    // --- Setup Steps ---
    preconditionPhaseFab(); // Modify the phase field if necessary (e.g., remove spots)
    setupGrids();           // Setup HYPRE grid structure
    setupStencil();         // Setup HYPRE stencil structure
    setupMatrixEquation();  // Setup HYPRE matrix and vectors
}

/**
 * @brief Destructor: Frees all allocated HYPRE resources. CRITICAL.
 */
TortuosityHypre::~TortuosityHypre()
{
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    // Set handles to NULL after destruction (optional but good practice)
    m_x = m_b = NULL;
    m_A = NULL;
    m_stencil = NULL;
    m_grid = NULL;
}


/**
 * @brief Sets up the HYPRE StructGrid based on the AMReX BoxArray.
 */
void TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;

    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr); // <<< Use HYPRE_CHECK >>>

    // Use MFIter on the *owned* phase multifab m_mf_phase after copy/precondition
    for ( amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi )
    {
        const amrex::Box& bx = mfi.validbox();
        auto lo = loV(bx); // Use static helper from header
        auto hi = hiV(bx); // Use static helper from header

        ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());
        HYPRE_CHECK(ierr);
    }

    // Finalize grid assembly
    ierr = HYPRE_StructGridAssemble(m_grid);
    HYPRE_CHECK(ierr);
}


/**
 * @brief Sets up the 7-point HYPRE StructStencil.
 */
void TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    constexpr int stencil_size = 7;
    // Standard 7-point stencil offsets: {Center, -x, +x, -y, +y, -z, +z}
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{0,0,0},
                                                     {-1,0,0}, {1,0,0},
                                                     {0,-1,0}, {0,1,0},
                                                     {0,0,-1}, {0,0,1}};

    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil);
    HYPRE_CHECK(ierr);

    for (int i = 0; i < stencil_size; i++)
    {
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]);
        HYPRE_CHECK(ierr);
    }
}

/**
 * @brief Modifies the phase MultiFab, e.g., removing isolated spots.
 * Calls the Fortran routine `tortuosity_remspot`.
 */
void TortuosityHypre::preconditionPhaseFab()
{
    // This routine modifies the *owned* copy m_mf_phase
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");

    // Ensure ghost cells are filled before modifying based on neighbors
    m_mf_phase.FillBoundary(m_geom.periodicity());

    const amrex::Box& domain_box = m_geom.Domain();

    // Use MFIter with tiling possibility
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Operate on the tilebox (valid region for this thread/iteration)
        const amrex::Box& tile_box = mfi.tilebox();
        amrex::IArrayBox& fab = m_mf_phase[mfi]; // Get non-const fab to modify

        // Pass the FAB itself, the tile box (valid region to iterate inside),
        // and the domain box (for boundary checks inside Fortran)
        tortuosity_remspot(BL_TO_FORTRAN_FAB(fab),
                           BL_TO_FORTRAN_BOX(tile_box),
                           BL_TO_FORTRAN_BOX(domain_box));
    }

    // Need to refill ghost cells after modification if subsequent steps rely on them
    m_mf_phase.FillBoundary(m_geom.periodicity());
}

/**
 * @brief Sets up the HYPRE StructMatrix and StructVectors (A, b, x).
 * Fills matrix coefficients and RHS using the Fortran routine `tortuosity_fillmtx`.
 */
void TortuosityHypre::setupMatrixEquation()
{
    HYPRE_Int ierr = 0;

    // Create and initialize matrix A
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructMatrixInitialize(m_A);
    HYPRE_CHECK(ierr);

    // Create and initialize RHS vector b
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b);
    HYPRE_CHECK(ierr);
    // Often initialize RHS to zero before setting boundary contributions
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0);
    HYPRE_CHECK(ierr);


    // Create and initialize solution vector x (initial guess)
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
     // Often initialize guess to zero
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0);
    HYPRE_CHECK(ierr);


    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[7] = {0,1,2,3,4,5,6}; // Indices matching setupStencil order
    const int dir_int = static_cast<int>(m_dir); // Cast direction enum once

    // Calculate dxinv^2 needed by Fortran
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize(); // Get cell sizes
    for(int i=0; i<AMREX_SPACEDIM; ++i) {
        dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]); // Calculate 1/dx^2 etc.
    }

    // Iterate over boxes owned by this rank to fill matrix/vectors
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Operate on tile box
        const int npts = static_cast<int>(bx.numPts()); // <<< Use int >>>

        // Check for potential overflow if size_t could be larger than int max
        if (static_cast<size_t>(npts) != bx.numPts()) {
            amrex::Abort("TortuosityHypre::setupMatrixEquation: Number of points in box exceeds INT_MAX");
        }
        if (npts == 0) continue; // Skip empty boxes

        // Use dynamic allocation (std::vector) for potentially large arrays
        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * 7);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        // Get phase data pointer and bounds
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi]; // Get IArrayBox
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = m_mf_phase.fabbox(mfi); // Get FAB box for phase data

        // --- CORRECTED Fortran Call for tortuosity_fillmtx ---
        tortuosity_fillmtx(matrix_values.data(),       // a
                           rhs_values.data(),          // rhs
                           initial_guess.data(),       // xinit
                           &npts,                      // nval (int*)
                           p_ptr,                      // p (int*)
                           pbox.loVect().getVect(),    // p_lo (int*)
                           pbox.hiVect().getVect(),    // p_hi (int*)
                           bx.loVect().getVect(),      // bxlo (int*) - Use tile box
                           bx.hiVect().getVect(),      // bxhi (int*) - Use tile box
                           domain.loVect().getVect(),  // domlo (int*)
                           domain.hiVect().getVect(),  // domhi (int*)
                           dxinv_sq.data(),            // dxinv (Real*) - Pass inverse SQUARED
                           &m_vlo,                     // vlo (Real*)
                           &m_vhi,                     // vhi (Real*)
                           &m_phase,                   // phase (int*)
                           &dir_int);                  // dir (int*)

        // Set values in HYPRE structures for this box
        // Note: HYPRE uses HYPRE_Int* for bounds, need temporary conversion if HYPRE_Int != int
        auto hypre_lo = loV(bx); // Use static helper
        auto hypre_hi = hiV(bx); // Use static helper

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), 7, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);

        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);

        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }

    // Finalize matrix assembly
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr);
    // Vectors b and x usually don't need explicit assembly after SetBoxValues
}

/**
 * @brief Solves the linear system Ax = b using the selected HYPRE solver.
 * Updates m_mf_phi with the solution and writes a plotfile.
 * @return true if solver converged successfully, false otherwise.
 */
bool TortuosityHypre::solve()
{
    // Generate timestamp for output file
    std::time_t strt_time;
    std::tm* timeinfo;
    char datetime [80];
    std::time(&strt_time);
    timeinfo = std::localtime(&strt_time);
    // Ensure buffer is large enough for format, e.g., YYYYMMDD_HHMMSS
    std::strftime(datetime, sizeof(datetime),"%Y%m%d_%H%M%S", timeinfo);

    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL; // Preconditioner handle, NULL means none used yet

    HYPRE_Int ierr = 0;
    HYPRE_Int num_iterations = 0;
    HYPRE_Real final_res_norm = -1.0;

    if (m_verbose > 0) {
        amrex::Print() << "Solving with HYPRE... Tolerance=" << m_eps << ", MaxIter=" << m_maxiter << std::endl;
    }

    // --- Setup Solver and Optional Preconditioner ---
    // Using SMG or PFMG as a preconditioner is common for Struct solvers
    // Example: Using PFMG as preconditioner for FlexGMRES/GMRES
    if (m_solvertype == SolverType::FlexGMRES || m_solvertype == SolverType::GMRES || m_solvertype == SolverType::PCG) {
        HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_StructPFMGSetMaxIter(precond, 1); // Use as preconditioner (1 cycle)
        HYPRE_StructPFMGSetTol(precond, 0.0);   // Tolerance doesn't matter for precond
        HYPRE_StructPFMGSetRelChange(precond, 0); // Don't check relative change
        // Set other PFMG parameters if needed (e.g., relaxation, num pre/post smooth)
    }

    // --- Select and Run the HYPRE solver ---
    switch (m_solvertype)
    {
        case SolverType::Jacobi: // Not generally recommended, very slow convergence
             ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructJacobiSetTol(solver, m_eps); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructJacobiSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
             // Set other Jacobi options (e.g., weight) if needed
             ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x); // Check ierr below
             ierr = HYPRE_StructJacobiGetNumIterations(solver, &num_iterations); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &final_res_norm); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructJacobiDestroy(solver); HYPRE_CHECK(ierr);
             break;

        case SolverType::FlexGMRES:
             ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructFlexGMRESSetTol(solver, m_eps); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
             if (precond) {
                 ierr = HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
             }
             ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x); // Check ierr below
             ierr = HYPRE_StructFlexGMRESGetNumIterations(solver, &num_iterations); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructFlexGMRESDestroy(solver); HYPRE_CHECK(ierr);
             break;

        case SolverType::PCG:
             ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructPCGSetTol(solver, m_eps); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructPCGSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
              if (precond) {
                 ierr = HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
             }
             ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x); // Check ierr below
             ierr = HYPRE_StructPCGGetNumIterations(solver, &num_iterations); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructPCGDestroy(solver); HYPRE_CHECK(ierr);
             break;

        case SolverType::GMRES: // Fallthrough intended for default case
        default:
             ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructGMRESSetTol(solver, m_eps); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructGMRESSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
             if (precond) {
                 ierr = HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
             }
             ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x); // Check ierr below
             ierr = HYPRE_StructGMRESGetNumIterations(solver, &num_iterations); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm); HYPRE_CHECK(ierr);
             ierr = HYPRE_StructGMRESDestroy(solver); HYPRE_CHECK(ierr);
             break;
    }

    // --- Clean up Preconditioner if used ---
    if (precond) {
        HYPRE_StructPFMGDestroy(precond);
        precond = NULL;
    }

    // --- Report Solver Outcome ---
    if (m_verbose > 0) {
        amrex::Print() << "Solver finished: " << num_iterations
                       << " Iterations, Final Relative Residual = "
                       << final_res_norm << std::endl;
    }

    // Check for HYPRE solver errors (ierr from the specific Solve call)
    bool converged_ok = true;
    if (ierr) {
        amrex::Print() << "ERROR: HYPRE solver returned error code: " << ierr << std::endl;
        if (HYPRE_CheckError(ierr, HYPRE_ERROR_CONV)) {
            amrex::Print() << "       (Solver did not converge within max iterations or tolerance)" << std::endl;
        } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_MEMORY)) {
            amrex::Print() << "       (Solver memory allocation error)" << std::endl;
        } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_ARG)) {
            amrex::Print() << "       (Solver argument error)" << std::endl;
        }
         // Consider checking for HYPRE_ERROR_INDEFINITE_PC if using certain preconditioners
         converged_ok = false; // Indicate solver failure
     }
     // Optional: Explicitly check convergence status based on tolerance
     if (final_res_norm > m_eps && num_iterations >= m_maxiter) {
         if (m_verbose > 0) { // Only warn if verbose
            amrex::Print() << "Warning: Solver reached max iterations without converging to tolerance." << std::endl;
         }
         converged_ok = false; // Or treat this as failure depending on needs
     }


    // Copy solution from HYPRE vector m_x to AMReX MultiFab m_mf_phi (Component SolnComp)
    getSolution(m_mf_phi, SolnComp); // Pass component explicitly

    // Fill Component PhaseComp of m_mf_phi with cell types based on phase
    getCellTypes(m_mf_phi, PhaseComp);

    // --- Write Plotfile ---
    // Ensure output directory exists
    if (amrex::ParallelDescriptor::IOProcessor()) { // Only IO Processor creates directory
        if (!m_resultspath.empty() && !amrex::UtilCreateDirectory(m_resultspath, 0755)) {
             amrex::Warning("Could not create results directory: " + m_resultspath);
        }
    }
    // Barrier to make sure directory is created before non-IO processors proceed
    amrex::ParallelDescriptor::Barrier();

    // Construct filename
    std::string plotfilename = m_resultspath + "/hypre_soln_" + std::string(datetime);
    const std::vector<std::string> varnames = {"potential", "cell_type"}; // Use component names

    if (m_verbose > 0) amrex::Print() << "Writing plotfile: " << plotfilename << std::endl;
    amrex::WriteSingleLevelPlotfile(plotfilename, m_mf_phi, varnames, m_geom, 0.0, 0);

    return converged_ok; // Indicate solver success/failure based on ierr and tolerance
}

/**
 * @brief Calculates the tortuosity value based on the solved potential field.
 * Calls solve() if necessary. Performs parallel reduction for flux sums.
 * @param refresh If true, forces a re-solve of the linear system.
 * @return The calculated tortuosity value (VF / RelativeDiffusivity).
 */
amrex::Real TortuosityHypre::value(const bool refresh)
{
    // Solve the system if needed
    if (refresh || m_first_call)
    {
        if (!solve()) {
            // Handle solver failure - maybe throw exception or return error code
             amrex::Warning("Solver failed in TortuosityHypre::value. Returning NaN.");
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN(); // Cache NaN
             m_first_call = true; // Allow retry
             return m_value;
        }
        m_first_call = false; // Solve succeeded
    } else if (std::isnan(m_value)) {
        // If previous solve failed (cached NaN) and refresh=false, return NaN immediately
        return m_value;
    }

    // --- Calculate Fluxes ---
    // Re-implement global_fluxes logic here or call a separate private method
    amrex::Real fluxin = 0.0, fluxout = 0.0;
    global_fluxes(fluxin, fluxout); // Call the private helper method

    // --- Calculate Tortuosity ---
    const amrex::Box& domain_box = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    amrex::Real length_dir = m_geom.ProbLength(static_cast<int>(m_dir));
    amrex::Real cross_sectional_area = 0.0;

    switch(m_dir)
    {
        case OpenImpala::Direction::X : cross_sectional_area = m_geom.ProbLength(1) * m_geom.ProbLength(2); break;
        case OpenImpala::Direction::Y : cross_sectional_area = m_geom.ProbLength(0) * m_geom.ProbLength(2); break;
        case OpenImpala::Direction::Z : cross_sectional_area = m_geom.ProbLength(0) * m_geom.ProbLength(1); break;
        default: amrex::Abort("TortuosityHypre::value: Invalid direction");
    }

     if (cross_sectional_area <= tiny_flux_threshold) {
         amrex::Warning("TortuosityHypre::value: Domain cross-sectional area is near zero. Cannot calculate tortuosity.");
         m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
         return m_value;
     }

    // Effective Diffusivity Calculation (assuming D_bulk = 1)
    // Flux = - D_eff * Area * Grad(Phi) = - D_eff * Area * (Vhi - Vlo) / Length
    // D_eff = - Flux * Length / (Area * (Vhi - Vlo))
    // If Vhi-Vlo != 0:
    // Relative Diffusivity D_rel = D_eff / D_bulk = - Flux * Length / (Area * (Vhi - Vlo)) (assuming D_bulk=1)
    amrex::Real delta_V = m_vhi - m_vlo;
    amrex::Real rel_diffusivity = 0.0;
    amrex::Real avg_flux = (fluxin + fluxout) / 2.0; // Should be equal if converged

    if (std::abs(delta_V) > tiny_flux_threshold && length_dir > tiny_flux_threshold) {
         rel_diffusivity = - avg_flux * length_dir / (cross_sectional_area * delta_V);
    } else {
        amrex::Warning("TortuosityHypre::value: Cannot calculate relative diffusivity due to zero length or zero potential difference.");
    }

    // Tortuosity Calculation: tau = VF / D_rel (assuming Bruggeman-like relation tau = VF / (D_eff/D_bulk))
    // Alternative definition: tau^2 = 1 / D_rel (Path length tortuosity squared) - Check definition required!
    // Using tau = VF / D_rel for now.
    if (std::abs(rel_diffusivity) > tiny_flux_threshold) {
         m_value = m_vf / rel_diffusivity;
         // Tortuosity should generally be >= VF. If rel_diffusivity is negative (unexpected flux direction?) or > 1, result might be odd.
         if (m_value < 0.0 && m_vf > tiny_flux_threshold) {
             amrex::Warning("Calculated negative tortuosity, check flux direction and BCs.");
         }
    } else {
         if (m_vf > tiny_flux_threshold) {
             amrex::Print() << "Warning: Relative diffusivity is near zero for non-zero VF. Tortuosity is effectively infinite." << std::endl;
             m_value = std::numeric_limits<amrex::Real>::infinity();
         } else {
             m_value = 0.0; // Tortuosity is 0 or undefined if VF is 0
         }
    }

    // Print diagnostics (only on IOProcessor to avoid redundant output)
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "------------------------------------------" << std::endl;
        amrex::Print() << " Tortuosity Calculation Results (Dir=" << static_cast<int>(m_dir) << ")" << std::endl; // Use int cast for enum
        amrex::Print() << std::fixed << std::setprecision(6); // Set precision for output
        amrex::Print() << "    Volume Fraction (VF)                : " << m_vf << std::endl;
        amrex::Print() << "    Relative Effective Diffusivity      : " << rel_diffusivity << std::endl;
        amrex::Print() << "    Tortuosity (tau = VF / D_rel)       : " << m_value << std::endl;
        amrex::Print() << "    --- Intermediate Values ---" << std::endl;
        amrex::Print() << "    Flux Low Face                       : " << fluxin << std::endl;
        amrex::Print() << "    Flux High Face                      : " << fluxout << std::endl;
        amrex::Print() << "    Flux Average                        : " << avg_flux << std::endl;
        amrex::Print() << "    Flux Conservation Check |In - Out|  : " << std::abs(fluxin - fluxout) << std::endl;
        amrex::Print() << "------------------------------------------" << std::endl;
    }

    return m_value;
}


/**
 * @brief Copies the solution vector from HYPRE (m_x) to a specified component
 * of the destination AMReX MultiFab (soln).
 * @param soln The destination MultiFab.
 * @param ncomp The component index within soln to store the result.
 */
void TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp)
{
    // Ensure soln has the required component
    AMREX_ASSERT(ncomp >= 0 && ncomp < soln.nComp());

    // Create temporary FArrayBox on host to receive data box-by-box
    amrex::FArrayBox host_fab;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(host_fab)
#endif
    for (amrex::MFIter mfi(soln); mfi.isValid(); ++mfi)
    {
        const amrex::Box &reg = mfi.validbox();
        // Resize temporary host FAB for the current box
        host_fab.resize(reg);

        auto reglo = loV(reg); // Use static helper
        auto reghi = hiV(reg); // Use static helper

        // Retrieve data from HYPRE vector m_x into the temporary host FArrayBox.
        HYPRE_Int ierr = HYPRE_StructVectorGetBoxValues(m_x, reglo.data(), reghi.data(), host_fab.dataPtr());
        HYPRE_CHECK(ierr); // Use macro for error checking

        // Copy data from the host FAB to the destination MultiFab component 'ncomp'
        // This handles potential GPU data movement if soln is on device.
        soln[mfi].copy(host_fab, 0, ncomp, 1, soln.nGrowVect(), soln.nGrowVect()); // Use copy with guard cells specified
    }
}


/**
 * @brief Fills a specified component of a MultiFab with cell type information based on phase.
 * Calls the Fortran routine `tortuosity_filct`.
 * @param phi The MultiFab to modify.
 * @param ncomp The component index within phi to fill (e.g., PhaseComp).
 */
void TortuosityHypre::getCellTypes(amrex::MultiFab& phi, const int ncomp)
{
    // Check if the component index is valid
    AMREX_ASSERT(ncomp >= 0 && ncomp < phi.nComp());

    // Requires Fortran interface from Tortuosity_filcc_F.H
    // Need to make sure m_mf_phase (iMultiFab) is accessible and provides integer data

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Use tilebox
        amrex::FArrayBox& fab_target = phi[mfi]; // Target FArrayBox (Real)
        const amrex::IArrayBox& fab_phase_src = m_mf_phase[mfi]; // Source phase data (Integer)

        // --- Call Fortran routine tortuosity_filct ---
        // WARNING: This assumes tortuosity_filct can write Real output (cell type 1.0 or 0.0)
        // based on Integer input phase data. Fortran interface should match.
        // Also assumes Fortran handles component indexing correctly based on base pointers.
        int q_ncomp = fab_target.nComp();
        int p_ncomp = fab_phase_src.nComp();
        const auto& qbox = phi.fabbox(mfi); // Use FAB box bounds for target array
        const auto& pbox = m_mf_phase.fabbox(mfi); // Use FAB box bounds for source array
        const auto& domain_box = m_geom.Domain(); // Domain bounds

        // Need to pass pointer to the specific component 'ncomp' of the target FAB
        amrex::Real* q_comp_ptr = fab_target.dataPtr(ncomp);

        tortuosity_filct(q_comp_ptr, // Pass pointer to component ncomp
                         qbox.loVect().getVect(), qbox.hiVect().getVect(), &q_ncomp,
                         fab_phase_src.dataPtr(), // Pass pointer to int data
                         pbox.loVect().getVect(), pbox.hiVect().getVect(), &p_ncomp,
                         domain_box.loVect().getVect(), domain_box.hiVect().getVect(),
                         &m_phase);
    }
}

/**
 * @brief Computes global fluxes across domain boundaries.
 * Assumes the solution (`m_mf_phi`) has been computed and stored.
 * Requires careful finite difference implementation at boundaries.
 * @param[out] fxin Total flux entering the domain.
 * @param[out] fxout Total flux exiting the domain.
 */
void TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const {
    fxin = 0.0;
    fxout = 0.0;
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

    const int idir = static_cast<int>(m_dir);
    const amrex::Real dxinv_dir = (idir == 0) ? (1.0 / m_geom.CellSize(0)) :
                                  (idir == 1) ? (1.0 / m_geom.CellSize(1)) :
                                                (1.0 / m_geom.CellSize(2));
    const amrex::Real area = (idir == 0) ? (m_geom.ProbLength(1) * m_geom.ProbLength(2)) :
                             (idir == 1) ? (m_geom.ProbLength(0) * m_geom.ProbLength(2)) :
                                           (m_geom.ProbLength(0) * m_geom.ProbLength(1));

    const amrex::Box& domain = m_geom.Domain();
    const int domlo_dir = domain.smallEnd(idir);
    const int domhi_dir = domain.bigEnd(idir);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_phi); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& phi = m_mf_phi.const_array(mfi, SolnComp);
        const auto& phase = m_mf_phase.const_array(mfi); // Use the owned, possibly preconditioned phase data

        amrex::Box lobox = amrex::adjCellLo(domain, idir, 1); // Box adjacent to low domain boundary
        lobox &= bx; // Intersect with current valid box

        if (lobox.ok()) { // If this box touches the low boundary
             amrex::Loop(lobox, [=, &local_fxin] (int i, int j, int k) noexcept {
                 // Check if cell AND neighbor inside domain are conducting phase
                 // Neighbor is at (i-1) for X, (j-1) for Y, (k-1) for Z
                 amrex::IntVect iv(i, j, k);
                 amrex::IntVect iv_neighbor = iv;
                 iv_neighbor[idir] -= 1; // Neighbor towards low boundary

                 if (phase(iv) == m_phase && domain.contains(iv_neighbor) && phase(iv_neighbor) == m_phase) {
                     // Central difference for flux at face: -D * (phi(i) - phi(i-1))/dx
                     // Assuming D=1
                     local_fxin += -1.0 * (phi(iv) - phi(iv_neighbor)) * dxinv_dir;
                 }
             });
        }

        amrex::Box hibox = amrex::adjCellHi(domain, idir, 0); // Box adjacent to high domain boundary (+1 face)
        hibox.shift(idir, 1); // Shift box up to index high face itself
        hibox &= bx; // Intersect with current valid box

        if (hibox.ok()) { // If this box touches the high boundary face
             amrex::Loop(hibox, [=, &local_fxout] (int i, int j, int k) noexcept {
                 // Check if cell AND neighbor inside domain are conducting phase
                 // Neighbor is at (i-1) for X, (j-1) for Y, (k-1) for Z
                 amrex::IntVect iv(i, j, k);
                 amrex::IntVect iv_neighbor = iv;
                 iv_neighbor[idir] -= 1; // Neighbor towards low boundary

                 if (phase(iv) == m_phase && domain.contains(iv_neighbor) && phase(iv_neighbor) == m_phase) {
                      // Central difference for flux at face: -D * (phi(i) - phi(i-1))/dx
                     // Assuming D=1
                     local_fxout += -1.0 * (phi(iv) - phi(iv_neighbor)) * dxinv_dir;
                 }
             });
        }
    } // End MFIter loop

    // Multiply sum of gradients by face area (constant)
    local_fxin *= area;
    local_fxout *= area;

    // Reduce across MPI processes
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout, amrex::ParallelDescriptor::IOProcessorNumber());

    // Only IOProc holds the correct sum now
    if (amrex::ParallelDescriptor::IOProcessor()) {
        fxin = local_fxin;
        fxout = local_fxout;
    }
    // Optional: Broadcast if all processes need the result
    // amrex::ParallelDescriptor::Bcast(&fxin, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    // amrex::ParallelDescriptor::Bcast(&fxout, 1, amrex::ParallelDescriptor::IOProcessorNumber());
}
