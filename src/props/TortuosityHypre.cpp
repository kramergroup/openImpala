#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"      // For tortuosity_remspot, tortuosity_filct
#include "TortuosityHypreFill_F.H"  // For tortuosity_fillmtx

#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <cmath>
#include <limits>   // For std::numeric_limits
#include <stdexcept> // For potential error throwing (optional)

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>       // For amrex::average_down (potentially needed elsewhere, good include)
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>          // For amrex::UtilCreateDirectory
#include <AMReX_BLassert.H>         // Corrected include path
#include <AMReX_ParmParse.H>        // Added for ParmParse
#include <AMReX_Vector.H>           // Added for amrex::Vector (needed for plotfile fix)
#include <AMReX_Array.H>            // Added for amrex::Array (for dxinv_sq fix & loV/hiV return)

// HYPRE includes (already in TortuosityHypre.H but good practice here too)
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
// #include <HYPRE_struct_mv.h> // Might be needed for SetConstantValues or GetBoxValues

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
// Helper Functions are static members defined in the header TortuosityHypre.H
//-----------------------------------------------------------------------------
// Implementation of static helpers (should ideally be in the header or defined here)
namespace OpenImpala { // Assuming TortuosityHypre is in OpenImpala namespace

inline amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::loV (const amrex::Box& b) {
    const int* lo_ptr = b.loVect();
    amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hypre_lo;
    for (int i=0; i<AMREX_SPACEDIM; ++i) hypre_lo[i] = static_cast<HYPRE_Int>(lo_ptr[i]);
    return hypre_lo;
}
inline amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::hiV (const amrex::Box& b) {
    const int* hi_ptr = b.hiVect();
    amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hypre_hi;
    for (int i=0; i<AMREX_SPACEDIM; ++i) hypre_hi[i] = static_cast<HYPRE_Int>(hi_ptr[i]);
    return hypre_hi;
}

//-----------------------------------------------------------------------------
// TortuosityHypre Class Implementation
//-----------------------------------------------------------------------------

/**
 * @brief Constructor: Sets up geometry, solver params, and HYPRE structures.
 */
OpenImpala::TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom,
                                             const amrex::BoxArray& ba,
                                             const amrex::DistributionMapping& dm,
                                             const amrex::iMultiFab& mf_phase_input, // Renamed for clarity
                                             const amrex::Real vf,
                                             const int phase,
                                             const OpenImpala::Direction dir,
                                             const SolverType st, // Namespace already implicitly OpenImpala::TortuosityHypre::
                                             const std::string& resultspath,
                                             const amrex::Real vlo, // Default args from H
                                             const amrex::Real vhi, // Default args from H
                                             int verbose)           // Default args from H
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase(mf_phase_input, amrex::make_alias, 0, mf_phase_input.nComp()), // Create owned copy/alias
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      // Initialize eps and maxiter from ParmParse or defaults
      m_eps(1e-9), m_maxiter(200),
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose),
      m_mf_phi(ba, dm, numComponents, 1), // Allocate solution multifab (SolnComp + PhaseComp), 1 ghost cell
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()), // Initialize value to NaN
      // Initialize HYPRE handles to NULL
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL)
{
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initializing..." << std::endl;
    }

    // --- Read potential missing params from ParmParse (Example) ---
    amrex::ParmParse pp("hypre"); // Look for params prefixed with "hypre."
    pp.query("eps", m_eps);
    pp.query("maxiter", m_maxiter);
    // Read verbosity specific to this class if needed, overriding argument default
    amrex::ParmParse pp_tort("tortuosity"); // Or a relevant prefix
    pp_tort.query("verbose", m_verbose);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  HYPRE Params: eps=" << m_eps << ", maxiter=" << m_maxiter << std::endl;
         amrex::Print() << "  Class Verbose Level: " << m_verbose << std::endl;
    }


    // --- Validate Inputs ---
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0, "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf_phase_input.nGrow() >= 1, "Input phase iMultiFab needs at least 1 ghost cell");

    // --- Setup Steps ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab..." << std::endl;
    preconditionPhaseFab(); // Modify the phase field if necessary (e.g., remove spots)

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();           // Setup HYPRE grid structure

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil();         // Setup HYPRE stencil structure

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation();  // Setup HYPRE matrix and vectors

     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
    }
}

/**
 * @brief Destructor: Frees all allocated HYPRE resources. CRITICAL.
 */
OpenImpala::TortuosityHypre::~TortuosityHypre()
{
    // Add verbose print if desired
    // if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Destroying HYPRE objects..." << std::endl;
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
void OpenImpala::TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;

    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr);

    // Use MFIter on the *owned* phase multifab m_mf_phase after copy/precondition
    for ( amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi )
    {
        const amrex::Box& bx = mfi.validbox();
        // Use static helpers qualified with class name
        auto lo = OpenImpala::TortuosityHypre::loV(bx); // Returns amrex::Array<HYPRE_Int,...>
        auto hi = OpenImpala::TortuosityHypre::hiV(bx); // Returns amrex::Array<HYPRE_Int,...>

        // Pass pointer from amrex::Array using .data()
        ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());
        HYPRE_CHECK(ierr);
    }

    // Finalize grid assembly
    ierr = HYPRE_StructGridAssemble(m_grid);
    HYPRE_CHECK(ierr);

    // +++ Add Check +++
    if (!m_grid) {
        amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!");
    }
    // +++ End Check +++
}


/**
 * @brief Sets up the 7-point HYPRE StructStencil.
 */
void OpenImpala::TortuosityHypre::setupStencil()
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
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
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

        // --- Call Fortran routine tortuosity_remspot ---
        // Pass FAB data pointer, tile box bounds, and domain bounds
        // Need Fortran function signature from Tortuosity_filcc_F.H
        // void tortuosity_remspot(int* q, const int* q_lo, const int* q_hi, const int* ncomp,
        //                         const int* bxlo, const int* bxhi,
        //                         const int* domlo, const int* domhi);
        int ncomp = fab.nComp();
        tortuosity_remspot(fab.dataPtr(),         // q
                           fab.loVect(),          // q_lo (Fortran bounds of fab, incl. ghost)
                           fab.hiVect(),          // q_hi (Fortran bounds of fab, incl. ghost)
                           &ncomp,                // ncomp
                           tile_box.loVect(),     // bxlo (valid box to iterate over)
                           tile_box.hiVect(),     // bxhi (valid box to iterate over)
                           domain_box.loVect(),   // domlo
                           domain_box.hiVect());  // domhi
    }

    // Need to refill ghost cells after modification if subsequent steps rely on them
    m_mf_phase.FillBoundary(m_geom.periodicity());
}

/**
 * @brief Sets up the HYPRE StructMatrix and StructVectors (A, b, x).
 * Fills matrix coefficients and RHS using the Fortran routine `tortuosity_fillmtx`.
 */
void OpenImpala::TortuosityHypre::setupMatrixEquation()
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
    // +++ Add Check +++
    if (!m_b) {
        amrex::Abort("FATAL: m_b handle is NULL after HYPRE_StructVectorInitialize!");
    }
    // +++ End Check +++
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); // Initialize RHS to zero
    HYPRE_CHECK(ierr);


    // Create and initialize solution vector x (initial guess)
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); // Initialize guess to zero
    HYPRE_CHECK(ierr);


    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[7] = {0,1,2,3,4,5,6}; // Indices matching setupStencil order
    const int dir_int = static_cast<int>(m_dir); // Cast direction enum once

    // Calculate dxinv^2 needed by Fortran
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize(); // Get cell sizes
    for(int i=0; i<AMREX_SPACEDIM; ++i) {
        dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]); // Calculate 1/dx^2 etc.
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine..." << std::endl;
    }

    // Iterate over boxes owned by this rank to fill matrix/vectors
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Operate on tile box
        const int npts = static_cast<int>(bx.numPts()); // Use int

        if (npts == 0) continue; // Skip empty boxes

        // Use dynamic allocation (std::vector) for potentially large arrays
        // Allocate on host memory
        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * 7);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        // Get phase data pointer and bounds
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi]; // Get IArrayBox
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex()); // Use index for box

        // --- Fortran Call for tortuosity_fillmtx ---
        tortuosity_fillmtx(matrix_values.data(),      // a
                           rhs_values.data(),         // rhs
                           initial_guess.data(),      // xinit
                           &npts,                     // nval (int*)
                           p_ptr,                     // p (int*)
                           pbox.loVect(),             // p_lo (int*)
                           pbox.hiVect(),             // p_hi (int*)
                           bx.loVect(),               // bxlo (int*) - Use tile box
                           bx.hiVect(),               // bxhi (int*) - Use tile box
                           domain.loVect(),           // domlo (int*)
                           domain.hiVect(),           // domhi (int*)
                           dxinv_sq.data(),           // dxinv (Real*) - Pass pointer from amrex::Array
                           &m_vlo,                    // vlo (Real*)
                           &m_vhi,                    // vhi (Real*)
                           &m_phase,                  // phase (int*)
                           &dir_int);                 // dir (int*)

        // +++ START DEBUG CHECK (for NaN/Inf) +++
        bool rhs_ok = true;
        for(size_t idx = 0; idx < npts; ++idx) {
            if (std::isnan(rhs_values[idx]) || std::isinf(rhs_values[idx])) {
                // Print only once per rank to avoid flooding logs
                if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::IOProcessorNumber()) {
                    amrex::Print() << "!!! Invalid value detected in rhs_values[" << idx << "] = " << rhs_values[idx]
                                   << " within box " << bx << " (Rank " << amrex::ParallelDescriptor::MyProc() << ")" << std::endl;
                    // Optional: Print neighbouring phase values from p_ptr around the error index for context
                }
                rhs_ok = false;
                // break; // Optional: stop checking after first error found
            }
        }
        // Ensure all ranks know if an error occurred
        int global_rhs_ok = rhs_ok; // Initialize with local status
        amrex::ParallelDescriptor::ReduceIntMin(global_rhs_ok); // Find if *any* rank failed (0=FAIL, 1=OK)

        if (global_rhs_ok == 0) { // If any rank found bad data
             amrex::Abort("NaN/Inf found in rhs_values after tortuosity_fillmtx Fortran call!");
        }
        // +++ END DEBUG CHECK (for NaN/Inf) +++

        // Set values in HYPRE structures for this box
        // Use static helpers qualified with class name
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx); // Use static helper
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx); // Use static helper

        // Pass pointers from amrex::Array using .data()
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), 7, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);

        // +++ Add Box Print +++
        // Print unconditionally for now, or use m_verbose > 1
        amrex::Print() << "Rank " << amrex::ParallelDescriptor::MyProc()
                       << " processing box " << bx << " before HYPRE_StructVectorSetBoxValues(m_b)" << std::endl;
        // Flush output to ensure it appears before potential crash
        amrex::OutStream().flush();
        // +++ End Box Print +++

        // Failing call:
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr); // This triggers the abort

        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    } // End MFIter loop

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Finished Fortran call. Assembling matrix..." << std::endl;
    }

    // Finalize matrix assembly
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr);
    // Vectors b and x usually don't need explicit assembly after SetBoxValues
    // ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr); // Optional
    // ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr); // Optional

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Matrix assembled." << std::endl;
    }
}

/**
 * @brief Solves the linear system Ax = b using the selected HYPRE solver.
 * Updates m_mf_phi with the solution and writes a plotfile.
 * @return true if solver converged successfully, false otherwise.
 */
bool OpenImpala::TortuosityHypre::solve()
{
    // Generate timestamp for output file
    std::time_t strt_time;
    std::tm* timeinfo;
    char datetime [80];
    std::time(&strt_time);
    timeinfo = std::localtime(&strt_time);
    std::strftime(datetime, sizeof(datetime),"%Y%m%d_%H%M%S", timeinfo);

    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL;

    HYPRE_Int ierr = 0;
    HYPRE_Int num_iterations = 0;
    HYPRE_Real final_res_norm = -1.0;

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre::solve(): Solving with HYPRE... Tolerance=" << m_eps << ", MaxIter=" << m_maxiter << std::endl;
    }

    // --- Setup Solver and Optional Preconditioner ---
    // Example: Using PFMG as preconditioner for iterative methods
    if (m_solvertype == OpenImpala::TortuosityHypre::SolverType::FlexGMRES ||
        m_solvertype == OpenImpala::TortuosityHypre::SolverType::GMRES ||
        m_solvertype == OpenImpala::TortuosityHypre::SolverType::PCG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Using PFMG Preconditioner (1 cycle)" << std::endl;
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        ierr = HYPRE_StructPFMGSetMaxIter(precond, 1); HYPRE_CHECK(ierr); // Use as preconditioner (1 cycle)
        ierr = HYPRE_StructPFMGSetTol(precond, 0.0); HYPRE_CHECK(ierr);    // Tolerance doesn't matter for precond
        ierr = HYPRE_StructPFMGSetRelChange(precond, 0); HYPRE_CHECK(ierr); // Don't check relative change
        // Add other PFMG settings if needed (e.g., relax type)
    }

    // --- Select and Run the HYPRE solver ---
    switch (m_solvertype)
    {
        case OpenImpala::TortuosityHypre::SolverType::Jacobi:
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Solver Type: Jacobi" << std::endl;
            ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructJacobiSetTol(solver, m_eps); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructJacobiSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
            // NOTE: HYPRE_StructJacobiSetLogging does not appear to exist in Hypre v2.30.0
            // Iteration progress for Jacobi is usually less critical to log than Krylov methods.
            // if (m_verbose > 1) { /* No specific Jacobi logging function found */ }

             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructJacobiSetup..." << std::endl;
            ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructJacobiSolve..." << std::endl;
            ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x); // Check ierr below
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  HYPRE_StructJacobiSolve finished (ierr=" << ierr << ")" << std::endl;

            HYPRE_StructJacobiGetNumIterations(solver, &num_iterations);
            HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructJacobiDestroy(solver);
            break;

        case OpenImpala::TortuosityHypre::SolverType::FlexGMRES:
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Solver Type: FlexGMRES" << std::endl;
            ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructFlexGMRESSetTol(solver, m_eps); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
            if (precond) {
                 if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting Preconditioner..." << std::endl;
                ierr = HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
            }
             if (m_verbose > 1) { HYPRE_StructFlexGMRESSetLogging(solver, 1); } // Enable iteration logging if verbose

             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructFlexGMRESSetup..." << std::endl;
            ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructFlexGMRESSolve..." << std::endl;
            ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x); // Check ierr below
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  HYPRE_StructFlexGMRESSolve finished (ierr=" << ierr << ")" << std::endl;

            HYPRE_StructFlexGMRESGetNumIterations(solver, &num_iterations);
            HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructFlexGMRESDestroy(solver);
            break;

        case OpenImpala::TortuosityHypre::SolverType::PCG:
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Solver Type: PCG" << std::endl;
            ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructPCGSetTol(solver, m_eps); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructPCGSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
            if (precond) {
                 if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting Preconditioner..." << std::endl;
                ierr = HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
            }
             if (m_verbose > 1) { HYPRE_StructPCGSetLogging(solver, 1); } // Enable iteration logging if verbose

             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructPCGSetup..." << std::endl;
            ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
              if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructPCGSolve..." << std::endl;
            ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x); // Check ierr below
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  HYPRE_StructPCGSolve finished (ierr=" << ierr << ")" << std::endl;

            HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
            HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructPCGDestroy(solver);
            break;

        case OpenImpala::TortuosityHypre::SolverType::GMRES: // Fallthrough intended for default case
        default: // Default to GMRES
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Solver Type: GMRES (Default)" << std::endl;
            ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructGMRESSetTol(solver, m_eps); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructGMRESSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
            if (precond) {
                 if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting Preconditioner..." << std::endl;
                ierr = HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
            }
              if (m_verbose > 1) { HYPRE_StructGMRESSetLogging(solver, 1); } // Enable iteration logging if verbose

             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructGMRESSetup..." << std::endl;
            ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
              if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructGMRESSolve..." << std::endl;
            ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x); // Check ierr below
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  HYPRE_StructGMRESSolve finished (ierr=" << ierr << ")" << std::endl;

            HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
            HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructGMRESDestroy(solver);
            break;
    }

    // --- Clean up Preconditioner if used ---
    if (precond) {
        HYPRE_StructPFMGDestroy(precond);
        precond = NULL;
    }

    // --- Report Solver Outcome ---
    bool converged_ok = true; // Assume converged unless error or tolerance check fails
    // Always print outcome if verbose > 0
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Solver finished: " << num_iterations
                       << " Iterations, Final Relative Residual = "
                       << final_res_norm << std::endl;
    }

    // Check for HYPRE solver errors (ierr from the specific Solve call)
    if (ierr != 0) { // Check if the Solve function itself returned an error
        if (amrex::ParallelDescriptor::IOProcessor()) { // Print only once
            amrex::Print() << "ERROR: HYPRE solver returned error code: " << ierr << std::endl;
            if (HYPRE_CheckError(ierr, HYPRE_ERROR_CONV)) {
                amrex::Print() << "       (Solver did not converge within max iterations or tolerance)" << std::endl;
            } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_MEMORY)) {
                amrex::Print() << "       (Solver memory allocation error)" << std::endl;
            } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_ARG)) {
                amrex::Print() << "       (Solver argument error)" << std::endl;
            }
            // Add more HYPRE_ERROR checks if needed
        }
        converged_ok = false; // Indicate solver failure
    }

    // Explicitly check convergence status based on reported residual norm
    // Check even if ierr was 0, as some solvers might finish maxiter without error but without converging
    if (final_res_norm > m_eps || std::isnan(final_res_norm) || std::isinf(final_res_norm)) { // Also check for NaN/Inf
        if (converged_ok && amrex::ParallelDescriptor::IOProcessor()) { // Only print warning if no HYPRE error was reported yet
             amrex::Print() << "Warning: Solver did not converge to tolerance " << m_eps
                            << " (Final Residual = " << final_res_norm << ")" << std::endl;
        }
        converged_ok = false; // Mark as not converged if tolerance not met or result invalid
    }

    // --- Copy solution and potentially write plotfile ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Copying solution to MultiFab..." << std::endl;
    getSolution(m_mf_phi, SolnComp); // Pass component explicitly

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Filling cell types..." << std::endl;
    getCellTypes(m_mf_phi, PhaseComp);

    // --- Write Plotfile ---
    if (write_plotfile != 0 && !m_resultspath.empty()) { // Check flag write_plotfile
        if (!amrex::UtilCreateDirectory(m_resultspath, 0755)) {
             amrex::Warning("Could not create results directory: " + m_resultspath);
             // Decide if plotfile writing failure should affect return status? Probably not.
        } else {
            std::string plotfilename = m_resultspath + "/hypre_soln_" + std::string(datetime);
            amrex::Vector<std::string> plot_varnames = {"potential", "cell_type"};

            if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Writing plotfile: " << plotfilename << std::endl;
            amrex::WriteSingleLevelPlotfile(plotfilename, m_mf_phi, plot_varnames, m_geom, 0.0, 0);
        }
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre::solve() finished. Converged OK: " << (converged_ok ? "Yes" : "No") << std::endl;
    }
    return converged_ok; // Indicate solver success/failure based on ierr and tolerance
}

/**
 * @brief Calculates the tortuosity value based on the solved potential field.
 * Calls solve() if necessary. Performs parallel reduction for flux sums.
 * @param refresh If true, forces a re-solve of the linear system.
 * @return The calculated tortuosity value (VF / RelativeDiffusivity).
 */
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
      if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
          amrex::Print() << "TortuosityHypre::value() called (refresh=" << refresh << ", first_call=" << m_first_call << ")" << std::endl;
    }

    // Solve the system if needed
    if (refresh || m_first_call)
    {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling solve()..." << std::endl;
        if (!solve()) {
            // Handle solver failure
             amrex::Warning("Solver failed in TortuosityHypre::value. Returning NaN.");
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN(); // Cache NaN
             m_first_call = true; // Allow retry if called again with refresh=true
             return m_value;
        }
        m_first_call = false; // Solve succeeded
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  solve() completed successfully." << std::endl;

    } else if (std::isnan(m_value)) {
        // If previous solve failed (cached NaN) and refresh=false, return NaN immediately
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Returning cached NaN value." << std::endl;
        return m_value;
    }

    // --- Calculate Fluxes ---
      if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling global_fluxes()..." << std::endl;
    amrex::Real fluxin = 0.0, fluxout = 0.0;
    global_fluxes(fluxin, fluxout); // Call the private helper method
      if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  global_fluxes() completed (fluxin=" << fluxin << ", fluxout=" << fluxout << ")" << std::endl;


    // --- Calculate Tortuosity ---
    const amrex::Box& domain_box = m_geom.Domain();
    amrex::Real length_dir = m_geom.ProbLength(static_cast<int>(m_dir));
    amrex::Real cross_sectional_area = 0.0;

    // Use member m_dir directly
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
    amrex::Real delta_V = m_vhi - m_vlo;
    amrex::Real rel_diffusivity = 0.0;
    amrex::Real avg_flux = (fluxin + fluxout) / 2.0; // Should be equal if converged

    if (std::abs(delta_V) > tiny_flux_threshold && length_dir > tiny_flux_threshold) {
          // Ensure non-zero area before dividing
          if (std::abs(cross_sectional_area) > tiny_flux_threshold) {
              rel_diffusivity = - avg_flux * length_dir / (cross_sectional_area * delta_V);
          } else {
              amrex::Warning("TortuosityHypre::value: Cross sectional area is zero.");
              // rel_diffusivity remains 0.0
          }
    } else {
        amrex::Warning("TortuosityHypre::value: Cannot calculate relative diffusivity due to zero length or zero potential difference.");
        // rel_diffusivity remains 0.0
    }

    // Tortuosity Calculation: tau = VF / D_rel
    if (std::abs(rel_diffusivity) > tiny_flux_threshold) {
          m_value = m_vf / rel_diffusivity;
          // Check for physically unreasonable values
          if (m_value < 0.0 && m_vf > tiny_flux_threshold) {
              amrex::Warning("Calculated negative tortuosity, check flux direction, BCs, or definition.");
          } else if (m_value < m_vf && m_vf > tiny_flux_threshold) {
              // Tortuosity should generally be >= 1. If VF < 1, then tau should be > VF.
              // This check might be too strict depending on definition, but useful warning.
              amrex::Warning("Calculated tortuosity is less than volume fraction. Check definition/calculation.");
          }
    } else {
          if (m_vf > tiny_flux_threshold) {
               if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: Relative diffusivity is near zero for non-zero VF. Tortuosity is effectively infinite." << std::endl;
               m_value = std::numeric_limits<amrex::Real>::infinity();
          } else {
                if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Note: Volume Fraction is near zero. Setting tortuosity to 0.0." << std::endl;
               m_value = 0.0; // Tortuosity is 0 or undefined if VF is 0
          }
    }

    // Print final diagnostics (only on IOProcessor to avoid redundant output)
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "------------------------------------------" << std::endl;
        amrex::Print() << " Tortuosity Calculation Results (Dir=" << static_cast<int>(m_dir) << ")" << std::endl;
        amrex::Print() << std::fixed << std::setprecision(6); // Set precision for output
        amrex::Print() << "   Volume Fraction (VF)                 : " << m_vf << std::endl;
        amrex::Print() << "   Relative Effective Diffusivity       : " << rel_diffusivity << std::endl;
        amrex::Print() << "   Tortuosity (tau = VF / D_rel)        : " << m_value << std::endl;
        amrex::Print() << "   --- Intermediate Values ---" << std::endl;
        amrex::Print() << "   Flux Low Face                        : " << fluxin << std::endl;
        amrex::Print() << "   Flux High Face                       : " << fluxout << std::endl;
        amrex::Print() << "   Flux Average                         : " << avg_flux << std::endl;
        amrex::Print() << "   Flux Conservation Check |In + Out|   : " << std::abs(fluxin + fluxout) << std::endl; // Should be near zero
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
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp)
{
    // Ensure soln has the required component and ghost cells >= 0
    AMREX_ASSERT(ncomp >= 0 && ncomp < soln.nComp());
    AMREX_ASSERT(soln.nGrowVect().min() >= 0);

    // Create temporary FArrayBox on host to receive data box-by-box
    amrex::FArrayBox host_fab;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(host_fab)
#endif
    for (amrex::MFIter mfi(soln); mfi.isValid(); ++mfi)
    {
        const amrex::Box &reg = mfi.validbox();

        // Ensure box is not empty before proceeding
        if (!reg.ok()) continue;

        // Resize temporary host FAB for the current box
        host_fab.resize(reg, 1); // Resize for 1 component

        // Use static helpers qualified with class name
        auto reglo = OpenImpala::TortuosityHypre::loV(reg); // Use static helper
        auto reghi = OpenImpala::TortuosityHypre::hiV(reg); // Use static helper

        // Retrieve data from HYPRE vector m_x into the temporary host FArrayBox.
        HYPRE_Int ierr = HYPRE_StructVectorGetBoxValues(m_x, reglo.data(), reghi.data(), host_fab.dataPtr());
        HYPRE_CHECK(ierr); // Use macro for error checking

        // Copy data from the host FAB (component 0) to the destination MultiFab component 'ncomp'
        soln[mfi].copy(host_fab, reg, 0, reg, ncomp, 1); // Copy valid region only
    }
}


/**
 * @brief Fills a specified component of a MultiFab with cell type information based on phase.
 * Calls the Fortran routine `tortuosity_filct`.
 * @param phi The MultiFab to modify.
 * @param ncomp The component index within phi to fill (e.g., PhaseComp).
 */
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, const int ncomp)
{
    // Check if the component index is valid
    AMREX_ASSERT(ncomp >= 0 && ncomp < phi.nComp());
    // Ensure input phase MF has sufficient ghost cells if Fortran needs them
    AMREX_ASSERT(m_mf_phase.nGrow() >= phi.nGrow());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab_target = phi[mfi]; // Target FArrayBox (Real)
        const amrex::IArrayBox& fab_phase_src = m_mf_phase[mfi]; // Source phase data (Integer)

        // --- Call Fortran routine tortuosity_filct ---
        int q_ncomp = fab_target.nComp();
        int p_ncomp = fab_phase_src.nComp();
        const auto& qbox = phi.box(mfi.LocalTileIndex()); // Use index for box
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex()); // Use index for box
        const auto& domain_box = m_geom.Domain(); // Domain bounds

        // Need to pass pointer to the specific component 'ncomp' of the target FAB
        amrex::Real* q_comp_ptr = fab_target.dataPtr(ncomp);

        // Get Fortran function signature from Tortuosity_filcc_F.H
        // void tortuosity_filct(amrex_real* q, const int* q_lo, const int* q_hi, const int* q_ncomp,
        //                       const int* p, const int* p_lo, const int* p_hi, const int* p_ncomp,
        //                       const int* domlo, const int* domhi, const int* phase);

        // Pass raw pointers from loVect/hiVect
        tortuosity_filct(q_comp_ptr,              // q (pointer to component ncomp)
                         fab_target.loVect(),     // q_lo (Fortran bounds of fab_target, incl. ghost)
                         fab_target.hiVect(),     // q_hi (Fortran bounds of fab_target, incl. ghost)
                         &q_ncomp,                // q_ncomp
                         fab_phase_src.dataPtr(), // p (pointer to integer phase data)
                         fab_phase_src.loVect(),  // p_lo (Fortran bounds of fab_phase_src, incl. ghost)
                         fab_phase_src.hiVect(),  // p_hi (Fortran bounds of fab_phase_src, incl. ghost)
                         &p_ncomp,                // p_ncomp
                         domain_box.loVect(),     // domlo
                         domain_box.hiVect(),     // domhi
                         &m_phase);               // phase
    }
}

/**
 * @brief Computes global fluxes across domain boundaries based on the solved potential field.
 * @param[out] fxin Total flux entering the domain at the low face.
 * @param[out] fxout Total flux exiting the domain at the high face.
 */
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const {
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre::global_fluxes() calculating..." << std::endl;
    }

    fxin = 0.0;
    fxout = 0.0;
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

    const int idir = static_cast<int>(m_dir);
    const amrex::Real dxinv_dir = (idir == 0) ? (1.0 / m_geom.CellSize(0)) :
                                  (idir == 1) ? (1.0 / m_geom.CellSize(1)) :
                                                (1.0 / m_geom.CellSize(2));
    // Calculate face area based on cell dimensions
    amrex::Real face_area = 1.0;
     if (idir == 0) { face_area = m_geom.CellSize(1) * m_geom.CellSize(2); }
     else if (idir == 1) { face_area = m_geom.CellSize(0) * m_geom.CellSize(2); }
     else { face_area = m_geom.CellSize(0) * m_geom.CellSize(1); }


    const amrex::Box& domain = m_geom.Domain();
    const int domlo_dir = domain.smallEnd(idir);
    const int domhi_dir = domain.bigEnd(idir);

    // Ensure solution ghost cells are up-to-date (needed for finite difference)
    // Create a temporary copy or alias to fill boundaries without modifying original m_mf_phi[SolnComp] state if needed elsewhere
    amrex::MultiFab phi_soln_local(m_mf_phi, amrex::make_alias, SolnComp, 1);
    phi_soln_local.FillBoundary(m_geom.periodicity());
    // Note: Applying physical BCs (Dirichlet m_vlo/m_vhi) might be needed here if FillBoundary
    // only handles internal boundaries, depending on AMReX version/setup.
    // However, the flux calculation below explicitly uses m_vlo/m_vhi at the boundary face.

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    // Use phi_soln_local which has filled boundaries
    for (amrex::MFIter mfi(phi_soln_local); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        // Need const access to potential (SolnComp) and phase (from m_mf_phase)
        const auto& phi    = phi_soln_local.const_array(mfi); // Use local copy with filled ghosts
        // Use the cell type component that was filled by getCellTypes
        const auto& cell_type = m_mf_phi.const_array(mfi, PhaseComp); // Assuming PhaseComp holds 0/1 for non-phase/phase

        // Calculate flux on the low domain face (idir = domlo_dir)
        amrex::Box lo_face_box = domain;
        lo_face_box.setBig(idir, domlo_dir);
        lo_face_box.setSmall(idir, domlo_dir);
        lo_face_box &= bx; // Intersect face box with current valid box

        if (lo_face_box.ok()) {
            // Use central difference across the low boundary face
            amrex::Loop(lo_face_box, [=, &local_fxin] (int i, int j, int k) noexcept {
                amrex::IntVect iv_cell1(i, j, k);   // Cell inside domain at index 'lo'

                // Check if cell INSIDE boundary is target phase (using PhaseComp)
                if (cell_type(iv_cell1) == 1) { // Assuming 1 marks the conducting phase here
                    // Flux = - D * (phi_cell1 - phi_ghost) / dx
                    // Here D=1, phi_ghost is effectively m_vlo
                    // Flux entering = -1.0 * (phi(iv_cell1) - m_vlo) / dx
                    local_fxin += -1.0 * (phi(iv_cell1) - m_vlo) * dxinv_dir;
                }
            });
        }

        // Calculate flux on the high domain face (face index is domhi_dir + 1)
        amrex::Box hi_face_box = domain;
        hi_face_box.setSmall(idir, domhi_dir + 1); // Face index
        hi_face_box.setBig(idir, domhi_dir + 1);
        // Intersect face box with current valid box. Need grow cell for phi access inside.
        // MFIter is over valid cells, need access to phi(domhi_dir)
        amrex::Box bx_grown = bx; bx_grown.grow(idir, 1); // Grow box in direction of interest
        amrex::Box hi_face_intersect = hi_face_box & bx_grown; // Intersect with grown box

         if (hi_face_intersect.ok()) {
              // Use central difference across the high boundary face
              amrex::Loop(hi_face_intersect, [=, &local_fxout] (int i, int j, int k) noexcept {
                  amrex::IntVect iv_cell1(i, j, k); iv_cell1[idir] -= 1; // Cell inside at index 'hi'

                  // Check if cell INSIDE boundary is target phase (using PhaseComp)
                  if (cell_type(iv_cell1) == 1) { // Assuming 1 marks the conducting phase here
                       // Flux = - D * (phi_ghost - phi_cell1) / dx
                       // Here D=1, phi_ghost is effectively m_vhi
                       // Flux exiting = -1.0 * (m_vhi - phi(iv_cell1)) / dx
                       local_fxout += -1.0 * (m_vhi - phi(iv_cell1)) * dxinv_dir;
                  }
              });
         }
    } // End MFIter loop

    // Multiply sum of gradients by face area (area of a single cell face)
    local_fxin *= face_area;
    local_fxout *= face_area;

    // Reduce across MPI processes - Sum results from all processes
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    // Assign the globally reduced values
    fxin = local_fxin;  // Flux entering at low face (should be positive if Vhi > Vlo)
    fxout = local_fxout; // Flux exiting at high face (should be positive if Vhi > Vlo)

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  Reduced fluxes: fxin=" << fxin << ", fxout=" << fxout << std::endl;
    }

}


} // End namespace OpenImpala
