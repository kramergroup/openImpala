#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"     // For tortuosity_remspot, tortuosity_filct
#include "TortuosityHypreFill_F.H" // For tortuosity_fillmtx

#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <cmath>
#include <limits>    // For std::numeric_limits
#include <stdexcept> // For potential error throwing (optional)
#include <iomanip>   // For std::setprecision
#include <iostream>  // For std::cout, std::flush

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H> // Needed for AMREX_GPU_DEVICE if used elsewhere
#include <AMReX_Box.H>           // Needed for Box operations
#include <AMReX_IntVect.H>       // Needed for amrex::IntVect

// HYPRE includes
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

// Define HYPRE error checking macro
#define HYPRE_CHECK(ierr) do { \
    if (ierr != 0) { \
        char hypre_error_msg[256]; \
        HYPRE_DescribeError(ierr, hypre_error_msg); \
        amrex::Abort("HYPRE Error: " + std::string(hypre_error_msg) + \
                     " - Error Code: " + std::to_string(ierr) + \
                     " File: " + __FILE__ + " Line: " + std::to_string(__LINE__)); \
    } \
} while (0)


// Define constants
namespace {
    constexpr int SolnComp = 0;
    constexpr int PhaseComp = 1;
    constexpr int numComponents = 2; // Now assuming SolnComp + PhaseComp
    constexpr amrex::Real tiny_flux_threshold = 1.e-15;
    constexpr int stencil_size = 7; // Standard 7-point stencil
}

// Helper Functions and Class Implementation
namespace OpenImpala {

// --- Helper functions loV, hiV ---
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

// --- Constructor ---
OpenImpala::TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom,
                                             const amrex::BoxArray& ba,
                                             const amrex::DistributionMapping& dm,
                                             const amrex::iMultiFab& mf_phase_input,
                                             const amrex::Real vf,
                                             const int phase,
                                             const OpenImpala::Direction dir,
                                             const SolverType st,
                                             const std::string& resultspath,
                                             const amrex::Real vlo,
                                             const amrex::Real vhi,
                                             int verbose,
                                             bool write_plotfile)
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase(mf_phase_input, amrex::make_alias, 0, mf_phase_input.nComp()), // Use alias
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      m_eps(1e-9), m_maxiter(200),
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose),
      m_write_plotfile(write_plotfile),
      m_mf_phi(ba, dm, numComponents, 1), // Ensure 1 ghost cell for phi if needed later
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL),
      m_num_iterations(-1), m_final_res_norm(std::numeric_limits<amrex::Real>::quiet_NaN())
{
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initializing..." << std::endl;
    }

    // Parse solver parameters
    amrex::ParmParse pp("hypre");
    pp.query("eps", m_eps);
    pp.query("maxiter", m_maxiter);

    // Parse general verbosity (allow override from hypre block or tortuosity block)
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("verbose", m_verbose); // Overrides hypre.verbose if present

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  HYPRE Params: eps=" << m_eps << ", maxiter=" << m_maxiter << std::endl;
         amrex::Print() << "  Class Verbose Level: " << m_verbose << std::endl;
         amrex::Print() << "  Write Plotfile Flag: " << m_write_plotfile << std::endl;
    }

    // Assertions for valid inputs
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0, "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf_phase_input.nGrow() >= 1, "Input phase iMultiFab needs at least 1 ghost cell");

    // Precondition phase field (e.g., remove isolated spots)
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab..." << std::endl;
    preconditionPhaseFab(); // Assuming this modifies m_mf_phase in place or fills a member fab

    // Setup HYPRE Grid
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();

    // Setup HYPRE Stencil
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil();

    // Setup HYPRE Matrix Equation (calls Fortran fill routine)
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation();

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
    }
}

// --- Destructor ---
OpenImpala::TortuosityHypre::~TortuosityHypre()
{
    // Destroy HYPRE objects in reverse order of creation (generally)
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);

    // Nullify pointers after destruction
    m_x = m_b = NULL;
    m_A = NULL;
    m_stencil = NULL;
    m_grid = NULL;
}


// --- Setup HYPRE Grid based on AMReX BoxArray ---
void OpenImpala::TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;

    // Create the grid object spanning ndim dimensions
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr);

    // Add each box from the BoxArray to the HYPRE grid
    // Note: HYPRE expects integer coordinates.
    for (int i = 0; i < m_ba.size(); ++i) {
        // Get the current box (valid region, no ghost cells for grid definition)
        amrex::Box bx = m_ba[i];
        auto lo = OpenImpala::TortuosityHypre::loV(bx); // Use helper
        auto hi = OpenImpala::TortuosityHypre::hiV(bx); // Use helper

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  DEBUG [setupGrids]: Setting extents for Box " << bx
                            << " (from m_ba[" << i << "])"
                            << " with lo = [" << lo[0] << "," << lo[1] << "," << lo[2] << "]"
                            << " hi = [" << hi[0] << "," << hi[1] << "," << hi[2] << "]" << std::endl;
        }

        // Set the extents for this box (processor's local portion)
        ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());
        HYPRE_CHECK(ierr);
    }

    // Assemble the grid across all processors
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Calling HYPRE_StructGridAssemble..." << std::endl;
    }
    ierr = HYPRE_StructGridAssemble(m_grid);
    HYPRE_CHECK(ierr);
    if (!m_grid) { amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!"); }
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Grid setup complete. m_grid pointer: " << m_grid << std::endl;
    }
}

// --- Setup HYPRE Stencil (Standard 7-point) ---
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    // Define the offsets for a 7-point stencil in 3D
    // Order: Center, West (-x), East (+x), South (-y), North (+y), Bottom (-z), Top (+z)
    // This order MUST match the Fortran code's istn_* parameters and C++ stencil_indices
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{ 0, 0, 0},
                                                       {-1, 0, 0}, { 1, 0, 0},
                                                       { 0,-1, 0}, { 0, 1, 0},
                                                       { 0, 0,-1}, { 0, 0, 1}};

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Using standard 7-point stencil." << std::endl;
    }
    // Create the stencil object
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil);
    HYPRE_CHECK(ierr);

    // Set the stencil entries
    for (int i = 0; i < stencil_size; i++)
    {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  DEBUG [setupStencil]: Setting element " << i
                            << " with offset = [" << offsets[i][0] << "," << offsets[i][1] << "," << offsets[i][2] << "]" << std::endl;
         }
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]);
        HYPRE_CHECK(ierr);
    }
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL after HYPRE_StructStencilCreate/SetElement!"); }
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Stencil setup complete. m_stencil pointer: " << m_stencil << std::endl;
    }
}

// --- Preprocess Phase Field (Example: Remove isolated spots) ---
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    // Ensure phase fab has necessary ghost cells filled
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");
    m_mf_phase.FillBoundary(m_geom.periodicity()); // Fill ghost cells

    const amrex::Box& domain_box = m_geom.Domain();

    // Example: Call a Fortran routine to modify the phase field (e.g., remove isolated spots)
    // This routine would operate on the FAB data directly.
    // Ensure the Fortran routine respects tileboxes if using Tiling.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tile_box = mfi.tilebox(); // Operate only on the valid region of this tile
        amrex::IArrayBox& fab = m_mf_phase[mfi];    // Get the FAB data (including ghost cells)

        // Assuming tortuosity_remspot takes raw pointer, lo/hi of fab, lo/hi of tilebox, lo/hi of domain
        int ncomp = fab.nComp(); // Assuming it operates on the first component only? Check Fortran interface.
        // tortuosity_remspot(fab.dataPtr(PhaseComp), // Pass pointer to the relevant component
        //                    fab.loVect(), fab.hiVect(), &ncomp,
        //                    tile_box.loVect(), tile_box.hiVect(),
        //                    domain_box.loVect(), domain_box.hiVect());
        // Note: PhaseComp is defined as 1, but C++ array access is 0-based.
        // Check if Fortran expects 1-based component index or if remspot handles multi-component.
        // If remspot operates on all components, pass fab.dataPtr() and adjust ncomp logic.
        // If only component 0 (C++ index), pass fab.dataPtr(0) if PhaseComp means the first component.
        // ** Using fab.dataPtr() assuming Fortran routine handles the component offset **
         tortuosity_remspot(fab.dataPtr(), fab.loVect(), fab.hiVect(), &ncomp,
                           tile_box.loVect(), tile_box.hiVect(),
                           domain_box.loVect(), domain_box.hiVect());
    }

    // Refill ghost cells after modification if necessary
    m_mf_phase.FillBoundary(m_geom.periodicity());
}


// --- Setup HYPRE Matrix and Vectors, Call Fortran Fill Routine ---
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    HYPRE_Int ierr = 0;

    // Create the matrix object
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Checking handles before HYPRE_StructMatrixCreate..." << std::endl;
        amrex::Print() << "    m_grid pointer:    " << m_grid << std::endl;
        amrex::Print() << "    m_stencil pointer: " << m_stencil << std::endl;
    }
    if (!m_grid)    { amrex::Abort("FATAL: m_grid handle is NULL before HYPRE_StructMatrixCreate!"); }
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL before HYPRE_StructMatrixCreate!"); }
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructMatrixCreate..." << std::endl;
    }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    HYPRE_CHECK(ierr);
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: HYPRE_StructMatrixCreate successful." << std::endl;
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructMatrixInitialize..." << std::endl;
    }
    // Initialize the matrix structure before setting values
    ierr = HYPRE_StructMatrixInitialize(m_A);
    HYPRE_CHECK(ierr);


    // Create and initialize RHS vector b
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructVectorCreate (b)..." << std::endl;
    }
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b);
    HYPRE_CHECK(ierr);
    // Set RHS vector b to zero initially (Fortran routine will fill it)
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0);
    HYPRE_CHECK(ierr);

    // Create and initialize solution vector x
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructVectorCreate (x)..." << std::endl;
    }
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
    // Set initial guess x to zero initially (Fortran routine might provide a better guess)
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0);
    HYPRE_CHECK(ierr);


    // Prepare data for Fortran call
    const amrex::Box& domain = m_geom.Domain();
    // Define stencil indices (0-based) matching the Fortran parameters (istn_*)
    // Order: Center, -x, +x, -y, +y, -z, +z
    int stencil_indices[stencil_size] = {0, 1, 2, 3, 4, 5, 6};

    // Get grid spacing for coefficient calculation
    const int dir_int = static_cast<int>(m_dir); // Convert enum direction to integer
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) { dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]); }


     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
          amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine..." << std::endl;
     }
    // Loop over FABs and call Fortran routine to fill matrix/vector values
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox(); // Operate on the valid cells of this tile
        const int npts = static_cast<int>(bx.numPts()); // Number of points in this box

        if (npts == 0) continue; // Skip empty boxes

        // Allocate temporary vectors to hold data for this box
        // HYPRE expects contiguous data for the box
        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * stencil_size);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        // Get phase data FAB for this box (including ghost cells needed by Fortran)
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr(); // Use correct component if necessary
        const auto& pbox = phase_iab.box(); // Box including ghost cells

        // Call the Fortran routine
        // Assumes Fortran routine uses 1-based indexing internally for rhs/xinit
        // Assumes 'a' is flat array [point0_stencil0, point0_stencil1, ..., point1_stencil0, ...]
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts, p_ptr, pbox.loVect(), pbox.hiVect(),
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);

        // --- Check for NaNs/Infs returned by Fortran ---
        bool data_ok = true;
        for(int idx = 0; idx < npts; ++idx) {
            if (std::isnan(rhs_values[idx]) || std::isinf(rhs_values[idx])) { data_ok = false; break; }
            if (std::isnan(initial_guess[idx]) || std::isinf(initial_guess[idx])) { data_ok = false; break; }
            for (int s_idx = 0; s_idx < stencil_size; ++s_idx) {
                 if (std::isnan(matrix_values[static_cast<size_t>(idx)*stencil_size + s_idx]) ||
                     std::isinf(matrix_values[static_cast<size_t>(idx)*stencil_size + s_idx])) {
                    data_ok = false; break;
                 }
            }
            if (!data_ok) break;
        }
        int global_data_ok = data_ok;
        amrex::ParallelDescriptor::ReduceIntMin(global_data_ok); // Check across all MPI ranks
        if (global_data_ok == 0) {
           amrex::Abort("NaN/Inf found in matrix/rhs/init_guess values returned from Fortran!");
        }
        // --- End NaN/Inf Check ---

        // Set the computed values into the HYPRE matrix and vectors for this box
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // Set matrix coefficients
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);

        // Set RHS values
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);

        // Set initial guess values
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Finished MFIter loop setting values from Fortran." << std::endl;
    }

    // Assemble the matrix across all processors
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling HYPRE_StructMatrixAssemble..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Matrix assembled." << std::endl;
    }
}


// --- Solve the Linear System using HYPRE ---
bool OpenImpala::TortuosityHypre::solve() {
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL; // Initialize preconditioner handle

    // Reset solve status members
    m_num_iterations = -1;
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();

    // --- PCG Solver ---
    if (m_solvertype == SolverType::PCG) {
        // Create PCG solver
        ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);

        // Set PCG parameters
        HYPRE_StructPCGSetTol(solver, m_eps);           // Convergence tolerance
        HYPRE_StructPCGSetMaxIter(solver, m_maxiter);   // Maximum iterations
        HYPRE_StructPCGSetTwoNorm(solver, 1);           // Use the 2-norm for residual
        HYPRE_StructPCGSetRelChange(solver, 0);         // Don't use relative change stopping criteria
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 2 : 0); // Print residual info if verbose

        // Set up SMG preconditioner
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetMemoryUse(precond, 0);        // Optimize for memory
        HYPRE_StructSMGSetMaxIter(precond, 1);          // Use one SMG cycle as preconditioner
        HYPRE_StructSMGSetTol(precond, 0.0);            // Tolerance for preconditioner solve (0.0 = use fixed iterations)
        HYPRE_StructSMGSetZeroGuess(precond);           // Use zero initial guess for preconditioner solve
        HYPRE_StructSMGSetNumPreRelax(precond, 1);      // Number of pre-relaxation sweeps
        HYPRE_StructSMGSetNumPostRelax(precond, 1);     // Number of post-relaxation sweeps

        // Set the preconditioner for PCG
        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);

        // Setup PCG solver (including preconditioner setup)
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);

        // Solve the system
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV) {
            if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE PCG solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else if (ierr != 0) { // Check for other errors (like NaN)
             HYPRE_CHECK(ierr); // Abort on other errors
        }

        // Get solve statistics
        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        // Clean up
        HYPRE_StructPCGDestroy(solver);
        if (precond) HYPRE_StructSMGDestroy(precond);
    }
    // --- GMRES Solver ---
    else if (m_solvertype == SolverType::GMRES) {
        // Create GMRES solver
        ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);

        // Set GMRES parameters
        HYPRE_StructGMRESSetTol(solver, m_eps);         // Convergence tolerance
        HYPRE_StructGMRESSetMaxIter(solver, m_maxiter); // Maximum iterations
        HYPRE_StructGMRESSetPrintLevel(solver, m_verbose > 1 ? 2 : 0); // Print residual info if verbose

        // --- TEMPORARILY DISABLE PRECONDITIONER FOR DEBUGGING NANs ---
        /*
        // Set up SMG preconditioner
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetMemoryUse(precond, 0);
        HYPRE_StructSMGSetMaxIter(precond, 1);
        HYPRE_StructSMGSetTol(precond, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond);
        HYPRE_StructSMGSetNumPreRelax(precond, 1);
        HYPRE_StructSMGSetNumPostRelax(precond, 1);

        // Set the preconditioner for GMRES
        HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
        */
        // --- END TEMPORARY DISABLE ---

        // Setup GMRES solver (will setup default preconditioner if none explicitly set, or none if SetPrecond commented out)
        ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);

        // Solve the system
        ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE GMRES solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
         } else if (ierr != 0) { // Check for other errors (like NaN)
              HYPRE_CHECK(ierr); // Abort on other errors
         }

        // Get solve statistics
        HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        // Clean up
        HYPRE_StructGMRESDestroy(solver);
        // if (precond) HYPRE_StructSMGDestroy(precond); // Keep commented if precond creation is commented
    }
    // --- FlexGMRES Solver ---
    else if (m_solvertype == SolverType::FlexGMRES) {
         amrex::Abort("FlexGMRES not fully implemented yet in TortuosityHypre::solve");
         // Need to implement HYPRE_StructFlexGMRES... functions
    }
    // --- Jacobi Solver ---
    else if (m_solvertype == SolverType::Jacobi) {
         // Create Jacobi solver
         ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
         HYPRE_CHECK(ierr);
         // Set Jacobi parameters
         HYPRE_StructJacobiSetTol(solver, m_eps);
         HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
         ierr = HYPRE_StructJacobiSetZeroGuess(solver); // Use zero initial guess
         HYPRE_CHECK(ierr);
         // Setup
         ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
         HYPRE_CHECK(ierr);
         // Solve
         ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) {
              if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE Jacobi solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
         } else if (ierr != 0) { // Check for other errors (like NaN)
               HYPRE_CHECK(ierr); // Abort on other errors
         }
         // Get stats
         HYPRE_StructJacobiGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         // Destroy
         HYPRE_StructJacobiDestroy(solver);
    }
    else {
        amrex::Abort("Unsupported solver type requested in TortuosityHypre::solve");
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << m_final_res_norm << std::endl;
    }

    // --- Write plot file if requested ---
    if (m_write_plotfile) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Writing solution plotfile..." << std::endl;
        }
        // Create a MultiFab to hold the solution and phase data for plotting
        // numComponents = 2 (solution, phase)
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponents, 0); // No ghost cells needed for plotfile

        // Temporary MultiFab to copy solution data into (needs only 1 component)
        amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 0); // No ghost cells needed
        mf_soln_temp.setVal(0.0); // Initialize

        // Buffer to hold data retrieved from HYPRE vector for each box
        std::vector<double> soln_buffer; // Use double as HYPRE functions often expect this

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
        for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) { // Use non-tiling MFIter is okay here
            const amrex::Box& bx = mfi.validbox(); // Get the valid box for this FAB
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0) continue;

            soln_buffer.resize(npts); // Resize buffer for current box size

            // Get HYPRE box coordinates
            auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

            // Get solution values from HYPRE vector for this box
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
            if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed in solve() for plotfile!"); }

            // Copy data from buffer to the temporary MultiFab's Array4 (CPU)
            // Assumes C-order (row-major) for HYPRE GetBoxValues buffer
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
            const amrex::IntVect lo = bx.smallEnd();
            const amrex::IntVect hi = bx.bigEnd();
            long long k_lin_idx = 0; // Linear index for the buffer
            for (int kk = lo[2]; kk <= hi[2]; ++kk) {
                for (int jj = lo[1]; jj <= hi[1]; ++jj) {
                    for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                        if (k_lin_idx < npts) { // Basic bounds check
                            // Cast from double buffer to amrex::Real Array4
                            soln_arr(ii,jj,kk) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]);
                        }
                        k_lin_idx++;
                    }
                }
            }
             if (k_lin_idx != npts) {
                amrex::Warning("Linear index mismatch during HYPRE->AMReX copy in solve()!");
             }
        }

        // Copy solution from temporary MF to plot MF (component SolnComp = 0)
        amrex::Copy(mf_plot, mf_soln_temp, 0, SolnComp, 1, 0);
        // Copy phase data from member variable MF to plot MF (component PhaseComp = 1)
        // Ensure m_mf_phase has data on component 0 (or adjust src_comp)
        // Need 0 ghost cells for Copy if mf_plot has 0 ghost cells.
        amrex::Copy(mf_plot, m_mf_phase, 0, PhaseComp, 1, 0); // Assumes phase data is in comp 0 of m_mf_phase


        // Write the plot file
        std::string plotfilename = m_resultspath + "/tortuosity_solution"; // Construct filename
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id"}; // Variable names
        amrex::Real time = 0.0; // Simulation time (arbitrary for this calculation)
        int level_step = 0; // Time step (arbitrary)
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, time, level_step);
    }

    m_first_call = false; // Mark that solve has been called

    // Return true if the solver converged within tolerance
    // Handle potential NaN in final_res_norm
    bool converged = (!std::isnan(m_final_res_norm)) && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);
    return converged;
}


// --- Calculate Tortuosity Value (Calls Solve if needed) ---
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
    if (m_first_call || refresh) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityHypre: Calling solve()..." << std::endl;
        }
        bool converged = solve(); // Run the solver

        // Check convergence status
        if (!converged && m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            // Print warning even if final_res_norm was NaN (handled inside solve already)
            amrex::Print() << "Warning: Solver did not converge (residual norm "
                           << m_final_res_norm << " > tolerance " << m_eps
                           << ", or NaN residual). Tortuosity value may be inaccurate." << std::endl;
        }

        // Calculate fluxes using the solution vector m_x
        amrex::Real flux_in = 0.0, flux_out = 0.0;
        global_fluxes(flux_in, flux_out); // Calculate fluxes based on solution m_x

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Global Flux In:  " << flux_in << std::endl;
             amrex::Print() << "  Global Flux Out: " << flux_out << std::endl;
             // Check flux balance as a sanity check (optional)
             // amrex::Real flux_diff = std::abs(flux_in + flux_out); // Note: flux_out might be negative
             // amrex::Print() << "  Abs Flux Difference: " << flux_diff << std::endl;
        }

        // --- Calculate Tortuosity ---
        // Check for conditions where tortuosity is ill-defined
        if (std::abs(flux_in) < tiny_flux_threshold) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Calculated input flux is near zero (" << flux_in
                                << "). Tortuosity is ill-defined or infinite. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else if (m_vf <= 0.0) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                  amrex::Print() << "Warning: Volume fraction is zero or negative. Tortuosity is ill-defined. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else {
            // Calculate geometric factors
            const amrex::Real* problo = m_geom.ProbLo();
            const amrex::Real* probhi = m_geom.ProbHi();
            amrex::Real area = 1.0;             // Cross-sectional area perpendicular to flow
            amrex::Real length_parallel = 1.0;  // Length parallel to flow
            int idir = static_cast<int>(m_dir); // Cast direction enum to int

            if (idir == 0) { // X-direction flow
                area = (probhi[1] - problo[1]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[0] - problo[0]);
            } else if (idir == 1) { // Y-direction flow
                area = (probhi[0] - problo[0]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[1] - problo[1]);
            } else { // Z-direction flow (idir == 2)
                area = (probhi[0] - problo[0]) * (probhi[1] - problo[1]);
                length_parallel = (probhi[2] - problo[2]);
            }

            // Avoid division by zero if length is zero
            if (std::abs(length_parallel) < std::numeric_limits<amrex::Real>::epsilon()) {
                 if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << "Warning: Domain length parallel to flow direction is near zero. Tortuosity is ill-defined. Returning NaN." << std::endl;
                 }
                 m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            } else {
                // Calculate effective diffusivity Deff = Flux_in / (Area * |Grad(Potential)|)
                // Assuming linear potential gradient: |Grad(Potential)| = |vhi - vlo| / length_parallel
                amrex::Real potential_diff = m_vhi - m_vlo;
                 if (std::abs(potential_diff) < tiny_flux_threshold) {
                      if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                          amrex::Print() << "Warning: Applied potential difference (vhi - vlo) is near zero. Tortuosity is ill-defined. Returning NaN." << std::endl;
                      }
                      m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
                 } else {
                    amrex::Real potential_gradient_mag = std::abs(potential_diff) / length_parallel;
                    amrex::Real effective_diffusivity = flux_in / (area * potential_gradient_mag);

                    // Tortuosity = VolumeFraction / (EffectiveDiffusivity / BulkDiffusivity)
                    // Assuming BulkDiffusivity = 1.0 (or factored out)
                    // Tortuosity = VF / Deff
                    if (effective_diffusivity <= 0.0) { // Should be positive if flux_in and grad_potential have same sign effectively
                         if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                             amrex::Print() << "Warning: Calculated effective diffusivity is zero or negative (" << effective_diffusivity
                                            << "). Check flux direction relative to potential gradient. Returning NaN." << std::endl;
                         }
                         m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
                    } else {
                        m_value = m_vf / effective_diffusivity;
                    }
                 }
            }
        }

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             // Use fixed and setprecision for potentially large/small values
             amrex::Print() << std::fixed << std::setprecision(8)
                            << "TortuosityHypre: Calculated tortuosity = " << m_value << std::endl;
        }
    }
    return m_value;
}


// --- Get Solution Field (Not fully implemented) ---
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) {
    amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
    // Implementation would involve copying data from m_x to the provided MultiFab 'soln'
    // similar to the copy logic used in solve() for plotfile writing, ensuring
    // correct component and ghost cell handling.
}

// --- Get Cell Types (Not implemented) ---
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
     amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
     // Implementation would likely involve classifying cells based on phase and boundary conditions
}


// --- Calculate Global Fluxes Across Domain Boundaries ---
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    fxin = 0.0;
    fxout = 0.0;
    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir); // Direction of flow

    // Need solution with ghost cells to calculate gradients at the boundary
    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1); // Need 1 ghost cell
    mf_soln_temp.setVal(0.0); // Initialize

    // --- Copy solution from HYPRE vector m_x to mf_soln_temp ---
    // Buffer to hold data retrieved from HYPRE vector for each box
    std::vector<double> soln_buffer; // Use double as HYPRE functions often expect this

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox(); // Get the valid box for this FAB
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        soln_buffer.resize(npts); // Resize buffer for current box size

        // Get HYPRE box coordinates
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // Get solution values from HYPRE vector for this box
        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
         if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed in global_fluxes!"); }

        // Copy data from buffer to the MultiFab's Array4 (CPU)
        // Assumes C-order (row-major) for HYPRE GetBoxValues buffer
        amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
        const amrex::IntVect lo = bx.smallEnd();
        const amrex::IntVect hi = bx.bigEnd();
        long long k_lin_idx = 0; // Linear index for the buffer
        for (int kk = lo[2]; kk <= hi[2]; ++kk) {
            for (int jj = lo[1]; jj <= hi[1]; ++jj) {
                for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                    if (k_lin_idx < npts) {
                        // Cast from double buffer to amrex::Real Array4
                        soln_arr(ii,jj,kk) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]);
                    }
                    k_lin_idx++;
                }
            }
        }
        if (k_lin_idx != npts) {
            amrex::Warning("Linear index mismatch during HYPRE->AMReX copy in global_fluxes()!");
        }
    }
    // --- End Copy ---

    // Fill ghost cells for the solution field needed for gradient calculation
    mf_soln_temp.FillBoundary(m_geom.periodicity());


    // --- Calculate Fluxes ---
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Use tilebox for iteration domain

        // Get const references to phase and solution arrays
        const auto phase = m_mf_phase.const_array(mfi); // Assuming component 0 holds the relevant phase ID
        const auto soln = mf_soln_temp.const_array(mfi); // Component 0 holds the potential

        // Box defining the low face of the domain in the flow direction
        amrex::Box lobox = amrex::adjCellLo(domain, idir, 1);
        lobox &= bx; // Intersect with current tilebox

        // Box defining the high face of the domain in the flow direction
        // Note: For flux calculation, we need cells *inside* the high boundary
        // to compute the gradient across the boundary face. adjCellHi gives cells *outside*.
        // Let's use adjCellLo(domain, idir, 0) which is the domain boundary itself,
        // then grow by -shift to get the cell just inside.
        amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);
        amrex::Box hibox_boundary = amrex::adjCellHi(domain, idir, 0); // High boundary face
        amrex::Box hibox = hibox_boundary;
        hibox.shift(shift*(-1)); // Get cells just inside the high boundary
        hibox &= bx; // Intersect with current tilebox

        amrex::Real grad, flux;

        // Calculate flux across the low boundary face (Flux In)
        // Centered difference gradient: (phi_face - phi_cell_behind) / dx
        // Flux = - grad
        const amrex::IntVect lo_flux = lobox.smallEnd();
        const amrex::IntVect hi_flux = lobox.bigEnd();
        for (int k = lo_flux[2]; k <= hi_flux[2]; ++k) {
            for (int j = lo_flux[1]; j <= hi_flux[1]; ++j) {
                for (int i = lo_flux[0]; i <= hi_flux[0]; ++i) {
                     // Check if the cell *inside* the boundary (i,j,k) is conductive
                     if (phase(i,j,k, PhaseComp) == m_phase) { // Check correct component
                         // Gradient uses cell i and neighbor i-1 (which is outside low boundary)
                         grad = (soln(i,j,k) - soln(i-shift[0], j-shift[1], k-shift[2])) / dx[idir];
                         flux = -grad;
                         local_fxin += flux;
                     }
                }
            }
        }

        // Calculate flux across the high boundary face (Flux Out)
        // Gradient uses cell i+1 (outside high boundary) and cell i (inside boundary)
        const amrex::IntVect lo_flux_hi = hibox.smallEnd();
        const amrex::IntVect hi_flux_hi = hibox.bigEnd();
        for (int k = lo_flux_hi[2]; k <= hi_flux_hi[2]; ++k) {
            for (int j = lo_flux_hi[1]; j <= hi_flux_hi[1]; ++j) {
                for (int i = lo_flux_hi[0]; i <= hi_flux_hi[0]; ++i) {
                     // Check if the cell *inside* the boundary (i,j,k) is conductive
                     if (phase(i,j,k, PhaseComp) == m_phase) { // Check correct component
                          // Gradient uses cell i+1 (which is outside high boundary) and cell i
                          grad = (soln(i+shift[0], j+shift[1], k+shift[2]) - soln(i,j,k)) / dx[idir];
                          flux = -grad;
                          local_fxout += flux; // Accumulate flux leaving through high face
                     }
                }
            }
        }
    } // End MFIter

    // Reduce fluxes across all MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    // Multiply sum of fluxes by face area dA
    amrex::Real face_area_element = 1.0;
    if (idir == 0) { // X-direction flow, Area = dy*dz
        face_area_element = dx[1] * dx[2];
    } else if (idir == 1) { // Y-direction flow, Area = dx*dz
        face_area_element = dx[0] * dx[2];
    } else { // Z-direction flow, Area = dx*dy
        face_area_element = dx[0] * dx[1];
    }

    // Final fluxes are sum over faces * face area element
    fxin = local_fxin * face_area_element;
    fxout = local_fxout * face_area_element;
}


} // End namespace OpenImpala
