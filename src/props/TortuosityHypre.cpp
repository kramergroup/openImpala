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
#include <HYPRE_struct_ls.h> // Includes headers for SMG, PFMG, Jacobi, PCG, GMRES, BiCGSTAB, FlexGMRES etc.
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
// (Unchanged)
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
// (Unchanged)
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
      m_eps(1e-6), m_maxiter(200),
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
    preconditionPhaseFab();

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
// (Unchanged)
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
// (Unchanged)
void OpenImpala::TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;

    // Create the grid object spanning ndim dimensions
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr);

    // Add each box from the BoxArray to the HYPRE grid
    for (int i = 0; i < m_ba.size(); ++i) {
        amrex::Box bx = m_ba[i];
        auto lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hi = OpenImpala::TortuosityHypre::hiV(bx);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  DEBUG [setupGrids]: Setting extents for Box " << bx
                            << " (from m_ba[" << i << "])"
                            << " with lo = [" << lo[0] << "," << lo[1] << "," << lo[2] << "]"
                            << " hi = [" << hi[0] << "," << hi[1] << "," << hi[2] << "]" << std::endl;
        }
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
// (Unchanged)
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{ 0, 0, 0},
                                                       {-1, 0, 0}, { 1, 0, 0},
                                                       { 0,-1, 0}, { 0, 1, 0},
                                                       { 0, 0,-1}, { 0, 0, 1}};

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Using standard 7-point stencil." << std::endl;
    }
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil);
    HYPRE_CHECK(ierr);

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

// --- Preprocess Phase Field (Example: Remove isolated spots iteratively) ---
// (Unchanged)
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();
    const int num_remspot_passes = 3; // Number of passes for remspot

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Applying tortuosity_remspot filter (" << num_remspot_passes << " passes)..." << std::endl;
    }

    for (int pass = 0; pass < num_remspot_passes; ++pass) {
        // Fill boundary/ghost cells before each pass might be necessary depending on stencil
        m_mf_phase.FillBoundary(m_geom.periodicity());

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.tilebox();
            amrex::IArrayBox& fab = m_mf_phase[mfi];
            int ncomp = fab.nComp(); // Assuming remspot might need ncomp, though likely operates on comp_phase

            // Call the Fortran routine to remove spots in-place
            tortuosity_remspot(fab.dataPtr(), fab.loVect(), fab.hiVect(), &ncomp,
                               tile_box.loVect(), tile_box.hiVect(),
                               domain_box.loVect(), domain_box.hiVect());
        } // End MFIter loop

         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "    DEBUG [preconditionPhaseFab]: Finished remspot pass " << pass + 1 << std::endl;
         }
    } // End pass loop

    // Final boundary fill after all passes are done
    m_mf_phase.FillBoundary(m_geom.periodicity());
     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  ...remspot filtering complete." << std::endl;
    }
}


// --- Setup HYPRE Matrix and Vectors, Call Fortran Fill Routine ---
// (Unchanged)
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
    ierr = HYPRE_StructMatrixInitialize(m_A);
    HYPRE_CHECK(ierr);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructVectorCreate (b)..." << std::endl;
    }
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); // Initialize RHS to zero before filling
    HYPRE_CHECK(ierr);

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructVectorCreate (x)..." << std::endl;
    }
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); // Initialize solution to zero before filling initial guess
    HYPRE_CHECK(ierr);

    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[stencil_size] = {0, 1, 2, 3, 4, 5, 6};
    const int dir_int = static_cast<int>(m_dir);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) { dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]); }


     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine..." << std::endl;
     }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());

        if (npts == 0) continue;

        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * stencil_size);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = phase_iab.box(); // Get box including ghost cells for Fortran access

        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts, p_ptr, pbox.loVect(), pbox.hiVect(),
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);

        // Check for NaNs/Infs returned from Fortran (important!)
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
        amrex::ParallelDescriptor::ReduceIntMin(global_data_ok); // Use Min reduction (1=OK, 0=FAIL)
        if (global_data_ok == 0) {
           amrex::Abort("NaN/Inf found in matrix/rhs/init_guess values returned from Fortran!");
        }

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);
        // Set the initial guess provided by Fortran
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Finished MFIter loop setting values from Fortran." << std::endl;
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling HYPRE_StructMatrixAssemble..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Matrix assembled." << std::endl;
    }

    // Assemble vectors after setting values
    ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr);
}


// --- Solve the Linear System using HYPRE ---
// <<< MODIFIED: Using Jacobi preconditioner for BiCGSTAB >>>
bool OpenImpala::TortuosityHypre::solve() {
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL; // Can represent different preconditioner types

    m_num_iterations = -1;
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();

    // --- PCG Solver ---
    if (m_solvertype == SolverType::PCG) {
        // NOTE: Using TUNED PFMG settings here based on previous attempt
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE PCG Solver with Tuned PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructPCGSetTol(solver, m_eps);
        HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPCGSetTwoNorm(solver, 1); // Use L2 norm for convergence check
        HYPRE_StructPCGSetRelChange(solver, 0); // Tol is relative to initial residual norm
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0); // Higher HYPRE verbosity if needed

        // --- Setup Tuned PFMG Preconditioner ---
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0);       // Solve to zero tolerance (relative) within precond
        HYPRE_StructPFMGSetMaxIter(precond, 1);      // Use one V-cycle (or other cycle) per application
        HYPRE_StructPFMGSetRelaxType(precond, 6);    // Tuned: Use Red-Black G-S type smoother
        HYPRE_StructPFMGSetNumPreRelax(precond, 2);  // Tuned: Increase pre-relaxation sweeps
        HYPRE_StructPFMGSetNumPostRelax(precond, 2); // Tuned: Increase post-relaxation sweeps
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG Preconditioner created and configured (tuned)." << std::endl;
        // --- End PFMG Setup ---

        // Set PFMG as the preconditioner for PCG
        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG set as preconditioner for PCG." << std::endl;

        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructPCGSetup..." << std::endl;
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructPCGSolve..." << std::endl;
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
        // Check for convergence issues
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE PCG solver did not converge within max iterations (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else if (ierr != 0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE PCG solve returned error code " << ierr << ". Possible divergence or other issue.\n";
             // HYPRE_CHECK(ierr); // Optionally abort on any error
        }

        // Get stats
        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        // Clean up
        HYPRE_StructPCGDestroy(solver);
        if (precond) HYPRE_StructPFMGDestroy(precond); // Destroy PFMG
    }
    // --- GMRES Solver ---
    else if (m_solvertype == SolverType::GMRES) {
        // NOTE: Using DEFAULT PFMG settings here
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE GMRES Solver with Default PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructGMRESSetTol(solver, m_eps);
        HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0); // Higher HYPRE verbosity
        // HYPRE_StructGMRESSetKDim(solver, k_dim);

        // --- Setup Default PFMG Preconditioner ---
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0);
        HYPRE_StructPFMGSetMaxIter(precond, 1);
        HYPRE_StructPFMGSetRelaxType(precond, 1); // Default: Weighted Jacobi
        HYPRE_StructPFMGSetNumPreRelax(precond, 1); // Default: 1 sweep
        HYPRE_StructPFMGSetNumPostRelax(precond, 1);// Default: 1 sweep
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG Preconditioner created and configured (default settings)." << std::endl;
        // --- End PFMG Setup ---

        // Set PFMG as the preconditioner for GMRES
        HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG set as preconditioner for GMRES." << std::endl;

        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructGMRESSetup..." << std::endl;
        ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructGMRESSolve..." << std::endl;
        ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
         // Check for convergence issues
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE GMRES solver did not converge within max iterations (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else if (ierr != 0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE GMRES solve returned error code " << ierr << ". Possible divergence or other issue (e.g., memory error).\n";
             // HYPRE_CHECK(ierr); // Optionally abort on any error
        }

        // Get stats
        HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        // Clean up
        HYPRE_StructGMRESDestroy(solver);
        if (precond) HYPRE_StructPFMGDestroy(precond); // Destroy PFMG
    }
    // --- FlexGMRES Solver ---
    else if (m_solvertype == SolverType::FlexGMRES) {
        // NOTE: Using DEFAULT PFMG settings here
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with Tuned PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver, m_eps);
        HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        // HYPRE_StructFlexGMRESSetKDim(solver, k_dim);

        // --- Setup Tuned PFMG Preconditioner ---
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0);
        HYPRE_StructPFMGSetMaxIter(precond, 1);
        HYPRE_StructPFMGSetRelaxType(precond, 6);   // Tuned: Red-Black G-S
        HYPRE_StructPFMGSetNumPreRelax(precond, 2); // Tuned: 2 sweeps
        HYPRE_StructPFMGSetNumPostRelax(precond, 2);// Tuned: 2 sweeps
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG Preconditioner created and configured (TUNED settings)." << std::endl;
        // --- End PFMG Setup ---

        // Set PFMG as the preconditioner for FlexGMRES
        HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG set as preconditioner for FlexGMRES." << std::endl;

        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructFlexGMRESSetup..." << std::endl;
        ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructFlexGMRESSolve..." << std::endl;
        ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
         // Check for convergence issues
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE FlexGMRES solver did not converge within max iterations (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else if (ierr != 0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE FlexGMRES solve returned error code " << ierr << ". Possible divergence or other issue.\n";
             // HYPRE_CHECK(ierr); // Optionally abort on any error
        }

        // Get stats
        HYPRE_StructFlexGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        // Clean up
        HYPRE_StructFlexGMRESDestroy(solver);
        if (precond) HYPRE_StructPFMGDestroy(precond); // Destroy PFMG
    }
    // --- BiCGSTAB Solver --- <<< SECTION MODIFIED FOR JACOBI PRECONDITIONER >>>
    else if (m_solvertype == SolverType::BiCGSTAB) {
        // *** Using Jacobi Preconditioner ***
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE BiCGSTAB Solver with Jacobi Preconditioner..." << std::endl; // <<< MODIFIED LOG MSG
        ierr = HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructBiCGSTABSetTol(solver, m_eps);
        HYPRE_StructBiCGSTABSetMaxIter(solver, m_maxiter);
        HYPRE_StructBiCGSTABSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);

        // --- Setup Jacobi Preconditioner --- // <<< MODIFIED >>>
        precond = NULL; // Ensure precond starts NULL
        ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        // Configure Jacobi - typically just need to set max iterations to 1 for use as preconditioner
        HYPRE_StructJacobiSetMaxIter(precond, 1);
        // HYPRE_StructJacobiSetTol(precond, 0.0); // Usually tolerance is 0 for preconditioner solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Jacobi Preconditioner created and configured." << std::endl; // <<< MODIFIED LOG MSG
        // --- End Jacobi Setup ---

        // Set Jacobi as preconditioner for BiCGSTAB // <<< MODIFIED >>>
        HYPRE_StructBiCGSTABSetPrecond(solver, HYPRE_StructJacobiSolve, HYPRE_StructJacobiSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Jacobi set as preconditioner for BiCGSTAB." << std::endl; // <<< MODIFIED LOG MSG

        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructBiCGSTABSetup..." << std::endl;
        ierr = HYPRE_StructBiCGSTABSetup(solver, m_A, m_b, m_x); // Setup WITH preconditioner
        HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructBiCGSTABSolve..." << std::endl;
        ierr = HYPRE_StructBiCGSTABSolve(solver, m_A, m_b, m_x); // Solve WITH preconditioner
        // Check for convergence issues
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE BiCGSTAB solver did not converge within max iterations (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else if (ierr != 0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE BiCGSTAB solve returned error code " << ierr << ". Possible divergence or other issue.\n";
             // HYPRE_CHECK(ierr); // Optionally abort on any error
        }

        // Get stats
        HYPRE_StructBiCGSTABGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        // Clean up
        HYPRE_StructBiCGSTABDestroy(solver);
        if (precond) HYPRE_StructJacobiDestroy(precond); // <<< MODIFIED >>>
    }
    // --- Jacobi Solver ---
    else if (m_solvertype == SolverType::Jacobi) {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE Jacobi Solver (NO preconditioner)..." << std::endl;
         // Jacobi does not use a separate preconditioner object
         ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
         HYPRE_CHECK(ierr);
         HYPRE_StructJacobiSetTol(solver, m_eps);
         HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
         // HYPRE_StructJacobiSetZeroGuess(solver); // Use initial guess from fillmtx
         ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
         HYPRE_CHECK(ierr);
         ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE Jacobi solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
         } else if (ierr != 0) { HYPRE_CHECK(ierr); } // Abort on other Jacobi errors
         HYPRE_StructJacobiGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         HYPRE_StructJacobiDestroy(solver);
    }
    else {
        amrex::Abort("Unsupported solver type requested in TortuosityHypre::solve: " + std::to_string(static_cast<int>(m_solvertype)));
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << m_final_res_norm << std::endl;
    }

    // --- Write plot file if requested ---
    // (Unchanged)
    if (m_write_plotfile) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Writing solution plotfile..." << std::endl;
        }
        // Create MultiFab for plotting (potential + phase)
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponents, 0); // No ghost cells needed for plotfile

        // Create temporary MultiFab to copy solution from HYPRE vector
        amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 0); // No ghost cells needed here either
        mf_soln_temp.setVal(0.0); // Initialize just in case

        std::vector<double> soln_buffer; // Buffer for HYPRE data
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
        for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
             const amrex::Box& bx = mfi.validbox(); // Use validbox for copying data
             const int npts = static_cast<int>(bx.numPts());
             if (npts == 0) continue;

             soln_buffer.resize(npts); // Resize buffer for current box size
             auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
             auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

             // Get solution data from HYPRE vector m_x
             HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
             if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed in solve() for plotfile!"); }

             // Copy data from buffer to AMReX MultiFab mf_soln_temp
             amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
             const amrex::IntVect lo = bx.smallEnd();
             const amrex::IntVect hi = bx.bigEnd();
             long long k_lin_idx = 0; // Linear index into soln_buffer
             for (int kk = lo[2]; kk <= hi[2]; ++kk) {
                 for (int jj = lo[1]; jj <= hi[1]; ++jj) {
                     for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                          if (k_lin_idx < npts) {
                               soln_arr(ii,jj,kk) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]);
                          }
                         k_lin_idx++;
                     }
                 }
             }
              if (k_lin_idx != npts) {
                   amrex::Warning("Linear index mismatch during HYPRE->AMReX copy in solve()!");
              }
        } // End MFIter loop for copying solution

        // Copy solution and phase data into the plot MultiFab
        amrex::Copy(mf_plot, mf_soln_temp, 0, SolnComp, 1, 0); // Copy solution to component 0
        amrex::Copy(mf_plot, m_mf_phase, 0, PhaseComp, 1, 0); // Copy phase ID to component 1

        // Define plotfile name and variable names
        std::string plotfilename = m_resultspath + "/tortuosity_solution";
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id"};

        // Write the plotfile
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
         if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
              amrex::Print() << "  Plotfile written to " << plotfilename << std::endl;
         }
    }

    m_first_call = false; // Mark that solve has been called
    bool converged = (!std::isnan(m_final_res_norm)) && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);
    return converged;
}


// --- Calculate Tortuosity Value (Calls Solve if needed) ---
// (Unchanged)
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
    if (m_first_call || refresh) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityHypre: Calling solve()..." << std::endl;
        }
        bool converged = solve(); // Run the solver

        // ===> Handle non-convergence BEFORE calculating flux <===
        if (!converged) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Solver did not converge (residual norm "
                                << m_final_res_norm << " > tolerance " << m_eps
                                << ", or NaN residual). Cannot calculate tortuosity. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
             return m_value; // Return NaN immediately
        }
        // ===> End non-convergence check <===

        // If converged, proceed to calculate fluxes
        amrex::Real flux_in = 0.0, flux_out = 0.0;
        global_fluxes(flux_in, flux_out); // Calculate fluxes based on converged solution m_x

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Global Flux In:  " << flux_in << std::endl;
             amrex::Print() << "  Global Flux Out: " << flux_out << std::endl;
        }

        // Calculate Tortuosity (logic remains the same, but now only runs if converged)
        if (std::abs(flux_in) < tiny_flux_threshold) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Calculated input flux is near zero (" << flux_in
                                << ") despite solver convergence. Tortuosity is ill-defined or infinite. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else if (m_vf <= 0.0) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Volume fraction is zero or negative. Tortuosity is ill-defined. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else {
            const amrex::Real* problo = m_geom.ProbLo();
            const amrex::Real* probhi = m_geom.ProbHi();
            amrex::Real area = 1.0;
            amrex::Real length_parallel = 1.0;
            int idir = static_cast<int>(m_dir);

            if (idir == 0) { // Direction::X
                area = (probhi[1] - problo[1]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[0] - problo[0]);
            } else if (idir == 1) { // Direction::Y
                area = (probhi[0] - problo[0]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[1] - problo[1]);
            } else { // Direction::Z
                area = (probhi[0] - problo[0]) * (probhi[1] - problo[1]);
                length_parallel = (probhi[2] - problo[2]);
            }

            if (std::abs(length_parallel) < std::numeric_limits<amrex::Real>::epsilon()) {
                 if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "Warning: Domain length parallel to flow direction is near zero. Tortuosity is ill-defined. Returning NaN." << std::endl;
                 }
                 m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            } else {
                amrex::Real potential_diff = m_vhi - m_vlo;
                 if (std::abs(potential_diff) < tiny_flux_threshold) {
                      if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                          amrex::Print() << "Warning: Applied potential difference (vhi - vlo) is near zero. Tortuosity is ill-defined. Returning NaN." << std::endl;
                      }
                      m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
                 } else {
                     amrex::Real potential_gradient_mag = std::abs(potential_diff) / length_parallel;
                     // Assuming intrinsic diffusivity/conductivity D_0 = 1 for the phase
                     amrex::Real effective_diffusivity = flux_in / (area * potential_gradient_mag);

                     if (effective_diffusivity <= 0.0) {
                          if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                              amrex::Print() << "Warning: Calculated effective diffusivity is zero or negative (" << effective_diffusivity
                                             << "). Check flux direction relative to potential gradient. Returning NaN." << std::endl;
                          }
                          m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
                     } else {
                         // Tortuosity = VolumeFraction / EffectiveDiffusivity (assuming D_intrinsic = 1)
                        m_value = m_vf / effective_diffusivity;
                     }
                 }
            }
        }

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << std::fixed << std::setprecision(8)
                            << "TortuosityHypre: Calculated tortuosity = " << m_value << std::endl;
        }
    }
    return m_value;
}


// --- Get Solution Field (Not fully implemented) ---
// (Unchanged)
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) {
    amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
}

// --- Get Cell Types (Not implemented) ---
// (Unchanged)
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
     amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
}


// --- Calculate Global Fluxes Across Domain Boundaries ---
// (Unchanged)
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    fxin = 0.0;
    fxout = 0.0;
    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);

    // Need 1 ghost cell for finite differencing flux at boundary
    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1);
    mf_soln_temp.setVal(0.0); // Initialize

    // Copy solution from HYPRE vector to AMReX MultiFab with ghost cells
    std::vector<double> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox(); // Get data for the valid region
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;
        soln_buffer.resize(npts);
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
        if (get_ierr != 0) {
             char hypre_error_msg[256] = "Unknown HYPRE Error";
             HYPRE_DescribeError(get_ierr, hypre_error_msg);
             amrex::Warning("HYPRE_StructVectorGetBoxValues failed in global_fluxes! Error: " + std::string(hypre_error_msg) + " (Code: " + std::to_string(get_ierr) + ")");
        }

        amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
        const amrex::IntVect lo = bx.smallEnd();
        const amrex::IntVect hi = bx.bigEnd();
        long long k_lin_idx = 0;
        for (int kk = lo[2]; kk <= hi[2]; ++kk) {
            for (int jj = lo[1]; jj <= hi[1]; ++jj) {
                for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                     if (k_lin_idx < npts) {
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
    // Fill ghost cells of the solution MultiFab using domain periodicity (or boundary conditions if needed)
    mf_soln_temp.FillBoundary(m_geom.periodicity()); // Use domain periodicity info

    // --- Calculate flux using finite differences on mf_soln_temp ---
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Iterate over tiles
        // Need const access to phase and solution arrays within the tile
        const auto phase = m_mf_phase.const_array(mfi);
        const auto soln = mf_soln_temp.const_array(mfi); // Use the solution MF with ghost cells

        // Box defining the low boundary face for flux calculation within this tile
        amrex::Box lobox = amrex::adjCellLo(domain, idir, 1); // Cells just inside low domain boundary
        lobox &= bx; // Intersect with current tile box

        // Box defining the high boundary face for flux calculation
        amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);
        amrex::Box hibox_ghost_cell = amrex::adjCellHi(domain, idir, 1); // Cell just outside high boundary
        amrex::Box hibox_internal_cell = hibox_ghost_cell;
        hibox_internal_cell.shift(shift * (-1)); // Cell just inside high boundary
        hibox_internal_cell &= bx; // Intersect with current tile box

        amrex::Real grad, flux;
        amrex::IntVect iv; // Reusable IntVect

        // Calculate flux entering at the low boundary (Dirichlet face)
        const amrex::IntVect lo_flux = lobox.smallEnd();
        const amrex::IntVect hi_flux = lobox.bigEnd();
        for (int k = lo_flux[2]; k <= hi_flux[2]; ++k) {
             iv[2]=k;
             for (int j = lo_flux[1]; j <= hi_flux[1]; ++j) {
                 iv[1]=j;
                 for (int i = lo_flux[0]; i <= hi_flux[0]; ++i) {
                      iv[0]=i;
                      if (phase(iv, PhaseComp) == m_phase) { // Is it conductive phase?
                           grad = (soln(iv) - soln(iv - shift)) / dx[idir];
                           flux = -grad; // Assumes D=1
                           local_fxin += flux;
                      }
                 }
             }
        }

        // Calculate flux exiting at the high boundary (Dirichlet face)
        const amrex::IntVect lo_flux_hi = hibox_internal_cell.smallEnd();
        const amrex::IntVect hi_flux_hi = hibox_internal_cell.bigEnd();
        for (int k = lo_flux_hi[2]; k <= hi_flux_hi[2]; ++k) {
            iv[2]=k;
            for (int j = lo_flux_hi[1]; j <= hi_flux_hi[1]; ++j) {
                iv[1]=j;
                for (int i = lo_flux_hi[0]; i <= hi_flux_hi[0]; ++i) {
                      iv[0]=i;
                      if (phase(iv, PhaseComp) == m_phase) { // Is it conductive phase?
                           grad = (soln(iv + shift) - soln(iv)) / dx[idir];
                           flux = -grad; // Assumes D=1
                           local_fxout += flux;
                      }
                }
            }
        }
    } // End MFIter

    // Reduce fluxes across all processors
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    // Scale by face area element (dx*dy, dx*dz, or dy*dz)
    amrex::Real face_area_element = 1.0;
    if (idir == 0) { // X-direction flux -> YZ face area
        face_area_element = dx[1] * dx[2];
    } else if (idir == 1) { // Y-direction flux -> XZ face area
        face_area_element = dx[0] * dx[2];
    } else { // Z-direction flux -> XY face area
        face_area_element = dx[0] * dx[1];
    }

    // Assign final global fluxes
    fxin = local_fxin * face_area_element;
    fxout = local_fxout * face_area_element;
}


} // End namespace OpenImpala
