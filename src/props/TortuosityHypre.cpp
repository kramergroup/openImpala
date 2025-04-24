#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"     // For tortuosity_remspot, tortuosity_filct
#include "TortuosityHypreFill_F.H"  // For tortuosity_fillmtx

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

// HYPRE includes
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

// Define HYPRE error checking macro (Keep this robust version)
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
    constexpr int numComponents = 2;
    constexpr amrex::Real tiny_flux_threshold = 1.e-15;
    // Define stencil size (standard 7-point for 3D)
    constexpr int stencil_size = 7;
}

// Helper Functions
namespace OpenImpala {

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

// Constructor Implementation
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
      m_mf_phase(mf_phase_input, amrex::make_alias, 0, mf_phase_input.nComp()),
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      m_eps(1e-9), m_maxiter(200),
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose),
      m_write_plotfile(write_plotfile),
      m_mf_phi(ba, dm, numComponents, 1), // Ensure enough ghost cells if needed later
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL)
{
    // Constructor body remains the same as before...
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initializing..." << std::endl;
    }
    amrex::ParmParse pp("hypre");
    pp.query("eps", m_eps);
    pp.query("maxiter", m_maxiter);
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("verbose", m_verbose);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  HYPRE Params: eps=" << m_eps << ", maxiter=" << m_maxiter << std::endl;
         amrex::Print() << "  Class Verbose Level: " << m_verbose << std::endl;
         amrex::Print() << "  Write Plotfile Flag: " << m_write_plotfile << std::endl;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0, "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf_phase_input.nGrow() >= 1, "Input phase iMultiFab needs at least 1 ghost cell");

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab..." << std::endl;
    preconditionPhaseFab();

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil(); // <-- This will now use the restored 7-point version

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation(); // <-- This will now use the restored version

     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
     }
}

// Destructor Implementation
OpenImpala::TortuosityHypre::~TortuosityHypre()
{
    // Destructor body remains the same...
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A); // This might still crash based on standalone test
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    m_x = m_b = NULL;
    m_A = NULL;
    m_stencil = NULL;
    m_grid = NULL;
}

// setupGrids Implementation
void OpenImpala::TortuosityHypre::setupGrids()
{
    // setupGrids body remains the same...
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr);
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
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Calling HYPRE_StructGridAssemble..." << std::endl;
    }
    ierr = HYPRE_StructGridAssemble(m_grid);
    HYPRE_CHECK(ierr);
    // Add explicit check after assemble
    if (!m_grid) { amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!"); }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Grid setup complete. m_grid pointer: " << m_grid << std::endl;
    }
}


/**
 * @brief Sets up the HYPRE StructStencil.
 * *** RESTORED: Uses a standard 7-point stencil for 3D Laplacian ***
 */
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    // Standard 7-point stencil offsets for 3D Laplacian: (center, W, E, S, N, B, T)
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{ 0, 0, 0},
                                                       {-1, 0, 0}, { 1, 0, 0},
                                                       { 0,-1, 0}, { 0, 1, 0},
                                                       { 0, 0,-1}, { 0, 0, 1}};

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Using standard 7-point stencil." << std::endl;
    }

    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil);
    HYPRE_CHECK(ierr);

    // Set stencil entries
    for (int i = 0; i < stencil_size; i++)
    {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
              amrex::Print() << "  DEBUG [setupStencil]: Setting element " << i
                             << " with offset = [" << offsets[i][0] << "," << offsets[i][1] << "," << offsets[i][2] << "]" << std::endl;
         }
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]);
        HYPRE_CHECK(ierr);
    }

    // Add explicit check after creation
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL after HYPRE_StructStencilCreate/SetElement!"); }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Stencil setup complete. m_stencil pointer: " << m_stencil << std::endl;
    }
}

// preconditionPhaseFab Implementation
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    // preconditionPhaseFab body remains the same...
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");
    m_mf_phase.FillBoundary(m_geom.periodicity());
    const amrex::Box& domain_box = m_geom.Domain();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tile_box = mfi.tilebox();
        amrex::IArrayBox& fab = m_mf_phase[mfi];
        int ncomp = fab.nComp(); // Should be 1 for phase data
        tortuosity_remspot(fab.dataPtr(), fab.loVect(), fab.hiVect(), &ncomp,
                           tile_box.loVect(), tile_box.hiVect(),
                           domain_box.loVect(), domain_box.hiVect());
    }
    m_mf_phase.FillBoundary(m_geom.periodicity());
}

// setupMatrixEquation Implementation
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    // *** RESTORED: Uses 7-point stencil and calls Fortran routine ***
    HYPRE_Int ierr = 0;

    // --- Add explicit checks for grid and stencil handles ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Checking handles before HYPRE_StructMatrixCreate..." << std::endl;
        amrex::Print() << "    m_grid pointer:    " << m_grid << std::endl;
        amrex::Print() << "    m_stencil pointer: " << m_stencil << std::endl;
    }
    if (!m_grid)    { amrex::Abort("FATAL: m_grid handle is NULL before HYPRE_StructMatrixCreate!"); }
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL before HYPRE_StructMatrixCreate!"); }
    // --- End explicit checks ---


    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructMatrixCreate..." << std::endl;
    }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    HYPRE_CHECK(ierr); // This is the line that was failing (Line 241 in previous context)

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
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); // Initialize RHS to zero
    HYPRE_CHECK(ierr);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructVectorCreate (x)..." << std::endl;
    }
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); // Initialize solution guess to zero
    HYPRE_CHECK(ierr);

    const amrex::Box& domain = m_geom.Domain();
    // Stencil indices corresponding to the 7-point stencil definition in setupStencil
    // {center, W, E, S, N, B, T}
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

        // Allocate space for matrix values (7 entries per point) and RHS/guess
        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * stencil_size);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts); // Can be used or ignored by Fortran

        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr();
        // Need the box corresponding to the FAB, including ghost cells, for the Fortran routine
        const auto& pbox = phase_iab.box();

        // *** RESTORED CALL to Fortran routine ***
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts, p_ptr, pbox.loVect(), pbox.hiVect(),
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);

        // NaN/Inf Check (still useful)
        bool data_ok = true;
        for(int idx = 0; idx < npts; ++idx) {
            if (std::isnan(rhs_values[idx]) || std::isinf(rhs_values[idx])) { data_ok = false; break; }
            for (int s_idx = 0; s_idx < stencil_size; ++s_idx) {
                 if (std::isnan(matrix_values[static_cast<size_t>(idx)*stencil_size + s_idx]) ||
                     std::isinf(matrix_values[static_cast<size_t>(idx)*stencil_size + s_idx])) {
                    data_ok = false; break;
                 }
            }
            if (!data_ok) break;
        }
        int global_data_ok = data_ok;
        amrex::ParallelDescriptor::ReduceIntMin(global_data_ok);
        if (global_data_ok == 0) { amrex::Abort("NaN/Inf found in matrix/rhs values returned from Fortran!"); }


        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // Set matrix values for the current box and all stencil entries
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        // Set RHS values for the current box
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);
        // Set initial guess values for the current box
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }

    // Matrix Assembly Section
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Finished MFIter loop setting values from Fortran." << std::endl;
    }

    // Optional: Matrix print attempt (might be useful, but could be large)
    // std::string matrix_debug_filename = "debug_matrix_state_7pt.log";
    // if (amrex::ParallelDescriptor::IOProcessor()) {
    //     std::cout << "[Rank " << amrex::ParallelDescriptor::MyProc()
    //               << "] Attempting HYPRE_StructMatrixPrint to " << matrix_debug_filename << "..." << std::endl << std::flush;
    //     HYPRE_Int print_ierr = HYPRE_StructMatrixPrint(matrix_debug_filename.c_str(), m_A, 0);
    //     if (print_ierr != 0) { /* ... handle print error ... */ }
    //     else { /* ... print success ... */ }
    // }
    // amrex::ParallelDescriptor::Barrier();

    // Finalize matrix assembly
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling HYPRE_StructMatrixAssemble..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr); // Check assembly result

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Matrix assembled." << std::endl;
     }
}


// --- solve(), value(), getSolution(), getCellTypes(), global_fluxes() ---
// Implementations remain the same as before.
// (Skipped for brevity)
bool OpenImpala::TortuosityHypre::solve() {
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL; // Use NULL for no preconditioner initially

    if (m_solvertype == SolverType::PCG) {
        ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructPCGSetTol(solver, m_eps);
        HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPCGSetTwoNorm(solver, 1); // Use 2-norm for convergence check
        HYPRE_StructPCGSetRelChange(solver, 0); // Use relative residual norm
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 2 : 0); // Print iterations if verbose >= 2
        // Setup Preconditioner (Example: SMG)
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetMemoryUse(precond, 0);
        HYPRE_StructSMGSetMaxIter(precond, 1); // Use as preconditioner (1 iteration)
        HYPRE_StructSMGSetTol(precond, 0.0);   // Tolerate 0 for preconditioning
        HYPRE_StructSMGSetZeroGuess(precond);
        HYPRE_StructSMGSetNumPreRelax(precond, 1);
        HYPRE_StructSMGSetNumPostRelax(precond, 1);
        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
        // Setup Solver
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        // Solve
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
        // Check for convergence issues (specific error codes)
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE PCG solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else {
             HYPRE_CHECK(ierr); // Check for other errors
        }
        // Get stats
        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Destroy
        HYPRE_StructPCGDestroy(solver);
        if (precond) HYPRE_StructSMGDestroy(precond);

    } else if (m_solvertype == SolverType::GMRES) {
        ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructGMRESSetTol(solver, m_eps);
        HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructGMRESSetPrintLevel(solver, m_verbose > 1 ? 2 : 0);
        // Setup Preconditioner (Example: SMG)
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetMemoryUse(precond, 0);
        HYPRE_StructSMGSetMaxIter(precond, 1);
        HYPRE_StructSMGSetTol(precond, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond);
        HYPRE_StructSMGSetNumPreRelax(precond, 1);
        HYPRE_StructSMGSetNumPostRelax(precond, 1);
        HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
        // Setup Solver
        ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        // Solve
        ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE GMRES solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else {
             HYPRE_CHECK(ierr);
        }
        // Get stats
        HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Destroy
        HYPRE_StructGMRESDestroy(solver);
        if (precond) HYPRE_StructSMGDestroy(precond);

    } else if (m_solvertype == SolverType::FlexGMRES) {
         // Similar setup for FlexGMRES...
         amrex::Abort("FlexGMRES not fully implemented yet in TortuosityHypre::solve");
         // HYPRE_StructFlexGMRESCreate...
         // ... setup ...
         // HYPRE_StructFlexGMRESDestroy...
    } else if (m_solvertype == SolverType::Jacobi) {
         // Use StructJacobi solver (often used as a smoother/preconditioner, maybe not ideal as standalone)
         ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
         HYPRE_CHECK(ierr);
         HYPRE_StructJacobiSetTol(solver, m_eps);
         HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
         HYPRE_StructJacobiSetZeroGuess(solver, 1); // Assume zero initial guess
         // Setup
         ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
         HYPRE_CHECK(ierr);
         // Solve
         ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) {
              if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE Jacobi solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
         } else {
              HYPRE_CHECK(ierr);
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

    // Optional: Write plotfile if requested
    if (m_write_plotfile) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Writing solution plotfile..." << std::endl;
        }
        // Need to copy solution from m_x (HYPRE vector) back to an AMReX MultiFab
        // This requires iterating and using HYPRE_StructVectorGetBoxValues
        // For now, just copy the phase data and the initial guess (which is 0)
        // TODO: Implement proper copy back from m_x if plotfile writing is needed
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponents, 0); // 0 ghost cells for plotfile
        amrex::Copy(mf_plot, m_mf_phi, 0, SolnComp, 1, 0); // Copy initial guess (0) to SolnComp
        amrex::Copy(mf_plot, m_mf_phase, 0, PhaseComp, 1, 0); // Copy phase data to PhaseComp

        std::string plotfilename = m_resultspath + "/tortuosity_solution";
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
    }

    m_first_call = false; // Mark that solve has been called
    return (m_final_res_norm <= m_eps); // Return true if converged
}

amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
    if (m_first_call || refresh) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TortuosityHypre: Calling solve()..." << std::endl;
        }
        bool converged = solve();
        if (!converged && m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Warning: Solver did not converge. Tortuosity value may be inaccurate." << std::endl;
        }

        // Calculate fluxes
        amrex::Real flux_in = 0.0, flux_out = 0.0;
        global_fluxes(flux_in, flux_out);

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Global Flux In:  " << flux_in << std::endl;
             amrex::Print() << "  Global Flux Out: " << flux_out << std::endl;
        }

        // Calculate Tortuosity = VolumeFraction / EffectiveDiffusivity
        // EffectiveDiffusivity = Flux / (Area * Grad(Potential))
        // Area = L_perp1 * L_perp2
        // Grad(Potential) = (Vhi - Vlo) / L_parallel
        // Flux = flux_in (should equal -flux_out for steady state)

        if (std::abs(flux_in) < tiny_flux_threshold) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Calculated input flux is near zero (" << flux_in << "). Tortuosity is ill-defined or infinite. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else if (m_vf <= 0.0) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                  amrex::Print() << "Warning: Volume fraction is zero or negative. Tortuosity is ill-defined. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        }
        else {
            const amrex::Real* problo = m_geom.ProbLo();
            const amrex::Real* probhi = m_geom.ProbHi();
            amrex::Real area = 1.0;
            amrex::Real length_parallel = 1.0;

            if (m_dir == Direction::X) {
                area = (probhi[1] - problo[1]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[0] - problo[0]);
            } else if (m_dir == Direction::Y) {
                area = (probhi[0] - problo[0]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[1] - problo[1]);
            } else { // Direction::Z
                area = (probhi[0] - problo[0]) * (probhi[1] - problo[1]);
                length_parallel = (probhi[2] - problo[2]);
            }

            amrex::Real potential_gradient = (m_vhi - m_vlo) / length_parallel;
            if (std::abs(potential_gradient) < tiny_flux_threshold) {
                 if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "Warning: Potential gradient is near zero. Tortuosity is ill-defined. Returning NaN." << std::endl;
                 }
                 m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            } else {
                 amrex::Real effective_diffusivity = flux_in / (area * potential_gradient);
                 if (effective_diffusivity <= 0.0) {
                     if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                         amrex::Print() << "Warning: Calculated effective diffusivity is zero or negative (" << effective_diffusivity << "). Tortuosity is ill-defined or infinite. Returning NaN." << std::endl;
                     }
                     m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
                 } else {
                     m_value = m_vf / effective_diffusivity;
                 }
            }
        }
         if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "TortuosityHypre: Calculated tortuosity = " << m_value << std::endl;
         }
    }
    return m_value;
}

void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) {
     // TODO: Implement copying solution from HYPRE vector m_x to AMReX MultiFab soln
     amrex::Abort("TortuosityHypre::getSolution not implemented yet!");
}

void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
     // TODO: Implement copying cell types (phase info) if needed
     amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
}

void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    // Calculate fluxes across the low and high boundaries in the specified direction
    fxin = 0.0;
    fxout = 0.0;
    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir); // 0 for X, 1 for Y, 2 for Z

    // Create temporary MultiFab to hold solution copied from HYPRE vector
    // This is inefficient but necessary until getSolution is properly implemented
    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1); // Need 1 ghost cell for gradient calc
    mf_soln_temp.setVal(0.0); // Initialize

    // --- Copy solution from HYPRE vector m_x to mf_soln_temp ---
    // This requires iterating over boxes and using HYPRE_StructVectorGetBoxValues
    // Allocate buffer for each box
    std::vector<double> soln_buffer;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) { // Use non-tiling iterator
        const amrex::Box& bx = mfi.validbox(); // Get the valid box for this FAB
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        // Resize buffer for the current box's data
        soln_buffer.resize(npts);

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // Get values from HYPRE vector for this box
        HYPRE_Int ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
        // We probably shouldn't abort here, but maybe log a warning if needed
        if (ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed in global_fluxes!"); }

        // Copy data from buffer to the MultiFab FAB
        amrex::FArrayBox& soln_fab = mf_soln_temp[mfi];
        // AMReX Array4 for easier access
        auto soln_arr = soln_fab.array();
        size_t k = 0; // Index for the flat buffer
        AMREX_HOST_DEVICE_FOR_BOX(bx, idx) {
             soln_arr(idx) = soln_buffer[k++];
        }
    }
    // --- End copy from HYPRE vector ---

    // Fill ghost cells for the solution MultiFab
    mf_soln_temp.FillBoundary(m_geom.periodicity());


    // Now calculate fluxes using the mf_soln_temp and m_mf_phase
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        const amrex::IArrayBox& phasefab = m_mf_phase[mfi];
        const amrex::FArrayBox& solnfab = mf_soln_temp[mfi]; // Use the copied solution
        const auto phase = phasefab.array();
        const auto soln = solnfab.array();

        amrex::Box lobox = amrex::adjCellLo(domain, idir, 1); // Cells adjacent to low boundary
        lobox &= bx; // Intersect with current tile box

        amrex::Box hibox = amrex::adjCellHi(domain, idir, 1); // Cells adjacent to high boundary
        hibox &= bx; // Intersect with current tile box

        amrex::Real grad, flux;
        amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir); // Shift in flux direction

        // Flux across low boundary (idir direction) - Central difference
        AMREX_HOST_DEVICE_FOR_BOX(lobox, i) {
            if (phase(i) == m_phase) { // Check if cell is in the phase of interest
                 // Gradient at face between i-shift and i
                 grad = (soln(i) - soln(i - shift)) / dx[idir];
                 flux = -grad; // Fick's law with D=1
                 local_fxin += flux;
            }
        }

        // Flux across high boundary (idir direction) - Central difference
        AMREX_HOST_DEVICE_FOR_BOX(hibox, i) {
            if (phase(i) == m_phase) { // Check if cell is in the phase of interest
                 // Gradient at face between i and i+shift
                 grad = (soln(i + shift) - soln(i)) / dx[idir];
                 flux = -grad; // Fick's law with D=1
                 local_fxout += flux;
            }
        }
    } // End MFIter

    // Sum fluxes across all processes
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout, amrex::ParallelDescriptor::IOProcessorNumber());

    // Multiply by face area perpendicular to flux direction
    amrex::Real area = 1.0;
    const amrex::Real* problo = m_geom.ProbLo();
    const amrex::Real* probhi = m_geom.ProbHi();
    if (idir == 0) { // X-direction
        area = (probhi[1] - problo[1]) * (probhi[2] - problo[2]) / (domain.length(1) * domain.length(2));
    } else if (idir == 1) { // Y-direction
        area = (probhi[0] - problo[0]) * (probhi[2] - problo[2]) / (domain.length(0) * domain.length(2));
    } else { // Z-direction
        area = (probhi[0] - problo[0]) * (probhi[1] - problo[1]) / (domain.length(0) * domain.length(1));
    }

    fxin = local_fxin * area;
    fxout = local_fxout * area;
}


} // End namespace OpenImpala
