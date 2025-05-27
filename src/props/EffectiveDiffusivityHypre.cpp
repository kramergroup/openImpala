// --- EffectiveDiffusivityHypre.cpp ---

#include "EffectiveDiffusivityHypre.H"
// Assuming your new Fortran interface header will be:
#include "EffDiffFillMtx_F.H" // For effdiff_fillmtx (needs to be created)

#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <sstream>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_VisMF.H>         // For debug writes
#include <AMReX_Loop.H>          // For amrex::LoopOnCpu

// HYPRE includes
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

// MPI include
#include <mpi.h>

// HYPRE_CHECK macro
#define HYPRE_CHECK(ierr) do { \
    if (ierr != 0) { \
        char hypre_error_msg[256]; \
        HYPRE_DescribeError(ierr, hypre_error_msg); \
        amrex::Abort("HYPRE Error: " + std::string(hypre_error_msg) + \
                     " - Error Code: " + std::to_string(ierr) + \
                     " File: " + __FILE__ + " Line: " + std::to_string(__LINE__)); \
    } \
} while (0)

// Constants namespace
namespace {
    constexpr int ChiComp = 0;     // Component for the corrector function chi
    constexpr int MaskComp = 0;    // Component for the phase mask
    constexpr int numComponentsChi = 1; // Solving for one scalar chi_k at a time
    constexpr int stencil_size = 7;
    constexpr int cell_inactive = 0; // Represents solid phase (D=0)
    constexpr int cell_active = 1;   // Represents pore phase (D=1, or D_material)
    // Stencil entry indices (0-based for HYPRE)
    constexpr int istn_c  = 0; // Center
    constexpr int istn_mx = 1; // Minus X (West)
    constexpr int istn_px = 2; // Plus X  (East)
    constexpr int istn_my = 3; // Minus Y (South)
    constexpr int istn_py = 4; // Plus Y  (North)
    constexpr int istn_mz = 5; // Minus Z (Bottom)
    constexpr int istn_pz = 6; // Plus Z  (Top)
}

namespace OpenImpala {

// Helper to convert AMReX Box lo/hi to HYPRE_Int arrays
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> EffectiveDiffusivityHypre::loV (const amrex::Box& b) {
    const int* lo_ptr = b.loVect();
    amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hypre_lo;
    for (int i=0; i<AMREX_SPACEDIM; ++i) hypre_lo[i] = static_cast<HYPRE_Int>(lo_ptr[i]);
    return hypre_lo;
}
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> EffectiveDiffusivityHypre::hiV (const amrex::Box& b) {
    const int* hi_ptr = b.hiVect();
    amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hypre_hi;
    for (int i=0; i<AMREX_SPACEDIM; ++i) hypre_hi[i] = static_cast<HYPRE_Int>(hi_ptr[i]);
    return hypre_hi;
}

// --- Constructor ---
EffectiveDiffusivityHypre::EffectiveDiffusivityHypre(
    const amrex::Geometry& geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    const amrex::iMultiFab& mf_phase_input, // Input segmented image
    const int phase_id,                     // ID of the phase where D=D_material (e.g., 1)
    const OpenImpala::Direction dir_of_chi_k, // Direction k for current chi_k (X, Y, or Z)
    const SolverType solver_type,
    const std::string& resultspath,
    int verbose_level,
    bool write_plotfile_flag)
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase_original(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()), // Store a copy
      m_phase_id(phase_id),
      m_dir_solve(dir_of_chi_k),
      m_solvertype(solver_type),
      m_eps(1e-9), m_maxiter(1000), // Default, can be overridden by ParmParse
      m_resultspath(resultspath),
      m_verbose(verbose_level),
      m_write_plotfile(write_plotfile_flag),
      m_mf_chi(ba, dm, numComponentsChi, 1), // For storing the solution chi_k, 1 ghost cell for gradient calculation later
      m_mf_active_mask(ba, dm, 1, 1),      // Mask where D=D_material, 1 ghost cell for stencil
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL),
      m_num_iterations(-1),
      m_final_res_norm(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_converged(false)
{
    BL_PROFILE("EffectiveDiffusivityHypre::Constructor");

    // Copy input phase data
    amrex::Copy(m_mf_phase_original, mf_phase_input, 0, 0, m_mf_phase_original.nComp(), m_mf_phase_original.nGrow());
    // Ensure ghost cells are filled for the original phase data if needed by generateActiveMask
    m_mf_phase_original.FillBoundary(m_geom.periodicity());


    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Initializing for chi_k in direction "
                       << static_cast<int>(m_dir_solve) << "..." << std::endl;
    }

    amrex::ParmParse pp_hypre("hypre"); // Standard HYPRE params
    pp_hypre.query("eps", m_eps);
    pp_hypre.query("maxiter", m_maxiter);

    // Specific ParmParse for this solver if needed, e.g., "effdiff"
    // amrex::ParmParse pp_effdiff("effdiff");
    // pp_effdiff.query("verbose", m_verbose); // Could override general verbose

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");

    // Store cell sizes, needed for RHS in Fortran and D_eff calculation
    const amrex::Real* dx_tmp = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) {
        m_dx[i] = dx_tmp[i];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_dx[i] > 0.0, "Cell size must be positive.");
    }

    // Generate the active mask (D=1 in phase_id, D=0 otherwise)
    // This is simpler than TortuosityHypre's flood fill version
    generateActiveMask(); // Uses m_mf_phase_original and m_phase_id

    // Check if there's any active phase. If not, solving is pointless.
    long num_active_cells = m_mf_active_mask.sum(MaskComp, true); // Sum over valid cells
    if (num_active_cells == 0) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "WARNING: No active cells found for phase_id " << m_phase_id
                           << ". Effective diffusivity will likely be zero or undefined." << std::endl;
        }
        // Set converged to false, chi will be zero, D_eff calculation will handle this.
        m_converged = false; // Though technically no solve was attempted.
        m_mf_chi.setVal(0.0); // Set chi to zero
        // No HYPRE setup needed
        return;
    }


    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "EffectiveDiffusivityHypre: Setting up HYPRE structures..." << std::endl;
    setupGrids();
    setupStencil();
    setupMatrixEquation(); // This will call the new Fortran kernel

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Initialization complete." << std::endl;
    }
}

// --- Destructor ---
EffectiveDiffusivityHypre::~EffectiveDiffusivityHypre()
{
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    m_x = m_b = NULL; m_A = NULL; m_stencil = NULL; m_grid = NULL;
}

// --- generateActiveMask ---
// Creates a binary mask: 1 where mf_phase_original == phase_id, 0 otherwise.
void EffectiveDiffusivityHypre::generateActiveMask()
{
    BL_PROFILE("EffectiveDiffusivityHypre::generateActiveMask");
    AMREX_ASSERT(m_mf_active_mask.nGrow() >= m_mf_phase_original.nGrow()); // Ensure enough ghost cells

    // Ensure original phase data has filled boundaries if it has ghost cells
    // This should have been done in constructor after copy if m_mf_phase_original has nGrow > 0
    // m_mf_phase_original.FillBoundary(m_geom.periodicity()); // Redundant if done in constructor

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    #endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Iterate over cells including ghost cells if needed by stencil
        amrex::Array4<int> const mask_arr = m_mf_active_mask.array(mfi);
        amrex::Array4<const int> const phase_arr = m_mf_phase_original.const_array(mfi);

        amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept
        {
            // Assuming phase data is in component 0 of m_mf_phase_original
            if (phase_arr(i,j,k,0) == m_phase_id) {
                mask_arr(i,j,k,MaskComp) = cell_active;
            } else {
                mask_arr(i,j,k,MaskComp) = cell_inactive;
            }
        });
    }
    // Fill ghost cells of the newly created mask using periodicity
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        long active_count = m_mf_active_mask.sum(MaskComp, true); // Sum over valid region
        amrex::Print() << "  Active mask generated. Number of active cells: " << active_count << std::endl;
    }
}


// --- setupGrids ---
// Sets up the HYPRE_StructGrid based on the BoxArray.
// This is identical to TortuosityHypre.
void EffectiveDiffusivityHypre::setupGrids()
{
    BL_PROFILE("EffectiveDiffusivityHypre::setupGrids");
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid); HYPRE_CHECK(ierr);
    for (int i = 0; i < m_ba.size(); ++i) {
        if (m_dm[i] == amrex::ParallelDescriptor::MyProc()) { // Process only local boxes
            amrex::Box bx = m_ba[i]; // Valid box for this MPI rank
            auto lo = EffectiveDiffusivityHypre::loV(bx);
            auto hi = EffectiveDiffusivityHypre::hiV(bx);
            if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  setupGrids: Rank " << amrex::ParallelDescriptor::MyProc()
                               << " adding box " << bx << std::endl;
            }
            ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data()); HYPRE_CHECK(ierr);
        }
    }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupGrids: Calling HYPRE_StructGridAssemble..." << std::endl;
    }
    ierr = HYPRE_StructGridAssemble(m_grid); HYPRE_CHECK(ierr);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_grid != NULL, "m_grid is NULL after HYPRE_StructGridAssemble!");
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupGrids: Assemble complete." << std::endl;
    }
}

// --- setupStencil ---
// Defines the 7-point stencil for HYPRE.
// This is identical to TortuosityHypre.
void EffectiveDiffusivityHypre::setupStencil()
{
    BL_PROFILE("EffectiveDiffusivityHypre::setupStencil");
    HYPRE_Int ierr = 0;
    // Standard 7-point stencil offsets
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {
        { 0, 0, 0},  // Center
        {-1, 0, 0},  // West
        { 1, 0, 0},  // East
        { 0,-1, 0},  // South
        { 0, 1, 0},  // North
        { 0, 0,-1},  // Bottom
        { 0, 0, 1}   // Top
    };
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupStencil: Creating " << stencil_size << "-point stencil..." << std::endl;
    }
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil); HYPRE_CHECK(ierr);
    for (int i = 0; i < stencil_size; i++) {
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]); HYPRE_CHECK(ierr);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stencil != NULL, "m_stencil is NULL after HYPRE_StructStencilCreate/SetElement!");
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupStencil: Complete." << std::endl;
    }
}

// --- setupMatrixEquation ---
// Sets up the HYPRE matrix A, RHS vector b, and initial guess x.
// This will call the new Fortran kernel for the cell problem.
void EffectiveDiffusivityHypre::setupMatrixEquation()
{
    BL_PROFILE("EffectiveDiffusivityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Creating HYPRE Matrix and Vectors..." << std::endl;
    }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructMatrixInitialize(m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); HYPRE_CHECK(ierr); // Initialize RHS to 0
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); HYPRE_CHECK(ierr); // Initialize solution chi_k to 0

    const amrex::Box& domain = m_geom.Domain();
    // Stencil indices for Fortran (HYPRE expects 0 to nstencil-1)
    int stencil_indices_hypre[stencil_size];
    for(int i=0; i<stencil_size; ++i) stencil_indices_hypre[i] = i;

    const int current_dir_int = static_cast<int>(m_dir_solve); // Pass X=0, Y=1, Z=2 to Fortran

    // Ensure active mask has filled ghost cells, as Fortran kernel might access neighbors
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Calling Fortran kernel 'effdiff_fillmtx'..." << std::endl;
    }

    std::vector<amrex::Real> matrix_values_buffer;
    std::vector<amrex::Real> rhs_values_buffer;
    std::vector<amrex::Real> initial_guess_buffer; // chi_k initial guess

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) \
                 private(matrix_values_buffer, rhs_values_buffer, initial_guess_buffer)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Operate on tilebox to fill matrix corresponding to valid cells
        const int npts_tile = static_cast<int>(bx.numPts()); // Number of points in the tilebox
        if (npts_tile == 0) continue;

        // Resize buffers for the current tile
        // The Fortran kernel will fill these for cells within bx that are part of mfi.validbox()
        // The HYPRE SetBoxValues will use mfi.validbox()
        const amrex::Box& valid_bx = mfi.validbox();
        const int npts_valid = static_cast<int>(valid_bx.numPts());
        if (npts_valid == 0) continue;

        matrix_values_buffer.resize(static_cast<size_t>(npts_valid) * stencil_size);
        rhs_values_buffer.resize(npts_valid);
        initial_guess_buffer.resize(npts_valid); // For chi_k, usually 0

        // Get pointers to underlying data for the Fortran kernel
        // The kernel will iterate over valid_bx
        const amrex::IArrayBox& mask_fab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_fab.dataPtr(MaskComp);
        const auto& mask_box_bounds = mask_fab.box(); // Box of the FAB itself (might include ghosts)

        // Call the Fortran kernel
        // It needs to know the bounds of the mask_fab it receives (mask_box_bounds)
        // and the specific region (valid_bx) it should fill coefficients for.
        effdiff_fillmtx(
            matrix_values_buffer.data(), rhs_values_buffer.data(), initial_guess_buffer.data(),
            &npts_valid, // Number of cells in valid_bx
            mask_ptr, mask_box_bounds.loVect(), mask_box_bounds.hiVect(), // Active mask data and its bounds
            valid_bx.loVect(), valid_bx.hiVect(),      // Region to fill (current valid box)
            domain.loVect(), domain.hiVect(),          // Overall domain bounds
            m_dx.dataPtr(),                               // Cell sizes [dx, dy, dz]
            &current_dir_int,                          // Direction of chi_k being solved
            &m_verbose                                 // Verbosity level for Fortran debug
        );
        // Note: The Fortran kernel needs access to neighbor information from active_mask,
        // so mask_ptr and its bounds should cover valid_bx grown by 1.
        // m_mf_active_mask has 1 ghost cell, FillBoundary was called.

        // Set values in HYPRE structure for the validbox of this MFIter
        auto hypre_lo_valid = EffectiveDiffusivityHypre::loV(valid_bx);
        auto hypre_hi_valid = EffectiveDiffusivityHypre::hiV(valid_bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              stencil_size, stencil_indices_hypre, matrix_values_buffer.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              rhs_values_buffer.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              initial_guess_buffer.data()); // Initial guess for chi_k
        HYPRE_CHECK(ierr);
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Assembling HYPRE Matrix and Vectors..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Assembly complete." << std::endl;
    }
}


// --- solve ---
// Solves the linear system A*x = b for chi_k.
// This is largely similar to TortuosityHypre.
bool EffectiveDiffusivityHypre::solve()
{
    BL_PROFILE("EffectiveDiffusivityHypre::solve");

    // If no active cells, no solve needed, solution is effectively zero.
    long num_active_cells = m_mf_active_mask.sum(MaskComp, true);
    if (num_active_cells == 0) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Skipping HYPRE solve: No active cells for phase " << m_phase_id << std::endl;
        }
        m_mf_chi.setVal(0.0); // Ensure chi is zero
        m_converged = true; // Or false, depending on desired state. True=no iterations needed.
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        return m_converged;
    }


    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL; // Preconditioner

    m_num_iterations = -1; // Reset solver stats
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();
    m_converged = false;

    // --- Solver Setup (Example: FlexGMRES with SMG preconditioner) ---
    if (m_solvertype == SolverType::FlexGMRES) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with SMG Preconditioner..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver, m_eps);
        HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver, (m_verbose > 2) ? 3 : 0); // Detailed HYPRE print for high verbose

        // Setup SMG preconditioner
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetTol(precond, 0.0); // Preconditioner solves to machine precision or fixed iterations
        HYPRE_StructSMGSetMaxIter(precond, 1); // Typically 1 iteration for SMG as preconditioner
        HYPRE_StructSMGSetNumPreRelax(precond, 1);
        HYPRE_StructSMGSetNumPostRelax(precond, 1);
        HYPRE_StructSMGSetPrintLevel(precond, (m_verbose > 3) ? 1 : 0); // Minimal print from precond
        HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Running HYPRE_StructFlexGMRESSetup..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Running HYPRE_StructFlexGMRESSolve..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
        // Check for convergence errors (HYPRE_ERROR_CONV is common)
        if (ierr != 0 && ierr != HYPRE_ERROR_CONV) { HYPRE_CHECK(ierr); }

        HYPRE_StructFlexGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);

        m_converged = !(std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm));
        m_converged = m_converged && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);

        if (ierr == HYPRE_ERROR_CONV && !m_converged && m_verbose >=0) {
            amrex::Warning("HYPRE FlexGMRES solver did not converge within tolerance!");
        } else if (ierr !=0 && m_verbose >=0) {
             amrex::Warning("HYPRE FlexGMRES solver returned error code: " + std::to_string(ierr));
        }
        HYPRE_StructFlexGMRESDestroy(solver);
        if (precond) HYPRE_StructSMGDestroy(precond);
    }
    // TODO: Add other solver types (PCG, GMRES, Jacobi) as in TortuosityHypre if needed
    else {
        amrex::Abort("Unsupported solver type requested in EffectiveDiffusivityHypre::solve: "
                     + std::to_string(static_cast<int>(m_solvertype)));
    }

    // --- Post-solve operations ---
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver Iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << std::scientific << m_final_res_norm << std::defaultfloat << std::endl;
        amrex::Print() << "  Solver Converged Status: " << (m_converged ? "Yes" : "No") << std::endl;
    }

    if (std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm)) {
        amrex::Warning("HYPRE solve resulted in NaN or Inf residual norm!");
        m_converged = false; // Ensure it's marked as not converged
    }

    // Copy solution from HYPRE vector m_x to AMReX MultiFab m_mf_chi
    if (m_converged) {
        getChiSolution(m_mf_chi); // Populate m_mf_chi
    } else {
        m_mf_chi.setVal(0.0); // Or some other indicator for non-convergence
        if (m_verbose >=0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning("Solver did not converge. Chi solution (m_mf_chi) set to 0.");
        }
    }


    // Optional: Write plotfile for chi_k
    if (m_write_plotfile && m_converged) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Writing solution plotfile for chi_k in direction "
                            << static_cast<int>(m_dir_solve) << "..." << std::endl;
        }
        // Create a temporary MultiFab for plotting, could include mask etc.
        amrex::MultiFab mf_plot(m_ba, m_dm, 2, 0); // Plot chi and active_mask
        amrex::Copy(mf_plot, m_mf_chi, ChiComp, 0, 1, 0);
                // Copy m_mf_active_mask (iMultiFab) to component 1 of mf_plot (MultiFab)
        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(mf_plot, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<amrex::Real> const plot_arr = mf_plot.array(mfi);
            amrex::Array4<const int> const mask_arr = m_mf_active_mask.const_array(mfi);

            amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept
            {
                plot_arr(i,j,k,1) = static_cast<amrex::Real>(mask_arr(i,j,k,MaskComp));
            });
        }

        std::string plot_filename_str = "effdiff_chi_dir" + std::to_string(static_cast<int>(m_dir_solve));
        std::string full_plot_path = m_resultspath + "/" + plot_filename_str;

        amrex::Vector<std::string> varnames = {"chi_k", "active_mask"};
        amrex::WriteSingleLevelPlotfile(full_plot_path, mf_plot, varnames, m_geom, 0.0, 0);

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Plotfile written to " << full_plot_path << std::endl;
        }
    } else if (m_write_plotfile && !m_converged) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning("Skipping plotfile write for chi_k because solver did not converge.");
        }
    }
    return m_converged;
}

// --- getChiSolution ---
// Copies the solution from HYPRE vector m_x to an AMReX MultiFab.
void EffectiveDiffusivityHypre::getChiSolution(amrex::MultiFab& chi_field)
{
    BL_PROFILE("EffectiveDiffusivityHypre::getChiSolution");

    if (!m_x) { // If HYPRE solution vector was never created
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  getChiSolution: HYPRE solution vector m_x is NULL. Setting chi_field to 0." << std::endl;
        }
        chi_field.setVal(0.0);
        // Ensure ghost cells are also zeroed if any, or fill boundary if appropriate for a zero field
        // For a zero field, setVal(0.0) on valid region is often enough, FillBoundary might not be needed
        // or could be called by the user of chi_field if they expect it.
        // For safety, explicitly fill if chi_field has ghosts:
        if (chi_field.nGrow() > 0) {
             chi_field.FillBoundary(m_geom.periodicity()); // Fill with 0s based on periodicity
        }
        return;
    }
    AMREX_ASSERT(chi_field.nComp() >= numComponentsChi);
    AMREX_ASSERT(chi_field.boxArray() == m_ba);
    AMREX_ASSERT(chi_field.DistributionMap() == m_dm);

    // Ensure chi_field has at least one ghost cell if values at faces will be needed later (e.g. for gradient)
    // The m_mf_chi member is defined with 1 ghost cell.
    // AMREX_ASSERT(chi_field.nGrow() >= 1);


    std::vector<amrex::Real> soln_buffer; // Use amrex::Real for consistency
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(chi_field, false); mfi.isValid(); ++mfi) // Iterate without tiling, over valid cells
    {
        const amrex::Box& bx = mfi.validbox(); // Get data for valid cells
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        soln_buffer.resize(npts); // Buffer for HYPRE data

        auto hypre_lo = EffectiveDiffusivityHypre::loV(bx);
        auto hypre_hi = EffectiveDiffusivityHypre::hiV(bx);

        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
        if (get_ierr != 0) {
            amrex::Warning("HYPRE_StructVectorGetBoxValues failed during getChiSolution!");
            // Set this fab to zero or handle error appropriately
            chi_field[mfi].setVal(0.0, bx, ChiComp, numComponentsChi);
            continue;
        }

        // Copy from buffer to MultiFab
        amrex::Array4<amrex::Real> const chi_arr = chi_field.array(mfi);
        long long k_lin_idx = 0; // Linear index into soln_buffer

        amrex::LoopOnCpu(bx, [=,&k_lin_idx] (int i, int j, int k) noexcept // Capture k_lin_idx by reference
        {
            if (k_lin_idx < npts) { // Bounds check
                chi_arr(i,j,k, ChiComp) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]);
            }
            k_lin_idx++;
        });
        AMREX_ASSERT_WITH_MESSAGE(k_lin_idx == npts, "Point count mismatch during HYPRE GetBoxValues copy in getChiSolution");
    }
    // Fill ghost cells of chi_field if it has them, using periodicity.
    // This is important if gradients of chi are computed later.
    chi_field.FillBoundary(m_geom.periodicity());
}


// Placeholder for the actual D_eff calculation method.
// This would be called *after* solving for chi_x, chi_y, and chi_z.
// It would take the three chi MultiFabs as input.
/*
amrex::Array2D<amrex::Real, 0, AMREX_SPACEDIM-1, 0, AMREX_SPACEDIM-1>
EffectiveDiffusivityHypre::calculateEffectiveDiffusivityTensor(
    const amrex::MultiFab& chi_x_field,
    const amrex::MultiFab& chi_y_field,
    const amrex::MultiFab& chi_z_field)
{
    BL_PROFILE("EffectiveDiffusivityHypre::calculateEffectiveDiffusivityTensor");
    amrex::Array2D<amrex::Real, 0, AMREX_SPACEDIM-1, 0, AMREX_SPACEDIM-1> D_eff_tensor;
    // Initialize tensor to zero
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        for (int j=0; j<AMREX_SPACEDIM; ++j) {
            D_eff_tensor(i,j) = 0.0;
        }
    }

    // 1. Ensure all chi fields and m_mf_active_mask have filled ghost cells
    //    (should be done after they are populated/created)

    // 2. Loop over MFIter for m_mf_active_mask (or any of the chi fields)
    //    For each cell (i,j,k):
    //    a. Check if active_mask(i,j,k) == cell_active. If not, skip (D=0).
    //    b. Compute gradients: grad_chi_x, grad_chi_y, grad_chi_z at (i,j,k)
    //       (e.g., using central differences, needs access to neighbors in chi fields)
    //    c. Form the tensor term for integration: D * ( (grad_chi_x @ ê_x) + (grad_chi_y @ ê_y) + (grad_chi_z @ ê_z) + I )
    //       Since D=1 in active cells, this is just ( (grad_chi_x @ ê_x) + ... + I )
    //       Example for D_eff_xy component:
    //         term_xy = ( (d(chi_x)/dy)*1 + (d(chi_y)/dy)*0 + (d(chi_z)/dy)*0 ) + (0 if x!=y else 1 if x==y)
    //                 = d(chi_x)/dy
    //       Example for D_eff_xx component:
    //         term_xx = ( (d(chi_x)/dx)*1 + (d(chi_y)/dx)*0 + (d(chi_z)/dx)*0 ) + 1
    //                 = d(chi_x)/dx + 1

    //    d. Accumulate these terms for each component of D_eff_tensor, weighted by cell volume (usually 1 if dx*dy*dz taken out)

    // 3. After iterating all cells, divide accumulated sums by total domain volume (N_total_cells * V_voxel)
    //    Or, if formula is (1/V_pore) * integral_over_pore( D * (...) ), then average over pore cells and multiply by porosity.
    //    The formula given is (1/||Omega_i||) integral_Omega_i D (...). Omega_i is the REV.
    //    So, sum (D_cell * (...)_cell * V_voxel) / V_REV.
    //    Since D_cell is 0 or 1, this is sum_pore ( (...)_cell * V_voxel) / V_REV
    //    = (N_pore * V_voxel / V_REV) * (1/N_pore) * sum_pore ( (...)_cell )
    //    = porosity * average_of_integrand_in_pore_phase.

    amrex::Print() << "ERROR: calculateEffectiveDiffusivityTensor is not fully implemented yet." << std::endl;
    return D_eff_tensor;
}
*/

} // End namespace OpenImpala
