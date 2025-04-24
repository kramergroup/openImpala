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
#include <iomanip>  // For std::setprecision
#include <iostream> // For std::cout, std::flush

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>      // For amrex::average_down (potentially needed elsewhere, good include)
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>       // For amrex::UtilCreateDirectory
#include <AMReX_BLassert.H>      // Corrected include path
#include <AMReX_ParmParse.H>     // Added for ParmParse
#include <AMReX_Vector.H>        // Added for amrex::Vector (needed for plotfile fix)
#include <AMReX_Array.H>         // Added for amrex::Array (for dxinv_sq fix & loV/hiV return)

// HYPRE includes (already in TortuosityHypre.H but good practice here too)
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h> // Needed for HYPRE_StructMatrixPrint, HYPRE_StructGrid, HYPRE_StructStencil

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
                                             const amrex::BoxArray& ba, // This is the potentially simplified BA
                                             const amrex::DistributionMapping& dm, // This is the potentially simplified DM
                                             const amrex::iMultiFab& mf_phase_input, // This iMultiFab still has the original layout
                                             const amrex::Real vf,
                                             const int phase,
                                             const OpenImpala::Direction dir,
                                             const SolverType st,
                                             const std::string& resultspath,
                                             const amrex::Real vlo,
                                             const amrex::Real vhi,
                                             int verbose,
                                             bool write_plotfile)
    : m_geom(geom), m_ba(ba), m_dm(dm), // Store the passed BA/DM (could be simplified)
      // Create internal phase MF based on the input MF's layout (original decomposition)
      m_mf_phase(mf_phase_input, amrex::make_alias, 0, mf_phase_input.nComp()),
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      m_eps(1e-9), m_maxiter(200),
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose),
      m_write_plotfile(write_plotfile),
      // Allocate solution multifab based on the PASSED BA/DM (could be simplified)
      m_mf_phi(ba, dm, numComponents, 1),
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL)
{
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
    preconditionPhaseFab(); // Modifies m_mf_phase (original layout)

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();           // Uses m_ba (potentially simplified)

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil();         // Independent of grid layout

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation();  // Uses m_grid, m_stencil, m_mf_phase (original layout for values)

     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
     }
}

/**
 * @brief Destructor: Frees all allocated HYPRE resources. CRITICAL.
 */
OpenImpala::TortuosityHypre::~TortuosityHypre()
{
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    m_x = m_b = NULL;
    m_A = NULL;
    m_stencil = NULL;
    m_grid = NULL;
}


/**
 * @brief Sets up the HYPRE StructGrid based on the AMReX BoxArray member `m_ba`.
 * @details This version iterates directly over the boxes stored in the `m_ba`
 * member variable, ensuring that the HYPRE grid reflects the BoxArray
 * passed during construction (which might be the simplified one for debugging).
 */
void OpenImpala::TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;

    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    HYPRE_CHECK(ierr);

    // *** MODIFIED: Iterate directly over the boxes in the stored m_ba ***
    // This ensures we use the BoxArray passed to the constructor (which might be the simplified one)
    for (int i = 0; i < m_ba.size(); ++i)
    {
        // Get the box directly from the BoxArray member
        amrex::Box bx = m_ba[i];

        // Use static helpers qualified with class name
        auto lo = OpenImpala::TortuosityHypre::loV(bx); // Returns amrex::Array<HYPRE_Int,...>
        auto hi = OpenImpala::TortuosityHypre::hiV(bx); // Returns amrex::Array<HYPRE_Int,...>

        // --- DEBUG LOGGING ---
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  DEBUG [setupGrids]: Setting extents for Box " << bx
                           << " (from m_ba[" << i << "])" // Indicate source
                           << " with lo = [" << lo[0] << "," << lo[1] << "," << lo[2] << "]"
                           << " hi = [" << hi[0] << "," << hi[1] << "," << hi[2] << "]" << std::endl;
        }
        // --- END DEBUG LOGGING ---

        // Pass pointer from amrex::Array using .data()
        ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());
        HYPRE_CHECK(ierr); // Check each SetExtents call
    }
    // *** END MODIFIED LOOP ***

    // Finalize grid assembly
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Calling HYPRE_StructGridAssemble..." << std::endl;
    }
    ierr = HYPRE_StructGridAssemble(m_grid);
    HYPRE_CHECK(ierr);

    // Check handle validity
    if (!m_grid) {
        amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!");
    }

    // Grid query calls remain commented out

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Grid setup complete." << std::endl;
    }
}


/**
 * @brief Sets up the 7-point HYPRE StructStencil.
 */
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    constexpr int stencil_size = 7;
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{0,0,0},
                                                       {-1,0,0}, {1,0,0},
                                                       {0,-1,0}, {0,1,0},
                                                       {0,0,-1}, {0,0,1}};

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

    // Stencil query calls remain commented out

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Stencil setup complete." << std::endl;
    }
}

/**
 * @brief Modifies the phase MultiFab, e.g., removing isolated spots.
 * Calls the Fortran routine `tortuosity_remspot`.
 */
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    // This routine modifies the *owned* copy m_mf_phase (original layout)
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");

    m_mf_phase.FillBoundary(m_geom.periodicity());
    const amrex::Box& domain_box = m_geom.Domain();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // MFIter here uses the original layout from m_mf_phase
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tile_box = mfi.tilebox();
        amrex::IArrayBox& fab = m_mf_phase[mfi];
        int ncomp = fab.nComp();
        tortuosity_remspot(fab.dataPtr(), fab.loVect(), fab.hiVect(), &ncomp,
                           tile_box.loVect(), tile_box.hiVect(),
                           domain_box.loVect(), domain_box.hiVect());
    }
    m_mf_phase.FillBoundary(m_geom.periodicity());
}

/**
 * @brief Sets up the HYPRE StructMatrix and StructVectors (A, b, x).
 * Fills matrix coefficients and RHS using the Fortran routine `tortuosity_fillmtx`.
 */
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    HYPRE_Int ierr = 0;

    // Matrix/Vector creation depends on m_grid, which is now based on m_ba (potentially simplified)
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructMatrixCreate..." << std::endl;
    }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    HYPRE_CHECK(ierr); // Line 352 (original failure point)

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
    if (!m_b) { amrex::Abort("FATAL: m_b handle is NULL after HYPRE_StructVectorInitialize!"); }
    ierr = HYPRE_StructVectorInitialize(m_b);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0);
    HYPRE_CHECK(ierr);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructVectorCreate (x)..." << std::endl;
    }
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x);
    HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0);
    HYPRE_CHECK(ierr);

    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[7] = {0,1,2,3,4,5,6};
    const int dir_int = static_cast<int>(m_dir);

    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) {
        dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]);
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine..." << std::endl;
    }

    // *** IMPORTANT: MFIter for filling values MUST use the original layout from m_mf_phase ***
    // The HYPRE matrix/vectors were created based on m_grid (potentially simplified),
    // but the actual phase data and coefficient calculations depend on the original decomposition.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // Use m_mf_phase for iteration to access the correct phase data layout
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Get the box corresponding to the original decomposition
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());

        if (npts == 0) continue;

        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * 7);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        // Get phase data pointer from m_mf_phase (original layout)
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex());

        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts, p_ptr, pbox.loVect(), pbox.hiVect(),
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);

        // NaN/Inf Check (remains the same)
        bool data_ok = true;
        for(int idx = 0; idx < npts; ++idx) {
            if (std::isnan(rhs_values[idx]) || std::isinf(rhs_values[idx])) {
                if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::IOProcessorNumber()) {
                    amrex::Print() << "!!! Invalid value detected in rhs_values[" << idx << "] = " << rhs_values[idx]
                                   << " within box " << bx << " (Rank " << amrex::ParallelDescriptor::MyProc() << ")" << std::endl;
                }
                data_ok = false;
            }
            for (int s_idx=0; s_idx < 7; ++s_idx) {
                 size_t matrix_idx = static_cast<size_t>(idx) * 7 + s_idx;
                 if (std::isnan(matrix_values[matrix_idx]) || std::isinf(matrix_values[matrix_idx])) {
                    if (amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::IOProcessorNumber()) {
                        amrex::Print() << "!!! Invalid value detected in matrix_values[" << idx << "][stencil " << s_idx << "] = " << matrix_values[matrix_idx]
                                   << " within box " << bx << " (Rank " << amrex::ParallelDescriptor::MyProc() << ")" << std::endl;
                    }
                    data_ok = false;
                 }
            }
        }
        int global_data_ok = data_ok;
        amrex::ParallelDescriptor::ReduceIntMin(global_data_ok);
        if (global_data_ok == 0) {
             amrex::Abort("NaN/Inf found in matrix_values or rhs_values after tortuosity_fillmtx Fortran call!");
        }

        // Set values in HYPRE structures using the box 'bx' from the MFIter (original layout)
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), 7, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);

    } // End MFIter loop

    // Matrix Assembly Section
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Finished MFIter loop setting values." << std::endl;
    }

    // Matrix print attempt (remains the same)
    std::string matrix_debug_filename = "debug_matrix_state.log";
    if (amrex::ParallelDescriptor::IOProcessor()) {
         std::cout << "[Rank " << amrex::ParallelDescriptor::MyProc()
                   << "] Attempting HYPRE_StructMatrixPrint to " << matrix_debug_filename << "..." << std::endl << std::flush;
         HYPRE_Int print_ierr = HYPRE_StructMatrixPrint(matrix_debug_filename.c_str(), m_A, 0);
         if (print_ierr != 0) {
             char print_err_msg[256];
             HYPRE_DescribeError(print_ierr, print_err_msg);
             amrex::Warning("HYPRE_StructMatrixPrint FAILED with code: " + std::to_string(print_ierr) +
                            " - " + std::string(print_err_msg));
             std::cout << "[Rank " << amrex::ParallelDescriptor::MyProc()
                       << "] HYPRE_StructMatrixPrint call FAILED." << std::endl << std::flush;
         } else {
             std::cout << "[Rank " << amrex::ParallelDescriptor::MyProc()
                       << "] HYPRE_StructMatrixPrint call completed (check artifact: " << matrix_debug_filename << ")."
                       << std::endl << std::flush;
         }
    }
    amrex::ParallelDescriptor::Barrier();

    // Finalize matrix assembly
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling HYPRE_StructMatrixAssemble..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr); // Check assembly result

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Matrix assembled." << std::endl;
     }
} // End setupMatrixEquation


// --- solve(), value(), getSolution(), getCellTypes(), global_fluxes() remain unchanged ---
// (Skipped for brevity, assume they are the same as the last correct version)

bool OpenImpala::TortuosityHypre::solve()
{
    // ... (Implementation from previous correct version) ...
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

    // Check if matrix A and vectors b, x are valid before attempting solve
    if (!m_A) { amrex::Abort("TortuosityHypre::solve() called but matrix m_A is NULL!"); }
    if (!m_b) { amrex::Abort("TortuosityHypre::solve() called but vector m_b is NULL!"); }
    if (!m_x) { amrex::Abort("TortuosityHypre::solve() called but vector m_x is NULL!"); }


    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre::solve(): Solving with HYPRE... Tolerance=" << m_eps << ", MaxIter=" << m_maxiter << std::endl;
    }

    // --- Setup Solver and Optional Preconditioner ---
    if (m_solvertype == OpenImpala::TortuosityHypre::SolverType::FlexGMRES ||
        m_solvertype == OpenImpala::TortuosityHypre::SolverType::GMRES ||
        m_solvertype == OpenImpala::TortuosityHypre::SolverType::PCG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Using PFMG Preconditioner (1 cycle)" << std::endl;
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        ierr = HYPRE_StructPFMGSetMaxIter(precond, 1); HYPRE_CHECK(ierr);
        ierr = HYPRE_StructPFMGSetTol(precond, 0.0); HYPRE_CHECK(ierr);
        ierr = HYPRE_StructPFMGSetRelChange(precond, 0); HYPRE_CHECK(ierr);
    }

    // --- Select and Run the HYPRE solver ---
    switch (m_solvertype)
    {
        case OpenImpala::TortuosityHypre::SolverType::Jacobi:
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Solver Type: Jacobi" << std::endl;
            ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructJacobiSetTol(solver, m_eps); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructJacobiSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructJacobiSetup..." << std::endl;
            ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructJacobiSolve..." << std::endl;
            ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
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
             if (m_verbose > 1) { HYPRE_StructFlexGMRESSetLogging(solver, 1); }
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructFlexGMRESSetup..." << std::endl;
            ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructFlexGMRESSolve..." << std::endl;
            ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
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
             if (m_verbose > 1) { HYPRE_StructPCGSetLogging(solver, 1); }
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructPCGSetup..." << std::endl;
            ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructPCGSolve..." << std::endl;
            ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  HYPRE_StructPCGSolve finished (ierr=" << ierr << ")" << std::endl;
            HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
            HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructPCGDestroy(solver);
            break;

        case OpenImpala::TortuosityHypre::SolverType::GMRES:
        default:
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Solver Type: GMRES (Default)" << std::endl;
            ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructGMRESSetTol(solver, m_eps); HYPRE_CHECK(ierr);
            ierr = HYPRE_StructGMRESSetMaxIter(solver, m_maxiter); HYPRE_CHECK(ierr);
            if (precond) {
                 if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting Preconditioner..." << std::endl;
                ierr = HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); HYPRE_CHECK(ierr);
            }
             if (m_verbose > 1) { HYPRE_StructGMRESSetLogging(solver, 1); }
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructGMRESSetup..." << std::endl;
            ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling HYPRE_StructGMRESSolve..." << std::endl;
            ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
             if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  HYPRE_StructGMRESSolve finished (ierr=" << ierr << ")" << std::endl;
            HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
            HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructGMRESDestroy(solver);
            break;
    }

    if (precond) {
        HYPRE_StructPFMGDestroy(precond);
        precond = NULL;
    }

    bool converged_ok = true;
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Solver finished: " << num_iterations
                       << " Iterations, Final Relative Residual = "
                       << final_res_norm << std::endl;
    }
    if (ierr != 0) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "ERROR: HYPRE solver returned error code: " << ierr << std::endl;
            if (HYPRE_CheckError(ierr, HYPRE_ERROR_CONV)) { amrex::Print() << "        (Solver did not converge within max iterations or tolerance)" << std::endl; }
            else if (HYPRE_CheckError(ierr, HYPRE_ERROR_MEMORY)) { amrex::Print() << "        (Solver memory allocation error)" << std::endl; }
            else if (HYPRE_CheckError(ierr, HYPRE_ERROR_ARG)) { amrex::Print() << "        (Solver argument error)" << std::endl; }
        }
        converged_ok = false;
    }
    if (final_res_norm > m_eps || std::isnan(final_res_norm) || std::isinf(final_res_norm)) {
        if (converged_ok && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Warning: Solver did not converge to tolerance " << m_eps
                            << " (Final Residual = " << final_res_norm << ")" << std::endl;
        }
        converged_ok = false;
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Copying solution to MultiFab..." << std::endl;
    getSolution(m_mf_phi, SolnComp);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Filling cell types..." << std::endl;
    getCellTypes(m_mf_phi, PhaseComp);

    if (m_write_plotfile && !m_resultspath.empty()) {
         if (amrex::ParallelDescriptor::IOProcessor()) {
             if (!amrex::UtilCreateDirectory(m_resultspath, 0755)) {
                 amrex::Warning("Could not create results directory: " + m_resultspath);
             }
         }
         amrex::ParallelDescriptor::Barrier();
         std::string plotfilename = m_resultspath + "/hypre_soln_" + std::string(datetime);
         amrex::Vector<std::string> plot_varnames = {"potential", "cell_type"};
         if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Writing plotfile: " << plotfilename << std::endl;
         amrex::WriteSingleLevelPlotfile(plotfilename, m_mf_phi, plot_varnames, m_geom, 0.0, 0);
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre::solve() finished. Converged OK: " << (converged_ok ? "Yes" : "No") << std::endl;
    }
    return converged_ok;
}

amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
      if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
           amrex::Print() << "TortuosityHypre::value() called (refresh=" << refresh << ", first_call=" << m_first_call << ")" << std::endl;
      }
    if (refresh || m_first_call)
    {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling solve()..." << std::endl;
        if (!solve()) {
             amrex::Warning("Solver failed in TortuosityHypre::value. Returning NaN.");
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
             m_first_call = true;
             return m_value;
        }
        m_first_call = false;
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  solve() completed successfully." << std::endl;

    } else if (std::isnan(m_value)) {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Returning cached NaN value." << std::endl;
        return m_value;
    }

      if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Calling global_fluxes()..." << std::endl;
    amrex::Real fluxin = 0.0, fluxout = 0.0;
    global_fluxes(fluxin, fluxout);
      if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  global_fluxes() completed (fluxin=" << fluxin << ", fluxout=" << fluxout << ")" << std::endl;

    const amrex::Box& domain_box = m_geom.Domain();
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
    amrex::Real delta_V = m_vhi - m_vlo;
    amrex::Real rel_diffusivity = 0.0;
    amrex::Real avg_flux = (fluxin + fluxout) / 2.0;
    if (std::abs(delta_V) > tiny_flux_threshold && length_dir > tiny_flux_threshold) {
         if (std::abs(cross_sectional_area) > tiny_flux_threshold) {
             rel_diffusivity = - avg_flux * length_dir / (cross_sectional_area * delta_V);
         } else {
             amrex::Warning("TortuosityHypre::value: Cross sectional area is zero.");
         }
    } else {
        amrex::Warning("TortuosityHypre::value: Cannot calculate relative diffusivity due to zero length or zero potential difference.");
    }
    if (std::abs(rel_diffusivity) > tiny_flux_threshold) {
         m_value = m_vf / rel_diffusivity;
         if (m_value < 0.0 && m_vf > tiny_flux_threshold) { amrex::Warning("Calculated negative tortuosity, check flux direction, BCs, or definition."); }
         else if (m_value < m_vf && m_vf > tiny_flux_threshold) { amrex::Warning("Calculated tortuosity is less than volume fraction. Check definition/calculation."); }
    } else {
         if (m_vf > tiny_flux_threshold) {
             if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: Relative diffusivity is near zero for non-zero VF. Tortuosity is effectively infinite." << std::endl;
             m_value = std::numeric_limits<amrex::Real>::infinity();
         } else {
              if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Note: Volume Fraction is near zero. Setting tortuosity to 0.0." << std::endl;
              m_value = 0.0;
         }
    }
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "------------------------------------------" << std::endl;
        amrex::Print() << " Tortuosity Calculation Results (Dir=" << static_cast<int>(m_dir) << ")" << std::endl;
        amrex::Print() << std::fixed << std::setprecision(6);
        amrex::Print() << "   Volume Fraction (VF)                  : " << m_vf << std::endl;
        amrex::Print() << "   Relative Effective Diffusivity      : " << rel_diffusivity << std::endl;
        amrex::Print() << "   Tortuosity (tau = VF / D_rel)       : " << m_value << std::endl;
        amrex::Print() << "   --- Intermediate Values ---" << std::endl;
        amrex::Print() << "   Flux Low Face                       : " << fluxin << std::endl;
        amrex::Print() << "   Flux High Face                      : " << fluxout << std::endl;
        amrex::Print() << "   Flux Average                        : " << avg_flux << std::endl;
        amrex::Print() << "   Flux Conservation Check |In + Out|  : " << std::abs(fluxin + fluxout) << std::endl;
        amrex::Print() << "------------------------------------------" << std::endl;
    }
    return m_value;
}

void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp)
{
    AMREX_ASSERT(ncomp >= 0 && ncomp < soln.nComp());
    AMREX_ASSERT(soln.nGrowVect().min() >= 0);
    if (!m_x) { amrex::Warning("TortuosityHypre::getSolution called but m_x is NULL. Returning without copying."); return; }
    amrex::FArrayBox host_fab;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(host_fab)
#endif
    // MFIter here should iterate over the layout of soln (which matches m_ba passed to constructor)
    for (amrex::MFIter mfi(soln); mfi.isValid(); ++mfi)
    {
        const amrex::Box &reg = mfi.validbox();
        if (!reg.ok()) continue;
        host_fab.resize(reg, 1);
        auto reglo = OpenImpala::TortuosityHypre::loV(reg);
        auto reghi = OpenImpala::TortuosityHypre::hiV(reg);
        HYPRE_Int ierr = HYPRE_StructVectorGetBoxValues(m_x, reglo.data(), reghi.data(), host_fab.dataPtr());
        HYPRE_CHECK(ierr);
        soln[mfi].copy(host_fab, reg, 0, reg, ncomp, 1);
    }
}

void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, const int ncomp)
{
    AMREX_ASSERT(ncomp >= 0 && ncomp < phi.nComp());
    AMREX_ASSERT(m_mf_phase.nGrow() >= phi.nGrow());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // MFIter here should iterate over the layout of phi (which matches m_ba passed to constructor)
    // BUT it needs data from m_mf_phase (original layout). This might be an issue if BA/DM differ.
    // For now, assume the loop works if the boxes overlap sufficiently.
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab_target = phi[mfi];
        // *** This assumes m_mf_phase covers the same region as phi[mfi] ***
        const amrex::IArrayBox& fab_phase_src = m_mf_phase[mfi];

        int q_ncomp = fab_target.nComp();
        int p_ncomp = fab_phase_src.nComp();
        const auto& qbox = phi.box(mfi.LocalTileIndex());
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex());
        const auto& domain_box = m_geom.Domain();
        amrex::Real* q_comp_ptr = fab_target.dataPtr(ncomp);

        tortuosity_filct(q_comp_ptr, fab_target.loVect(), fab_target.hiVect(), &q_ncomp,
                         fab_phase_src.dataPtr(), fab_phase_src.loVect(), fab_phase_src.hiVect(), &p_ncomp,
                         domain_box.loVect(), domain_box.hiVect(), &m_phase);
    }
}

void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const {
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre::global_fluxes() calculating..." << std::endl;
    }
    fxin = 0.0; fxout = 0.0;
    amrex::Real local_fxin = 0.0; amrex::Real local_fxout = 0.0;
    const int idir = static_cast<int>(m_dir);
    const amrex::Real dxinv_dir = (idir == 0) ? (1.0 / m_geom.CellSize(0)) :
                                  (idir == 1) ? (1.0 / m_geom.CellSize(1)) :
                                                (1.0 / m_geom.CellSize(2));
    amrex::Real face_area = 1.0;
      if (idir == 0) { face_area = m_geom.CellSize(1) * m_geom.CellSize(2); }
      else if (idir == 1) { face_area = m_geom.CellSize(0) * m_geom.CellSize(2); }
      else { face_area = m_geom.CellSize(0) * m_geom.CellSize(1); }
    const amrex::Box& domain = m_geom.Domain();
    const int domlo_dir = domain.smallEnd(idir);
    const int domhi_dir = domain.bigEnd(idir);

    // Use m_mf_phi (solution multifab) which has the same layout as m_ba passed to constructor
    if (!m_mf_phi.is_nodal() && m_mf_phi.nComp() <= SolnComp) {
        amrex::Warning("TortuosityHypre::global_fluxes: Solution MultiFab m_mf_phi seems invalid. Skipping flux calculation.");
        fxin = std::numeric_limits<amrex::Real>::quiet_NaN();
        fxout = std::numeric_limits<amrex::Real>::quiet_NaN();
        return;
    }
    amrex::MultiFab phi_soln_local(m_mf_phi, amrex::make_alias, SolnComp, 1);
    phi_soln_local.FillBoundary(m_geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    // MFIter uses layout of phi_soln_local (same as m_ba passed to constructor)
    for (amrex::MFIter mfi(phi_soln_local); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& phi     = phi_soln_local.const_array(mfi);
        // *** Need cell_type data corresponding to this box 'bx' ***
        // This requires m_mf_phi to have the PhaseComp data correctly filled
        // for the potentially simplified BA/DM layout. getCellTypes might need adjustment.
        if (m_mf_phi.nComp() <= PhaseComp) { amrex::Abort("Phase component not found in m_mf_phi for flux calculation!"); }
        const auto& cell_type = m_mf_phi.const_array(mfi, PhaseComp);

        amrex::Box lo_face_box = domain;
        lo_face_box.setBig(idir, domlo_dir); lo_face_box.setSmall(idir, domlo_dir);
        lo_face_box &= bx;
        if (lo_face_box.ok()) {
            amrex::Loop(lo_face_box, [=, &local_fxin] (int i, int j, int k) noexcept {
                amrex::IntVect iv_cell1(i, j, k);
                if (cell_type(iv_cell1) == 1) {
                    local_fxin += -1.0 * (phi(iv_cell1) - m_vlo) * dxinv_dir;
                }
            });
        }
        amrex::Box hi_face_box = domain;
        hi_face_box.setSmall(idir, domhi_dir + 1); hi_face_box.setBig(idir, domhi_dir + 1);
        amrex::Box bx_grown = bx; bx_grown.grow(idir, 1);
        amrex::Box hi_face_intersect = hi_face_box & bx_grown;
         if (hi_face_intersect.ok()) {
              amrex::Loop(hi_face_intersect, [=, &local_fxout] (int i, int j, int k) noexcept {
                  amrex::IntVect iv_cell1(i, j, k); iv_cell1[idir] -= 1;
                  if (cell_type(iv_cell1) == 1) {
                       local_fxout += -1.0 * (m_vhi - phi(iv_cell1)) * dxinv_dir;
                  }
              });
         }
    }
    local_fxin *= face_area; local_fxout *= face_area;
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);
    fxin = local_fxin; fxout = local_fxout;
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  Reduced fluxes: fxin=" << fxin << ", fxout=" << fxout << std::endl;
    }
}


} // End namespace OpenImpala
