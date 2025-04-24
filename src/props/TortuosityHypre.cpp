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
    constexpr int numComponents = 2;
    constexpr amrex::Real tiny_flux_threshold = 1.e-15;
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
      m_mf_phi(ba, dm, numComponents, 1),
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL)
{
    // ... (Constructor body remains the same as the last working version) ...
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
    setupStencil(); // <-- This will now use the modified version below
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation();
     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
     }
}

// Destructor Implementation
OpenImpala::TortuosityHypre::~TortuosityHypre()
{
    // ... (Destructor body remains the same) ...
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

// setupGrids Implementation
void OpenImpala::TortuosityHypre::setupGrids()
{
    // ... (setupGrids body remains the same as the last working version, iterating over m_ba) ...
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
    if (!m_grid) { amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!"); }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupGrids]: Grid setup complete." << std::endl;
    }
}


/**
 * @brief Sets up the HYPRE StructStencil.
 * *** MODIFIED FOR DEBUGGING: Uses only a 1-point stencil ***
 */
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    // *** DEBUG: Use a 1-point stencil (center only) ***
    constexpr int stencil_size = 1;
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{0,0,0}}; // Only the center point

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupStencil]: Using DEBUG 1-point stencil." << std::endl;
    }

    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil);
    HYPRE_CHECK(ierr);

    // Loop only runs once for i=0
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

// preconditionPhaseFab Implementation
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    // ... (preconditionPhaseFab body remains the same) ...
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");
    m_mf_phase.FillBoundary(m_geom.periodicity());
    const amrex::Box& domain_box = m_geom.Domain();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& tile_box = mfi.tilebox();
        amrex::IArrayBox& fab = m_mf_phase[mfi];
        int ncomp = fab.nComp();
        tortuosity_remspot(fab.dataPtr(), fab.loVect(), fab.hiVect(), &ncomp,
                           tile_box.loVect(), tile_box.hiVect(),
                           domain_box.loVect(), domain_box.hiVect());
    }
    m_mf_phase.FillBoundary(m_geom.periodicity());
}

// setupMatrixEquation Implementation
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    // ... (setupMatrixEquation body remains the same as the last working version) ...
    // ... It will now use the 1-point stencil created above ...
    // ... The stencil_indices array and the call to tortuosity_fillmtx will likely need adjustment ...
    // ... if this test passes and we need to make the 1-point stencil work further, ...
    // ... but for now, just test if MatrixCreate passes. ...
    HYPRE_Int ierr = 0;
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG [setupMatrixEquation]: Calling HYPRE_StructMatrixCreate..." << std::endl;
    }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    HYPRE_CHECK(ierr); // Still the main check point

    // If MatrixCreate passes, the rest of this function might fail later
    // because the Fortran code expects a 7-point stencil. That's okay for now.
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
    // *** DEBUG: Adjust stencil indices for 1-point stencil ***
    int stencil_indices[1] = {0}; // Only index 0 (center) exists
    const int dir_int = static_cast<int>(m_dir);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) { dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]); }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine (with DEBUG 1-point stencil)..." << std::endl;
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        // *** DEBUG: Adjust matrix_values size for 1-point stencil ***
        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * 1); // Only 1 value per point
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = m_mf_phase.box(mfi.LocalTileIndex());

        // *** WARNING: Fortran call tortuosity_fillmtx likely expects 7 stencil values. ***
        // *** This call might now fail or produce incorrect results. ***
        // *** We are only testing if HYPRE_StructMatrixSetBoxValues passes with the 1-point stencil setup. ***
        // tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
        //                    &npts, p_ptr, pbox.loVect(), pbox.hiVect(),
        //                    bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
        //                    dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);

        // *** DEBUG: Instead of calling Fortran, just fill matrix_values with 1.0 for the center point ***
        for(size_t idx = 0; idx < static_cast<size_t>(npts); ++idx) {
            matrix_values[idx] = 1.0; // Set diagonal to 1
            rhs_values[idx] = 0.0; // Set RHS to 0
            initial_guess[idx] = 0.0; // Set guess to 0
        }
        // *** END DEBUG FILL ***

        // NaN/Inf Check (still useful)
        // ... (NaN/Inf check code remains the same) ...
        bool data_ok = true;
        for(int idx = 0; idx < npts; ++idx) {
            if (std::isnan(rhs_values[idx]) || std::isinf(rhs_values[idx])) { data_ok = false; break; }
            // Check only the single matrix value per point now
            if (std::isnan(matrix_values[idx]) || std::isinf(matrix_values[idx])) { data_ok = false; break; }
        }
        int global_data_ok = data_ok;
        amrex::ParallelDescriptor::ReduceIntMin(global_data_ok);
        if (global_data_ok == 0) { amrex::Abort("NaN/Inf found in DEBUG matrix/rhs values!"); }


        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // *** DEBUG: Pass stencil size 1 to SetBoxValues ***
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), 1, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }

    // Matrix Assembly Section
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Finished MFIter loop setting values (DEBUG 1-point stencil)." << std::endl;
    }

    // Matrix print attempt (might be useful)
    std::string matrix_debug_filename = "debug_matrix_state.log";
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::cout << "[Rank " << amrex::ParallelDescriptor::MyProc()
                  << "] Attempting HYPRE_StructMatrixPrint to " << matrix_debug_filename << " (DEBUG 1-point stencil)..." << std::endl << std::flush;
        HYPRE_Int print_ierr = HYPRE_StructMatrixPrint(matrix_debug_filename.c_str(), m_A, 0);
        if (print_ierr != 0) { /* ... handle print error ... */ }
        else { /* ... print success ... */ }
    }
    amrex::ParallelDescriptor::Barrier();

    // Finalize matrix assembly
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling HYPRE_StructMatrixAssemble (DEBUG 1-point stencil)..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A);
    HYPRE_CHECK(ierr); // Check assembly result

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Matrix assembled (DEBUG 1-point stencil)." << std::endl;
     }
}


// --- solve(), value(), getSolution(), getCellTypes(), global_fluxes() ---
// These will likely NOT produce meaningful results with the 1-point stencil,
// but they need to exist to compile. Keep their implementations as they were.
// (Skipped for brevity)
bool OpenImpala::TortuosityHypre::solve() { /* ... */ return false; } // Likely won't converge
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh) { /* ... */ return std::numeric_limits<amrex::Real>::quiet_NaN(); }
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) { /* ... */ }
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) { /* ... */ }
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const { /* ... */ fxin=0; fxout=0; }


} // End namespace OpenImpala
