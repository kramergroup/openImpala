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
// #include <AMReX_ParallelFor.H> // <<< REMOVED THIS INCLUDE >>>

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
    constexpr int stencil_size = 7;
}

// Helper Functions
namespace OpenImpala {

// --- loV, hiV, Constructor, Destructor, setupGrids, setupStencil, preconditionPhaseFab ---
// --- setupMatrixEquation ---
// --- (These functions remain unchanged from the previous version) ---
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
    setupStencil();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation();
     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
     }
}
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
void OpenImpala::TortuosityHypre::setupGrids()
{
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
        amrex::Print() << "  DEBUG [setupGrids]: Grid setup complete. m_grid pointer: " << m_grid << std::endl;
    }
}
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
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
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
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    HYPRE_Int ierr = 0;
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
        const auto& pbox = phase_iab.box();
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts, p_ptr, pbox.loVect(), pbox.hiVect(),
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);
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
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);
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
}
bool OpenImpala::TortuosityHypre::solve() {
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL;
    m_num_iterations = -1;
    m_final_res_norm = -1.0;
    if (m_solvertype == SolverType::PCG) {
        ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructPCGSetTol(solver, m_eps);
        HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPCGSetTwoNorm(solver, 1);
        HYPRE_StructPCGSetRelChange(solver, 0);
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 2 : 0);
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetMemoryUse(precond, 0);
        HYPRE_StructSMGSetMaxIter(precond, 1);
        HYPRE_StructSMGSetTol(precond, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond);
        HYPRE_StructSMGSetNumPreRelax(precond, 1);
        HYPRE_StructSMGSetNumPostRelax(precond, 1);
        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE PCG solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else {
             HYPRE_CHECK(ierr);
        }
        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructPCGDestroy(solver);
        if (precond) HYPRE_StructSMGDestroy(precond);
    } else if (m_solvertype == SolverType::GMRES) {
        ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructGMRESSetTol(solver, m_eps);
        HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructGMRESSetPrintLevel(solver, m_verbose > 1 ? 2 : 0);
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetMemoryUse(precond, 0);
        HYPRE_StructSMGSetMaxIter(precond, 1);
        HYPRE_StructSMGSetTol(precond, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond);
        HYPRE_StructSMGSetNumPreRelax(precond, 1);
        HYPRE_StructSMGSetNumPostRelax(precond, 1);
        HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructSMGSolve, HYPRE_StructSMGSetup, precond);
        ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE GMRES solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
        } else {
             HYPRE_CHECK(ierr);
        }
        HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructGMRESDestroy(solver);
        if (precond) HYPRE_StructSMGDestroy(precond);
    } else if (m_solvertype == SolverType::FlexGMRES) {
         amrex::Abort("FlexGMRES not fully implemented yet in TortuosityHypre::solve");
    } else if (m_solvertype == SolverType::Jacobi) {
         ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
         HYPRE_CHECK(ierr);
         HYPRE_StructJacobiSetTol(solver, m_eps);
         HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
         ierr = HYPRE_StructJacobiSetZeroGuess(solver);
         HYPRE_CHECK(ierr);
         ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
         HYPRE_CHECK(ierr);
         ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) {
              if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Warning: HYPRE Jacobi solver did not converge (Error Code " << ierr << "). Tortuosity may be inaccurate.\n";
         } else {
              HYPRE_CHECK(ierr);
         }
         HYPRE_StructJacobiGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         HYPRE_StructJacobiDestroy(solver);
    }
    else {
        amrex::Abort("Unsupported solver type requested in TortuosityHypre::solve");
    }
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << m_final_res_norm << std::endl;
    }
    if (m_write_plotfile) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Writing solution plotfile..." << std::endl;
        }
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponents, 0);
        amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 0);
        mf_soln_temp.setVal(0.0);
        std::vector<double> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
        for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0) continue;
            soln_buffer.resize(npts);
            auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
            if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed in solve() for plotfile!"); }
            // *** FIX: Use AMREX_HOST_DEVICE_FOR_BOX for copy ***
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
            AMREX_HOST_DEVICE_FOR_BOX(bx, i, j, k) { // Use the macro here
                amrex::IntVect iv(i,j,k);
                long long linear_offset = bx.index(iv);
                if (linear_offset >= 0 && linear_offset < npts) {
                     soln_arr(i,j,k) = soln_buffer[linear_offset];
                }
            }
        }
        amrex::Copy(mf_plot, mf_soln_temp, 0, SolnComp, 1, 0);
        amrex::Copy(mf_plot, m_mf_phase, 0, PhaseComp, 1, 0);
        std::string plotfilename = m_resultspath + "/tortuosity_solution";
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
    }
    m_first_call = false;
    return (m_final_res_norm >= 0.0 && m_final_res_norm <= m_eps);
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
        amrex::Real flux_in = 0.0, flux_out = 0.0;
        global_fluxes(flux_in, flux_out);
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Global Flux In:  " << flux_in << std::endl;
             amrex::Print() << "  Global Flux Out: " << flux_out << std::endl;
        }
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
            int idir = static_cast<int>(m_dir);
            if (idir == 0) {
                area = (probhi[1] - problo[1]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[0] - problo[0]);
            } else if (idir == 1) {
                area = (probhi[0] - problo[0]) * (probhi[2] - problo[2]);
                length_parallel = (probhi[1] - problo[1]);
            } else {
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
     amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
}
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
     amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
}
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const
{
    fxin = 0.0;
    fxout = 0.0;
    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);

    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1);
    mf_soln_temp.setVal(0.0);

    std::vector<double> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;
        soln_buffer.resize(npts);
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
        if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed in global_fluxes!"); }

        // *** FIX: Use AMREX_HOST_DEVICE_FOR_BOX for copy ***
        amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
        AMREX_HOST_DEVICE_FOR_BOX(bx, i, j, k) { // Use the macro here
             amrex::IntVect iv(i,j,k);
             long long linear_offset = bx.index(iv);
             if (linear_offset >= 0 && linear_offset < npts) {
                  soln_arr(i,j,k) = soln_buffer[linear_offset];
             }
        }
    }
    mf_soln_temp.FillBoundary(m_geom.periodicity());


    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        const auto phase = m_mf_phase.const_array(mfi);
        const auto soln = mf_soln_temp.const_array(mfi);

        amrex::Box lobox = amrex::adjCellLo(domain, idir, 1);
        lobox &= bx;

        amrex::Box hibox = amrex::adjCellHi(domain, idir, 1);
        hibox &= bx;

        amrex::Real grad, flux;
        amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);

        // *** FIX: Use AMREX_HOST_DEVICE_FOR_BOX for flux calculation loops ***
        AMREX_HOST_DEVICE_FOR_BOX(lobox, i, j, k) { // Use the macro here
             if (phase(i,j,k) == m_phase) {
                 grad = (soln(i,j,k) - soln(i-shift[0], j-shift[1], k-shift[2])) / dx[idir];
                 flux = -grad;
                 local_fxin += flux;
             }
        }

        AMREX_HOST_DEVICE_FOR_BOX(hibox, i, j, k) { // Use the macro here
             if (phase(i,j,k) == m_phase) {
                 grad = (soln(i+shift[0], j+shift[1], k+shift[2]) - soln(i,j,k)) / dx[idir];
                 flux = -grad;
                 local_fxout += flux;
             }
        }
    } // End MFIter

    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    amrex::Real face_area = 1.0;
    if (idir == 0) {
        face_area = dx[1] * dx[2];
    } else if (idir == 1) {
        face_area = dx[0] * dx[2];
    } else {
        face_area = dx[0] * dx[1];
    }

    fxin = local_fxin * face_area;
    fxout = local_fxout * face_area;
}


} // End namespace OpenImpala
