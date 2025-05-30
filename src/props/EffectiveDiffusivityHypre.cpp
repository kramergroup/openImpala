// --- EffectiveDiffusivityHypre.cpp ---

#include "EffectiveDiffusivityHypre.H"
#include "EffDiffFillMtx_F.H" // For effdiff_fillmtx

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
#include <atomic> 

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
#include <AMReX_VisMF.H>
#include <AMReX_Loop.H>

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
    constexpr int ChiComp = 0;
    constexpr int MaskComp = 0;
    constexpr int numComponentsChi = 1;
    constexpr int stencil_size = 7;
    constexpr int cell_inactive = 0;
    constexpr int cell_active = 1;
    constexpr int istn_c  = 0;
    constexpr int istn_mx = 1;
    constexpr int istn_px = 2;
    constexpr int istn_my = 3;
    constexpr int istn_py = 4;
    constexpr int istn_mz = 5;
    constexpr int istn_pz = 6;
}

namespace OpenImpala {

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

EffectiveDiffusivityHypre::EffectiveDiffusivityHypre(
    const amrex::Geometry& geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    const amrex::iMultiFab& mf_phase_input,
    const int phase_id_arg, // Renamed argument to avoid confusion with member
    const OpenImpala::Direction dir_of_chi_k,
    const SolverType solver_type,
    const std::string& resultspath,
    int verbose_level,
    bool write_plotfile_flag)
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase_original(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()),
      m_phase_id(phase_id_arg), 
      m_dir_solve(dir_of_chi_k),
      m_solvertype(solver_type),
      m_eps(1e-9), m_maxiter(1000),
      m_resultspath(resultspath),
      m_verbose(verbose_level),
      m_write_plotfile(write_plotfile_flag),
      m_mf_chi(ba, dm, numComponentsChi, 1),
      m_mf_active_mask(ba, dm, 1, 1),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL),
      m_num_iterations(-1),
      m_final_res_norm(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_converged(false)
{
    BL_PROFILE("EffectiveDiffusivityHypre::Constructor");

    amrex::Copy(m_mf_phase_original, mf_phase_input, 0, 0, m_mf_phase_original.nComp(), m_mf_phase_original.nGrow());
    if (m_mf_phase_original.nGrow() > 0) {
        m_mf_phase_original.FillBoundary(m_geom.periodicity());
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Initializing for chi_k in direction "
                       << static_cast<int>(m_dir_solve) << "..." << std::endl;
        amrex::Print() << "  DEBUG HYPRE: Constructor received m_phase_id (member) = " << m_phase_id << std::endl;
        amrex::Print() << "  DEBUG HYPRE: Constructor received phase_id_arg (argument) = " << phase_id_arg << std::endl;
    }

    long initial_phase_count_debug = 0;
    // Manual sum for debug clarity
    for (amrex::MFIter mfi(m_mf_phase_original, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<const int> const phase_arr = m_mf_phase_original.const_array(mfi);
        long local_count = 0;
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) noexcept {
            if (phase_arr(i,j,k,0) == m_phase_id) {
                local_count++;
            }
        });
        initial_phase_count_debug += local_count;
    }
    amrex::ParallelDescriptor::ReduceLongSum(initial_phase_count_debug);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG HYPRE: Number of cells in input m_mf_phase_original matching m_phase_id (" << m_phase_id
                       << ") before generateActiveMask: " << initial_phase_count_debug << std::endl;
        amrex::Print() << "  DEBUG HYPRE: m_mf_phase_original nComp: " << m_mf_phase_original.nComp()
                       << ", nGrow: " << m_mf_phase_original.nGrow() << std::endl;
    }

    amrex::ParmParse pp_hypre("hypre");
    pp_hypre.query("eps", m_eps);
    pp_hypre.query("maxiter", m_maxiter);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");

    const amrex::Real* dx_tmp = m_geom.CellSize();
    for(int i_dim=0; i_dim<AMREX_SPACEDIM; ++i_dim) { 
        m_dx[i_dim] = dx_tmp[i_dim];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_dx[i_dim] > 0.0, "Cell size must be positive.");
    }

    m_mf_active_mask.setVal(cell_inactive);
    generateActiveMask(); 

    // Use standard AMReX sum now that we've confirmed its behavior in this context via manual sums
    long num_active_cells = m_mf_active_mask.sum(MaskComp, true); // Sum over valid cells only, local sum
    amrex::ParallelDescriptor::ReduceLongSum(num_active_cells); // Reduce to get global sum

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()){
        amrex::Print() << "  Active mask generated. Number of active cells (from m_mf_active_mask.sum()): " << num_active_cells << std::endl;
    }

    if (num_active_cells == 0) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "WARNING: No active cells found for phase_id " << m_phase_id
                           << ". HYPRE setup will be skipped." << std::endl;
        }
        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        m_mf_chi.setVal(0.0);
        return;
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Setting up HYPRE structures..." << std::endl;
    }
    setupGrids();
    setupStencil();
    setupMatrixEquation();

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "EffectiveDiffusivityHypre: Initialization complete." << std::endl;
    }
}

EffectiveDiffusivityHypre::~EffectiveDiffusivityHypre()
{
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    m_x = m_b = NULL; m_A = NULL; m_stencil = NULL; m_grid = NULL;
}

void EffectiveDiffusivityHypre::generateActiveMask()
{
    BL_PROFILE("EffectiveDiffusivityHypre::generateActiveMask");

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        long phase_original_sum_debug_again = 0;
        for (amrex::MFIter mfi_po(m_mf_phase_original, false); mfi_po.isValid(); ++mfi_po) {
            const amrex::Box& bx_po_valid = mfi_po.validbox();
            amrex::Array4<const int> const phase_arr_po = m_mf_phase_original.const_array(mfi_po);
            long local_count_po = 0;
            amrex::LoopOnCpu(bx_po_valid, [&] (int i, int j, int k) noexcept {
                if (phase_arr_po(i,j,k,0) == m_phase_id) { 
                    local_count_po++;
                }
            });
            phase_original_sum_debug_again += local_count_po;
        }
        amrex::ParallelDescriptor::ReduceLongSum(phase_original_sum_debug_again);
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Sum of m_mf_phase_original for m_phase_id (" << m_phase_id
                       << ") *immediately before* mask generation loop: " << phase_original_sum_debug_again << std::endl;
    }

    if (m_mf_phase_original.nGrow() > 0) {
         m_mf_phase_original.FillBoundary(m_geom.periodicity());
    }

    std::atomic<long> phase0_became_active_count_debug(0);
    std::atomic<long> phase1_became_active_count_debug(0);
    std::atomic<long> phase1_became_inactive_count_debug(0);

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    #endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& current_tile_box = mfi.tilebox();
        amrex::Array4<int> const mask_arr = m_mf_active_mask.array(mfi);
        amrex::Array4<const int> const phase_arr = m_mf_phase_original.const_array(mfi);
        
        const int local_m_phase_id = m_phase_id;

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor() && mfi.LocalIndex() == 0 && amrex::ParallelDescriptor::MyProc() == amrex::ParallelDescriptor::IOProcessorNumber()) {
            amrex::Print() << "  DEBUG HYPRE: generateActiveMask MFIter loop is using local_m_phase_id = " << local_m_phase_id
                           << " (from m_phase_id = " << m_phase_id << ") for comparison." << std::endl;
        }

        const amrex::Box& valid_bx_for_debug = mfi.validbox();

        amrex::LoopOnCpu(current_tile_box, [=, &phase0_became_active_count_debug, &phase1_became_active_count_debug, &phase1_became_inactive_count_debug] (int i, int j, int k) noexcept
        {
            int original_phase_val = phase_arr(i,j,k,0);
            bool should_be_active = (original_phase_val == local_m_phase_id);
            
            if (should_be_active) {
                mask_arr(i,j,k,MaskComp) = cell_active;
                if (valid_bx_for_debug.contains(i,j,k)) { 
                    if (original_phase_val == 1) phase1_became_active_count_debug++; // Assuming phase 1 is the target for this counter
                }
            } else {
                mask_arr(i,j,k,MaskComp) = cell_inactive;
                 if (valid_bx_for_debug.contains(i,j,k)) { 
                    if (original_phase_val == 1) phase1_became_inactive_count_debug++; // Assuming phase 1 is the target
                 }
            }

            if (valid_bx_for_debug.contains(i,j,k)) {
                // Specifically check if a cell that should be inactive (original phase 0 when target is 1) becomes active
                if (original_phase_val == 0 && local_m_phase_id == 1 && mask_arr(i,j,k,MaskComp) == cell_active) {
                    phase0_became_active_count_debug++;
                }
            }
        });
    }
    
    long phase0_became_active_val = phase0_became_active_count_debug.load();
    long phase1_became_active_val = phase1_became_active_count_debug.load();
    long phase1_became_inactive_val = phase1_became_inactive_count_debug.load();

    amrex::ParallelDescriptor::ReduceLongSum(phase0_became_active_val);
    amrex::ParallelDescriptor::ReduceLongSum(phase1_became_active_val);
    amrex::ParallelDescriptor::ReduceLongSum(phase1_became_inactive_val);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { 
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Count of (valid) cells originally phase 0 that BECAME ACTIVE: "
                       << phase0_became_active_val << std::endl;
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Count of (valid) cells originally phase 1 that BECAME ACTIVE: "
                       << phase1_became_active_val << std::endl;
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: Count of (valid) cells originally phase 1 that BECAME INACTIVE: "
                       << phase1_became_inactive_val << std::endl;
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        long active_mask_sum_before_fillboundary = m_mf_active_mask.sum(MaskComp, true);
        amrex::ParallelDescriptor::ReduceLongSum(active_mask_sum_before_fillboundary);
        amrex::Print() << "  DEBUG HYPRE generateActiveMask: m_mf_active_mask.sum() (valid cells) *after* loop, *before* FillBoundary: "
                       << active_mask_sum_before_fillboundary << std::endl;
    }

    if (m_mf_active_mask.nGrow() > 0) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  DEBUG HYPRE generateActiveMask: Calling m_mf_active_mask.FillBoundary()..." << std::endl;
        }
        m_mf_active_mask.FillBoundary(m_geom.periodicity()); 
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  DEBUG HYPRE generateActiveMask: Returned from m_mf_active_mask.FillBoundary()." << std::endl;
        }

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { 
            long active_mask_sum_after_fillboundary = m_mf_active_mask.sum(MaskComp, true);
            amrex::ParallelDescriptor::ReduceLongSum(active_mask_sum_after_fillboundary);
            amrex::Print() << "  DEBUG HYPRE generateActiveMask: m_mf_active_mask.sum() (valid cells) *immediately after* FillBoundary: "
                           << active_mask_sum_after_fillboundary << std::endl;
        }
    } else {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  DEBUG HYPRE generateActiveMask: Skipped m_mf_active_mask.FillBoundary() as nGrow is 0." << std::endl;
         }
    }
}

void EffectiveDiffusivityHypre::setupGrids()
{
    BL_PROFILE("EffectiveDiffusivityHypre::setupGrids");
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid); HYPRE_CHECK(ierr);

    bool any_dim_periodic = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (m_geom.isPeriodic(d)) {
            any_dim_periodic = true;
            break;
        }
    }

    if (any_dim_periodic) {
        const amrex::Box& domain_geom_box = m_geom.Domain();
        HYPRE_Int periodic_hyp[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (m_geom.isPeriodic(d)) {
                periodic_hyp[d] = static_cast<HYPRE_Int>(domain_geom_box.length(d));
                if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "  setupGrids: Dim " << d << " is periodic with length " << periodic_hyp[d] << std::endl;
                }
            } else {
                periodic_hyp[d] = 0;
                 if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "  setupGrids: Dim " << d << " is NOT periodic (in AMReX geom)." << std::endl;
                }
            }
        }
         for (int i_ba = 0; i_ba < m_ba.size(); ++i_ba) {
            if (m_dm[i_ba] == amrex::ParallelDescriptor::MyProc()) {
                amrex::Box bx = m_ba[i_ba];
                auto lo = EffectiveDiffusivityHypre::loV(bx);
                auto hi = EffectiveDiffusivityHypre::hiV(bx);
                if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "  setupGrids: Rank " << amrex::ParallelDescriptor::MyProc()
                                    << " adding box " << bx << std::endl;
                }
                ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data()); HYPRE_CHECK(ierr);
            }
        }
        ierr = HYPRE_StructGridSetPeriodic(m_grid, periodic_hyp); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  setupGrids: HYPRE_StructGridSetPeriodic called." << std::endl;
        }
    } else { 
         for (int i_ba = 0; i_ba < m_ba.size(); ++i_ba) {
            if (m_dm[i_ba] == amrex::ParallelDescriptor::MyProc()) {
                amrex::Box bx = m_ba[i_ba];
                auto lo = EffectiveDiffusivityHypre::loV(bx);
                auto hi = EffectiveDiffusivityHypre::hiV(bx);
                if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "  setupGrids (Non-Periodic AMReX Geom): Rank " << amrex::ParallelDescriptor::MyProc()
                                    << " adding box " << bx << std::endl;
                }
                ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data()); HYPRE_CHECK(ierr);
            }
        }
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  setupGrids: SKIPPING HYPRE_StructGridSetPeriodic (AMReX geom is non-periodic)." << std::endl;
        }
    }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupGrids: Calling HYPRE_StructGridAssemble..." << std::endl;
    }
    ierr = HYPRE_StructGridAssemble(m_grid); HYPRE_CHECK(ierr);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_grid != NULL, "m_grid is NULL after HYPRE_StructGridAssemble!");

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupGrids: Assemble complete." << std::endl;
    }
}

void EffectiveDiffusivityHypre::setupStencil()
{
    BL_PROFILE("EffectiveDiffusivityHypre::setupStencil");
    HYPRE_Int ierr = 0;
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {
        { 0, 0, 0}, {-1, 0, 0}, { 1, 0, 0}, { 0,-1, 0}, { 0, 1, 0}, { 0, 0,-1}, { 0, 0, 1}
    };
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupStencil: Creating " << stencil_size << "-point stencil..." << std::endl;
    }
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil); HYPRE_CHECK(ierr);
    for (int i_stn = 0; i_stn < stencil_size; ++i_stn) { 
        ierr = HYPRE_StructStencilSetElement(m_stencil, i_stn, offsets[i_stn]); HYPRE_CHECK(ierr);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stencil != NULL, "m_stencil is NULL after HYPRE_StructStencilCreate/SetElement!");
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupStencil: Complete." << std::endl;
    }
}

void EffectiveDiffusivityHypre::setupMatrixEquation()
{
    BL_PROFILE("EffectiveDiffusivityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Creating HYPRE Matrix and Vectors..." << std::endl;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_grid != NULL, "m_grid is NULL in setupMatrixEquation. Call setupGrids first.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stencil != NULL, "m_stencil is NULL in setupMatrixEquation. Call setupStencil first.");

    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructMatrixCreate: OK" << std::endl;

    ierr = HYPRE_StructMatrixInitialize(m_A); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructMatrixInitialize: OK" << std::endl;

    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructVectorCreate (b): OK" << std::endl;

    ierr = HYPRE_StructVectorInitialize(m_b); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructVectorInitialize (b): OK" << std::endl;

    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructVectorCreate (x): OK" << std::endl;

    ierr = HYPRE_StructVectorInitialize(m_x); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructVectorInitialize (x): OK" << std::endl;

    const amrex::Box& domain_for_kernel = m_geom.Domain();
    int stencil_indices_hypre[stencil_size]; 
    for(int i_loop=0; i_loop<stencil_size; ++i_loop) {
        stencil_indices_hypre[i_loop] = i_loop;
    }
    const int current_dir_int = static_cast<int>(m_dir_solve);

    if (m_mf_active_mask.nGrow() > 0) {
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Calling Fortran kernel 'effdiff_fillmtx' and SetBoxValues..." << std::endl;
    }

    std::vector<amrex::Real> matrix_values_buffer;
    std::vector<amrex::Real> rhs_values_buffer;
    std::vector<amrex::Real> initial_guess_buffer;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) \
                 private(matrix_values_buffer, rhs_values_buffer, initial_guess_buffer)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& valid_bx = mfi.validbox();
        const int npts_valid = static_cast<int>(valid_bx.numPts());
        if (npts_valid == 0) continue;

        matrix_values_buffer.resize(static_cast<size_t>(npts_valid) * stencil_size);
        rhs_values_buffer.resize(npts_valid);
        initial_guess_buffer.resize(npts_valid);

        const amrex::IArrayBox& mask_fab = m_mf_active_mask[mfi]; 
        const int* mask_ptr = mask_fab.dataPtr(MaskComp);
        const auto* mask_fab_lo = mask_fab.loVect();
        const auto* mask_fab_hi = mask_fab.hiVect();

        effdiff_fillmtx(
            matrix_values_buffer.data(), rhs_values_buffer.data(), initial_guess_buffer.data(),
            &npts_valid,
            mask_ptr, mask_fab_lo, mask_fab_hi,
            valid_bx.loVect(), valid_bx.hiVect(),
            domain_for_kernel.loVect(), domain_for_kernel.hiVect(),
            m_dx.dataPtr(),
            &current_dir_int, 
            &m_verbose
        );

        auto hypre_lo_valid = EffectiveDiffusivityHypre::loV(valid_bx);
        auto hypre_hi_valid = EffectiveDiffusivityHypre::hiV(valid_bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              stencil_size, stencil_indices_hypre, matrix_values_buffer.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              rhs_values_buffer.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo_valid.data(), hypre_hi_valid.data(),
                                              initial_guess_buffer.data());
        HYPRE_CHECK(ierr);
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
       amrex::Print() << "  setupMatrixEquation: Attempting HYPRE_StructMatrixAssemble(m_A)..." << std::endl;
    }
    ierr = HYPRE_StructMatrixAssemble(m_A); HYPRE_CHECK(ierr);
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
       amrex::Print() << "    HYPRE_StructMatrixAssemble(m_A): OK" << std::endl;
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
       amrex::Print() << "  setupMatrixEquation: Attempting Vector Assembles..." << std::endl;
    }
    ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr); 
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructVectorAssemble(m_b): OK" << std::endl;
    ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr); 
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    HYPRE_StructVectorAssemble(m_x): OK" << std::endl;

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEquation: Setup complete (All Assembles Done)." << std::endl;
    }
}

bool EffectiveDiffusivityHypre::solve()
{
    BL_PROFILE("EffectiveDiffusivityHypre::solve");

    long num_active_cells_in_solve = m_mf_active_mask.sum(MaskComp, true);
    amrex::ParallelDescriptor::ReduceLongSum(num_active_cells_in_solve);


    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  DEBUG HYPRE solve: num_active_cells_in_solve (from m_mf_active_mask.sum()) = " << num_active_cells_in_solve << std::endl;
    }

    if (num_active_cells_in_solve == 0) {
        if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "EffectiveDiffusivityHypre::solve: Skipping HYPRE solve as no active cells were found for phase "
                           << m_phase_id << std::endl;
        }
        m_mf_chi.setVal(0.0); 
        if (m_mf_chi.nGrow() > 0) m_mf_chi.FillBoundary(m_geom.periodicity()); 

        m_converged = true;
        m_num_iterations = 0;
        m_final_res_norm = 0.0;
        return m_converged;
    }

    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver_hypre;
    HYPRE_StructSolver precond = NULL;

    m_num_iterations = -1;
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();
    m_converged = false;

    if (m_solvertype == SolverType::FlexGMRES) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  solve: Setting up HYPRE FlexGMRES Solver with PFMG Preconditioner..." << std::endl; 
        }
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver_hypre); HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver_hypre, m_eps);
        HYPRE_StructFlexGMRESSetMaxIter(solver_hypre, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver_hypre, (m_verbose > 2) ? 3 : 0);

        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0);      
        HYPRE_StructPFMGSetMaxIter(precond, 1);    
        HYPRE_StructPFMGSetNumPreRelax(precond, 1); 
        HYPRE_StructPFMGSetNumPostRelax(precond, 1); 
        HYPRE_StructPFMGSetPrintLevel(precond, (m_verbose > 3) ? 1 : 0);
        HYPRE_StructFlexGMRESSetPrecond(solver_hypre, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond); 

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  solve: Running HYPRE_StructFlexGMRESSetup (with PFMG precond)..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESSetup(solver_hypre, m_A, m_b, m_x); HYPRE_CHECK(ierr); 

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  solve: Running HYPRE_StructFlexGMRESSolve..." << std::endl;
        }
        ierr = HYPRE_StructFlexGMRESSolve(solver_hypre, m_A, m_b, m_x);
        if (ierr != 0 && ierr != HYPRE_ERROR_CONV) { HYPRE_CHECK(ierr); }

        HYPRE_StructFlexGMRESGetNumIterations(solver_hypre, &m_num_iterations);
        HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver_hypre, &m_final_res_norm);

        m_converged = !(std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm));
        m_converged = m_converged && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);

        if (ierr == HYPRE_ERROR_CONV && !m_converged && m_verbose >=0) {
            amrex::Warning("HYPRE FlexGMRES solver (with PFMG precond) did not converge within tolerance!");
        } else if (ierr !=0 && ierr != HYPRE_ERROR_CONV && m_verbose >=0) {
             amrex::Warning("HYPRE FlexGMRES solver (with PFMG precond) returned error code: " + std::to_string(ierr));
        }
        HYPRE_StructFlexGMRESDestroy(solver_hypre);
        if (precond) HYPRE_StructPFMGDestroy(precond); 
    }
    else {
        amrex::Abort("Unsupported solver type requested in EffectiveDiffusivityHypre::solve: "
                     + std::to_string(static_cast<int>(m_solvertype)));
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver Iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << std::scientific << m_final_res_norm << std::defaultfloat << std::endl;
        amrex::Print() << "  Solver Converged Status: " << (m_converged ? "Yes" : "No") << std::endl;
    }

    if (std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm)) {
        amrex::Warning("HYPRE solve resulted in NaN or Inf residual norm!");
        m_converged = false;
    }

    if (m_converged) {
        getChiSolution(m_mf_chi); 
    } else {
        m_mf_chi.setVal(0.0); 
        if (m_mf_chi.nGrow() > 0) m_mf_chi.FillBoundary(m_geom.periodicity());
        if (m_verbose >=0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Warning("Solver did not converge. Chi solution (m_mf_chi) set to 0.");
        }
    }

    if (m_write_plotfile && (m_converged || num_active_cells_in_solve == 0) ) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Writing solution plotfile for chi_k in direction "
                            << static_cast<int>(m_dir_solve) << "..." << std::endl;
        }
        amrex::MultiFab mf_plot(m_ba, m_dm, 2, 0);
        amrex::Copy(mf_plot, m_mf_chi, ChiComp, 0, 1, 0);

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(mf_plot, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx_plot = mfi.tilebox(); 
            amrex::Array4<amrex::Real> const plot_arr = mf_plot.array(mfi);
            amrex::Array4<const int> const mask_arr_plot = m_mf_active_mask.const_array(mfi); 

            amrex::LoopOnCpu(bx_plot, [=] (int i, int j, int k) noexcept
            {
                plot_arr(i,j,k,1) = static_cast<amrex::Real>(mask_arr_plot(i,j,k,MaskComp));
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
            amrex::Warning("Skipping plotfile write for chi_k because solver did not converge and had active cells.");
        }
    }
    return m_converged;
}

void EffectiveDiffusivityHypre::getChiSolution(amrex::MultiFab& chi_field)
{
    BL_PROFILE("EffectiveDiffusivityHypre::getChiSolution");

    if (!m_x) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  getChiSolution: HYPRE solution vector m_x is NULL. Setting chi_field to 0." << std::endl;
        }
        chi_field.setVal(0.0);
        if (chi_field.nGrow() > 0) {
             chi_field.FillBoundary(m_geom.periodicity());
        }
        return;
    }
    AMREX_ALWAYS_ASSERT(chi_field.nComp() >= numComponentsChi); 
    AMREX_ALWAYS_ASSERT(chi_field.boxArray() == m_ba);
    AMREX_ALWAYS_ASSERT(chi_field.DistributionMap() == m_dm);

    std::vector<amrex::Real> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(chi_field, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx_getsol = mfi.validbox(); 
        const int npts = static_cast<int>(bx_getsol.numPts());
        if (npts == 0) continue;

        soln_buffer.resize(npts);

        auto hypre_lo = EffectiveDiffusivityHypre::loV(bx_getsol);
        auto hypre_hi = EffectiveDiffusivityHypre::hiV(bx_getsol);

        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
        if (get_ierr != 0) {
            amrex::Warning("HYPRE_StructVectorGetBoxValues failed during getChiSolution!");
            chi_field[mfi].setVal(0.0, bx_getsol, ChiComp, numComponentsChi);
            continue;
        }

        amrex::Array4<amrex::Real> const chi_arr = chi_field.array(mfi);
        long long k_lin_idx = 0;

        amrex::LoopOnCpu(bx_getsol, [=,&k_lin_idx] (int i, int j, int k) noexcept
        {
            if (k_lin_idx < npts) { 
                chi_arr(i,j,k, ChiComp) = soln_buffer[k_lin_idx]; 
            }
            k_lin_idx++;
        });
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k_lin_idx == npts, "Point count mismatch during HYPRE GetBoxValues copy in getChiSolution");
    }

    if (chi_field.nGrow() > 0) {
        chi_field.FillBoundary(m_geom.periodicity());
    }
}

} // End namespace OpenImpala
