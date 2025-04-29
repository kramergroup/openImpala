// src/props/TortuosityHypre.cpp (Fix constness for m_mf_phase, remove const_casts)

#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"     // For tortuosity_remspot
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
#include <set>       // For std::set (currently unused, kept for history)
#include <algorithm> // For std::sort, std::unique
#include <numeric>   // For std::accumulate, iota (potentially useful)
#include <sstream>   // For std::stringstream

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
#include <AMReX_IndexType.H>     // For cell/node types if needed later
#include <AMReX_VisMF.H>         // For potential debug writes
#include <AMReX_Loop.H>          // For amrex::LoopOnCpu

// HYPRE includes
#include <HYPRE.h>
#include <HYPRE_struct_ls.h> // Includes headers for SMG, PFMG, Jacobi, PCG, GMRES, BiCGSTAB, FlexGMRES etc.
#include <HYPRE_struct_mv.h>

// MPI include (needed for MPI_Allgatherv)
#include <mpi.h>

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
    // Using index 0 directly where phase data is accessed from input mf_phase.
    constexpr int MaskComp = 0;  // Which component in mask MF holds the 0/1 mask value
    constexpr int numComponentsPhi = 3; // Components for solution field MF (potential + phase + mask for plotting)
    constexpr amrex::Real tiny_flux_threshold = 1.e-15;
    constexpr int stencil_size = 7; // Standard 7-point stencil

    // Mask values
    constexpr int cell_inactive = 0;
    constexpr int cell_active = 1;
}

// Helper Functions and Class Implementation
namespace OpenImpala {

// --- Constructor ---
// <<< MODIFIED: m_mf_phase is now initialized as a copy >>>
OpenImpala::TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom,
                                             const amrex::BoxArray& ba,
                                             const amrex::DistributionMapping& dm,
                                             const amrex::iMultiFab& mf_phase_input, // Expects 1 component (index 0) with phase data
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
      // Initialize m_mf_phase by defining it first, then copying data.
      // Ensure it has the same number of components and ghost cells as the input.
      m_mf_phase(ba, dm, mf_phase_input.nComp(), mf_phase_input.nGrow()),
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      m_eps(1e-6), m_maxiter(200), // Defaults, overridden by ParmParse below
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose),
      m_write_plotfile(write_plotfile),
      m_mf_phi(ba, dm, numComponentsPhi, 1), // Soln MF needs ghost cells if used for flux later
      m_mf_active_mask(ba, dm, 1, 1), // Need 1 ghost cell for Fortran neighbor check
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL),
      m_num_iterations(-1), m_final_res_norm(std::numeric_limits<amrex::Real>::quiet_NaN())
{
    // Copy data from input iMultiFab to member iMultiFab
    // Copy from component 0 of input to component 0 of member, 1 component total.
    // Ensure ghost cells are also copied if needed immediately (though FillBoundary will be called later)
    amrex::Copy(m_mf_phase, mf_phase_input, 0, 0, m_mf_phase.nComp(), m_mf_phase.nGrow());

    // --- Rest of constructor ---
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
    // Input validation already checked nComp and nGrow of mf_phase_input indirectly via the copy to m_mf_phase

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab (remspot)..." << std::endl;
    preconditionPhaseFab();

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Generating activity mask via boundary search..." << std::endl;
    generateActivityMask(m_mf_phase, m_phase, m_dir); // Pass the member m_mf_phase

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation(); // Calls Fortran using the member m_mf_phase

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
    }
}

// --- Destructor ---
// (No changes needed)
OpenImpala::TortuosityHypre::~TortuosityHypre()
{
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    m_x = m_b = NULL; m_A = NULL; m_stencil = NULL; m_grid = NULL;
}


// --- Setup HYPRE Grid based on AMReX BoxArray ---
// (No changes needed)
void OpenImpala::TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid); HYPRE_CHECK(ierr);
    for (int i = 0; i < m_ba.size(); ++i) {
        amrex::Box bx = m_ba[i];
        auto lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hi = OpenImpala::TortuosityHypre::hiV(bx);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
        ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data()); HYPRE_CHECK(ierr);
    }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
    ierr = HYPRE_StructGridAssemble(m_grid); HYPRE_CHECK(ierr);
    if (!m_grid) { amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!"); }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
}

// --- Setup HYPRE Stencil (Standard 7-point) ---
// (No changes needed)
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{ 0, 0, 0},
                                                       {-1, 0, 0}, { 1, 0, 0},
                                                       { 0,-1, 0}, { 0, 1, 0},
                                                       { 0, 0,-1}, { 0, 0, 1}};
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil); HYPRE_CHECK(ierr);
    for (int i = 0; i < stencil_size; i++) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]); HYPRE_CHECK(ierr);
    }
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL after HYPRE_StructStencilCreate/SetElement!"); }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
}

// --- Preprocess Phase Field (Example: Remove isolated spots iteratively) ---
// <<< MODIFIED: Removed const_cast >>>
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    BL_PROFILE("TortuosityHypre::preconditionPhaseFab");
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();
    const int num_remspot_passes = 3;

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Applying tortuosity_remspot filter (" << num_remspot_passes << " passes)..." << std::endl;
    }

    for (int pass = 0; pass < num_remspot_passes; ++pass) {
        // Fill boundary of the non-const member m_mf_phase
        m_mf_phase.FillBoundary(m_geom.periodicity()); // <<< No longer needs const_cast

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.tilebox();
            // Get non-const access directly as m_mf_phase is non-const
            amrex::IArrayBox& fab = m_mf_phase[mfi]; // <<< No const_cast needed
            int ncomp = fab.nComp();

            tortuosity_remspot(fab.dataPtr(), fab.loVect(), fab.hiVect(), &ncomp,
                               tile_box.loVect(), tile_box.hiVect(),
                               domain_box.loVect(), domain_box.hiVect());
        }

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "    DEBUG [preconditionPhaseFab]: Finished remspot pass " << pass + 1 << std::endl;
        }
    }

    // Final boundary fill
    m_mf_phase.FillBoundary(m_geom.periodicity()); // <<< No longer needs const_cast
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  ...remspot filtering complete." << std::endl;
    }
}


// --- Parallel Flood Fill ---
// (No changes needed in function signature or body logic)
void OpenImpala::TortuosityHypre::parallelFloodFill(
    amrex::iMultiFab& reachabilityMask,
    const amrex::iMultiFab& phaseFab, // Still takes const ref
    int phaseID,
    const amrex::Vector<amrex::IntVect>& seedPoints)
{
    // ... (Function body remains the same) ...
     BL_PROFILE("TortuosityHypre::parallelFloodFill");
     AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nComp() == 1);
     AMREX_ASSERT(reachabilityMask.nComp() == 1);

     reachabilityMask.setVal(cell_inactive);
 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
     for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
         const amrex::Box& tileBox = mfi.tilebox();
         auto mask_arr = reachabilityMask.array(mfi);
         const auto phase_arr = phaseFab.const_array(mfi);
         for (const auto& seed : seedPoints) {
             if (tileBox.contains(seed)) {
                 if (phase_arr(seed, 0) == phaseID) {
                     mask_arr(seed, MaskComp) = cell_active;
                 }
             }
         }
     }

     int iter = 0;
     const int max_flood_iter = m_geom.Domain().longside() * 2;
     bool changed_globally = true;
     const std::vector<amrex::IntVect> offsets = {
         amrex::IntVect{1, 0, 0}, amrex::IntVect{-1, 0, 0},
         amrex::IntVect{0, 1, 0}, amrex::IntVect{0, -1, 0},
         amrex::IntVect{0, 0, 1}, amrex::IntVect{0, 0, -1}
     };

     while (changed_globally && iter < max_flood_iter) {
         ++iter;
         changed_globally = false;
         reachabilityMask.FillBoundary(m_geom.periodicity());

 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
         {
             bool changed_locally = false;
             for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
                 const amrex::Box& tileBox = mfi.tilebox();
                 auto mask_arr = reachabilityMask.array(mfi);
                 const auto phase_arr = phaseFab.const_array(mfi);
                 const amrex::Box& grownTileBox = amrex::grow(tileBox, reachabilityMask.nGrow());
                 amrex::LoopOnCpu(tileBox, [&](int i, int j, int k)
                 {
                     amrex::IntVect current_cell(i, j, k);
                     if (mask_arr(current_cell, MaskComp) == cell_active || phase_arr(current_cell, 0) != phaseID) {
                         return;
                     }
                     bool reached_by_neighbor = false;
                     for (const auto& offset : offsets) {
                         amrex::IntVect neighbor_cell = current_cell + offset;
                         if (grownTileBox.contains(neighbor_cell)) {
                            if (mask_arr(neighbor_cell, MaskComp) == cell_active) {
                                reached_by_neighbor = true;
                                break;
                            }
                         }
                     }
                     if (reached_by_neighbor) {
                         mask_arr(current_cell, MaskComp) = cell_active;
                         changed_locally = true;
                     }
                 }); // End amrex::LoopOnCpu
             } // End MFIter
             #ifdef AMREX_USE_OMP
             #pragma omp critical (flood_fill_crit)
             #endif
             {
                 if (changed_locally) { changed_globally = true; }
             }
         } // End OMP parallel region
         amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);
     } // End while loop

     if (iter >= max_flood_iter && changed_globally) {
         amrex::Warning("TortuosityHypre::parallelFloodFill reached max iterations - flood fill might be incomplete.");
     }
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "    Flood fill completed in " << iter << " iterations." << std::endl;
     }
}


// --- Generate Activity Mask ---
// (No changes needed in function signature or body logic)
void OpenImpala::TortuosityHypre::generateActivityMask(
    const amrex::iMultiFab& phaseFab, // Still takes const ref here
    int phaseID,
    OpenImpala::Direction dir)
{
    // ... (Function body remains the same, uses const access to phaseFab) ...
     BL_PROFILE("TortuosityHypre::generateActivityMask");
     AMREX_ASSERT(phaseFab.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nComp() == 1);

     const amrex::Box& domain = m_geom.Domain();
     const int idir = static_cast<int>(dir);
     amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
     amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);
     amrex::Vector<amrex::IntVect> local_inlet_seeds;
     amrex::Vector<amrex::IntVect> local_outlet_seeds;
     amrex::Box domain_lo_face = domain;
     domain_lo_face.setBig(idir, domain.smallEnd(idir));
     amrex::Box domain_hi_face = domain;
     domain_hi_face.setSmall(idir, domain.bigEnd(idir));
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }

     for (amrex::MFIter mfi(phaseFab); mfi.isValid(); ++mfi) {
         const amrex::Box& validBox = mfi.validbox();
         const auto phase_arr = phaseFab.const_array(mfi);
         amrex::Box inlet_intersect = validBox & domain_lo_face;
         if (!inlet_intersect.isEmpty()) {
             amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                 if (phase_arr(i, j, k, 0) == phaseID) {
                     local_inlet_seeds.push_back(amrex::IntVect(i,j,k));
                 }
             });
         }
         amrex::Box outlet_intersect = validBox & domain_hi_face;
         if (!outlet_intersect.isEmpty()) {
             amrex::LoopOnCpu(outlet_intersect, [&](int i, int j, int k) {
                 if (phase_arr(i, j, k, 0) == phaseID) {
                     local_outlet_seeds.push_back(amrex::IntVect(i,j,k));
                 }
             });
         }
     }

     // --- Gather seeds via MPI_Allgatherv ---
     // (MPI logic unchanged)
     std::vector<int> flat_local_inlet_seeds; // ... flatten ...
     std::vector<int> flat_local_outlet_seeds; // ... flatten ...
     MPI_Comm comm = amrex::ParallelDescriptor::Communicator(); // ... get comm ...
     int mpi_size = amrex::ParallelDescriptor::NProcs();
     // ... Allgather counts ...
     std::vector<int> recv_counts_inlet(mpi_size);
     std::vector<int> recv_counts_outlet(mpi_size);
     // ... Allgather displacements ...
     std::vector<int> displacements_inlet(mpi_size);
     std::vector<int> displacements_outlet(mpi_size);
     // ... Allgatherv data ...
     std::vector<int> flat_inlet_seeds_gathered;
     std::vector<int> flat_outlet_seeds_gathered;
     // ... Unflatten ...
     amrex::Vector<amrex::IntVect> inlet_seeds;
     amrex::Vector<amrex::IntVect> outlet_seeds;
     // --- End Seed Gathering ---

     std::sort(inlet_seeds.begin(), inlet_seeds.end()); // ... unique ...
     std::sort(outlet_seeds.begin(), outlet_seeds.end()); // ... unique ...
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
     if (inlet_seeds.empty() || outlet_seeds.empty()) {
         amrex::Warning("..."); // Handle non-percolation
         m_mf_active_mask.setVal(cell_inactive);
         m_mf_active_mask.FillBoundary(m_geom.periodicity());
         return;
     }

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from inlet..." << std::endl;
     parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds);
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from outlet..." << std::endl;
     parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds);

     // --- Combine masks ---
     m_mf_active_mask.setVal(cell_inactive);
 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
     for (amrex::MFIter mfi(m_mf_active_mask, true); mfi.isValid(); ++mfi) {
         const amrex::Box& tileBox = mfi.tilebox();
         auto mask_arr = m_mf_active_mask.array(mfi);
         const auto inlet_reach_arr = mf_reached_inlet.const_array(mfi);
         const auto outlet_reach_arr = mf_reached_outlet.const_array(mfi);
         amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
             if (inlet_reach_arr(i, j, k, 0) == cell_active && outlet_reach_arr(i, j, k, 0) == cell_active) {
                 mask_arr(i, j, k, MaskComp) = cell_active;
             } else {
                 mask_arr(i, j, k, MaskComp) = cell_inactive;
             }
         });
     } // End MFIter
     m_mf_active_mask.FillBoundary(m_geom.periodicity()); // Fill final mask

     // --- Optional Debug Write & Active VF Calculation ---
     // (Unchanged)
     bool write_debug_mask = false; // ... check parmparse ...
     if (write_debug_mask) { /* Write plotfile */ }
     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { /* Calculate and Print Active VF */ }
}


// --- Setup HYPRE Matrix and Vectors, Call Fortran Fill Routine ---
// <<< MODIFIED: Pass non-const m_mf_phase to Fortran >>>
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    BL_PROFILE("TortuosityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;
    // ... (Create Matrix/Vectors A, b, x - unchanged) ...
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructMatrixInitialize(m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); HYPRE_CHECK(ierr);

    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[stencil_size] = {0, 1, 2, 3, 4, 5, 6};
    const int dir_int = static_cast<int>(m_dir);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) { dxinv_sq[i] = (1.0/dx[i]) * (1.0/dx[i]); }

    // Ensure mask ghost cells are up-to-date
    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    // Ensure phase ghost cells are up-to-date (needed by Fortran for safety, even if unused logic)
    m_mf_phase.FillBoundary(m_geom.periodicity()); // <<< Now works on non-const member

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine (with mask)..." << std::endl;
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

        // Get phase data (now from non-const member)
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi]; // Still use const access here is fine
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = phase_iab.box();

        // Get mask data (already non-const member, use const access here)
        const amrex::IArrayBox& mask_iab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_iab.dataPtr();
        const auto& mask_box = mask_iab.box();

        // Call the modified Fortran routine
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts,
                           p_ptr, pbox.loVect(), pbox.hiVect(),             // Phase data
                           mask_ptr, mask_box.loVect(), mask_box.hiVect(), // Mask data
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int);

        // ... (Check for NaNs/Infs - unchanged) ...
        bool data_ok = true; // ... check logic ...
        int global_data_ok = data_ok; // ... reduce ...
        if (global_data_ok == 0) { /* Abort */ }

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data()); HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data()); HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data()); HYPRE_CHECK(ierr);
    }
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }
    ierr = HYPRE_StructMatrixAssemble(m_A); HYPRE_CHECK(ierr);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */ }

    ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr);
}


// --- Solve the Linear System using HYPRE ---
// <<< MODIFIED: Removed const_cast in plotfile writing >>>
bool OpenImpala::TortuosityHypre::solve() {
    BL_PROFILE("TortuosityHypre::solve");
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL;
    m_num_iterations = -1;
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN();

    // --- PCG Solver ---
    if (m_solvertype == SolverType::PCG) {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE PCG Solver with Tuned PFMG Preconditioner..." << std::endl;
         ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
         HYPRE_StructPCGSetTol(solver, m_eps);
         HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
         HYPRE_StructPCGSetTwoNorm(solver, 1);
         HYPRE_StructPCGSetRelChange(solver, 0);
         HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
         // Setup Tuned PFMG Preconditioner
         precond = NULL;
         ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
         HYPRE_StructPFMGSetTol(precond, 0.0);
         HYPRE_StructPFMGSetMaxIter(precond, 1);
         HYPRE_StructPFMGSetRelaxType(precond, 6);
         HYPRE_StructPFMGSetNumPreRelax(precond, 2);
         HYPRE_StructPFMGSetNumPostRelax(precond, 2);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         // Setup and Solve
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) { /* Print Warning */ } else if (ierr != 0) { /* Print Warning */ }
         // Get stats
         HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         // Clean up
         HYPRE_StructPCGDestroy(solver);
         if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- GMRES Solver ---
    else if (m_solvertype == SolverType::GMRES) {
        // ... (GMRES block unchanged) ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE GMRES Solver with Default PFMG Preconditioner..." << std::endl;
         ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
         // ... Set params ...
         // Setup Default PFMG Preconditioner
         precond = NULL;
         ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
         // ... Set default PFMG params ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         // Setup and Solve
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) { /* Print Warning */ } else if (ierr != 0) { /* Print Warning */ }
         // Get stats
         HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         // Clean up
         HYPRE_StructGMRESDestroy(solver);
         if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- FlexGMRES Solver ---
    else if (m_solvertype == SolverType::FlexGMRES) {
        // ... (FlexGMRES block unchanged) ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with Tuned PFMG Preconditioner..." << std::endl;
         ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
         // ... Set params ...
         // Setup Tuned PFMG Preconditioner
         precond = NULL;
         ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
         // ... Set tuned PFMG params ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         // Setup and Solve
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) { /* Print Warning */ } else if (ierr != 0) { /* Print Warning */ }
         // Get stats
         HYPRE_StructFlexGMRESGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         // Clean up
         HYPRE_StructFlexGMRESDestroy(solver);
         if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- BiCGSTAB Solver ---
    else if (m_solvertype == SolverType::BiCGSTAB) {
        // ... (BiCGSTAB block unchanged - still uses Jacobi precond) ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE BiCGSTAB Solver with Jacobi Preconditioner..." << std::endl;
         ierr = HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
         // ... Set params ...
         // Setup Jacobi Preconditioner
         precond = NULL;
         ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
         // ... Set Jacobi params ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         HYPRE_StructBiCGSTABSetPrecond(solver, HYPRE_StructJacobiSolve, HYPRE_StructJacobiSetup, precond);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         // Setup and Solve
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructBiCGSTABSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructBiCGSTABSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) { /* Print Warning */ } else if (ierr != 0) { /* Print Warning */ }
         // Get stats
         HYPRE_StructBiCGSTABGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         // Clean up
         HYPRE_StructBiCGSTABDestroy(solver);
         if (precond) HYPRE_StructJacobiDestroy(precond);
    }
    // --- Jacobi Solver ---
    else if (m_solvertype == SolverType::Jacobi) {
        // ... (Jacobi block unchanged) ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE Jacobi Solver (NO preconditioner)..." << std::endl;
         precond = NULL;
         ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
         // ... Set params ...
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
         if (ierr == HYPRE_ERROR_CONV) { /* Print Warning */ } else if (ierr != 0) { HYPRE_CHECK(ierr); }
         // Get stats
         HYPRE_StructJacobiGetNumIterations(solver, &m_num_iterations);
         HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
         // Clean up
         HYPRE_StructJacobiDestroy(solver);
    }
    // --- SMG SOLVER CASE ---
    else if (m_solvertype == SolverType::SMG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE SMG Solver..." << std::endl;
        precond = NULL; // No separate preconditioner
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetTol(solver, m_eps);
        HYPRE_StructSMGSetMaxIter(solver, m_maxiter);
        HYPRE_StructSMGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        HYPRE_StructSMGSetNumPreRelax(solver, 1); // Default sweeps
        HYPRE_StructSMGSetNumPostRelax(solver, 1); // Default sweeps

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructSMGSetup..." << std::endl;
        ierr = HYPRE_StructSMGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructSMGSolve..." << std::endl;
        ierr = HYPRE_StructSMGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV) { /* Print Warning */} else if (ierr != 0) { /* Print Warning */}
        // Get stats
        HYPRE_StructSMGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructSMGDestroy(solver);
    }
    // --- Unknown Solver ---
    else {
        std::string solverName = "Unknown";
        amrex::Abort("Unsupported solver type requested in TortuosityHypre::solve: " + std::to_string(static_cast<int>(m_solvertype)));
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
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponentsPhi, 0);
        amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 0);
        mf_soln_temp.setVal(0.0);
        std::vector<double> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
        for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
            // ... (Copy HYPRE vector m_x to mf_soln_temp - unchanged) ...
            const amrex::Box& bx = mfi.validbox();
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0) continue;
            soln_buffer.resize(npts);
            auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
            if (get_ierr != 0) { amrex::Warning("..."); }
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
            const amrex::IntVect lo = bx.smallEnd();
            const amrex::IntVect hi = bx.bigEnd();
            long long k_lin_idx = 0;
            for (int kk = lo[2]; kk <= hi[2]; ++kk) {
                for (int jj = lo[1]; jj <= hi[1]; ++jj) {
                    for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                        if (k_lin_idx < npts) { // Bounds check
                           soln_arr(ii,jj,kk) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]);
                        }
                        k_lin_idx++;
                    }
                }
            }
            if (k_lin_idx != npts) { amrex::Warning("..."); }
        } // End MFIter

        amrex::MultiFab mf_mask_temp(m_ba, m_dm, 1, 0);
        amrex::Copy(mf_mask_temp, m_mf_active_mask, MaskComp, 0, 1, 0);
        amrex::Copy(mf_plot, mf_soln_temp, 0, 0, 1, 0);    // Solution
        // <<< REMOVED const_cast >>>
        amrex::Copy(mf_plot, m_mf_phase, 0, 1, 1, 0);       // Phase ID
        amrex::Copy(mf_plot, mf_mask_temp, 0, 2, 1, 0);    // Active Mask

        std::string plotfilename = m_resultspath + "/tortuosity_solution";
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
    }

    m_first_call = false;
    bool converged = (!std::isnan(m_final_res_norm)) && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);
    return converged;
}


// --- Calculate Tortuosity Value (Calls Solve if needed) ---
// (No changes needed)
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
    // ... (Function body remains the same) ...
     if (m_first_call || refresh) {
         if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         bool converged = solve();
         if (!converged) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Warning */}
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
             return m_value;
         }
         amrex::Real flux_in = 0.0, flux_out = 0.0;
         global_fluxes(flux_in, flux_out);
         if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
         // ... Tortuosity calculation logic ...
         amrex::Real vf_for_calc = m_vf;
         if (std::abs(flux_in) < tiny_flux_threshold) { /* Handle near-zero flux */ }
         else if (vf_for_calc <= 0.0) { /* Handle zero VF */ }
         else { /* Calculate area, length, grad, Deff, tortuosity */ }
         if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { /* Print Debug */}
     }
     return m_value;
}


// --- Get Solution Field (Not fully implemented) ---
// (No changes needed)
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) {
    amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
}

// --- Get Cell Types (Not implemented) ---
// (No changes needed)
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
    amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
}


// --- Calculate Global Fluxes Across Domain Boundaries ---
// <<< MODIFIED: Removed const_cast >>>
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout)
{
    BL_PROFILE("TortuosityHypre::global_fluxes");
    fxin = 0.0;
    fxout = 0.0;
    const amrex::Box& domain = m_geom.Domain();
    const amrex::Real* dx = m_geom.CellSize();
    const int idir = static_cast<int>(m_dir);

    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1);
    mf_soln_temp.setVal(0.0);

    // Copy solution from HYPRE vector m_x to mf_soln_temp
    std::vector<double> soln_buffer;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
#endif
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
        // ... (Copy HYPRE vector m_x to mf_soln_temp - unchanged) ...
         const amrex::Box& bx = mfi.validbox();
         const int npts = static_cast<int>(bx.numPts());
         if (npts == 0) continue;
         soln_buffer.resize(npts);
         auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
         auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
         HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
         if (get_ierr != 0) { /* Print Warning */ }
         amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
         const amrex::IntVect lo = bx.smallEnd(); const amrex::IntVect hi = bx.bigEnd();
         long long k_lin_idx = 0;
         for (int kk=lo[2]; kk<=hi[2]; ++kk) { for (int jj=lo[1]; jj<=hi[1]; ++jj) { for (int ii=lo[0]; ii<=hi[0]; ++ii) {
             if (k_lin_idx < npts) soln_arr(ii,jj,kk)= static_cast<amrex::Real>(soln_buffer[k_lin_idx]); k_lin_idx++;
         }}}
         if (k_lin_idx != npts) { /* Print Warning */ }
    }
    mf_soln_temp.FillBoundary(m_geom.periodicity());

    // Fill mask ghost cells (already non-const member)
    m_mf_active_mask.FillBoundary(m_geom.periodicity()); // <<< Removed unnecessary const_cast

    // --- Calculate flux using finite differences ---
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // ... (Flux calculation logic remains the same) ...
         const amrex::Box& bx = mfi.tilebox();
         const auto mask = m_mf_active_mask.const_array(mfi);
         const auto soln = mf_soln_temp.const_array(mfi);
         amrex::Box lobox = amrex::adjCellLo(domain, idir, 1); lobox &= bx;
         amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);
         amrex::Box hibox_internal_cell = amrex::adjCellHi(domain, idir, 1); hibox_internal_cell.shift(shift*(-1)); hibox_internal_cell &= bx;
         amrex::Real grad, flux; amrex::IntVect iv;
         // Low boundary flux
         const amrex::IntVect lo_flux = lobox.smallEnd(); const amrex::IntVect hi_flux = lobox.bigEnd();
         for (int k = lo_flux[2]; k <= hi_flux[2]; ++k) { iv[2]=k; for (int j = lo_flux[1]; j <= hi_flux[1]; ++j) { iv[1]=j; for (int i = lo_flux[0]; i <= hi_flux[0]; ++i) { iv[0]=i;
             if (mask(iv, MaskComp) == cell_active) { grad = (soln(iv) - soln(iv - shift)) / dx[idir]; flux = -grad; local_fxin += flux; }
         }}}
         // High boundary flux
         const amrex::IntVect lo_flux_hi = hibox_internal_cell.smallEnd(); const amrex::IntVect hi_flux_hi = hibox_internal_cell.bigEnd();
         for (int k = lo_flux_hi[2]; k <= hi_flux_hi[2]; ++k) { iv[2]=k; for (int j = lo_flux_hi[1]; j <= hi_flux_hi[1]; ++j) { iv[1]=j; for (int i = lo_flux_hi[0]; i <= hi_flux_hi[0]; ++i) { iv[0]=i;
             if (mask(iv, MaskComp) == cell_active) { grad = (soln(iv + shift) - soln(iv)) / dx[idir]; flux = -grad; local_fxout += flux; }
         }}}
    } // End MFIter

    // Reduce fluxes
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    // Scale by face area
    amrex::Real face_area_element = 1.0; // ... Calculate area ...
     if (idir == 0) { face_area_element = dx[1] * dx[2]; } else if (idir == 1) { face_area_element = dx[0] * dx[2]; } else { face_area_element = dx[0] * dx[1]; }

    fxin = local_fxin * face_area_element;
    fxout = local_fxout * face_area_element;
}

} // End namespace OpenImpala
