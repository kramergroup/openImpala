// src/props/TortuosityHypre.cpp (Fix phase array access in generateActivityMask)

#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"     // For tortuosity_remspot
#include "TortuosityHypreFill_F.H" // For tortuosity_fillmtx

#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <cmath>     // Required for std::isnan, std::isinf, std::abs
#include <limits>    // Required for std::numeric_limits
#include <stdexcept> // For potential error throwing (optional)
#include <iomanip>   // For std::setprecision
#include <iostream>  // For std::cout, std::flush
#include <set>       // For std::set (currently unused, kept for history)
#include <algorithm> // For std::sort, std::unique
#include <numeric>   // For std::accumulate, iota (potentially useful)
#include <sstream>   // For std::stringstream

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H> // Needed for amrex::Copy
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

    // Stencil Indices (matching Fortran)
    constexpr int istn_c  = 0; // Center
    constexpr int istn_mx = 1; // -X (West)
    constexpr int istn_px = 2; // +X (East)
    constexpr int istn_my = 3; // -Y (South)
    constexpr int istn_py = 4; // +Y (North)
    constexpr int istn_mz = 5; // -Z (Bottom)
    constexpr int istn_pz = 6; // +Z (Top)
}

// Helper Functions and Class Implementation
namespace OpenImpala {

amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::loV (const amrex::Box& b) {
    const int* lo_ptr = b.loVect();
    amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hypre_lo;
    for (int i=0; i<AMREX_SPACEDIM; ++i) hypre_lo[i] = static_cast<HYPRE_Int>(lo_ptr[i]);
    return hypre_lo;
}

amrex::Array<HYPRE_Int,AMREX_SPACEDIM> TortuosityHypre::hiV (const amrex::Box& b) {
    const int* hi_ptr = b.hiVect();
    amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hypre_hi;
    for (int i=0; i<AMREX_SPACEDIM; ++i) hypre_hi[i] = static_cast<HYPRE_Int>(hi_ptr[i]);
    return hypre_hi;
}

// --- Constructor ---
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
      m_eps(1e-6), m_maxiter(1000), // Increased default maxiter
      m_vlo(vlo), m_vhi(vhi),
      m_resultspath(resultspath), m_verbose(verbose),
      m_write_plotfile(write_plotfile),
      m_mf_phi(ba, dm, numComponentsPhi, 1), // Soln MF needs ghost cells if used for flux later
      m_mf_active_mask(ba, dm, 1, 1), // Need 1 ghost cell for Fortran neighbor check & mask checks
      m_first_call(true), m_value(std::numeric_limits<amrex::Real>::quiet_NaN()),
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL),
      m_num_iterations(-1), m_final_res_norm(std::numeric_limits<amrex::Real>::quiet_NaN())
{
    // Copy data from input iMultiFab to member iMultiFab
    amrex::Copy(m_mf_phase, mf_phase_input, 0, 0, m_mf_phase.nComp(), m_mf_phase.nGrow());

    // --- Rest of constructor ---
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initializing..." << std::endl;
    }

    amrex::ParmParse pp("hypre");
    pp.query("eps", m_eps);
    pp.query("maxiter", m_maxiter);
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("verbose", m_verbose); // Allow overriding verbose level

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Params: eps=" << m_eps << ", maxiter=" << m_maxiter << std::endl;
        amrex::Print() << "  Class Verbose Level: " << m_verbose << std::endl;
        amrex::Print() << "  Write Plotfile Flag: " << m_write_plotfile << std::endl;
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0, "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab (remspot)..." << std::endl;
    preconditionPhaseFab(); // Modifies m_mf_phase

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Generating activity mask via boundary search..." << std::endl;
    generateActivityMask(m_mf_phase, m_phase, m_dir); // Pass the (potentially modified) member m_mf_phase, fills m_mf_active_mask

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation(); // Fills m_A, m_b, m_x using m_mf_phase and m_mf_active_mask

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Initialization complete." << std::endl;
    }
}

// --- Destructor ---
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
void OpenImpala::TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;
    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid); HYPRE_CHECK(ierr);
    for (int i = 0; i < m_ba.size(); ++i) {
        if (m_dm[i] == amrex::ParallelDescriptor::MyProc()) { // Process only local boxes
            amrex::Box bx = m_ba[i];
            auto lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hi = OpenImpala::TortuosityHypre::hiV(bx);
            if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupGrids: Rank " << amrex::ParallelDescriptor::MyProc() << " adding box " << bx << std::endl; }
            ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data()); HYPRE_CHECK(ierr);
        }
    }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupGrids: Calling Assemble..." << std::endl; }
    ierr = HYPRE_StructGridAssemble(m_grid); HYPRE_CHECK(ierr);
    if (!m_grid) { amrex::Abort("FATAL: m_grid handle is NULL after HYPRE_StructGridAssemble!"); }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupGrids: Assemble complete." << std::endl; }
}

// --- Setup HYPRE Stencil (Standard 7-point) ---
void OpenImpala::TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    // Define offsets matching Fortran indices: C, -X, +X, -Y, +Y, -Z, +Z
    HYPRE_Int offsets[stencil_size][AMREX_SPACEDIM] = {{ 0, 0, 0},
                                                       {-1, 0, 0}, { 1, 0, 0},
                                                       { 0,-1, 0}, { 0, 1, 0},
                                                       { 0, 0,-1}, { 0, 0, 1}};
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupStencil: Creating..." << std::endl; }
    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil); HYPRE_CHECK(ierr);
    for (int i = 0; i < stencil_size; i++) {
        if (m_verbose > 3 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    Setting stencil element " << i << std::endl; }
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]); HYPRE_CHECK(ierr);
    }
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL after HYPRE_StructStencilCreate/SetElement!"); }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupStencil: Complete." << std::endl; }
}

// --- Preprocess Phase Field (Example: Remove isolated spots iteratively) ---
void OpenImpala::TortuosityHypre::preconditionPhaseFab()
{
    BL_PROFILE("TortuosityHypre::preconditionPhaseFab");
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();
    const int num_remspot_passes = 3; // Number of filter passes

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Applying tortuosity_remspot filter (" << num_remspot_passes << " passes)..." << std::endl;
    }

    for (int pass = 0; pass < num_remspot_passes; ++pass) {
        // Fill boundary of the non-const member m_mf_phase
        m_mf_phase.FillBoundary(m_geom.periodicity());

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.tilebox();
            // Get non-const access directly as m_mf_phase is non-const
            amrex::IArrayBox& fab = m_mf_phase[mfi];
            int ncomp = fab.nComp(); // Should be 1

            // --- CORRECTED Fortran CALL ---
            // Pass pointers directly, remove AMREX_ARLIM for vector bounds
            tortuosity_remspot(fab.dataPtr(0),         // Data pointer (int*)
                               fab.loVect(), fab.hiVect(), // Data bounds (const int*)
                               &ncomp,                 // Number of components (const int*)
                               tile_box.loVect(), tile_box.hiVect(), // Tile bounds (const int*)
                               domain_box.loVect(), domain_box.hiVect()); // Domain bounds (const int*)
        }

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "    DEBUG [preconditionPhaseFab]: Finished remspot pass " << pass + 1 << std::endl;
        }
    }

    // Final boundary fill
    m_mf_phase.FillBoundary(m_geom.periodicity());
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  ...remspot filtering complete." << std::endl;
    }
}


// --- Parallel Flood Fill ---
void OpenImpala::TortuosityHypre::parallelFloodFill(
    amrex::iMultiFab& reachabilityMask,
    const amrex::iMultiFab& phaseFab, // Still takes const ref
    int phaseID,
    const amrex::Vector<amrex::IntVect>& seedPoints)
{
     BL_PROFILE("TortuosityHypre::parallelFloodFill");
     AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nComp() > 0); // Phase must have at least comp 0
     AMREX_ASSERT(reachabilityMask.nComp() == 1); // Mask must have 1 comp

     reachabilityMask.setVal(cell_inactive);

 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
     for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
         const amrex::Box& tileBox = mfi.tilebox();
         auto mask_arr = reachabilityMask.array(mfi);
         const auto phase_arr = phaseFab.const_array(mfi, 0); // Use component 0 explicitly
         for (const auto& seed : seedPoints) {
             if (tileBox.contains(seed)) {
                 if (phase_arr(seed) == phaseID) { // phase_arr here is already for comp 0
                     mask_arr(seed, MaskComp) = cell_active;
                 }
             }
         }
     }

     int iter = 0;
     amrex::IntVect domain_size = m_geom.Domain().size();
     const int max_flood_iter = domain_size[0] + domain_size[1] + domain_size[2] + 2;
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
                 const auto phase_arr = phaseFab.const_array(mfi, 0); // Use component 0 explicitly
                 const amrex::Box& grownTileBox = amrex::grow(tileBox, reachabilityMask.nGrow());

                 amrex::LoopOnCpu(tileBox, [&](int i, int j, int k)
                 {
                     amrex::IntVect current_cell(i, j, k);
                     // Check using mask comp 0 and phase comp 0
                     if (mask_arr(current_cell, MaskComp) == cell_active || phase_arr(current_cell) != phaseID) {
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
     reachabilityMask.FillBoundary(m_geom.periodicity());
}


// --- Generate Activity Mask ---
void OpenImpala::TortuosityHypre::generateActivityMask(
    const amrex::iMultiFab& phaseFab, // Still takes const ref here
    int phaseID,
    OpenImpala::Direction dir)
{
     BL_PROFILE("TortuosityHypre::generateActivityMask");
     AMREX_ASSERT(phaseFab.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nComp() > 0); // Must have at least component 0

     const amrex::Box& domain = m_geom.Domain();
     const int idir = static_cast<int>(dir);

     amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
     amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);

     amrex::Vector<amrex::IntVect> local_inlet_seeds;
     amrex::Vector<amrex::IntVect> local_outlet_seeds;

     // Manual construction of cell-centered boundary faces
     amrex::Box domain_lo_face = domain;
     domain_lo_face.setBig(idir, domain.smallEnd(idir));

     amrex::Box domain_hi_face = domain;
     domain_hi_face.setSmall(idir, domain.bigEnd(idir));

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "    generateActivityMask: Searching for seeds on faces..." << std::endl;
        amrex::Print() << "     Inlet Face Box: " << domain_lo_face << std::endl;
        amrex::Print() << "     Outlet Face Box: " << domain_hi_face << std::endl;
     }

     #ifdef AMREX_USE_OMP
     #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
     #endif
     for (amrex::MFIter mfi(phaseFab); mfi.isValid(); ++mfi) {
         const amrex::Box& validBox = mfi.validbox();
         const auto phase_arr = phaseFab.const_array(mfi); // Get Array4 for all components

         amrex::Box inlet_intersect = validBox & domain_lo_face;
         if (!inlet_intersect.isEmpty()) {
             amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                 // **** CORRECTED ACCESS ****
                 if (phase_arr(i, j, k, 0) == phaseID) { // Check component 0
                     #ifdef AMREX_USE_OMP
                     #pragma omp critical (inlet_seed_crit)
                     #endif
                     local_inlet_seeds.push_back(amrex::IntVect(i,j,k));
                 }
             });
         }

         amrex::Box outlet_intersect = validBox & domain_hi_face;
         if (!outlet_intersect.isEmpty()) {
             amrex::LoopOnCpu(outlet_intersect, [&](int i, int j, int k) {
                 // **** CORRECTED ACCESS ****
                 if (phase_arr(i, j, k, 0) == phaseID) { // Check component 0
                     #ifdef AMREX_USE_OMP
                     #pragma omp critical (outlet_seed_crit)
                     #endif
                     local_outlet_seeds.push_back(amrex::IntVect(i,j,k));
                 }
             });
         }
     }

     // --- Gather seeds across all MPI ranks ---
     // Flatten local_inlet_seeds
     std::vector<int> flat_local_inlet_seeds(local_inlet_seeds.size() * AMREX_SPACEDIM);
     for (size_t i = 0; i < local_inlet_seeds.size(); ++i) {
         for (int d=0; d<AMREX_SPACEDIM; ++d) flat_local_inlet_seeds[i*AMREX_SPACEDIM+d]=local_inlet_seeds[i][d];
     }
     // Flatten local_outlet_seeds
     std::vector<int> flat_local_outlet_seeds(local_outlet_seeds.size() * AMREX_SPACEDIM);
     for (size_t i = 0; i < local_outlet_seeds.size(); ++i) {
         for (int d=0; d<AMREX_SPACEDIM; ++d) flat_local_outlet_seeds[i*AMREX_SPACEDIM+d]=local_outlet_seeds[i][d];
     }

     MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
     int mpi_size = amrex::ParallelDescriptor::NProcs();
     int my_rank = amrex::ParallelDescriptor::MyProc();

     // Gather inlet counts
     int local_inlet_count = static_cast<int>(flat_local_inlet_seeds.size());
     std::vector<int> recv_counts_inlet(mpi_size);
     MPI_Allgather(&local_inlet_count, 1, MPI_INT, recv_counts_inlet.data(), 1, MPI_INT, comm);
     // Gather outlet counts
     int local_outlet_count = static_cast<int>(flat_local_outlet_seeds.size());
     std::vector<int> recv_counts_outlet(mpi_size);
     MPI_Allgather(&local_outlet_count, 1, MPI_INT, recv_counts_outlet.data(), 1, MPI_INT, comm);

     // Calculate displacements and total counts
     std::vector<int> displacements_inlet(mpi_size, 0);
     std::vector<int> displacements_outlet(mpi_size, 0);
     int total_inlet_seeds = recv_counts_inlet[0];
     int total_outlet_seeds = recv_counts_outlet[0];
     for (int i = 1; i < mpi_size; ++i) {
         displacements_inlet[i]=displacements_inlet[i-1]+recv_counts_inlet[i-1];
         displacements_outlet[i]=displacements_outlet[i-1]+recv_counts_outlet[i-1];
         total_inlet_seeds+=recv_counts_inlet[i];
         total_outlet_seeds+=recv_counts_outlet[i];
     }

     // Gather inlet data
     std::vector<int> flat_inlet_seeds_gathered(total_inlet_seeds);
     MPI_Allgatherv(flat_local_inlet_seeds.data(), local_inlet_count, MPI_INT, flat_inlet_seeds_gathered.data(), recv_counts_inlet.data(), displacements_inlet.data(), MPI_INT, comm);
     // Gather outlet data
     std::vector<int> flat_outlet_seeds_gathered(total_outlet_seeds);
     MPI_Allgatherv(flat_local_outlet_seeds.data(), local_outlet_count, MPI_INT, flat_outlet_seeds_gathered.data(), recv_counts_outlet.data(), displacements_outlet.data(), MPI_INT, comm);

     // Unflatten inlet seeds
     amrex::Vector<amrex::IntVect> inlet_seeds;
     inlet_seeds.reserve(total_inlet_seeds / AMREX_SPACEDIM);
     for (size_t i = 0; i < flat_inlet_seeds_gathered.size(); i += AMREX_SPACEDIM) {
         inlet_seeds.emplace_back(flat_inlet_seeds_gathered[i], flat_inlet_seeds_gathered[i+1], flat_inlet_seeds_gathered[i+2]);
     }
     // Unflatten outlet seeds
     amrex::Vector<amrex::IntVect> outlet_seeds;
     outlet_seeds.reserve(total_outlet_seeds / AMREX_SPACEDIM);
     for (size_t i = 0; i < flat_outlet_seeds_gathered.size(); i += AMREX_SPACEDIM) {
         outlet_seeds.emplace_back(flat_outlet_seeds_gathered[i], flat_outlet_seeds_gathered[i+1], flat_outlet_seeds_gathered[i+2]);
     }
     // --- End Seed Gathering ---

     std::sort(inlet_seeds.begin(), inlet_seeds.end());
     inlet_seeds.erase(std::unique(inlet_seeds.begin(), inlet_seeds.end()), inlet_seeds.end());
     std::sort(outlet_seeds.begin(), outlet_seeds.end());
     outlet_seeds.erase(std::unique(outlet_seeds.begin(), outlet_seeds.end()), outlet_seeds.end());

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "    generateActivityMask: Found " << inlet_seeds.size() << " unique inlet seeds." << std::endl;
         amrex::Print() << "    generateActivityMask: Found " << outlet_seeds.size() << " unique outlet seeds." << std::endl;
     }

     if (inlet_seeds.empty() || outlet_seeds.empty()) {
         amrex::Warning("TortuosityHypre::generateActivityMask: No percolating path found (zero seeds on inlet or outlet face for the specified phase). Mask will be empty.");
         m_mf_active_mask.setVal(cell_inactive);
         m_mf_active_mask.FillBoundary(m_geom.periodicity());
         return;
     }

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from inlet..." << std::endl;
     parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds);

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from outlet..." << std::endl;
     parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds);

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
             // Use component 0 for the single-component reachability masks
             if (inlet_reach_arr(i, j, k, 0) == cell_active && outlet_reach_arr(i, j, k, 0) == cell_active) {
                 mask_arr(i, j, k, MaskComp) = cell_active;
             } else {
                 mask_arr(i, j, k, MaskComp) = cell_inactive;
             }
         });
     } // End MFIter

     m_mf_active_mask.FillBoundary(m_geom.periodicity());

     bool write_debug_mask = false;
     amrex::ParmParse pp_debug("debug");
     pp_debug.query("write_active_mask", write_debug_mask);

     if (write_debug_mask) {
         // --- CORRECTED Namespace ---
         if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Writing debug active mask plotfile..." << std::endl;
         std::string mask_plotfile = m_resultspath + "/debug_active_mask";
         amrex::MultiFab mf_mask_plot(m_ba, m_dm, 1, 0);
         // --- CORRECTED Function Name (convert -> Copy) ---
         amrex::Copy(mf_mask_plot, m_mf_active_mask, 0, 0, 1, 0); // Copy int mask to real MF
         amrex::Vector<std::string> mask_vn = {"active_mask"};
         amrex::WriteSingleLevelPlotfile(mask_plotfile, mf_mask_plot, mask_vn, m_geom, 0.0, 0);
     }

     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         long num_active = m_mf_active_mask.sum(MaskComp);
         long total_cells = m_geom.Domain().numPts();
         amrex::Real active_vf = (total_cells > 0) ? static_cast<amrex::Real>(num_active) / total_cells : 0.0;
         amrex::Print() << "  Active Volume Fraction (percolating phase " << m_phase << "): " << active_vf << std::endl;
     }
}


// --- Setup HYPRE Matrix and Vectors, Call Fortran Fill Routine ---
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    BL_PROFILE("TortuosityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Creating HYPRE Matrix/Vectors..." << std::endl; }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructMatrixInitialize(m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); HYPRE_CHECK(ierr);

    const amrex::Box& domain = m_geom.Domain();
    int stencil_indices[stencil_size] = {istn_c, istn_mx, istn_px, istn_my, istn_py, istn_mz, istn_pz};
    const int dir_int = static_cast<int>(m_dir);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq;
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) { dxinv_sq[i] = (dx[i] > 0.0) ? (1.0/dx[i]) * (1.0/dx[i]) : 0.0; }

    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_phase.FillBoundary(m_geom.periodicity()); // Phase still needed by Fortran (for unused phase_id arg)

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEq: Calling tortuosity_fillmtx Fortran routine (using mask)..." << std::endl;
    }

    std::vector<amrex::Real> matrix_values;
    std::vector<amrex::Real> rhs_values;
    std::vector<amrex::Real> initial_guess;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) \
                      private(matrix_values, rhs_values, initial_guess)
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        matrix_values.resize(static_cast<size_t>(npts) * stencil_size);
        rhs_values.resize(npts);
        initial_guess.resize(npts);

        // Use component 0 of phase fab
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr(0);
        const auto& pbox = phase_iab.box();

        // Use component MaskComp (0) of mask fab
        const amrex::IArrayBox& mask_iab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_iab.dataPtr(MaskComp);
        const auto& mask_box = mask_iab.box();

        // --- CORRECTED Fortran CALL ---
        // Pass pointers directly, remove AMREX_ARLIM for vector bounds
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts,
                           p_ptr, pbox.loVect(), pbox.hiVect(),      // Phase data + bounds (comp 0)
                           mask_ptr, mask_box.loVect(), mask_box.hiVect(), // Mask data + bounds (comp 0)
                           bx.loVect(), bx.hiVect(),                 // Box to compute on
                           domain.loVect(), domain.hiVect(),         // Domain bounds
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int); // Other params


        bool data_ok = true;
        for(size_t i = 0; i < matrix_values.size(); ++i) { if (std::isnan(matrix_values[i]) || std::isinf(matrix_values[i])) data_ok = false; }
        for(size_t i = 0; i < rhs_values.size(); ++i) { if (std::isnan(rhs_values[i]) || std::isinf(rhs_values[i])) data_ok = false; }
        if (!data_ok) { amrex::Warning("NaN/Inf detected in Fortran output before HYPRE SetBoxValues!"); }

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Finished MFIter loop." << std::endl; }

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Assembling HYPRE Matrix/Vectors..." << std::endl; }
    ierr = HYPRE_StructMatrixAssemble(m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Assembly complete." << std::endl; }
}


// --- Solve the Linear System using HYPRE ---
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
        HYPRE_StructPCGSetTol(solver, m_eps); HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPCGSetTwoNorm(solver, 1); HYPRE_StructPCGSetRelChange(solver, 0);
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        precond = NULL; ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0); HYPRE_StructPFMGSetMaxIter(precond, 1);
        HYPRE_StructPFMGSetRelaxType(precond, 6); HYPRE_StructPFMGSetNumPreRelax(precond, 2); HYPRE_StructPFMGSetNumPostRelax(precond, 2);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    PFMG Preconditioner created." << std::endl; }
        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    PCG Preconditioner set." << std::endl; }
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructPCGSetup..." << std::endl; }
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructPCGSolve..." << std::endl; }
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE PCG solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE PCG solver returned error code: " + std::to_string(ierr)); }
        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations); HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructPCGDestroy(solver); if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- GMRES Solver ---
    else if (m_solvertype == SolverType::GMRES) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE GMRES Solver with Default PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructGMRESSetTol(solver, m_eps); HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        precond = NULL; ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    PFMG Preconditioner created." << std::endl; }
        HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    GMRES Preconditioner set." << std::endl; }
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructGMRESSetup..." << std::endl; }
        ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructGMRESSolve..." << std::endl; }
        ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE GMRES solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE GMRES solver returned error code: " + std::to_string(ierr)); }
        HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations); HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructGMRESDestroy(solver); if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- FlexGMRES Solver ---
    else if (m_solvertype == SolverType::FlexGMRES) {
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with Tuned PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver, m_eps); HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        precond = NULL; ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0); HYPRE_StructPFMGSetMaxIter(precond, 1);
        HYPRE_StructPFMGSetRelaxType(precond, 6); HYPRE_StructPFMGSetNumPreRelax(precond, 2); HYPRE_StructPFMGSetNumPostRelax(precond, 2);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    PFMG Preconditioner created." << std::endl; }
        HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    FlexGMRES Preconditioner set." << std::endl; }
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructFlexGMRESSetup..." << std::endl; }
        ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructFlexGMRESSolve..." << std::endl; }
        ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE FlexGMRES solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE FlexGMRES solver returned error code: " + std::to_string(ierr)); }
        HYPRE_StructFlexGMRESGetNumIterations(solver, &m_num_iterations); HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructFlexGMRESDestroy(solver); if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- BiCGSTAB Solver ---
    else if (m_solvertype == SolverType::BiCGSTAB) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE BiCGSTAB Solver with Jacobi Preconditioner..." << std::endl;
        ierr = HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructBiCGSTABSetTol(solver, m_eps); HYPRE_StructBiCGSTABSetMaxIter(solver, m_maxiter);
        HYPRE_StructBiCGSTABSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        precond = NULL; ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructJacobiSetTol(precond, 0.0); HYPRE_StructJacobiSetMaxIter(precond, 2);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    Jacobi Preconditioner created." << std::endl; }
        HYPRE_StructBiCGSTABSetPrecond(solver, HYPRE_StructJacobiSolve, HYPRE_StructJacobiSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    BiCGSTAB Preconditioner set." << std::endl; }
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructBiCGSTABSetup..." << std::endl; }
        ierr = HYPRE_StructBiCGSTABSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructBiCGSTABSolve..." << std::endl; }
        ierr = HYPRE_StructBiCGSTABSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE BiCGSTAB solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE BiCGSTAB solver returned error code: " + std::to_string(ierr)); }
        HYPRE_StructBiCGSTABGetNumIterations(solver, &m_num_iterations); HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructBiCGSTABDestroy(solver); if (precond) HYPRE_StructJacobiDestroy(precond);
    }
    // --- Jacobi Solver ---
    else if (m_solvertype == SolverType::Jacobi) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE Jacobi Solver (NO preconditioner)..." << std::endl;
        precond = NULL; ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructJacobiSetTol(solver, m_eps); HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructJacobiSetup..." << std::endl; }
        ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructJacobiSolve..." << std::endl; }
        ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE Jacobi solver did not converge within max iterations!"); }
        else if (ierr != 0) { HYPRE_CHECK(ierr); }
        HYPRE_StructJacobiGetNumIterations(solver, &m_num_iterations); HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructJacobiDestroy(solver);
    }
    // --- SMG SOLVER CASE ---
    else if (m_solvertype == SolverType::SMG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE SMG Solver..." << std::endl;
        precond = NULL; ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetTol(solver, m_eps); HYPRE_StructSMGSetMaxIter(solver, m_maxiter);
        HYPRE_StructSMGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        HYPRE_StructSMGSetNumPreRelax(solver, 1); HYPRE_StructSMGSetNumPostRelax(solver, 1);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructSMGSetup..." << std::endl; }
        ierr = HYPRE_StructSMGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructSMGSolve..." << std::endl; }
        ierr = HYPRE_StructSMGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE SMG solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE SMG solver returned error code: " + std::to_string(ierr)); }
        HYPRE_StructSMGGetNumIterations(solver, &m_num_iterations); HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructSMGDestroy(solver);
    }
     // --- PFMG SOLVER CASE ---
    else if (m_solvertype == SolverType::PFMG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE PFMG Solver..." << std::endl;
        precond = NULL; ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(solver, m_eps); HYPRE_StructPFMGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPFMGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructPFMGSetup..." << std::endl; }
        ierr = HYPRE_StructPFMGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructPFMGSolve..." << std::endl; }
        ierr = HYPRE_StructPFMGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE PFMG solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE PFMG solver returned error code: " + std::to_string(ierr)); }
        HYPRE_StructPFMGGetNumIterations(solver, &m_num_iterations); HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        HYPRE_StructPFMGDestroy(solver);
    }
    // --- Unknown Solver ---
    else {
        std::string solverName = "Unknown";
        amrex::Abort("Unsupported solver type requested in TortuosityHypre::solve: " + std::to_string(static_cast<int>(m_solvertype)));
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver iterations: " << m_num_iterations << std::endl;
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << std::scientific << m_final_res_norm << std::defaultfloat << std::endl;
    }

    if (std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm)) {
        amrex::Warning("HYPRE solve resulted in NaN or Inf residual norm!");
    }

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
            const amrex::Box& bx = mfi.validbox();
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0) continue;
            soln_buffer.resize(npts);
            auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
            if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed during plotfile writing!"); }
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
            const amrex::IntVect lo = bx.smallEnd(); const amrex::IntVect hi = bx.bigEnd();
            long long k_lin_idx = 0;
            for (int kk = lo[2]; kk <= hi[2]; ++kk) { for (int jj = lo[1]; jj <= hi[1]; ++jj) { for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                if (k_lin_idx < npts) { soln_arr(ii,jj,kk, 0) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]); }
                else { amrex::Warning("Buffer overrun detected during HYPRE GetBoxValues copy!"); }
                k_lin_idx++;
            }}}
            if (k_lin_idx != npts) { amrex::Warning("Point count mismatch during HYPRE GetBoxValues copy!"); }
        } // End MFIter

        amrex::MultiFab mf_mask_temp(m_ba, m_dm, 1, 0);
        // --- CORRECTED Function Name (convert -> Copy) ---
        amrex::Copy(mf_mask_temp, m_mf_active_mask, MaskComp, 0, 1, 0);
        amrex::MultiFab mf_phase_temp(m_ba, m_dm, 1, 0);
        // --- CORRECTED Function Name (convert -> Copy) ---
        amrex::Copy(mf_phase_temp, m_mf_phase, 0, 0, 1, 0); // Copy phase component 0

        amrex::Copy(mf_plot, mf_soln_temp,   0, 0, 1, 0); // Solution potential to comp 0
        amrex::Copy(mf_plot, mf_phase_temp,  0, 1, 1, 0); // Phase ID to comp 1
        amrex::Copy(mf_plot, mf_mask_temp,   0, 2, 1, 0); // Active Mask to comp 2

        std::string plotfilename = m_resultspath + "/tortuosity_solution_" + std::to_string(static_cast<int>(m_dir));
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0);

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Plotfile written to " << plotfilename << std::endl;}
    } // End if m_write_plotfile

    m_first_call = false;
    bool converged = !(std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm));
    converged = converged && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);
    return converged;
}


// --- Calculate Tortuosity Value (Calls Solve if needed) ---
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
    if (m_first_call || refresh) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "Calculating Tortuosity (solve required)..." << std::endl; }
        bool converged = solve();
        if (!converged) {
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "WARNING: Solver did not converge or failed. Tortuosity is NaN." << std::endl; }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            return m_value;
        }
        amrex::Real flux_in = 0.0, flux_out = 0.0;
        global_fluxes(flux_in, flux_out);
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Calculated Fluxes: In = " << flux_in << ", Out = " << flux_out << std::endl;
             if (std::abs(flux_in + flux_out) > 1e-6 * (std::abs(flux_in) + std::abs(flux_out)) && std::abs(flux_in) > tiny_flux_threshold) {
                 amrex::Warning("Flux conservation check failed: |flux_in + flux_out| / (|flux_in|+|flux_out|) > 1e-6");
             }
        }
        amrex::Real vf_for_calc = m_vf; amrex::Real L = m_geom.ProbLength(static_cast<int>(m_dir));
        amrex::Real A = 1.0; if (AMREX_SPACEDIM == 3) { if (m_dir == OpenImpala::Direction::X) A = m_geom.ProbLength(1) * m_geom.ProbLength(2); else if (m_dir == OpenImpala::Direction::Y) A = m_geom.ProbLength(0) * m_geom.ProbLength(2); else A = m_geom.ProbLength(0) * m_geom.ProbLength(1); } else if (AMREX_SPACEDIM == 2) { if (m_dir == OpenImpala::Direction::X) A = m_geom.ProbLength(1); else A = m_geom.ProbLength(0); }
        amrex::Real gradPhi = (m_vhi - m_vlo) / L; amrex::Real Deff = 0.0;
        if (std::abs(flux_in) < tiny_flux_threshold) { if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "WARNING: Input flux is near zero (" << flux_in << "). Tortuosity set to Inf (or NaN if VF=0)." << std::endl; } m_value = (vf_for_calc > 0.0) ? std::numeric_limits<amrex::Real>::infinity() : std::numeric_limits<amrex::Real>::quiet_NaN(); }
        else if (vf_for_calc <= 0.0) { if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "WARNING: Volume fraction is zero. Tortuosity set to NaN." << std::endl; } m_value = std::numeric_limits<amrex::Real>::quiet_NaN(); }
        else if (std::abs(gradPhi) < tiny_flux_threshold) { if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "WARNING: Potential gradient is zero (vlo=vhi). Tortuosity set to Inf." << std::endl; } m_value = std::numeric_limits<amrex::Real>::infinity(); }
        else { Deff = - (flux_in / A) / gradPhi; m_value = vf_for_calc / Deff; }
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Calculation Details: Vf=" << vf_for_calc << ", L=" << L << ", A=" << A << ", gradPhi=" << gradPhi << ", Deff=" << Deff << std::endl; amrex::Print() << "  Calculated Tortuosity: " << m_value << std::endl; }
    }
    return m_value;
}

// --- NEW: Mathematical check function ---
bool OpenImpala::TortuosityHypre::checkMatrixProperties() {
    BL_PROFILE("TortuosityHypre::checkMatrixProperties");
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Checking assembled matrix/vector properties..." << std::endl;
    }

    HYPRE_Int ierr = 0;
    bool checks_passed_local = true;
    const double tol = 1.e-14;

    HYPRE_Int hypre_stencil_size = stencil_size;
    HYPRE_Int stencil_indices[stencil_size];
    for(int i=0; i<stencil_size; ++i) stencil_indices[i] = i;
    const int center_stencil_index = istn_c;

    const amrex::Box& domain = m_geom.Domain();
    const int idir = static_cast<int>(m_dir);

    std::vector<double> matrix_buffer;
    std::vector<double> rhs_buffer;

    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) \
                         reduction(&:checks_passed_local) \
                         private(matrix_buffer, rhs_buffer)
    #endif
    for (amrex::MFIter mfi(m_mf_active_mask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        matrix_buffer.resize(static_cast<size_t>(npts) * stencil_size);
        rhs_buffer.resize(npts);

        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        bool hypre_get_ok = true;
        ierr = HYPRE_StructMatrixGetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_buffer.data());
        if (ierr != 0) { hypre_get_ok = false; }
        ierr = HYPRE_StructVectorGetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_buffer.data());
        if (ierr != 0) { hypre_get_ok = false; }

        if (!hypre_get_ok) {
             checks_passed_local = false;
             if (m_verbose > 0) amrex::Print() << "CHECK FAILED: HYPRE_GetBoxValues error on rank " << amrex::ParallelDescriptor::MyProc() << " for box " << bx << std::endl;
             continue;
        }

        const amrex::IArrayBox& mask_fab = m_mf_active_mask[mfi];
        amrex::Array4<const int> const mask_arr = mask_fab.const_array();

        long long linear_idx = 0;
        const amrex::IntVect box_lo = bx.smallEnd();
        const amrex::IntVect box_hi = bx.bigEnd();

        for (int k = box_lo[2]; k <= box_hi[2]; ++k) {
        for (int j = box_lo[1]; j <= box_hi[1]; ++j) {
        for (int i = box_lo[0]; i <= box_hi[0]; ++i) {
            amrex::IntVect current_cell(i, j, k);
            size_t matrix_start_idx = linear_idx * stencil_size;
            double rhs_val = rhs_buffer[linear_idx];
            double diag_val = matrix_buffer[matrix_start_idx + center_stencil_index];

            // Check 1: NaN / Inf
            bool has_nan_inf = std::isnan(rhs_val) || std::isinf(rhs_val);
            for (int s = 0; s < stencil_size; ++s) { has_nan_inf = has_nan_inf || std::isnan(matrix_buffer[matrix_start_idx + s]) || std::isinf(matrix_buffer[matrix_start_idx + s]); }
            if (has_nan_inf) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: NaN/Inf found at cell " << current_cell << std::endl; checks_passed_local = false; }

            // Determine Cell Status
            int cell_activity = mask_arr(i, j, k);
            bool is_dirichlet = false;
            if (cell_activity == cell_active) { if ((idir == 0 && (i == domain.smallEnd(0) || i == domain.bigEnd(0))) || (idir == 1 && (j == domain.smallEnd(1) || j == domain.bigEnd(1))) || (idir == 2 && (k == domain.smallEnd(2) || k == domain.bigEnd(2)))) { is_dirichlet = true; } }

            // Check 2: Diagonal and RHS Value
            if (cell_activity == cell_inactive) { if (std::abs(diag_val - 1.0) > tol || std::abs(rhs_val) > tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Inactive cell check fail at " << current_cell << " (Aii=" << diag_val << ", b=" << rhs_val << ")" << std::endl; checks_passed_local = false; } for (int s=1; s<stencil_size; ++s) { if (std::abs(matrix_buffer[matrix_start_idx + s]) > tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero off-diag [" << s << "] at inactive cell " << current_cell << " (Aij=" << matrix_buffer[matrix_start_idx + s] << ")" << std::endl; checks_passed_local = false; }} }
            else if (is_dirichlet) { double expected_rhs = ((idir == 0 && i == domain.smallEnd(0)) || (idir == 1 && j == domain.smallEnd(1)) || (idir == 2 && k == domain.smallEnd(2))) ? m_vlo : m_vhi; if (std::abs(diag_val - 1.0) > tol || std::abs(rhs_val - expected_rhs) > tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Dirichlet cell check fail at " << current_cell << " (Aii=" << diag_val << ", b=" << rhs_val << ", exp_b=" << expected_rhs << ")" << std::endl; checks_passed_local = false; } for (int s=1; s<stencil_size; ++s) { if (std::abs(matrix_buffer[matrix_start_idx + s]) > tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero off-diag [" << s << "] at Dirichlet cell " << current_cell << " (Aij=" << matrix_buffer[matrix_start_idx + s] << ")" << std::endl; checks_passed_local = false; }} }
            else { // Active Interior
                 if (diag_val <= tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-positive diagonal at active interior cell " << current_cell << " (Aii=" << diag_val << ")" << std::endl; checks_passed_local = false; } if (std::abs(rhs_val) > tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero RHS at active interior cell " << current_cell << " (b=" << rhs_val << ")" << std::endl; checks_passed_local = false; }
                // Check 3: Row Sum = 0
                double row_sum = 0.0; for (int s = 0; s < stencil_size; ++s) { row_sum += matrix_buffer[matrix_start_idx + s]; } if (std::abs(row_sum) > tol) { if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero row sum at active interior cell " << current_cell << " (sum=" << row_sum << ")" << std::endl; checks_passed_local = false; }
            }
            linear_idx++;
        } // i
        } // j
        } // k
    } // End MFIter loop

    amrex::ParallelDescriptor::ReduceBoolAnd(checks_passed_local);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        if (checks_passed_local) { amrex::Print() << "TortuosityHypre: Matrix/vector property checks passed." << std::endl; }
        else { amrex::Print() << "TortuosityHypre: Matrix/vector property checks FAILED." << std::endl; }
    }
    return checks_passed_local;
}


// --- Get Solution Field (Not fully implemented) ---
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) {
    amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
}

// --- Get Cell Types (Not implemented) ---
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
    amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
}


// --- Calculate Global Fluxes Across Domain Boundaries ---
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout)
{
    BL_PROFILE("TortuosityHypre::global_fluxes");
    fxin = 0.0; fxout = 0.0;
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
        const amrex::Box& bx = mfi.validbox(); const int npts = static_cast<int>(bx.numPts()); if (npts == 0) continue;
        soln_buffer.resize(npts); auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx); auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);
        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data()); if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed during flux calculation copy!"); }
        amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi); const amrex::IntVect lo = bx.smallEnd(); const amrex::IntVect hi = bx.bigEnd(); long long k_lin_idx = 0;
        for (int kk=lo[2]; kk<=hi[2]; ++kk) { for (int jj=lo[1]; jj<=hi[1]; ++jj) { for (int ii=lo[0]; ii<=hi[0]; ++ii) { if (k_lin_idx < npts) soln_arr(ii,jj,kk)= static_cast<amrex::Real>(soln_buffer[k_lin_idx]); k_lin_idx++; }}} if (k_lin_idx != npts) { amrex::Warning("Point count mismatch during flux calc copy!"); }
    }

    mf_soln_temp.FillBoundary(m_geom.periodicity());
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;
    const amrex::Real dx_dir = dx[idir];
    if (dx_dir <= 0.0) amrex::Abort("Zero cell size in flux calculation direction!");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tileBox = mfi.tilebox();
        const auto mask = m_mf_active_mask.const_array(mfi, MaskComp); // Use MaskComp (0)
        const auto soln = mf_soln_temp.const_array(mfi, 0); // Use comp 0
        amrex::Box lobox_face = amrex::bdryLo(domain, idir); lobox_face &= tileBox;
        amrex::Box hibox_face = amrex::bdryHi(domain, idir); hibox_face &= tileBox;
        amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir);

        // Flux In (Low Face): Calculate grad across the face (cell i and i-1)
        if (!lobox_face.isEmpty()) {
            amrex::LoopOnCpu(lobox_face, [&](int i, int j, int k) {
                amrex::IntVect iv(i,j,k);
                if (mask(iv) == cell_active) { // Check mask on the cell *at* the boundary
                    amrex::Real grad = (soln(iv) - soln(iv - shift)) / dx_dir;
                    amrex::Real flux = -grad; // Assumes D=1
                    local_fxin += flux;
                }
            });
        }
        // Flux Out (High Face): Calculate grad across the face (cell i+1 and i)
        if (!hibox_face.isEmpty()) {
             amrex::LoopOnCpu(hibox_face, [&](int i, int j, int k) {
                amrex::IntVect iv(i,j,k);
                 if (mask(iv) == cell_active) { // Check mask on the cell *at* the boundary
                    amrex::Real grad = (soln(iv + shift) - soln(iv)) / dx_dir;
                    amrex::Real flux = -grad; // Assumes D=1
                    local_fxout += flux;
                }
            });
        }
    } // End MFIter loop

    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    amrex::Real face_area_element = 1.0;
    if (AMREX_SPACEDIM == 3) { if (idir == 0) { face_area_element = dx[1] * dx[2]; } else if (idir == 1) { face_area_element = dx[0] * dx[2]; } else { face_area_element = dx[0] * dx[1]; } }
    else if (AMREX_SPACEDIM == 2) { if (idir == 0) { face_area_element = dx[1]; } else { face_area_element = dx[0]; } }

    fxin = local_fxin * face_area_element;
    fxout = local_fxout * face_area_element;
}

} // End namespace OpenImpala
