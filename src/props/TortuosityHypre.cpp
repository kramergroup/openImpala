// src/props/TortuosityHypre.cpp (Fix constness for m_mf_phase, remove const_casts, add checkMatrixProperties)

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
    pp_tort.query("verbose", m_verbose); // Allow overriding verbose level

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
    // NOTE: Mathematical checks via checkMatrixProperties() are intended to be called
    // externally (e.g., in the test routine) AFTER construction is complete.
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
// (No changes needed)
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
        if (m_verbose > 3 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   Setting stencil element " << i << std::endl; }
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]); HYPRE_CHECK(ierr);
    }
    if (!m_stencil) { amrex::Abort("FATAL: m_stencil handle is NULL after HYPRE_StructStencilCreate/SetElement!"); }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupStencil: Complete." << std::endl; }
}

// --- Preprocess Phase Field (Example: Remove isolated spots iteratively) ---
// <<< MODIFIED: Removed const_cast >>>
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
        m_mf_phase.FillBoundary(m_geom.periodicity()); // <<< No longer needs const_cast

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tile_box = mfi.tilebox();
            // Get non-const access directly as m_mf_phase is non-const
            amrex::IArrayBox& fab = m_mf_phase[mfi]; // <<< No const_cast needed
            int ncomp = fab.nComp(); // Should be 1

            // Pass tile_box as the box to iterate over
            tortuosity_remspot(fab.dataPtr(0), // Pass pointer to component 0
                               AMREX_ARLIM(fab.loVect()), AMREX_ARLIM(fab.hiVect()), &ncomp,
                               AMREX_ARLIM(tile_box.loVect()), AMREX_ARLIM(tile_box.hiVect()),
                               AMREX_ARLIM(domain_box.loVect()), AMREX_ARLIM(domain_box.hiVect()));
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
     BL_PROFILE("TortuosityHypre::parallelFloodFill");
     AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
     AMREX_ASSERT(phaseFab.nGrow() >= 1);
     // Ensure input phaseFab has enough components (comp 0 = phase)
     AMREX_ASSERT(phaseFab.nComp() > 0);
     AMREX_ASSERT(reachabilityMask.nComp() == 1);

     // Initialize mask to inactive
     reachabilityMask.setVal(cell_inactive);

     // Set initial seeds in the mask
     // Note: Seeds are global, need to check if they fall in local tilebox
 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
     for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
         const amrex::Box& tileBox = mfi.tilebox();
         auto mask_arr = reachabilityMask.array(mfi);
         // Get const access to the correct component (0) of phaseFab
         const auto phase_arr = phaseFab.const_array(mfi, 0);
         for (const auto& seed : seedPoints) {
             if (tileBox.contains(seed)) {
                 // Check phase *at* the seed location
                 if (phase_arr(seed) == phaseID) { // Access component 0
                     mask_arr(seed, MaskComp) = cell_active; // Mark active in component 0 of mask
                 }
             }
         }
     }

     // Iterative expansion
     int iter = 0;
     // Use domain diagonal length as a safe upper bound estimate
     amrex::IntVect domain_size = m_geom.Domain().size();
     const int max_flood_iter = domain_size[0] + domain_size[1] + domain_size[2] + 2; // Generous estimate
     bool changed_globally = true;
     const std::vector<amrex::IntVect> offsets = {
         amrex::IntVect{1, 0, 0}, amrex::IntVect{-1, 0, 0},
         amrex::IntVect{0, 1, 0}, amrex::IntVect{0, -1, 0},
         amrex::IntVect{0, 0, 1}, amrex::IntVect{0, 0, -1}
     };

     while (changed_globally && iter < max_flood_iter) {
         ++iter;
         changed_globally = false;
         // Fill ghost cells of the mask to propagate reachability across proc boundaries
         reachabilityMask.FillBoundary(m_geom.periodicity());

 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
         {
             bool changed_locally = false;
             for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
                 const amrex::Box& tileBox = mfi.tilebox();
                 auto mask_arr = reachabilityMask.array(mfi); // Access component 0 of mask
                 const auto phase_arr = phaseFab.const_array(mfi, 0); // Access component 0 of phase
                 // Need to check neighbors within the mask's ghost cells
                 const amrex::Box& grownTileBox = amrex::grow(tileBox, reachabilityMask.nGrow());

                 amrex::LoopOnCpu(tileBox, [&](int i, int j, int k)
                 {
                     amrex::IntVect current_cell(i, j, k);
                     // If already reached or not the right phase, skip
                     if (mask_arr(current_cell, MaskComp) == cell_active || phase_arr(current_cell) != phaseID) {
                         return;
                     }

                     // Check neighbors
                     bool reached_by_neighbor = false;
                     for (const auto& offset : offsets) {
                         amrex::IntVect neighbor_cell = current_cell + offset;
                         // Check if neighbor is within the FAB's box (including ghost cells)
                         if (grownTileBox.contains(neighbor_cell)) {
                             // Check if neighbor mask is active
                             if (mask_arr(neighbor_cell, MaskComp) == cell_active) {
                                 reached_by_neighbor = true;
                                 break; // Found one active neighbor, enough to activate current cell
                             }
                         }
                         // Note: No explicit check for neighbor phase needed here; if a neighbor
                         // mask is active, it must have been the correct phase previously.
                     }

                     // If reached by an active neighbor, mark current cell as active
                     if (reached_by_neighbor) {
                         mask_arr(current_cell, MaskComp) = cell_active;
                         changed_locally = true;
                     }
                 }); // End amrex::LoopOnCpu
             } // End MFIter

             // Combine local change status using OMP reduction if nested parallelism is used,
             // or just update shared flag if outer parallel region handles it.
             // Using critical section for safety in this example.
             #ifdef AMREX_USE_OMP
             #pragma omp critical (flood_fill_crit)
             #endif
             {
                 if (changed_locally) { changed_globally = true; }
             }
         } // End OMP parallel region

         // Reduce the 'changed_globally' flag across all MPI ranks
         amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);

     } // End while loop

     // Check for convergence failure
     if (iter >= max_flood_iter && changed_globally) {
         amrex::Warning("TortuosityHypre::parallelFloodFill reached max iterations - flood fill might be incomplete.");
     }
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "    Flood fill completed in " << iter << " iterations." << std::endl;
     }
     // Final boundary fill might be needed if mask is used immediately after
     reachabilityMask.FillBoundary(m_geom.periodicity());
}


// --- Generate Activity Mask ---
// (No changes needed in function signature or body logic - except MPI part expanded)
void OpenImpala::TortuosityHypre::generateActivityMask(
    const amrex::iMultiFab& phaseFab, // Still takes const ref here
    int phaseID,
    OpenImpala::Direction dir)
{
     BL_PROFILE("TortuosityHypre::generateActivityMask");
     AMREX_ASSERT(phaseFab.nGrow() >= 1);
     // Assuming phaseFab component 0 holds the phase data
     AMREX_ASSERT(phaseFab.nComp() > 0);

     const amrex::Box& domain = m_geom.Domain();
     const int idir = static_cast<int>(dir);

     // Temporary MultiFabs to store reachability from each end
     amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1); // 1 component, 1 ghost cell
     amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);

     // --- Collect Seed Points on Domain Faces ---
     amrex::Vector<amrex::IntVect> local_inlet_seeds;
     amrex::Vector<amrex::IntVect> local_outlet_seeds;

     // Define face boxes
     amrex::Box domain_lo_face = amrex::bdryLo(domain, idir);
     amrex::Box domain_hi_face = amrex::bdryHi(domain, idir);

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "   generateActivityMask: Searching for seeds on faces..." << std::endl;
        amrex::Print() << "    Inlet Face Box: " << domain_lo_face << std::endl;
        amrex::Print() << "    Outlet Face Box: " << domain_hi_face << std::endl;
     }

     #ifdef AMREX_USE_OMP
     #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
     #endif
     for (amrex::MFIter mfi(phaseFab); mfi.isValid(); ++mfi) {
         const amrex::Box& validBox = mfi.validbox();
         const auto phase_arr = phaseFab.const_array(mfi, 0); // Component 0 = phase

         // Find intersection of valid box with domain faces
         amrex::Box inlet_intersect = validBox & domain_lo_face;
         if (!inlet_intersect.isEmpty()) {
             amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                 if (phase_arr(i, j, k) == phaseID) {
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
                 if (phase_arr(i, j, k) == phaseID) {
                     #ifdef AMREX_USE_OMP
                     #pragma omp critical (outlet_seed_crit)
                     #endif
                     local_outlet_seeds.push_back(amrex::IntVect(i,j,k));
                 }
             });
         }
     }

     // --- Gather seeds across all MPI ranks ---
     // Flatten local seeds for MPI communication
     std::vector<int> flat_local_inlet_seeds(local_inlet_seeds.size() * AMREX_SPACEDIM);
     for (size_t i = 0; i < local_inlet_seeds.size(); ++i) {
         for (int d = 0; d < AMREX_SPACEDIM; ++d) {
             flat_local_inlet_seeds[i * AMREX_SPACEDIM + d] = local_inlet_seeds[i][d];
         }
     }
     std::vector<int> flat_local_outlet_seeds(local_outlet_seeds.size() * AMREX_SPACEDIM);
      for (size_t i = 0; i < local_outlet_seeds.size(); ++i) {
         for (int d = 0; d < AMREX_SPACEDIM; ++d) {
             flat_local_outlet_seeds[i * AMREX_SPACEDIM + d] = local_outlet_seeds[i][d];
         }
     }

     // Get MPI size and communicator
     MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
     int mpi_size = amrex::ParallelDescriptor::NProcs();
     int my_rank = amrex::ParallelDescriptor::MyProc();

     // Gather the number of seeds from each rank
     int local_inlet_count = static_cast<int>(flat_local_inlet_seeds.size());
     std::vector<int> recv_counts_inlet(mpi_size);
     MPI_Allgather(&local_inlet_count, 1, MPI_INT, recv_counts_inlet.data(), 1, MPI_INT, comm);

     int local_outlet_count = static_cast<int>(flat_local_outlet_seeds.size());
     std::vector<int> recv_counts_outlet(mpi_size);
     MPI_Allgather(&local_outlet_count, 1, MPI_INT, recv_counts_outlet.data(), 1, MPI_INT, comm);

     // Calculate displacements for Allgatherv
     std::vector<int> displacements_inlet(mpi_size, 0);
     std::vector<int> displacements_outlet(mpi_size, 0);
     int total_inlet_seeds = recv_counts_inlet[0];
     int total_outlet_seeds = recv_counts_outlet[0];
     for (int i = 1; i < mpi_size; ++i) {
         displacements_inlet[i] = displacements_inlet[i-1] + recv_counts_inlet[i-1];
         displacements_outlet[i] = displacements_outlet[i-1] + recv_counts_outlet[i-1];
         total_inlet_seeds += recv_counts_inlet[i];
         total_outlet_seeds += recv_counts_outlet[i];
     }

     // Allocate space for gathered seeds
     std::vector<int> flat_inlet_seeds_gathered(total_inlet_seeds);
     std::vector<int> flat_outlet_seeds_gathered(total_outlet_seeds);

     // Gather all seeds using Allgatherv
     MPI_Allgatherv(flat_local_inlet_seeds.data(), local_inlet_count, MPI_INT,
                    flat_inlet_seeds_gathered.data(), recv_counts_inlet.data(), displacements_inlet.data(),
                    MPI_INT, comm);
     MPI_Allgatherv(flat_local_outlet_seeds.data(), local_outlet_count, MPI_INT,
                    flat_outlet_seeds_gathered.data(), recv_counts_outlet.data(), displacements_outlet.data(),
                    MPI_INT, comm);

     // Unflatten the gathered seeds
     amrex::Vector<amrex::IntVect> inlet_seeds;
     inlet_seeds.reserve(total_inlet_seeds / AMREX_SPACEDIM);
     for (size_t i = 0; i < flat_inlet_seeds_gathered.size(); i += AMREX_SPACEDIM) {
         inlet_seeds.emplace_back(flat_inlet_seeds_gathered[i], flat_inlet_seeds_gathered[i+1], flat_inlet_seeds_gathered[i+2]);
     }
     amrex::Vector<amrex::IntVect> outlet_seeds;
     outlet_seeds.reserve(total_outlet_seeds / AMREX_SPACEDIM);
      for (size_t i = 0; i < flat_outlet_seeds_gathered.size(); i += AMREX_SPACEDIM) {
         outlet_seeds.emplace_back(flat_outlet_seeds_gathered[i], flat_outlet_seeds_gathered[i+1], flat_outlet_seeds_gathered[i+2]);
     }
     // --- End Seed Gathering ---


     // Remove duplicate seeds (important after gathering)
     std::sort(inlet_seeds.begin(), inlet_seeds.end());
     inlet_seeds.erase(std::unique(inlet_seeds.begin(), inlet_seeds.end()), inlet_seeds.end());
     std::sort(outlet_seeds.begin(), outlet_seeds.end());
     outlet_seeds.erase(std::unique(outlet_seeds.begin(), outlet_seeds.end()), outlet_seeds.end());

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "   generateActivityMask: Found " << inlet_seeds.size() << " unique inlet seeds." << std::endl;
         amrex::Print() << "   generateActivityMask: Found " << outlet_seeds.size() << " unique outlet seeds." << std::endl;
     }

     // Check for non-percolation
     if (inlet_seeds.empty() || outlet_seeds.empty()) {
         amrex::Warning("TortuosityHypre::generateActivityMask: No percolating path found (zero seeds on inlet or outlet face for the specified phase). Mask will be empty.");
         // Set the final mask to all inactive
         m_mf_active_mask.setVal(cell_inactive);
         // Fill ghost cells too
         m_mf_active_mask.FillBoundary(m_geom.periodicity());
         return; // Exit early if no path exists
     }

     // --- Perform Flood Fills ---
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from inlet..." << std::endl;
     parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds);

     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from outlet..." << std::endl;
     parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds);

     // --- Combine masks: Active = Reachable from BOTH inlet and outlet ---
     // Initialize final mask to inactive
     m_mf_active_mask.setVal(cell_inactive);

 #ifdef AMREX_USE_OMP
 #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
 #endif
     for (amrex::MFIter mfi(m_mf_active_mask, true); mfi.isValid(); ++mfi) {
         const amrex::Box& tileBox = mfi.tilebox();
         auto mask_arr = m_mf_active_mask.array(mfi); // Final mask array (component 0)
         const auto inlet_reach_arr = mf_reached_inlet.const_array(mfi); // Inlet reach (component 0)
         const auto outlet_reach_arr = mf_reached_outlet.const_array(mfi); // Outlet reach (component 0)

         amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
             // Active only if reachable from both directions
             if (inlet_reach_arr(i, j, k, 0) == cell_active && outlet_reach_arr(i, j, k, 0) == cell_active) {
                 mask_arr(i, j, k, MaskComp) = cell_active; // Set component 0 of final mask
             } else {
                 mask_arr(i, j, k, MaskComp) = cell_inactive;
             }
         });
     } // End MFIter

     // Fill ghost cells of the final combined mask
     m_mf_active_mask.FillBoundary(m_geom.periodicity());

     // --- Optional Debug Write & Active VF Calculation ---
     bool write_debug_mask = false;
     amrex::ParmParse pp_debug("debug");
     pp_debug.query("write_active_mask", write_debug_mask);

     if (write_debug_mask) {
        if (ParallelDescriptor::IOProcessor()) Print() << "  Writing debug active mask plotfile..." << std::endl;
        std::string mask_plotfile = m_resultspath + "/debug_active_mask";
        amrex::MultiFab mf_mask_plot(m_ba, m_dm, 1, 0);
        amrex::Convert(mf_mask_plot, m_mf_active_mask, 0, 0, 1, 0); // Copy int mask to real MF
        amrex::Vector<std::string> mask_vn = {"active_mask"};
        amrex::WriteSingleLevelPlotfile(mask_plotfile, mf_mask_plot, mask_vn, m_geom, 0.0, 0);
     }

     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
         // Calculate volume fraction of the *active* region
         long num_active = m_mf_active_mask.sum(MaskComp); // Sum component 0
         long total_cells = m_geom.Domain().numPts();
         amrex::Real active_vf = (total_cells > 0) ? static_cast<amrex::Real>(num_active) / total_cells : 0.0;
         amrex::Print() << "  Active Volume Fraction (percolating phase " << m_phase << "): " << active_vf << std::endl;
     }
}


// --- Setup HYPRE Matrix and Vectors, Call Fortran Fill Routine ---
// <<< MODIFIED: Pass non-const m_mf_phase to Fortran (though Fortran uses mask now) >>>
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    BL_PROFILE("TortuosityHypre::setupMatrixEquation");
    HYPRE_Int ierr = 0;

    // --- Create Matrix & Vectors ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Creating HYPRE Matrix/Vectors..." << std::endl; }
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructMatrixInitialize(m_A); HYPRE_CHECK(ierr);

    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_b); HYPRE_CHECK(ierr);
    // Initialize RHS to 0; Fortran will overwrite where needed
    ierr = HYPRE_StructVectorSetConstantValues(m_b, 0.0); HYPRE_CHECK(ierr);

    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorInitialize(m_x); HYPRE_CHECK(ierr);
    // Initialize solution guess to 0; Fortran will overwrite
    ierr = HYPRE_StructVectorSetConstantValues(m_x, 0.0); HYPRE_CHECK(ierr);


    // --- Prepare data for Fortran call ---
    const amrex::Box& domain = m_geom.Domain();
    // Stencil indices must match Fortran expectation/HYPRE setup order
    int stencil_indices[stencil_size] = {istn_c, istn_mx, istn_px, istn_my, istn_py, istn_mz, istn_pz}; // {0, 1, 2, 3, 4, 5, 6}
    const int dir_int = static_cast<int>(m_dir);
    amrex::Array<amrex::Real, AMREX_SPACEDIM> dxinv_sq; // dxinv contains 1/dx^2 etc.
    const amrex::Real* dx = m_geom.CellSize();
    for(int i=0; i<AMREX_SPACEDIM; ++i) { dxinv_sq[i] = (dx[i] > 0.0) ? (1.0/dx[i]) * (1.0/dx[i]) : 0.0; }

    // Ensure mask and phase ghost cells are up-to-date before passing to Fortran
    m_mf_active_mask.FillBoundary(m_geom.periodicity());
    m_mf_phase.FillBoundary(m_geom.periodicity()); // <<< Now works on non-const member

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  setupMatrixEq: Calling tortuosity_fillmtx Fortran routine (using mask)..." << std::endl;
    }

    // Buffers for Fortran output (allocated once per thread if OpenMP used)
    std::vector<amrex::Real> matrix_values;
    std::vector<amrex::Real> rhs_values;
    std::vector<amrex::Real> initial_guess;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) \
                     private(matrix_values, rhs_values, initial_guess)
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox(); // Fortran routine iterates over this box
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        // Resize buffers for this tile
        // Fortran expects 0-based indexing for the flattened array: size = npts * nstencil
        matrix_values.resize(static_cast<size_t>(npts) * stencil_size);
        rhs_values.resize(npts);
        initial_guess.resize(npts);

        // Get phase data pointer (now from non-const member, but access is logically const here)
        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        // Phase data might not be needed by Fortran anymore if mask is sufficient,
        // but pass it for now to match signature. Use component 0.
        const int* p_ptr = phase_iab.dataPtr(0);
        const auto& pbox = phase_iab.box(); // Box including ghost cells

        // Get mask data pointer (use component 0 of the mask iMultiFab)
        const amrex::IArrayBox& mask_iab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_iab.dataPtr(MaskComp);
        const auto& mask_box = mask_iab.box(); // Box including ghost cells

        // --- Call the Fortran routine ---
        // Pass the tilebox (bx) as the region to compute coefficients for.
        // Pass the FAB boxes (pbox, mask_box) to define the bounds of the input arrays.
        // Pass the component pointers (p_ptr, mask_ptr).
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts,
                           p_ptr, AMREX_ARLIM(pbox.loVect()), AMREX_ARLIM(pbox.hiVect()), // Phase data (comp 0) + bounds
                           mask_ptr, AMREX_ARLIM(mask_box.loVect()), AMREX_ARLIM(mask_box.hiVect()), // Mask data (comp 0) + bounds
                           AMREX_ARLIM(bx.loVect()), AMREX_ARLIM(bx.hiVect()), // Box to compute on
                           AMREX_ARLIM(domain.loVect()), AMREX_ARLIM(domain.hiVect()), // Domain bounds
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int); // Other params

        // --- Check Fortran Output (Optional but recommended) ---
        // Add checks here for NaNs/Infs in matrix_values, rhs_values before sending to HYPRE
        bool data_ok = true;
        for(size_t i = 0; i < matrix_values.size(); ++i) {
            if (std::isnan(matrix_values[i]) || std::isinf(matrix_values[i])) data_ok = false;
        }
        for(size_t i = 0; i < rhs_values.size(); ++i) {
            if (std::isnan(rhs_values[i]) || std::isinf(rhs_values[i])) data_ok = false;
        }
        if (!data_ok) {
             amrex::Warning("NaN/Inf detected in Fortran output before HYPRE SetBoxValues!");
             // Potentially abort or handle error
        }
        // int global_data_ok = data_ok; // Needs reduction across ranks if aborting
        // amrex::ParallelDescriptor::ReduceBoolAnd(global_data_ok);
        // if (global_data_ok == 0) { amrex::Abort("NaN/Inf detected in Fortran output!"); }


        // --- Set HYPRE Values for this Box ---
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // Matrix: Pass stencil indices {0..6}
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(), stencil_size, stencil_indices, matrix_values.data());
        HYPRE_CHECK(ierr); // Check immediately after Set

        // RHS Vector
        ierr = HYPRE_StructVectorSetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_values.data());
        HYPRE_CHECK(ierr);

        // Initial Guess Vector
        ierr = HYPRE_StructVectorSetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), initial_guess.data());
        HYPRE_CHECK(ierr);
    }
    if (m_verbose > 2 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Finished MFIter loop." << std::endl; }

    // --- Assemble HYPRE Objects ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Assembling HYPRE Matrix/Vectors..." << std::endl; }
    ierr = HYPRE_StructMatrixAssemble(m_A); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_b); HYPRE_CHECK(ierr);
    ierr = HYPRE_StructVectorAssemble(m_x); HYPRE_CHECK(ierr);
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  setupMatrixEq: Assembly complete." << std::endl; }
}


// --- Solve the Linear System using HYPRE ---
// <<< MODIFIED: Removed const_cast in plotfile writing >>>
bool OpenImpala::TortuosityHypre::solve() {
    BL_PROFILE("TortuosityHypre::solve");
    HYPRE_Int ierr = 0;
    HYPRE_StructSolver solver;
    HYPRE_StructSolver precond = NULL; // Preconditioner handle
    m_num_iterations = -1; // Reset solve stats
    m_final_res_norm = std::numeric_limits<amrex::Real>::quiet_NaN(); // Reset solve stats

    // --- Choose Solver based on m_solvertype ---

    // --- PCG Solver ---
    if (m_solvertype == SolverType::PCG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE PCG Solver with Tuned PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructPCGSetTol(solver, m_eps);
        HYPRE_StructPCGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPCGSetTwoNorm(solver, 1); // Use 2-norm for convergence check
        HYPRE_StructPCGSetRelChange(solver, 0); // Use relative residual norm, not relative change
        HYPRE_StructPCGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0); // Print debug info if verbose > 1
        // Setup Tuned PFMG Preconditioner
        precond = NULL;
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0); // Use as preconditioner, not solver
        HYPRE_StructPFMGSetMaxIter(precond, 1); // Typically 1 cycle is enough for preconditioning
        HYPRE_StructPFMGSetRelaxType(precond, 6); // Weighted Jacobi (often good)
        HYPRE_StructPFMGSetNumPreRelax(precond, 2); // Number of pre-smoothing steps
        HYPRE_StructPFMGSetNumPostRelax(precond, 2); // Number of post-smoothing steps
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   PFMG Preconditioner created." << std::endl; }
        // Set preconditioner functions for PCG
        HYPRE_StructPCGSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   PCG Preconditioner set." << std::endl; }
        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructPCGSetup..." << std::endl; }
        ierr = HYPRE_StructPCGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructPCGSolve..." << std::endl; }
        ierr = HYPRE_StructPCGSolve(solver, m_A, m_b, m_x); // Solve the system
        // Check solve status (convergence failure is not necessarily a fatal error)
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE PCG solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE PCG solver returned error code: " + std::to_string(ierr)); }
        // Get stats
        HYPRE_StructPCGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructPCGDestroy(solver);
        if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- GMRES Solver ---
    else if (m_solvertype == SolverType::GMRES) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE GMRES Solver with Default PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructGMRESSetTol(solver, m_eps);
        HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        // Setup Default PFMG Preconditioner
        precond = NULL;
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        // Use default PFMG settings for preconditioning
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   PFMG Preconditioner created." << std::endl; }
        HYPRE_StructGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   GMRES Preconditioner set." << std::endl; }
        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructGMRESSetup..." << std::endl; }
        ierr = HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructGMRESSolve..." << std::endl; }
        ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE GMRES solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE GMRES solver returned error code: " + std::to_string(ierr)); }
        // Get stats
        HYPRE_StructGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructGMRESDestroy(solver);
        if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- FlexGMRES Solver ---
    else if (m_solvertype == SolverType::FlexGMRES) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with Tuned PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver, m_eps);
        HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        // Setup Tuned PFMG Preconditioner (same as PCG)
        precond = NULL;
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0); HYPRE_StructPFMGSetMaxIter(precond, 1);
        HYPRE_StructPFMGSetRelaxType(precond, 6); HYPRE_StructPFMGSetNumPreRelax(precond, 2); HYPRE_StructPFMGSetNumPostRelax(precond, 2);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   PFMG Preconditioner created." << std::endl; }
        HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   FlexGMRES Preconditioner set." << std::endl; }
        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructFlexGMRESSetup..." << std::endl; }
        ierr = HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructFlexGMRESSolve..." << std::endl; }
        ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE FlexGMRES solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE FlexGMRES solver returned error code: " + std::to_string(ierr)); }
        // Get stats
        HYPRE_StructFlexGMRESGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructFlexGMRESDestroy(solver);
        if (precond) HYPRE_StructPFMGDestroy(precond);
    }
    // --- BiCGSTAB Solver ---
    else if (m_solvertype == SolverType::BiCGSTAB) {
        // Note: Often paired with simpler preconditioners like Jacobi or SMG/PFMG
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE BiCGSTAB Solver with Jacobi Preconditioner..." << std::endl;
        ierr = HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructBiCGSTABSetTol(solver, m_eps);
        HYPRE_StructBiCGSTABSetMaxIter(solver, m_maxiter);
        HYPRE_StructBiCGSTABSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        // Setup Jacobi Preconditioner
        precond = NULL;
        ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond); HYPRE_CHECK(ierr);
        HYPRE_StructJacobiSetTol(precond, 0.0); // Use as preconditioner
        HYPRE_StructJacobiSetMaxIter(precond, 2); // Apply a few sweeps
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   Jacobi Preconditioner created." << std::endl; }
        HYPRE_StructBiCGSTABSetPrecond(solver, HYPRE_StructJacobiSolve, HYPRE_StructJacobiSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   BiCGSTAB Preconditioner set." << std::endl; }
        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructBiCGSTABSetup..." << std::endl; }
        ierr = HYPRE_StructBiCGSTABSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructBiCGSTABSolve..." << std::endl; }
        ierr = HYPRE_StructBiCGSTABSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE BiCGSTAB solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE BiCGSTAB solver returned error code: " + std::to_string(ierr)); }
        // Get stats
        HYPRE_StructBiCGSTABGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructBiCGSTABDestroy(solver);
        if (precond) HYPRE_StructJacobiDestroy(precond);
    }
    // --- Jacobi Solver --- (Mainly for testing, usually slow)
    else if (m_solvertype == SolverType::Jacobi) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE Jacobi Solver (NO preconditioner)..." << std::endl;
        precond = NULL; // No preconditioner for Jacobi itself
        ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructJacobiSetTol(solver, m_eps);
        HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
        // HYPRE_StructJacobiSetPrintLevel(solver, m_verbose > 1 ? 3 : 0); // Jacobi doesn't have PrintLevel
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructJacobiSetup..." << std::endl; }
        ierr = HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Running HYPRE_StructJacobiSolve..." << std::endl; }
        ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
        // Jacobi often returns HYPRE_ERROR_CONV if maxiter is reached before tol
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE Jacobi solver did not converge within max iterations!"); }
        else if (ierr != 0) { HYPRE_CHECK(ierr); } // Check other potential errors
        // Get stats
        HYPRE_StructJacobiGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructJacobiDestroy(solver);
    }
    // --- SMG SOLVER CASE ---
    else if (m_solvertype == SolverType::SMG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE SMG Solver..." << std::endl;
        precond = NULL; // No separate preconditioner needed for SMG
        ierr = HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructSMGSetTol(solver, m_eps);
        HYPRE_StructSMGSetMaxIter(solver, m_maxiter);
        HYPRE_StructSMGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0); // SMG uses PrintLevel
        HYPRE_StructSMGSetNumPreRelax(solver, 1); // Default sweeps often ok
        HYPRE_StructSMGSetNumPostRelax(solver, 1); // Default sweeps often ok
        // HYPRE_StructSMGSetMemoryUse(solver, 0); // Optional: Try reducing memory if needed

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructSMGSetup..." << std::endl;
        ierr = HYPRE_StructSMGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr); // <<< Potential failure point
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructSMGSolve..." << std::endl;
        ierr = HYPRE_StructSMGSolve(solver, m_A, m_b, m_x); // <<< Potential failure point (NaN residual)
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE SMG solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE SMG solver returned error code: " + std::to_string(ierr)); }
        // Get stats
        HYPRE_StructSMGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructSMGDestroy(solver);
    }
     // --- PFMG SOLVER CASE --- (Added based on user trying it)
    else if (m_solvertype == SolverType::PFMG) {
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE PFMG Solver..." << std::endl;
        precond = NULL; // No separate preconditioner needed for PFMG
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver); HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(solver, m_eps);
        HYPRE_StructPFMGSetMaxIter(solver, m_maxiter);
        HYPRE_StructPFMGSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        // Optional: Set specific PFMG parameters if defaults fail
        // HYPRE_StructPFMGSetRelaxType(solver, 6); // Weighted Jacobi
        // HYPRE_StructPFMGSetNumPreRelax(solver, 1);
        // HYPRE_StructPFMGSetNumPostRelax(solver, 1);

        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructPFMGSetup..." << std::endl;
        ierr = HYPRE_StructPFMGSetup(solver, m_A, m_b, m_x); HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructPFMGSolve..." << std::endl;
        ierr = HYPRE_StructPFMGSolve(solver, m_A, m_b, m_x);
        if (ierr == HYPRE_ERROR_CONV && m_verbose >= 0) { amrex::Warning("HYPRE PFMG solver did not converge!"); }
        else if (ierr != 0) { amrex::Warning("HYPRE PFMG solver returned error code: " + std::to_string(ierr)); }
        // Get stats
        HYPRE_StructPFMGGetNumIterations(solver, &m_num_iterations);
        HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &m_final_res_norm);
        // Clean up
        HYPRE_StructPFMGDestroy(solver);
    }
    // --- Unknown Solver ---
    else {
        std::string solverName = "Unknown"; // Add logic to get name from enum if available
        amrex::Abort("Unsupported solver type requested in TortuosityHypre::solve: " + std::to_string(static_cast<int>(m_solvertype)));
    }

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  HYPRE Solver iterations: " << m_num_iterations << std::endl;
        // Use std::scientific for potentially small residuals
        amrex::Print() << "  HYPRE Final Relative Residual Norm: " << std::scientific << m_final_res_norm << std::defaultfloat << std::endl;
    }

    // --- Check for NaN/Inf in residual norm ---
    if (std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm)) {
        amrex::Warning("HYPRE solve resulted in NaN or Inf residual norm!");
        // No need to abort here, allow calculation to proceed and fail later if needed
        // but set converged flag to false.
    }

    // --- Write plot file if requested ---
    if (m_write_plotfile) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Writing solution plotfile..." << std::endl;
        }
        // Create a MultiFab for plotting (potential, phase, mask) - no ghost cells needed for plotfile
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponentsPhi, 0);

        // Create a temporary MultiFab to hold the solution vector (no ghost cells needed here)
        amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 0);
        mf_soln_temp.setVal(0.0); // Initialize

        // Buffer for HYPRE GetBoxValues
        std::vector<double> soln_buffer;

        // Copy solution from HYPRE vector m_x to mf_soln_temp
        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
        #endif
        for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox(); // Use validbox as mf_soln_temp has no ghosts
            const int npts = static_cast<int>(bx.numPts());
            if (npts == 0) continue;

            soln_buffer.resize(npts); // Resize buffer for this box
            auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
            auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

            // Get solution values from HYPRE vector m_x
            HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
            if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed during plotfile writing!"); }

            // Copy buffer data into mf_soln_temp MultiFab
            amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
            const amrex::IntVect lo = bx.smallEnd();
            const amrex::IntVect hi = bx.bigEnd();
            long long k_lin_idx = 0; // Linear index for the buffer

            // Use AMReX Box iterator for simplicity and potential GPU compatibility later
            amrex::LoopConcurrentOnCpu(bx, [&](int i, int j, int k) {
                 // Calculate linear index corresponding to (i,j,k) within the box's Fortran order
                 // This requires careful calculation or use the simpler sequential access below.
             });
            // Simpler sequential access matching HYPRE's likely ordering:
            for (int kk = lo[2]; kk <= hi[2]; ++kk) {
                for (int jj = lo[1]; jj <= hi[1]; ++jj) {
                    for (int ii = lo[0]; ii <= hi[0]; ++ii) {
                        if (k_lin_idx < npts) { // Bounds check
                            soln_arr(ii,jj,kk, 0) = static_cast<amrex::Real>(soln_buffer[k_lin_idx]);
                        } else {
                             amrex::Warning("Buffer overrun detected during HYPRE GetBoxValues copy!");
                        }
                        k_lin_idx++;
                    }
                }
            }
            // Sanity check
            if (k_lin_idx != npts) { amrex::Warning("Point count mismatch during HYPRE GetBoxValues copy!"); }
        } // End MFIter

        // Copy solution, phase, and mask into the plot MultiFab
        amrex::MultiFab mf_mask_temp(m_ba, m_dm, 1, 0); // Temp MF for mask conversion
        amrex::Convert(mf_mask_temp, m_mf_active_mask, MaskComp, 0, 1, 0); // Convert int mask to real

        amrex::MultiFab mf_phase_temp(m_ba, m_dm, 1, 0); // Temp MF for phase conversion
        amrex::Convert(mf_phase_temp, m_mf_phase, 0, 0, 1, 0); // Convert int phase to real

        amrex::Copy(mf_plot, mf_soln_temp,    0, 0, 1, 0); // Solution potential to comp 0
        amrex::Copy(mf_plot, mf_phase_temp,   0, 1, 1, 0); // Phase ID to comp 1
        amrex::Copy(mf_plot, mf_mask_temp,    0, 2, 1, 0); // Active Mask to comp 2

        // Write the plotfile
        std::string plotfilename = m_resultspath + "/tortuosity_solution_" + std::to_string(static_cast<int>(m_dir));
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"};
        amrex::WriteSingleLevelPlotfile(plotfilename, mf_plot, varnames, m_geom, 0.0, 0); // Time = 0.0, Level = 0

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "  Plotfile written to " << plotfilename << std::endl;}
    } // End if m_write_plotfile

    m_first_call = false; // Mark that solve has been attempted

    // Define convergence based on residual norm and iteration count
    // Check against NaN/Inf first
    bool converged = !(std::isnan(m_final_res_norm) || std::isinf(m_final_res_norm));
    // Then check against tolerance (only if not NaN/Inf)
    converged = converged && (m_final_res_norm >= 0.0) && (m_final_res_norm <= m_eps);
    // Could also add check: m_num_iterations < m_maxiter

    return converged;
}


// --- Calculate Tortuosity Value (Calls Solve if needed) ---
// (No changes needed)
amrex::Real OpenImpala::TortuosityHypre::value(const bool refresh)
{
    // If first time called or refresh requested, solve the system
    if (m_first_call || refresh) {
        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Calculating Tortuosity (solve required)..." << std::endl;
        }
        bool converged = solve(); // Attempt to solve the system

        if (!converged) {
            // If solver failed or didn't converge, return NaN
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "WARNING: Solver did not converge or failed. Tortuosity is NaN." << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
            return m_value; // Return NaN immediately
        }

        // If converged, calculate fluxes
        amrex::Real flux_in = 0.0, flux_out = 0.0;
        global_fluxes(flux_in, flux_out); // Calculate fluxes using the solution m_x

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Calculated Fluxes: In = " << flux_in << ", Out = " << flux_out << std::endl;
            // Check flux conservation as a sanity check
             if (std::abs(flux_in + flux_out) > 1e-6 * (std::abs(flux_in) + std::abs(flux_out)) && std::abs(flux_in) > tiny_flux_threshold) {
                  amrex::Warning("Flux conservation check failed: |flux_in + flux_out| / (|flux_in|+|flux_out|) > 1e-6");
             }
        }

        // --- Tortuosity Calculation ---
        // Using flux_in (should be equal to -flux_out for converged solution)
        amrex::Real vf_for_calc = m_vf; // Use the provided volume fraction
        amrex::Real L = m_geom.ProbLength(static_cast<int>(m_dir)); // Length in flow direction
        amrex::Real A = 1.0; // Cross-sectional area
        if (AMREX_SPACEDIM == 3) {
            if (m_dir == OpenImpala::Direction::X) A = m_geom.ProbLength(1) * m_geom.ProbLength(2);
            else if (m_dir == OpenImpala::Direction::Y) A = m_geom.ProbLength(0) * m_geom.ProbLength(2);
            else A = m_geom.ProbLength(0) * m_geom.ProbLength(1);
        } else if (AMREX_SPACEDIM == 2) {
             if (m_dir == OpenImpala::Direction::X) A = m_geom.ProbLength(1);
             else A = m_geom.ProbLength(0);
        } // Add 1D case if needed

        amrex::Real gradPhi = (m_vhi - m_vlo) / L; // Macroscopic potential gradient = DeltaV / L
        amrex::Real Deff = 0.0;

        if (std::abs(flux_in) < tiny_flux_threshold) {
            // Handle near-zero flux (likely non-percolating or very low conductivity)
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "WARNING: Input flux is near zero (" << flux_in << "). Tortuosity set to Inf (or NaN if VF=0)." << std::endl;
            }
             m_value = (vf_for_calc > 0.0) ? std::numeric_limits<amrex::Real>::infinity() : std::numeric_limits<amrex::Real>::quiet_NaN();

        } else if (vf_for_calc <= 0.0) {
            // Handle zero volume fraction
            if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "WARNING: Volume fraction is zero. Tortuosity set to NaN." << std::endl;
            }
            m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else if (std::abs(gradPhi) < tiny_flux_threshold) {
             // Handle zero potential gradient (vlo == vhi)
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                  amrex::Print() << "WARNING: Potential gradient is zero (vlo=vhi). Tortuosity set to Inf." << std::endl;
             }
              m_value = std::numeric_limits<amrex::Real>::infinity();
        }
        else {
            // Calculate effective diffusivity: Deff = - (Flux / Area) / GradPhi
            Deff = - (flux_in / A) / gradPhi;
            // Calculate Tortuosity: tau = Vf / Deff (assuming D0 = 1)
            m_value = vf_for_calc / Deff;
        }

        if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Calculation Details: Vf=" << vf_for_calc << ", L=" << L << ", A=" << A << ", gradPhi=" << gradPhi << ", Deff=" << Deff << std::endl;
            amrex::Print() << "  Calculated Tortuosity: " << m_value << std::endl;
        }
    }
    // Return the stored or newly calculated value
    return m_value;
}

// --- NEW: Mathematical check function ---
// NOTE: Declaration should be added to TortuosityHypre.H
bool OpenImpala::TortuosityHypre::checkMatrixProperties() {
    BL_PROFILE("TortuosityHypre::checkMatrixProperties");
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "TortuosityHypre: Checking assembled matrix/vector properties..." << std::endl;
    }

    HYPRE_Int ierr = 0;
    bool checks_passed_local = true; // Assume passes locally until failure
    const double tol = 1.e-14; // Tolerance for floating point comparisons

    // --- Get stencil info (assuming 7-point as defined elsewhere) ---
    HYPRE_Int hypre_stencil_size = stencil_size; // = 7
    HYPRE_Int stencil_indices[stencil_size];
    for(int i=0; i<stencil_size; ++i) stencil_indices[i] = i;
    const int center_stencil_index = istn_c; // = 0

    // --- Get Domain Info ---
    const amrex::Box& domain = m_geom.Domain();
    const int idir = static_cast<int>(m_dir);

    // --- Buffers for HYPRE Get calls (allocate once per thread) ---
    std::vector<double> matrix_buffer;
    std::vector<double> rhs_buffer;

    // --- Ensure mask ghost cells are up-to-date ---
    // This should have been done in setupMatrixEquation before assembly,
    // but checking here ensures the mask used for checks is correct.
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) \
                         reduction(&:checks_passed_local) \
                         private(matrix_buffer, rhs_buffer)
    #endif
    for (amrex::MFIter mfi(m_mf_active_mask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox(); // Use tilebox for checks
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        // Resize buffers if needed (OMP private copies)
        matrix_buffer.resize(static_cast<size_t>(npts) * stencil_size);
        rhs_buffer.resize(npts);

        // Get HYPRE box extents
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        // --- Get Matrix & RHS Values ---
        // Use temporary error flag inside loop to avoid race conditions on checks_passed_local in OMP
        bool hypre_get_ok = true;
        ierr = HYPRE_StructMatrixGetBoxValues(m_A, hypre_lo.data(), hypre_hi.data(),
                                              stencil_size, stencil_indices, matrix_buffer.data());
        if (ierr != 0) { hypre_get_ok = false; /* Log specific error if needed */ }

        ierr = HYPRE_StructVectorGetBoxValues(m_b, hypre_lo.data(), hypre_hi.data(), rhs_buffer.data());
        if (ierr != 0) { hypre_get_ok = false; /* Log specific error if needed */ }

        if (!hypre_get_ok) {
             checks_passed_local = false; // Mark failure for this rank/thread
             if (m_verbose > 0) amrex::Print() << "CHECK FAILED: HYPRE_GetBoxValues error on rank " << ParallelDescriptor::MyProc() << " for box " << bx << std::endl;
             continue; // Skip checks for this box if HYPRE failed
        }

        // --- Get Mask Access ---
        const amrex::IArrayBox& mask_fab = m_mf_active_mask[mfi];
        amrex::Array4<const int> const mask_arr = mask_fab.const_array(); // Comp 0 = mask

        // --- Loop through cells in the box ---
        long long linear_idx = 0;
        const amrex::IntVect box_lo = bx.smallEnd();
        const amrex::IntVect box_hi = bx.bigEnd();

        // Iterate in Fortran order to match HYPRE buffer layout
        for (int k = box_lo[2]; k <= box_hi[2]; ++k) {
        for (int j = box_lo[1]; j <= box_hi[1]; ++j) {
        for (int i = box_lo[0]; i <= box_hi[0]; ++i) {
            amrex::IntVect current_cell(i, j, k);
            size_t matrix_start_idx = linear_idx * stencil_size;
            double rhs_val = rhs_buffer[linear_idx];
            double diag_val = matrix_buffer[matrix_start_idx + center_stencil_index];

            // --- Check 1: NaN / Inf ---
            bool has_nan_inf = std::isnan(rhs_val) || std::isinf(rhs_val);
            for (int s = 0; s < stencil_size; ++s) {
                has_nan_inf = has_nan_inf || std::isnan(matrix_buffer[matrix_start_idx + s]) || std::isinf(matrix_buffer[matrix_start_idx + s]);
            }
            if (has_nan_inf) {
                if (m_verbose > 0) amrex::Print() << "CHECK FAILED: NaN/Inf found at cell " << current_cell << std::endl;
                checks_passed_local = false;
            }

            // --- Determine Cell Status ---
            int cell_activity = mask_arr(i, j, k);
            bool is_dirichlet = false;
            if (cell_activity == cell_active) {
                // Check if cell lies on the domain boundary face corresponding to the flow direction
                if ((idir == 0 && (i == domain.smallEnd(0) || i == domain.bigEnd(0))) ||
                    (idir == 1 && (j == domain.smallEnd(1) || j == domain.bigEnd(1))) ||
                    (idir == 2 && (k == domain.smallEnd(2) || k == domain.bigEnd(2)))) {
                    is_dirichlet = true;
                }
            }

            // --- Check 2: Diagonal and RHS Value based on Status ---
            if (cell_activity == cell_inactive) {
                // Expect Aii = 1.0, Aij = 0.0, b = 0.0
                if (std::abs(diag_val - 1.0) > tol || std::abs(rhs_val) > tol) {
                    if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Inactive cell check fail at " << current_cell << " (Aii=" << diag_val << ", b=" << rhs_val << ")" << std::endl;
                    checks_passed_local = false;
                }
                // Optionally check off-diagonals are zero
                for (int s = 1; s < stencil_size; ++s) { // Skip center index
                     if (std::abs(matrix_buffer[matrix_start_idx + s]) > tol) {
                          if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero off-diag [" << s << "] at inactive cell " << current_cell << " (Aij=" << matrix_buffer[matrix_start_idx + s] << ")" << std::endl;
                          checks_passed_local = false;
                     }
                }
            } else if (is_dirichlet) {
                // Expect Aii = 1.0, Aij = 0.0, b = vlo/vhi
                double expected_rhs = ((idir == 0 && i == domain.smallEnd(0)) || (idir == 1 && j == domain.smallEnd(1)) || (idir == 2 && k == domain.smallEnd(2))) ? m_vlo : m_vhi;
                if (std::abs(diag_val - 1.0) > tol || std::abs(rhs_val - expected_rhs) > tol) {
                    if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Dirichlet cell check fail at " << current_cell << " (Aii=" << diag_val << ", b=" << rhs_val << ", exp_b=" << expected_rhs << ")" << std::endl;
                    checks_passed_local = false;
                }
                 // Optionally check off-diagonals are zero
                for (int s = 1; s < stencil_size; ++s) {
                     if (std::abs(matrix_buffer[matrix_start_idx + s]) > tol) {
                          if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero off-diag [" << s << "] at Dirichlet cell " << current_cell << " (Aij=" << matrix_buffer[matrix_start_idx + s] << ")" << std::endl;
                          checks_passed_local = false;
                     }
                }
            } else { // Active Interior cell
                // Check Diagonal Sign (Should be > 0)
                if (diag_val <= tol) { // Allow slightly negative due to FP, but should be positive sum
                    if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-positive diagonal at active interior cell " << current_cell << " (Aii=" << diag_val << ")" << std::endl;
                    checks_passed_local = false;
                }
                // Check RHS Value (Should be 0 for Laplace eqn)
                 if (std::abs(rhs_val) > tol) {
                     if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero RHS at active interior cell " << current_cell << " (b=" << rhs_val << ")" << std::endl;
                     checks_passed_local = false;
                 }

                // --- Check 3: Row Sum (Should be zero for Laplace eqn) ---
                double row_sum = 0.0;
                for (int s = 0; s < stencil_size; ++s) {
                    row_sum += matrix_buffer[matrix_start_idx + s];
                }
                if (std::abs(row_sum) > tol) {
                    if (m_verbose > 0) amrex::Print() << "CHECK FAILED: Non-zero row sum at active interior cell " << current_cell << " (sum=" << row_sum << ")" << std::endl;
                    checks_passed_local = false;
                }
            } // End cell status checks

            linear_idx++; // Increment linear index for buffer access
        } // i
        } // j
        } // k
    } // End MFIter loop

    // Reduce the local pass/fail status across all MPI ranks
    amrex::ParallelDescriptor::ReduceBoolAnd(checks_passed_local);

    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        if (checks_passed_local) {
            amrex::Print() << "TortuosityHypre: Matrix/vector property checks passed." << std::endl;
        } else {
            amrex::Print() << "TortuosityHypre: Matrix/vector property checks FAILED." << std::endl;
        }
    }
    return checks_passed_local; // Return the global status
}

// --- Get Solution Field (Not fully implemented) ---
// (No changes needed)
void OpenImpala::TortuosityHypre::getSolution (amrex::MultiFab& soln, int ncomp) {
    amrex::Abort("TortuosityHypre::getSolution not fully implemented yet!");
    // Implementation would involve HYPRE_StructVectorGetBoxValues similar to plotfile writing
}

// --- Get Cell Types (Not implemented) ---
// (No changes needed)
void OpenImpala::TortuosityHypre::getCellTypes(amrex::MultiFab& phi, int ncomp) {
    amrex::Abort("TortuosityHypre::getCellTypes not implemented yet!");
    // Implementation might involve returning the active mask or phase info
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

    // Create a temporary MultiFab to hold the solution WITH ghost cells
    // Need 1 ghost cell to calculate finite difference flux at boundaries
    amrex::MultiFab mf_soln_temp(m_ba, m_dm, 1, 1);
    mf_soln_temp.setVal(0.0); // Initialize

    // Buffer for HYPRE GetBoxValues
    std::vector<double> soln_buffer;

    // Copy solution from HYPRE vector m_x to mf_soln_temp (valid region only first)
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) private(soln_buffer)
    #endif
    for (amrex::MFIter mfi(mf_soln_temp, false); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox(); // Get into valid region
        const int npts = static_cast<int>(bx.numPts());
        if (npts == 0) continue;

        soln_buffer.resize(npts); // Resize for this box
        auto hypre_lo = OpenImpala::TortuosityHypre::loV(bx);
        auto hypre_hi = OpenImpala::TortuosityHypre::hiV(bx);

        HYPRE_Int get_ierr = HYPRE_StructVectorGetBoxValues(m_x, hypre_lo.data(), hypre_hi.data(), soln_buffer.data());
        if (get_ierr != 0) { amrex::Warning("HYPRE_StructVectorGetBoxValues failed during flux calculation copy!"); }

        // Copy buffer to mf_soln_temp (valid region)
        amrex::Array4<amrex::Real> const soln_arr = mf_soln_temp.array(mfi);
        const amrex::IntVect lo = bx.smallEnd(); const amrex::IntVect hi = bx.bigEnd();
        long long k_lin_idx = 0;
        for (int kk=lo[2]; kk<=hi[2]; ++kk) { for (int jj=lo[1]; jj<=hi[1]; ++jj) { for (int ii=lo[0]; ii<=hi[0]; ++ii) {
            if (k_lin_idx < npts) soln_arr(ii,jj,kk)= static_cast<amrex::Real>(soln_buffer[k_lin_idx]); k_lin_idx++;
        }}}
        if (k_lin_idx != npts) { amrex::Warning("Point count mismatch during flux calc copy!"); }
    }

    // Fill ghost cells of the solution MultiFab using boundary conditions
    mf_soln_temp.FillBoundary(m_geom.periodicity());

    // Ensure mask ghost cells are also filled (needed for checking neighbors)
    m_mf_active_mask.FillBoundary(m_geom.periodicity()); // <<< Removed unnecessary const_cast

    // --- Calculate flux using finite differences ---
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

    const amrex::Real dx_dir = dx[idir]; // Cell size in flow direction
    if (dx_dir <= 0.0) amrex::Abort("Zero cell size in flux calculation direction!");

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tileBox = mfi.tilebox(); // Operate on tiles
        // Get const access to mask (component 0) and solution (component 0)
        const auto mask = m_mf_active_mask.const_array(mfi, MaskComp);
        const auto soln = mf_soln_temp.const_array(mfi, 0);

        // Define boxes intersecting the low and high domain faces within this tile
        // Low Face: Cells ON the low boundary face
        amrex::Box lobox_face = amrex::bdryLo(domain, idir);
        lobox_face &= tileBox; // Intersection with current tile

        // High Face: Cells ON the high boundary face
        amrex::Box hibox_face = amrex::bdryHi(domain, idir);
        hibox_face &= tileBox;

        amrex::IntVect shift = amrex::IntVect::TheDimensionVector(idir); // Shift vector in flow direction

        // Calculate INWARD flux at LOW boundary face (idir direction)
        // Flux = - d(phi)/dn = - (phi(i) - phi(i-1))/dx (for low face, normal points -idir)
        // Needs phi value inside (i) and in ghost cell (i-1)
        if (!lobox_face.isEmpty()) {
             amrex::LoopOnCpu(lobox_face, [&](int i, int j, int k) {
                amrex::IntVect iv(i,j,k);
                // Only calculate flux if the cell ON the boundary is active
                if (mask(iv) == cell_active) {
                    // Finite difference using cell center and ghost cell center
                    amrex::Real grad = (soln(iv) - soln(iv - shift)) / dx_dir;
                    amrex::Real flux = -grad; // Inward flux relative to domain
                    local_fxin += flux;
                }
             });
        }

        // Calculate OUTWARD flux at HIGH boundary face (idir direction)
        // Flux = - d(phi)/dn = - (phi(i+1) - phi(i))/dx (for high face, normal points +idir)
        // Needs phi value in ghost cell (i+1) and inside (i)
        if (!hibox_face.isEmpty()) {
             amrex::LoopOnCpu(hibox_face, [&](int i, int j, int k) {
                 amrex::IntVect iv(i,j,k);
                // Only calculate flux if the cell ON the boundary is active
                 if (mask(iv) == cell_active) {
                     // Finite difference using ghost cell center and boundary cell center
                     amrex::Real grad = (soln(iv + shift) - soln(iv)) / dx_dir;
                     amrex::Real flux = -grad; // Outward flux relative to domain
                     local_fxout += flux;
                 }
             });
        }
    } // End MFIter loop

    // Reduce sum across all MPI ranks
    amrex::ParallelDescriptor::ReduceRealSum(local_fxin);
    amrex::ParallelDescriptor::ReduceRealSum(local_fxout);

    // Multiply by face area element to get total flux
    // Note: This assumes dx[i] corresponds to cell size, not face area directly
    amrex::Real face_area_element = 1.0;
    if (AMREX_SPACEDIM == 3) {
         if (idir == 0)      { face_area_element = dx[1] * dx[2]; } // YZ plane
         else if (idir == 1) { face_area_element = dx[0] * dx[2]; } // XZ plane
         else                { face_area_element = dx[0] * dx[1]; } // XY plane
    } else if (AMREX_SPACEDIM == 2) {
         if (idir == 0)      { face_area_element = dx[1]; } // Y line
         else                { face_area_element = dx[0]; } // X line
    } // Add 1D if necessary (area is 1)

    fxin = local_fxin * face_area_element;
    fxout = local_fxout * face_area_element;
}

} // End namespace OpenImpala
