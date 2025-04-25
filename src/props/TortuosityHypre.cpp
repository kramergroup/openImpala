#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"      // For tortuosity_remspot, tortuosity_filct
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
#include <set>       // For std::set in generateActivityMask
#include <algorithm> // For std::sort, std::unique
#include <numeric>   // For std::accumulate

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
    constexpr int PhaseComp = 1; // Which component in input phase MF holds the phase ID
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
      m_mf_phi(ba, dm, numComponentsPhi, 1), // Soln MF needs ghost cells if used for flux later
      m_mf_active_mask(ba, dm, 1, 1), // Need 1 ghost cell for Fortran neighbor check
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
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running preconditionPhaseFab (remspot)..." << std::endl;
    preconditionPhaseFab(); // Optional: Keep or remove based on results

    // *** Generate the activity mask based on percolation ***
    if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Generating activity mask via boundary search..." << std::endl;
    generateActivityMask(m_mf_phase, m_phase, m_dir);
    // *** Mask generation complete ***

    // Setup HYPRE Grid, Stencil, and Matrix Equation
    // (These will now implicitly use the mask via the modified Fortran routine)
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupGrids..." << std::endl;
    setupGrids();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupStencil..." << std::endl;
    setupStencil();
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "TortuosityHypre: Running setupMatrixEquation..." << std::endl;
    setupMatrixEquation(); // This now calls the modified Fortran routine

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
    BL_PROFILE("TortuosityHypre::preconditionPhaseFab"); // Add profile tag
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() >= 1, "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();
    const int num_remspot_passes = 3; // Number of passes for remspot

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
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
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  ...remspot filtering complete." << std::endl;
    }
}


// --- *** NEW METHOD: Parallel Flood Fill *** ---
// (Unchanged)
void OpenImpala::TortuosityHypre::parallelFloodFill(
    amrex::iMultiFab& reachabilityMask, // Mask to fill (1=reached, 0=not). Must have 1 ghost cell.
    const amrex::iMultiFab& phaseFab,   // Phase field (only fill within specified phase). Must have 1 ghost cell.
    int phaseID,                        // The phase ID that allows filling.
    const amrex::Vector<amrex::IntVect>& seedPoints) // Starting points for the fill.
{
    BL_PROFILE("TortuosityHypre::parallelFloodFill");
    AMREX_ASSERT(reachabilityMask.nGrow() >= 1);
    AMREX_ASSERT(phaseFab.nGrow() >= 1);

    // Initialize: Mark seed points on the current process
    reachabilityMask.setVal(cell_inactive); // Ensure starts at 0
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = reachabilityMask.array(mfi);
        const auto phase_arr = phaseFab.const_array(mfi); // Need phase info here too
        for (const auto& seed : seedPoints) {
            // Check if seed point is within the current tile box
            if (tileBox.contains(seed)) {
                // Mark seed only if it's the correct phase
                if (phase_arr(seed, PhaseComp) == phaseID) { // Use PhaseComp
                    mask_arr(seed, MaskComp) = cell_active; // Use MaskComp
                }
            }
        }
    }

    // Iterative flood fill loop
    int iter = 0;
    const int max_flood_iter = m_geom.Domain().longside() * 2; // Heuristic max iterations
    bool changed_globally = true;

    // Neighbor offsets (6 neighbors)
    const std::vector<amrex::IntVect> offsets = {
        amrex::IntVect{1, 0, 0}, amrex::IntVect{-1, 0, 0},
        amrex::IntVect{0, 1, 0}, amrex::IntVect{0, -1, 0},
        amrex::IntVect{0, 0, 1}, amrex::IntVect{0, 0, -1}
    };

    while (changed_globally && iter < max_flood_iter) {
        ++iter;
        changed_globally = false; // Assume no change in this iteration

        // Fill ghost cells for reachability mask
        reachabilityMask.FillBoundary(m_geom.periodicity());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            bool changed_locally = false; // OMP thread-local change flag
            for (amrex::MFIter mfi(reachabilityMask, true); mfi.isValid(); ++mfi) {
                const amrex::Box& tileBox = mfi.tilebox();
                auto mask_arr = reachabilityMask.array(mfi); // Non-const Array4
                const auto phase_arr = phaseFab.const_array(mfi); // Const Array4

                // Use grown box which includes ghost cells for neighbor checks
                const amrex::Box& grownTileBox = amrex::grow(tileBox, reachabilityMask.nGrow());

                amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) // Loop only over valid cells
                {
                    amrex::IntVect current_cell(i, j, k);

                    // Skip if already reached or not the correct phase
                    // Use MaskComp and PhaseComp indices
                    if (mask_arr(current_cell, MaskComp) == cell_active || phase_arr(current_cell, PhaseComp) != phaseID) {
                        return; // equivalent to continue in standard loop
                    }

                    // Check neighbors
                    bool reached_by_neighbor = false;
                    for (const auto& offset : offsets) {
                        amrex::IntVect neighbor_cell = current_cell + offset;
                        // Ensure the neighbor cell is within the bounds accessible by the mask_arr
                        if (grownTileBox.contains(neighbor_cell)) { // Check if neighbor is valid index for mask_arr
                           // Use MaskComp index
                           if (mask_arr(neighbor_cell, MaskComp) == cell_active) {
                                reached_by_neighbor = true;
                                break; // Found a reached neighbor
                           }
                        }
                    }

                    // If reached by a neighbor, mark current cell and flag change
                    if (reached_by_neighbor) {
                        mask_arr(current_cell, MaskComp) = cell_active; // Use MaskComp
                        changed_locally = true;
                    }
                }); // End amrex::LoopOnCpu

            } // End MFIter

            // Combine OMP thread-local flags if using OpenMP
            #ifdef AMREX_USE_OMP
            #pragma omp critical (flood_fill_crit) // Use a named critical section
            #endif
            {
                if (changed_locally) {
                    changed_globally = true;
                }
            }

        } // End OMP parallel region

        // Reduce across MPI processes
        amrex::ParallelDescriptor::ReduceBoolOr(changed_globally);

    } // End while loop

    if (iter >= max_flood_iter && changed_globally) {
        amrex::Warning("TortuosityHypre::parallelFloodFill reached max iterations - flood fill might be incomplete.");
    }
     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "    Flood fill completed in " << iter << " iterations." << std::endl;
     }
}


// --- *** NEW METHOD: Generate Activity Mask *** ---
// Identifies percolating conducting phase using two boundary flood fills.
void OpenImpala::TortuosityHypre::generateActivityMask(
    const amrex::iMultiFab& phaseFab, // Phase field (must have ghost cells)
    int phaseID,                      // Conducting phase ID
    OpenImpala::Direction dir)        // Direction of flow
{
    BL_PROFILE("TortuosityHypre::generateActivityMask");
    AMREX_ASSERT(phaseFab.nGrow() >= 1);

    const amrex::Box& domain = m_geom.Domain();
    const int idir = static_cast<int>(dir);

    // Define temporary masks for reachability (need 1 ghost cell for flood fill)
    amrex::iMultiFab mf_reached_inlet(m_ba, m_dm, 1, 1);
    amrex::iMultiFab mf_reached_outlet(m_ba, m_dm, 1, 1);
    // Initialization happens within parallelFloodFill now

    // Find seed points on boundaries for the current MPI rank
    amrex::Vector<amrex::IntVect> local_inlet_seeds; // Collect seeds locally first
    amrex::Vector<amrex::IntVect> local_outlet_seeds;
    const amrex::Box domain_lo_face = amrex::bdryLo(domain, idir);
    const amrex::Box domain_hi_face = amrex::bdryHi(domain, idir);

    for (amrex::MFIter mfi(phaseFab); mfi.isValid(); ++mfi) { // Iterate only over valid boxes
        const amrex::Box& validBox = mfi.validbox();
        const auto phase_arr = phaseFab.const_array(mfi); // Use const_array

        // Find inlet seeds on this tile's valid box
        amrex::Box inlet_intersect = validBox & domain_lo_face;
        if (!inlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(inlet_intersect, [&](int i, int j, int k) {
                // Use PhaseComp consistently
                if (phase_arr(i, j, k, PhaseComp) == phaseID) {
                    local_inlet_seeds.push_back(amrex::IntVect(i,j,k));
                }
            });
        }

        // Find outlet seeds on this tile's valid box
        amrex::Box outlet_intersect = validBox & domain_hi_face;
         if (!outlet_intersect.isEmpty()) {
            amrex::LoopOnCpu(outlet_intersect, [&](int i, int j, int k) { // Correct signature
                // Use PhaseComp consistently
                if (phase_arr(i, j, k, PhaseComp) == phaseID) { // Use k
                    local_outlet_seeds.push_back(amrex::IntVect(i,j,k));
                }
            });
        }
    } // End MFIter for seeds

    // --- Gather all seeds using AllGather ---
    // <<< FIX: Flatten IntVects to ints for AllGather >>>

    // 1. Flatten local vectors of IntVects into vectors of ints
    std::vector<int> flat_local_inlet_seeds;
    flat_local_inlet_seeds.reserve(local_inlet_seeds.size() * AMREX_SPACEDIM);
    for (const auto& iv : local_inlet_seeds) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            flat_local_inlet_seeds.push_back(iv[d]);
        }
    }

    std::vector<int> flat_local_outlet_seeds;
    flat_local_outlet_seeds.reserve(local_outlet_seeds.size() * AMREX_SPACEDIM);
    for (const auto& iv : local_outlet_seeds) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            flat_local_outlet_seeds.push_back(iv[d]);
        }
    }

    // 2. Prepare vectors to receive gathered flattened data
    std::vector<int> flat_inlet_seeds_gathered;
    std::vector<int> flat_outlet_seeds_gathered;

    // 3. Call AllGather with the flattened integer vectors
    amrex::ParallelDescriptor::AllGather(flat_local_inlet_seeds, flat_inlet_seeds_gathered);
    amrex::ParallelDescriptor::AllGather(flat_local_outlet_seeds, flat_outlet_seeds_gathered);

    // 4. Unflatten the gathered integer vectors back into amrex::Vector<IntVect>
    amrex::Vector<amrex::IntVect> inlet_seeds;
    AMREX_ASSERT_WITH_MESSAGE(flat_inlet_seeds_gathered.size() % AMREX_SPACEDIM == 0,
                              "Gathered inlet seed integer count not divisible by AMREX_SPACEDIM");
    inlet_seeds.reserve(flat_inlet_seeds_gathered.size() / AMREX_SPACEDIM);
    for (size_t i = 0; i < flat_inlet_seeds_gathered.size(); i += AMREX_SPACEDIM) {
        amrex::IntVect iv;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            iv[d] = flat_inlet_seeds_gathered[i + d];
        }
        inlet_seeds.push_back(iv);
    }

    amrex::Vector<amrex::IntVect> outlet_seeds;
    AMREX_ASSERT_WITH_MESSAGE(flat_outlet_seeds_gathered.size() % AMREX_SPACEDIM == 0,
                              "Gathered outlet seed integer count not divisible by AMREX_SPACEDIM");
    outlet_seeds.reserve(flat_outlet_seeds_gathered.size() / AMREX_SPACEDIM);
    for (size_t i = 0; i < flat_outlet_seeds_gathered.size(); i += AMREX_SPACEDIM) {
        amrex::IntVect iv;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            iv[d] = flat_outlet_seeds_gathered[i + d];
        }
        outlet_seeds.push_back(iv);
    }
    // --- End Seed Gathering Fix ---


    // Make seeds unique (now on the globally gathered vectors)
    std::sort(inlet_seeds.begin(), inlet_seeds.end());
    inlet_seeds.erase(std::unique(inlet_seeds.begin(), inlet_seeds.end()), inlet_seeds.end());
    std::sort(outlet_seeds.begin(), outlet_seeds.end());
    outlet_seeds.erase(std::unique(outlet_seeds.begin(), outlet_seeds.end()), outlet_seeds.end());

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Found " << inlet_seeds.size() << " unique inlet seed points globally." << std::endl;
        amrex::Print() << "  Found " << outlet_seeds.size() << " unique outlet seed points globally." << std::endl;
    }
    if (inlet_seeds.empty() || outlet_seeds.empty()) {
        amrex::Warning("TortuosityHypre::generateActivityMask: No seed points found on inlet or outlet boundary (or both). Mask will be empty.");
        m_mf_active_mask.setVal(cell_inactive);
        m_mf_active_mask.FillBoundary(m_geom.periodicity());
        return; // Exit early
    }


    // --- Perform flood fills ---
    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from inlet..." << std::endl;
    parallelFloodFill(mf_reached_inlet, phaseFab, phaseID, inlet_seeds);

    if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Performing flood fill from outlet..." << std::endl;
    parallelFloodFill(mf_reached_outlet, phaseFab, phaseID, outlet_seeds);

    // --- Combine reachability masks to create the final activity mask ---
    m_mf_active_mask.setVal(cell_inactive); // Ensure initialized to 0
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, true); mfi.isValid(); ++mfi) {
        const amrex::Box& tileBox = mfi.tilebox();
        auto mask_arr = m_mf_active_mask.array(mfi);
        // Need const access to temporary reachability masks
        const auto inlet_reach_arr = mf_reached_inlet.const_array(mfi);
        const auto outlet_reach_arr = mf_reached_outlet.const_array(mfi);

        amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
            // A cell is active if it was reached from BOTH inlet and outlet
            // Access component 0 for these single-component MultiFabs
            if (inlet_reach_arr(i, j, k, 0) == cell_active && outlet_reach_arr(i, j, k, 0) == cell_active) {
                mask_arr(i, j, k, MaskComp) = cell_active; // Use MaskComp
            } else {
                mask_arr(i, j, k, MaskComp) = cell_inactive; // Use MaskComp
            }
        });
    } // End MFIter for combining

    // Fill ghost cells of the final mask - needed for Fortran neighbor checks
    m_mf_active_mask.FillBoundary(m_geom.periodicity());

    // Optional: Write mask for debugging
    bool write_debug_mask = false;
    amrex::ParmParse pp_debug("tortuosity");
    pp_debug.query("write_debug_mask", write_debug_mask);
    if (write_debug_mask) {
        if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Writing debug activity mask plotfile..." << std::endl;
        std::string maskfilename = m_resultspath + "/tortuosity_activity_mask";
        amrex::Vector<std::string> varnames = {"active_mask"};
        // Need MultiFab not iMultiFab for plotfile - create temporary copy
        amrex::MultiFab mf_mask_plot(m_ba, m_dm, 1, 0);
        // Use amrex::Copy
        amrex::Copy(mf_mask_plot, m_mf_active_mask, MaskComp, 0, 1, 0); // Copy active mask (comp 0) to plot MF (comp 0)
        amrex::WriteSingleLevelPlotfile(maskfilename, mf_mask_plot, varnames, m_geom, 0.0, 0);
    }

     if (m_verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
        // Correct volume calculation
        amrex::Long local_active_volume = 0; // Use amrex::Long
        // Use #pragma omp parallel reduction, NOT parallel for
        #ifdef AMREX_USE_OMP
        #pragma omp parallel reduction(+:local_active_volume) if (amrex::Gpu::notInLaunchRegion())
        #endif
        // Iterate using TilingIfNotGPU for potential OMP efficiency inside loop
        for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& tileBox = mfi.tilebox();
            const auto& mask_arr = m_mf_active_mask.const_array(mfi); // Use const_array
            // Using local reduction variable within parallel loop for summing
            amrex::Long loop_local_sum = 0;
            amrex::LoopOnCpu(tileBox, [&](int i, int j, int k) {
                if (mask_arr(i, j, k, MaskComp) == cell_active) {
                    loop_local_sum += 1;
                }
            });
            #ifdef AMREX_USE_OMP
            // No critical needed due to reduction clause
            #endif
             local_active_volume += loop_local_sum; // Add tile sum to overall sum for this thread/rank
        } // End MFIter
        amrex::Long global_active_volume = local_active_volume; // Use amrex::Long
        // Use ReduceLongSum
        amrex::ParallelDescriptor::ReduceLongSum(global_active_volume);
        // Use Domain().volume()
        amrex::Real total_domain_volume = static_cast<amrex::Real>(m_geom.Domain().volume());
        amrex::Real active_vf = (total_domain_volume > 0.0)
                                  ? static_cast<amrex::Real>(global_active_volume) / total_domain_volume
                                  : 0.0;
        amrex::Print() << "  Activity mask generated. Active Volume Fraction: " << active_vf << std::endl;
    }
}


// --- Setup HYPRE Matrix and Vectors, Call Fortran Fill Routine ---
// (Unchanged)
void OpenImpala::TortuosityHypre::setupMatrixEquation()
{
    BL_PROFILE("TortuosityHypre::setupMatrixEquation"); // Add profile tag
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

    // Ensure mask ghost cells are up-to-date before passing to Fortran
    m_mf_active_mask.FillBoundary(m_geom.periodicity());


     if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "TortuosityHypre: Calling tortuosity_fillmtx Fortran routine (with mask)..." << std::endl;
     }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_mf_phase, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) { // Iterate using phaseFab MFIter
        const amrex::Box& bx = mfi.tilebox();
        const int npts = static_cast<int>(bx.numPts());

        if (npts == 0) continue;

        std::vector<amrex::Real> matrix_values(static_cast<size_t>(npts) * stencil_size);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        const amrex::IArrayBox& phase_iab = m_mf_phase[mfi];
        const int* p_ptr = phase_iab.dataPtr();
        const auto& pbox = phase_iab.box(); // Phase box incl. ghost cells

        // Get corresponding mask data for this MFIter index
        const amrex::IArrayBox& mask_iab = m_mf_active_mask[mfi];
        const int* mask_ptr = mask_iab.dataPtr();
        const auto& mask_box = mask_iab.box(); // Mask box incl. ghost cells

        // Call the modified Fortran routine
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts,
                           p_ptr, pbox.loVect(), pbox.hiVect(),                 // Phase data
                           mask_ptr, mask_box.loVect(), mask_box.hiVect(), // Mask data <<< NEW
                           bx.loVect(), bx.hiVect(), domain.loVect(), domain.hiVect(),
                           dxinv_sq.data(), &m_vlo, &m_vhi, &m_phase, &dir_int); // Pass original phase ID too, though Fortran may ignore it

        // Check for NaNs/Infs returned from Fortran (important!)
        // (Error checking logic remains the same)
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
// (Unchanged)
bool OpenImpala::TortuosityHypre::solve() {
    BL_PROFILE("TortuosityHypre::solve"); // Add profile tag
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
        precond = NULL;
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
        precond = NULL;
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
        // *** Using TUNED PFMG settings (recommended to try first) ***
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE FlexGMRES Solver with Tuned PFMG Preconditioner..." << std::endl;
        ierr = HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructFlexGMRESSetTol(solver, m_eps);
        HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
        HYPRE_StructFlexGMRESSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);
        // HYPRE_StructFlexGMRESSetKDim(solver, k_dim);

        // --- Setup Tuned PFMG Preconditioner ---
        precond = NULL;
        ierr = HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructPFMGSetTol(precond, 0.0);
        HYPRE_StructPFMGSetMaxIter(precond, 1);
        HYPRE_StructPFMGSetRelaxType(precond, 6);    // Tuned: Red-Black G-S
        HYPRE_StructPFMGSetNumPreRelax(precond, 2); // Tuned: 2 sweeps
        HYPRE_StructPFMGSetNumPostRelax(precond, 2);// Tuned: 2 sweeps
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG Preconditioner created and configured (TUNED settings)." << std::endl;
        // --- End PFMG Setup ---

        // Set PFMG as the preconditioner for FlexGMRES
        HYPRE_StructFlexGMRESSetPrecond(solver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond);
         if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  PFMG (TUNED) set as preconditioner for FlexGMRES." << std::endl;

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
    // --- BiCGSTAB Solver ---
    else if (m_solvertype == SolverType::BiCGSTAB) {
        // *** Using Jacobi Preconditioner (as fallback from PFMG issues) ***
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Setting up HYPRE BiCGSTAB Solver with Jacobi Preconditioner..." << std::endl;
        ierr = HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver);
        HYPRE_CHECK(ierr);
        HYPRE_StructBiCGSTABSetTol(solver, m_eps);
        HYPRE_StructBiCGSTABSetMaxIter(solver, m_maxiter);
        HYPRE_StructBiCGSTABSetPrintLevel(solver, m_verbose > 1 ? 3 : 0);

        // --- Setup Jacobi Preconditioner ---
        precond = NULL;
        ierr = HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &precond);
        HYPRE_CHECK(ierr);
        HYPRE_StructJacobiSetMaxIter(precond, 1);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Jacobi Preconditioner created and configured." << std::endl;
        // --- End Jacobi Setup ---

        // Set Jacobi as preconditioner for BiCGSTAB
        HYPRE_StructBiCGSTABSetPrecond(solver, HYPRE_StructJacobiSolve, HYPRE_StructJacobiSetup, precond);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Jacobi set as preconditioner for BiCGSTAB." << std::endl;

        // Setup and Solve
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructBiCGSTABSetup..." << std::endl;
        ierr = HYPRE_StructBiCGSTABSetup(solver, m_A, m_b, m_x);
        HYPRE_CHECK(ierr);
        if (m_verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "  Running HYPRE_StructBiCGSTABSolve..." << std::endl;
        ierr = HYPRE_StructBiCGSTABSolve(solver, m_A, m_b, m_x);
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
        if (precond) HYPRE_StructJacobiDestroy(precond);
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
        // Create MultiFab for plotting (potential + phase + mask)
        amrex::MultiFab mf_plot(m_ba, m_dm, numComponentsPhi, 0); // No ghost cells needed for plotfile

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

        // Copy solution, phase, and mask data into the plot MultiFab
        amrex::MultiFab mf_mask_temp(m_ba, m_dm, 1, 0); // Temporary Real MF for mask
        amrex::Copy(mf_mask_temp, m_mf_active_mask, MaskComp, 0, 1, 0); // Copy active mask (comp 0) to temp MF (comp 0)

        amrex::Copy(mf_plot, mf_soln_temp, 0, 0, 1, 0);       // Solution to component 0
        amrex::Copy(mf_plot, m_mf_phase, PhaseComp, 1, 1, 0); // Phase ID (comp PhaseComp) to component 1
        amrex::Copy(mf_plot, mf_mask_temp, 0, 2, 1, 0);       // Active Mask to component 2


        // Define plotfile name and variable names
        std::string plotfilename = m_resultspath + "/tortuosity_solution";
        amrex::Vector<std::string> varnames = {"solution_potential", "phase_id", "active_mask"}; // <<< ADDED MASK

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
        amrex::Real vf_for_calc = m_vf;
        if (std::abs(flux_in) < tiny_flux_threshold) {
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Calculated input flux is near zero (" << flux_in
                                << ") despite solver convergence. Tortuosity is ill-defined or infinite. Returning NaN." << std::endl;
             }
             m_value = std::numeric_limits<amrex::Real>::quiet_NaN();
        } else if (vf_for_calc <= 0.0) { // Check the VF used for calculation
             if (m_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Warning: Volume fraction used for calculation is zero or negative. Tortuosity is ill-defined. Returning NaN." << std::endl;
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
                        m_value = vf_for_calc / effective_diffusivity;
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
void OpenImpala::TortuosityHypre::global_fluxes(amrex::Real& fxin, amrex::Real& fxout) /* const removed */
{
    BL_PROFILE("TortuosityHypre::global_fluxes");
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
    // Fill ghost cells of the solution MultiFab
    mf_soln_temp.FillBoundary(m_geom.periodicity());
    // Fill mask ghost cells as it's used below (needs to be non-const for this)
    m_mf_active_mask.FillBoundary(m_geom.periodicity()); // Call on non-const mask

    // --- Calculate flux using finite differences on mf_soln_temp ---
    amrex::Real local_fxin = 0.0;
    amrex::Real local_fxout = 0.0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) reduction(+:local_fxin, local_fxout)
#endif
    for (amrex::MFIter mfi(m_mf_active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) // Iterate using mask MFIter
    {
        const amrex::Box& bx = mfi.tilebox(); // Iterate over tiles
        // Need const access to mask and solution arrays within the tile (+ghosts)
        const auto mask = m_mf_active_mask.const_array(mfi);
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
        // Only sum flux if the cell *inside* the boundary is active
        const amrex::IntVect lo_flux = lobox.smallEnd();
        const amrex::IntVect hi_flux = lobox.bigEnd();
        for (int k = lo_flux[2]; k <= hi_flux[2]; ++k) {
             iv[2]=k;
             for (int j = lo_flux[1]; j <= hi_flux[1]; ++j) {
                 iv[1]=j;
                 for (int i = lo_flux[0]; i <= hi_flux[0]; ++i) {
                      iv[0]=i;
                      if (mask(iv, MaskComp) == cell_active) { // Use MaskComp
                          // Flux = - D * grad(phi) = -1 * (phi_i - phi_{i-1})/dx (for X dir)
                           grad = (soln(iv) - soln(iv - shift)) / dx[idir];
                           flux = -grad; // Assumes D=1
                           local_fxin += flux;
                      }
                 }
             }
        }

        // Calculate flux exiting at the high boundary (Dirichlet face)
        // Only sum flux if the cell *inside* the boundary is active
        const amrex::IntVect lo_flux_hi = hibox_internal_cell.smallEnd();
        const amrex::IntVect hi_flux_hi = hibox_internal_cell.bigEnd();
        for (int k = lo_flux_hi[2]; k <= hi_flux_hi[2]; ++k) {
            iv[2]=k;
            for (int j = lo_flux_hi[1]; j <= hi_flux_hi[1]; ++j) {
                iv[1]=j;
                for (int i = lo_flux_hi[0]; i <= hi_flux_hi[0]; ++i) {
                     iv[0]=i;
                     if (mask(iv, MaskComp) == cell_active) { // Use MaskComp
                         // Flux = - D * grad(phi) = -1 * (phi_{i+1} - phi_i)/dx (for X dir)
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
