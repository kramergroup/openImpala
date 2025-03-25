#include "TortuosityHypre.H"
#include "Tortuosity_filcc_F.H"     // Assuming Fortran routine for fill cell types
#include "TortuosityHypreFill_F.H" // Assuming Fortran routine for matrix/rhs fill
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <cmath>
#include <limits>   // For std::numeric_limits
#include <stdexcept> // For potential error throwing (optional)

#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelAllReduce.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H> // For amrex::UtilCreateDirectory
#include <AMReX_Assert.H> // For AMREX_ASSERT, AMREX_ALWAYS_ASSERT

// Define constants for clarity and maintainability
namespace {
    constexpr int SolnComp = 0; // Component index for the potential/concentration solution
    constexpr int PhaseComp = 1; // Component index for the cell type/phase visualization
    constexpr int numComponents = 2; // Total components in m_mf_phi
    constexpr amrex::Real tiny_flux_threshold = 1.e-15; // Tolerance for checking near-zero values
}

//-----------------------------------------------------------------------------
// Helper Functions (could be static members or in an anonymous namespace)
//-----------------------------------------------------------------------------

/**
 * @brief Generate lower bounds array in HYPRE_Int format for an AMReX Box.
 */
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> getHypreLo (const amrex::Box& b) {
    const auto& v = b.loVect();
    return {AMREX_D_DECL(static_cast<HYPRE_Int>(v[0]),
                         static_cast<HYPRE_Int>(v[1]),
                         static_cast<HYPRE_Int>(v[2]))};
}

/**
 * @brief Generate upper bounds array in HYPRE_Int format for an AMReX Box.
 */
amrex::Array<HYPRE_Int,AMREX_SPACEDIM> getHypreHi (const amrex::Box& b) {
    const auto& v = b.hiVect();
    return {AMREX_D_DECL(static_cast<HYPRE_Int>(v[0]),
                         static_cast<HYPRE_Int>(v[1]),
                         static_cast<HYPRE_Int>(v[2]))};
}

//-----------------------------------------------------------------------------
// TortuosityHypre Class Implementation
//-----------------------------------------------------------------------------

/**
 * @brief Constructor: Sets up geometry, solver params, and HYPRE structures.
 */
TortuosityHypre::TortuosityHypre(const amrex::Geometry& geom,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm,
                                 amrex::iMultiFab& mf_phase_input, // Renamed for clarity
                                 const amrex::Real vf,
                                 const int phase,
                                 const Direction dir,
                                 const SolverType st,
                                 const amrex::Real eps, // Solver tolerance
                                 const int maxiter,     // Solver max iterations
                                 const amrex::Real vlo, // Boundary value low side
                                 const amrex::Real vhi, // Boundary value high side
                                 std::string const& resultspath)
    : m_geom(geom), m_ba(ba), m_dm(dm),
      m_mf_phase(mf_phase_input), // Copy phase info (assuming iMultiFab has copy semantics)
      m_phase(phase), m_vf(vf),
      m_dir(dir), m_solvertype(st),
      m_eps(eps), m_maxiter(maxiter), // Initialize solver params
      m_vlo(vlo), m_vhi(vhi),         // Initialize boundary values
      m_resultspath(resultspath),
      m_mf_phi(ba, dm, numComponents, 0), // Allocate solution multifab (SolnComp + PhaseComp)
      m_first_call(true),
      // Initialize HYPRE handles to NULL
      m_grid(NULL), m_stencil(NULL), m_A(NULL), m_b(NULL), m_x(NULL)
{
    // --- Validate Inputs ---
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_vf >= 0.0 && m_vf <= 1.0, "Volume fraction must be between 0 and 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_eps > 0.0, "Solver tolerance (eps) must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_maxiter > 0, "Solver max iterations must be positive");
    // Add more validation if needed (e.g., for phase, dir)

    // --- Setup Steps ---
    preconditionPhaseFab(); // Modify the phase field if necessary (e.g., remove spots)
    setupGrids();           // Setup HYPRE grid structure
    setupStencil();         // Setup HYPRE stencil structure
    setupMatrixEquation();  // Setup HYPRE matrix and vectors
}

/**
 * @brief Destructor: Frees all allocated HYPRE resources. CRITICAL.
 */
TortuosityHypre::~TortuosityHypre()
{
    if (m_x)       HYPRE_StructVectorDestroy(m_x);
    if (m_b)       HYPRE_StructVectorDestroy(m_b);
    if (m_A)       HYPRE_StructMatrixDestroy(m_A);
    if (m_stencil) HYPRE_StructStencilDestroy(m_stencil);
    if (m_grid)    HYPRE_StructGridDestroy(m_grid);
    // Set handles to NULL after destruction (optional but good practice)
    m_x = m_b = NULL;
    m_A = NULL;
    m_stencil = NULL;
    m_grid = NULL;
}

// ** Add these to your TortuosityHypre.H header file **
// // Disable copy constructor and assignment operator (Rule of Three/Five)
// TortuosityHypre(const TortuosityHypre&) = delete;
// TortuosityHypre& operator=(const TortuosityHypre&) = delete;


/**
 * @brief Sets up the HYPRE StructGrid based on the AMReX BoxArray.
 */
void TortuosityHypre::setupGrids()
{
    HYPRE_Int ierr = 0;

    ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &m_grid);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructGridCreate failed");

    for ( amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi )
    {
        const amrex::Box bx = mfi.validbox();
        auto lo = getHypreLo(bx); // Use helper function
        auto hi = getHypreHi(bx); // Use helper function

        // Set the extents (lower/upper bounds) for the boxes owned by this MPI rank
        ierr = HYPRE_StructGridSetExtents(m_grid, lo.data(), hi.data());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructGridSetExtents failed");
    }

    // Finalize grid assembly
    ierr = HYPRE_StructGridAssemble(m_grid);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructGridAssemble failed");
}


/**
 * @brief Sets up the 7-point HYPRE StructStencil.
 */
void TortuosityHypre::setupStencil()
{
    HYPRE_Int ierr = 0;
    // Define 7-point stencil offsets for 3D
    constexpr int stencil_size = 7;
    int offsets[stencil_size][AMREX_SPACEDIM] = {{0,0,0},
                                                 {-1,0,0}, {1,0,0},
                                                 {0,-1,0}, {0,1,0},
                                                 {0,0,-1}, {0,0,1}};

    ierr = HYPRE_StructStencilCreate(AMREX_SPACEDIM, stencil_size, &m_stencil);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructStencilCreate failed");

    // Set stencil entries using the offsets
    for (int i = 0; i < stencil_size; i++)
    {
        ierr = HYPRE_StructStencilSetElement(m_stencil, i, offsets[i]);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructStencilSetElement failed");
    }
    // Stencil doesn't require an Assemble call
}

/**
 * @brief Modifies the phase MultiFab, e.g., removing isolated spots.
 * Calls the Fortran routine `tortuosity_remspot`.
 */
void TortuosityHypre::preconditionPhaseFab()
{
    AMREX_ASSERT_WITH_MESSAGE(m_mf_phase.nGrow() > 0, "Phase fab needs ghost cells for preconditionPhaseFab");

    const amrex::Box& domain_box = m_geom.Domain();

    for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi)
    {
        amrex::IArrayBox& fab = m_mf_phase[mfi];
        const amrex::Box& fab_box = mfi.validbox();

        // --- Fortran Call ---
        // Assumes: tortuosity_remspot(fab, fab_box, domain_box) modifies 'fab' in place.
        // Arguments:
        //   fab:        IArrayBox containing phase data (modified)
        //   fab_box:    The valid box for this Fab
        //   domain_box: The overall domain box
        tortuosity_remspot(BL_TO_FORTRAN_FAB(fab),
                           BL_TO_FORTRAN_BOX(fab_box),
                           BL_TO_FORTRAN_BOX(domain_box));
    }
}

/**
 * @brief Sets up the HYPRE StructMatrix and StructVectors (A, b, x).
 * Fills matrix coefficients and RHS using the Fortran routine `tortuosity_fillmtx`.
 */
void TortuosityHypre::setupMatrixEquation()
{
    HYPRE_Int ierr = 0;

    // Create and initialize matrix A
    ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, m_grid, m_stencil, &m_A);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructMatrixCreate failed");
    ierr = HYPRE_StructMatrixInitialize(m_A);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructMatrixInitialize failed");

    // Create and initialize RHS vector b
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_b);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructVectorCreate failed");
    ierr = HYPRE_StructVectorInitialize(m_b);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructVectorInitialize failed");

    // Create and initialize solution vector x (initial guess)
    ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, m_grid, &m_x);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructVectorCreate failed");
    ierr = HYPRE_StructVectorInitialize(m_x);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructVectorInitialize failed");

    const amrex::Box domain = m_geom.Domain();
    int stencil_indices[7] = {0,1,2,3,4,5,6}; // Indices matching setupStencil

    // Iterate over boxes owned by this rank to fill matrix/vectors
    for (amrex::MFIter mfi(m_mf_phase); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.validbox();
        auto lo = getHypreLo(bx);
        auto hi = getHypreHi(bx);

        const size_t npts = bx.numPts();

        // Use dynamic allocation (std::vector) to prevent stack overflow
        std::vector<amrex::Real> matrix_values(npts * 7);
        std::vector<amrex::Real> rhs_values(npts);
        std::vector<amrex::Real> initial_guess(npts);

        // --- Fortran Call ---
        // Assumes: tortuosity_fillmtx populates the provided vectors based on phase, geometry, BCs.
        // Arguments:
        //   matrix_values: Output vector for 7 stencil coeffs per cell (size npts*7)
        //   rhs_values:    Output vector for RHS value per cell (size npts)
        //   initial_guess: Output vector for initial guess per cell (size npts)
        //   npts:          Number of points in the box (pointer to size_t)
        //   phase_fab:     Input phase data for the box
        //   bx:            The valid box for this Fab
        //   domain:        The overall domain box
        //   m_vlo, m_vhi:  Input boundary potential values (pointers)
        //   m_phase:       Input phase ID to consider conductive (pointer)
        //   m_dir:         Input direction for BC application (pointer)
        tortuosity_fillmtx(matrix_values.data(), rhs_values.data(), initial_guess.data(),
                           &npts, // Pass address of size_t
                           BL_TO_FORTRAN_ANYD(m_mf_phase[mfi]),
                           BL_TO_FORTRAN_BOX(bx),
                           BL_TO_FORTRAN_BOX(domain),
                           &m_vlo, &m_vhi,
                           &m_phase, &m_dir);

        // Set values in HYPRE structures for this box
        ierr = HYPRE_StructMatrixSetBoxValues(m_A, lo.data(), hi.data(), 7, stencil_indices, matrix_values.data());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructMatrixSetBoxValues failed");

        ierr = HYPRE_StructVectorSetBoxValues(m_b, lo.data(), hi.data(), rhs_values.data());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructVectorSetBoxValues failed for b");

        ierr = HYPRE_StructVectorSetBoxValues(m_x, lo.data(), hi.data(), initial_guess.data());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructVectorSetBoxValues failed for x");
    }

    // Finalize matrix assembly
    ierr = HYPRE_StructMatrixAssemble(m_A);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr==0, "HYPRE_StructMatrixAssemble failed");
    // Vectors b and x usually don't need explicit assembly after SetBoxValues
}

/**
 * @brief Solves the linear system Ax = b using the selected HYPRE solver.
 * Updates m_mf_phi with the solution and writes a plotfile.
 * @return true if solver converged successfully, false otherwise.
 */
bool TortuosityHypre::solve()
{
    // Generate timestamp for output file
    std::time_t strt_time;
    std::tm* timeinfo;
    char datetime [80];
    std::time(&strt_time);
    timeinfo = std::localtime(&strt_time);
    std::strftime(datetime, sizeof(datetime),"%Y%m%d%H%M", timeinfo);

    HYPRE_StructSolver solver;
    HYPRE_Int ierr = 0;
    HYPRE_Int num_iterations = 0;
    amrex::Real final_res_norm = -1.0;

    amrex::Print() << "Solving with HYPRE... Tolerance=" << m_eps << ", MaxIter=" << m_maxiter << std::endl;

    // Select and run the HYPRE solver
    switch (m_solvertype)
    {
        case Jacobi:
            HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
            HYPRE_StructJacobiSetTol(solver, m_eps);
            HYPRE_StructJacobiSetMaxIter(solver, m_maxiter);
            HYPRE_StructJacobiSetup(solver, m_A, m_b, m_x);
            ierr = HYPRE_StructJacobiSolve(solver, m_A, m_b, m_x);
            // Get results BEFORE destroying
            HYPRE_StructJacobiGetNumIterations(solver, &num_iterations);
            HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &final_res_norm);
            HYPRE_StructJacobiDestroy(solver);
            break;

        case FlexGMRES:
             HYPRE_StructFlexGMRESCreate(MPI_COMM_WORLD, &solver);
             HYPRE_StructFlexGMRESSetTol(solver, m_eps);
             HYPRE_StructFlexGMRESSetMaxIter(solver, m_maxiter);
             // Note: Preconditioner usually needed for GMRES/FlexGMRES. None set here.
             // HYPRE_StructFlexGMRESSetPrecond(solver, ...);
             HYPRE_StructFlexGMRESSetup(solver, m_A, m_b, m_x);
             ierr = HYPRE_StructFlexGMRESSolve(solver, m_A, m_b, m_x);
             // Get results BEFORE destroying
             HYPRE_StructFlexGMRESGetNumIterations(solver, &num_iterations);
             HYPRE_StructFlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
             HYPRE_StructFlexGMRESDestroy(solver);
             break;

        case GMRES: // Fallthrough intended for default case
        default:
             HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &solver);
             HYPRE_StructGMRESSetTol(solver, m_eps);
             HYPRE_StructGMRESSetMaxIter(solver, m_maxiter);
             // Note: Preconditioner usually needed for GMRES/FlexGMRES. None set here.
             // HYPRE_StructGMRESSetPrecond(solver, ...);
             HYPRE_StructGMRESSetup(solver, m_A, m_b, m_x);
             ierr = HYPRE_StructGMRESSolve(solver, m_A, m_b, m_x);
             // Get results BEFORE destroying
             HYPRE_StructGMRESGetNumIterations(solver, &num_iterations);
             HYPRE_StructGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
             HYPRE_StructGMRESDestroy(solver);
             break;
    }

    amrex::Print() << "Solver finished: " << num_iterations
                   << " Iterations, Final Relative Residual = "
                   << final_res_norm << std::endl;

    // Check for HYPRE solver errors
    if (ierr) {
        amrex::Print() << "ERROR: HYPRE solver returned error code: " << ierr << std::endl;
        if (HYPRE_CheckError(ierr, HYPRE_ERROR_CONV)) {
            amrex::Print() << "       (Solver did not converge within max iterations or tolerance)" << std::endl;
        } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_MEMORY)) {
            amrex::Print() << "       (Solver memory allocation error)" << std::endl;
        } else if (HYPRE_CheckError(ierr, HYPRE_ERROR_ARG)) {
            amrex::Print() << "       (Solver argument error)" << std::endl;
        }
        // Consider checking for HYPRE_ERROR_INDEFINITE_PC if using certain preconditioners
        return false; // Indicate solver failure
    }
     // Optional: Check convergence status explicitly
     if (final_res_norm > m_eps && num_iterations >= m_maxiter) {
         amrex::Print() << "Warning: Solver reached max iterations without converging to tolerance." << std::endl;
         // Decide if this should return false or true depending on requirements
     }


    // Copy solution from HYPRE vector m_x to AMReX MultiFab m_mf_phi (Component SolnComp)
    getSolution(m_mf_phi);

    // Fill Component PhaseComp of m_mf_phi with cell types based on phase
    getCellTypes(m_mf_phi, PhaseComp);

    // --- Write Plotfile ---
    // Ensure output directory exists
    if (amrex::ParallelDescriptor::IOProcessor()) { // Only IO Processor creates directory
        amrex::UtilCreateDirectory(m_resultspath, 0755);
    }
    // Barrier to make sure directory is created before non-IO processors proceed
    amrex::ParallelDescriptor::Barrier();

    // Construct filename
    std::string plotfilename = m_resultspath + "/diffusionplot_" + std::string(datetime);
    const std::vector<std::string> varnames = {"concentration", "phase"}; // Use component names

    amrex::WriteSingleLevelPlotfile(plotfilename, m_mf_phi, varnames, m_geom, 0.0, 0);
    amrex::Print() << "Plotfile written to: " << plotfilename << std::endl;

    return true; // Indicate solver success
}

/**
 * @brief Calculates the tortuosity value based on the solved potential field.
 * Calls solve() if necessary. Performs parallel reduction for flux sums.
 * @param refresh If true, forces a re-solve of the linear system.
 * @return The calculated tortuosity value (VF / RelativeDiffusivity).
 */
amrex::Real TortuosityHypre::value(const bool refresh)
{
    // Solve the system if needed
    if (refresh || m_first_call)
    {
        if (!solve()) {
            // Handle solver failure - maybe throw exception or return error code
            amrex::Abort("Solver failed in TortuosityHypre::value. Aborting calculation.");
            return -1.0; // Error indicator
        }
        m_first_call = false;
    }

    amrex::Real phisumlo = 0.0; // Sum of potential differences on low face
    amrex::Real phisumhi = 0.0; // Sum of potential differences on high face

    // Geometry info
    const amrex::Real dx = m_geom.CellSize(0);
    const amrex::Real dy = m_geom.CellSize(1);
    const amrex::Real dz = m_geom.CellSize(2);
    const amrex::Real length_x = m_geom.ProbLength(0);
    const amrex::Real length_y = m_geom.ProbLength(1);
    const amrex::Real length_z = m_geom.ProbLength(2);
    const amrex::Box domain_box = m_geom.Domain();

    // --- Calculate sum of potential differences across domain faces ---
    // This loop calculates local contributions; reduction follows.
    for (amrex::MFIter mfi(m_mf_phi); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        amrex::Array4<int const> const& phase_fab_4 = m_mf_phase.const_array(mfi);
        amrex::Array4<amrex::Real const> const& phi_fab_4 = m_mf_phi.const_array(mfi, SolnComp);

        const auto lo = lbound(box);
        const auto hi = ubound(box);

        if ( m_dir == 0) { // X-direction Flux Calculation
            const int ilo = domain_box.loVect()[0];
            const int ihi = domain_box.hiVect()[0];
            if (lo.x == ilo) { // Cells adjacent to low X face
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    if ( phase_fab_4(ilo, j, k) == m_phase && phase_fab_4(ilo+1, j, k) == m_phase ) {
                        phisumlo += phi_fab_4(ilo+1, j, k) - phi_fab_4(ilo, j, k);
                    }
                }}
            }
            if (hi.x == ihi) { // Cells adjacent to high X face
                 for (int k = lo.z; k <= hi.z; ++k) {
                 for (int j = lo.y; j <= hi.y; ++j) {
                    if ( phase_fab_4(ihi, j, k) == m_phase && phase_fab_4(ihi-1, j, k) == m_phase ) {
                        phisumhi += phi_fab_4(ihi, j, k) - phi_fab_4(ihi-1, j, k);
                    }
                }}
            }
        } else if ( m_dir == 1) { // Y-direction Flux Calculation
            const int jlo = domain_box.loVect()[1];
            const int jhi = domain_box.hiVect()[1];
            if (lo.y == jlo) { // Cells adjacent to low Y face
                for (int k = lo.z; k <= hi.z; ++k) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ( phase_fab_4(i, jlo, k) == m_phase && phase_fab_4(i, jlo+1, k) == m_phase ) {
                        phisumlo += phi_fab_4(i, jlo+1, k) - phi_fab_4(i, jlo, k);
                    }
                }}
            }
            if (hi.y == jhi) { // Cells adjacent to high Y face
                 for (int k = lo.z; k <= hi.z; ++k) {
                 for (int i = lo.x; i <= hi.x; ++i) {
                    if ( phase_fab_4(i, jhi, k) == m_phase && phase_fab_4(i, jhi-1, k) == m_phase ) {
                        phisumhi += phi_fab_4(i, jhi, k) - phi_fab_4(i, jhi-1, k);
                    }
                }}
            }
        } else if ( m_dir == 2) { // Z-direction Flux Calculation
            const int klo = domain_box.loVect()[2];
            const int khi = domain_box.hiVect()[2];
             if (lo.z == klo) { // Cells adjacent to low Z face
                 for (int j = lo.y; j <= hi.y; ++j) {
                 for (int i = lo.x; i <= hi.x; ++i) {
                    if ( phase_fab_4(i, j, klo) == m_phase && phase_fab_4(i, j, klo+1) == m_phase ) {
                        phisumlo += phi_fab_4(i, j, klo+1) - phi_fab_4(i, j, klo);
                    }
                }}
            }
            if (hi.z == khi) { // Cells adjacent to high Z face
                for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ( phase_fab_4(i, j, khi) == m_phase && phase_fab_4(i, j, khi-1) == m_phase ) {
                        phisumhi += phi_fab_4(i, j, khi) - phi_fab_4(i, j, khi-1);
                    }
                }}
            }
        } // End directional calculation blocks
    } // End MFIter loop

    // --- Parallel Reduction ---
    // Sum contributions from all MPI ranks
    // Using CommunicatorSub() assumes calculations are within a sub-communicator context if applicable.
    // If using MPI_COMM_WORLD always, use ParallelContext::CommunicatorAll() or MPI_COMM_WORLD directly.
    amrex::ParallelAllReduce::Sum(phisumlo, amrex::ParallelContext::CommunicatorSub());
    amrex::ParallelAllReduce::Sum(phisumhi, amrex::ParallelContext::CommunicatorSub());


    // --- Calculate Fluxes and Tortuosity ---
    amrex::Real fluxlo = 0.0;
    amrex::Real fluxhi = 0.0;
    amrex::Real flux_max = 0.0; // Theoretical max flux (Bulk Diffusivity * Grad(Phi)_applied * Area)

    // Calculate flux = Sum(DeltaPhi) / DeltaLength * Area_face
    // Calculate flux_max = DeltaPhi_total / Length_bulk * Area_bulk
    // ** VERIFY THESE FORMULAS ARE CORRECT FOR THE PROBLEM DEFINITION **
    if ( m_dir==0) {
        amrex::Real area = length_y * length_z;
        if (dx > tiny_flux_threshold) { // Avoid division by zero cell size
             fluxlo = phisumlo / dx * area;
             fluxhi = phisumhi / dx * area;
        }
        if (length_x > tiny_flux_threshold) { // Avoid division by zero length
            flux_max = (m_vhi-m_vlo) / length_x * area;
        }
    }
    else if ( m_dir==1) {
        amrex::Real area = length_x * length_z;
         if (dy > tiny_flux_threshold) {
             fluxlo = phisumlo / dy * area;
             fluxhi = phisumhi / dy * area;
         }
         if (length_y > tiny_flux_threshold) {
             flux_max = (m_vhi-m_vlo) / length_y * area;
         }
    }
    else if ( m_dir==2) {
        amrex::Real area = length_x * length_y;
         if (dz > tiny_flux_threshold) {
             fluxlo = phisumlo / dz * area;
             fluxhi = phisumhi / dz * area;
         }
          if (length_z > tiny_flux_threshold) {
             flux_max = (m_vhi-m_vlo) / length_z * area;
          }
    }

    // Calculate relative diffusivity D_eff / D_bulk = Flux_calculated / Flux_max
    amrex::Real rel_diffusivity = 0.0;
    if (std::abs(flux_max) > tiny_flux_threshold) {
        amrex::Real avg_flux = (fluxlo + fluxhi) / 2.0; // Average fluxes (should be equal for conservation)
        rel_diffusivity = avg_flux / flux_max;
    } else if (std::abs(fluxlo) > tiny_flux_threshold || std::abs(fluxhi) > tiny_flux_threshold){
        amrex::Print() << "Warning: Maximum theoretical flux is near zero, but calculated flux is non-zero. Check BCs or formulas." << std::endl;
    } else {
         amrex::Print() << "Warning: Maximum theoretical flux is near zero. Cannot compute relative diffusivity." << std::endl;
    }

    // Calculate Tortuosity = VolumeFraction / RelativeDiffusivity
    amrex::Real tau = 0.0;
     if (std::abs(rel_diffusivity) > tiny_flux_threshold) {
         tau = m_vf / rel_diffusivity;
     } else {
         // Only warn/indicate error if conductive phase exists
         if (m_vf > tiny_flux_threshold) {
             amrex::Print() << "Warning: Relative diffusivity is near zero for non-zero VF. Tortuosity is effectively infinite." << std::endl;
             // Return infinity or a very large number, depending on desired behavior
             tau = std::numeric_limits<amrex::Real>::infinity();
         } else {
             // If VF is zero, tortuosity is typically undefined or zero.
             tau = 0.0;
         }
     }

    // Print diagnostics (only on IOProcessor to avoid redundant output)
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "------------------------------------------" << std::endl;
        amrex::Print() << " Tortuosity Calculation Results (Dir=" << m_dir << ")" << std::endl;
        amrex::Print() << "   Volume Fraction (VF)                    : " << m_vf << std::endl;
        amrex::Print() << "   Relative Effective Diffusivity (D_eff/D): " << rel_diffusivity << std::endl;
        amrex::Print() << "   Tortuosity (tau = VF / (D_eff/D))       : " << tau << std::endl;
        amrex::Print() << "   --- Intermediate Values ---" << std::endl;
        amrex::Print() << "   Flux Low Face                           : " << fluxlo << std::endl;
        amrex::Print() << "   Flux High Face                          : " << fluxhi << std::endl;
        amrex::Print() << "   Flux Max Theoretical                    : " << flux_max << std::endl ;
        amrex::Print() << "   Flux Conservation Check |Low - High|    : " << std::abs(fluxlo - fluxhi) << std::endl;
        amrex::Print() << "------------------------------------------" << std::endl;
    }

    return tau;
}


/**
 * @brief Copies the solution vector from HYPRE (m_x) to Component 0
 * of the destination AMReX MultiFab (soln).
 * @param soln The destination MultiFab. Assumed to have at least 1 component.
 */
void TortuosityHypre::getSolution (amrex::MultiFab& soln)
{
    // Ensure soln has the required component
    AMREX_ASSERT(soln.nComp() > SolnComp);

    amrex::FArrayBox temp_fab; // Temporary fab if soln has ghost cells

    for (amrex::MFIter mfi(soln); mfi.isValid(); ++mfi)
    {
        const amrex::Box &reg = mfi.validbox();
        amrex::FArrayBox *target_fab_ptr;

        if (soln.nGrow() == 0 && reg == soln[mfi].box()) {
            // If soln has no ghost cells and box matches, write directly
            target_fab_ptr = &soln[mfi];
        } else {
            // Otherwise, use a temporary fab matching the valid region
            temp_fab.resize(reg);
            target_fab_ptr = &temp_fab;
        }

        auto reglo = getHypreLo(reg);
        auto reghi = getHypreHi(reg);

        // Retrieve data from HYPRE vector m_x into the target FArrayBox's data pointer.
        // Assumes HYPRE vector stores data for a single field corresponding to component SolnComp.
        HYPRE_Int ierr = HYPRE_StructVectorGetBoxValues(m_x, reglo.data(), reghi.data(), target_fab_ptr->dataPtr());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ierr == 0, "HYPRE_StructVectorGetBoxValues failed in getSolution");

        // If we used a temporary fab, copy it to the destination MultiFab's SolnComp (Component 0)
        if (target_fab_ptr == &temp_fab) {
            soln[mfi].copy(temp_fab, 0, SolnComp, 1); // srcComp=0, destComp=SolnComp, numComp=1
        }
    }
}


/**
 * @brief Fills a specified component of a MultiFab with cell type information based on phase.
 * Calls the Fortran routine `tortuosity_filct`.
 * @param phi The MultiFab to modify.
 * @param ncomp The component index within phi to fill (e.g., PhaseComp).
 */
void TortuosityHypre::getCellTypes(amrex::MultiFab& phi, const int ncomp)
{
    // Check if the component index is valid
    AMREX_ASSERT(ncomp >= 0 && ncomp < phi.nComp());

    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        amrex::FArrayBox& fab_target = phi[mfi]; // Target FArrayBox (Real)
        const amrex::IArrayBox& fab_phase_src = m_mf_phase[mfi]; // Source phase data (Integer)
        const amrex::Box& box = mfi.validbox();

        // --- Fortran Call ---
        // Assumes: tortuosity_filct fills component 'ncomp' of 'fab_target' based on 'fab_phase_src'.
        // Arguments:
        //   fab_target:    Target FArrayBox (Real, modified)
        //   fab_phase_src: Source IArrayBox (Integer, const)
        //   box:           Box defining iteration space
        //   m_phase:       Conducting phase ID (pointer to int)
        //   ncomp:         Component index in fab_target to fill (pointer to int, 0-based C++ index)
        //                  ** Ensure Fortran expects 0-based index or adjust accordingly **
        int comp_idx_fortran = ncomp; // Adjust if Fortran is 1-based: = ncomp + 1;
        tortuosity_filct(BL_TO_FORTRAN_FAB(fab_target),
                         BL_TO_FORTRAN_FAB(fab_phase_src),
                         BL_TO_FORTRAN_BOX(box),
                         &m_phase,
                         &comp_idx_fortran); // Pass address of component index
    }
}
