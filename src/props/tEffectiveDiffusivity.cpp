#include "../io/TiffReader.H"
#include "EffectiveDiffusivityHypre.H" // Your new class
#include "Tortuosity.H"                // For OpenImpala::Direction enum

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H> // For writing plotfiles

#include <string>
#include <vector>
#include <stdexcept>
#include <memory> // For std::unique_ptr

// Helper function to convert string to Direction (can be shared or duplicated)
OpenImpala::Direction stringToDirection(const std::string& dir_str) {
    std::string lower_dir_str = dir_str;
    std::transform(lower_dir_str.begin(), lower_dir_str.end(), lower_dir_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (lower_dir_str == "x") return OpenImpala::Direction::X;
    if (lower_dir_str == "y") return OpenImpala::Direction::Y;
    if (lower_dir_str == "z") return OpenImpala::Direction::Z;
    amrex::Abort("Invalid direction string: " + dir_str + ". Use X, Y, or Z.");
    return OpenImpala::Direction::X; // Should not reach here
}

// Helper function to convert string to SolverType (can be shared or duplicated)
OpenImpala::EffectiveDiffusivityHypre::SolverType stringToSolverType(const std::string& solver_str) {
    std::string lower_solver_str = solver_str;
    std::transform(lower_solver_str.begin(), lower_solver_str.end(), lower_solver_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (lower_solver_str == "jacobi") return OpenImpala::EffectiveDiffusivityHypre::SolverType::Jacobi;
    if (lower_solver_str == "gmres") return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES;
    if (lower_solver_str == "flexgmres") return OpenImpala::EffectiveDiffusivityHypre::SolverType::FlexGMRES;
    if (lower_solver_str == "pcg") return OpenImpala::EffectiveDiffusivityHypre::SolverType::PCG;
    if (lower_solver_str == "bicgstab") return OpenImpala::EffectiveDiffusivityHypre::SolverType::BiCGSTAB;
    if (lower_solver_str == "smg") return OpenImpala::EffectiveDiffusivityHypre::SolverType::SMG;
    if (lower_solver_str == "pfmg") return OpenImpala::EffectiveDiffusivityHypre::SolverType::PFMG;
    amrex::Abort("Invalid solver string: '" + solver_str + "'. Supported: Jacobi, GMRES, FlexGMRES, PCG, BiCGSTAB, SMG, PFMG");
    return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES; // Should not reach here
}


int main (int argc, char* argv[])
{
        // Initialize HYPRE First
    int hypre_ierr = HYPRE_Init(); // ADD THIS LINE
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL ERROR: HYPRE_Init() failed with code %d\n", hypre_ierr);
        // It's good practice to ensure MPI is finalized if HYPRE_Init fails after MPI_Init
        // but before amrex::Initialize if HYPRE_Init itself requires MPI.
        // However, amrex::Initialize should handle MPI_Init.
        return 1;
    }
    // Initialize HYPRE First (if not already handled by AMReX with HYPRE enabled)
    // HYPRE_Init(); // AMReX_Initialize should handle this if built with HYPRE support.

    amrex::Initialize(argc, argv);
    { // Start AMReX scope
        amrex::Real strt_time = amrex::second();
        bool all_solves_converged = true;

        // --- Configuration via ParmParse ---
        std::string tifffile;
        std::string resultsdir = "./effdiff_test_results"; // Default results directory
        int phase_id = 1;      // Phase to consider as D=1 (e.g., pores)
        std::string solver_str = "FlexGMRES"; // Default solver
        int box_size = 32;
        int verbose = 1;
        int write_plotfile = 0; // 0 = no, 1 = yes
        amrex::Real threshold_val = 0.5; // For thresholding the input image

        {
            amrex::ParmParse pp; // Default ParmParse for command line (e.g. inputs file)
            pp.get("tifffile", tifffile); // Required
            pp.query("resultsdir", resultsdir);
            pp.query("phase_id", phase_id);
            pp.query("solver", solver_str);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("write_plotfile", write_plotfile);
            pp.query("threshold_val", threshold_val);

            // Hypre specific parameters from "hypre" prefix in inputs file
            // (These will be read by the EffectiveDiffusivityHypre constructor)
            // e.g., hypre.eps = 1.e-8, hypre.maxiter = 500
        }

        OpenImpala::EffectiveDiffusivityHypre::SolverType solver_type = stringToSolverType(solver_str);

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Effective Diffusivity Test Configuration ---\n";
            amrex::Print() << "  Input TIFF File:   " << tifffile << "\n";
            amrex::Print() << "  Results Dir:       " << resultsdir << "\n";
            amrex::Print() << "  Active Phase ID:   " << phase_id << "\n";
            amrex::Print() << "  Threshold Value:   " << threshold_val << "\n";
            amrex::Print() << "  Solver:            " << solver_str << "\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Write Plotfile:    " << (write_plotfile != 0 ? "Yes" : "No") << "\n";
            amrex::Print() << "  Verbose Level:     " << verbose << "\n";
            amrex::Print() << "----------------------------------------------\n\n";
        }

        // --- AMReX Grid and Data Setup ---
        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_input; // For the thresholded input image data

        try {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile); // Assuming TiffReader is robust
            if (!reader.isRead()) { throw std::runtime_error("TiffReader failed to read metadata."); }

            const amrex::Box domain_box = reader.box();
            if (domain_box.isEmpty()) { throw std::runtime_error("TiffReader returned empty domain box."); }

            // Define geometry (assuming physical dimensions match voxel dimensions for simplicity)
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)}; // Original
            geom.define(domain_box, &rb, 0, is_periodic.data());
            geom.define(domain_box, &rb, 0, is_periodic.data());

            ba_original.define(domain_box);
            ba_original.maxSize(box_size);
            dm_original.define(ba_original);

            // Phase MultiFab needs 1 ghost cell for the solver's generateActiveMask and Fortran kernel
            mf_phase_input.define(ba_original, dm_original, 1, 1);

            // Temporary MultiFab to read thresholded data without ghosts initially
            amrex::iMultiFab mf_temp_no_ghost(ba_original, dm_original, 1, 0);
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << " Thresholding data...\n"; }
            reader.threshold(threshold_val, phase_id, (phase_id == 0 ? 1 : 0) /*value for other phase*/, mf_temp_no_ghost);

            // Copy to mf_phase_input and fill ghost cells
            amrex::Copy(mf_phase_input, mf_temp_no_ghost, 0, 0, 1, 0); // Copy valid region
            mf_phase_input.FillBoundary(geom.periodicity());          // Fill ghost cells

            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Print() << "Error during TiffReader/grid setup: " << e.what() << std::endl;
            amrex::Abort("Test setup failed.");
        }

        // Create results directory
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();


        // --- Solve for Corrector Functions chi_x, chi_y, chi_z ---
        amrex::MultiFab mf_chi_x(ba_original, dm_original, 1, 1); // Store solved chi_x, 1 ghost for D_eff gradient
        amrex::MultiFab mf_chi_y(ba_original, dm_original, 1, 1); // Store solved chi_y
        amrex::MultiFab mf_chi_z(ba_original, dm_original, 1, 1); // Store solved chi_z

        std::vector<OpenImpala::Direction> directions_to_solve = {
            OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z
        };
        if (AMREX_SPACEDIM == 2) {
            directions_to_solve = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
        }


        for (const auto& dir_k : directions_to_solve) {
            std::string dir_k_str = (dir_k == OpenImpala::Direction::X) ? "X" :
                                    (dir_k == OpenImpala::Direction::Y) ? "Y" : "Z";
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Solving for Corrector Function chi_" << dir_k_str << " ---\n";
            }

            std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> eff_diff_solver;
            bool current_solve_converged = false;

            try {
                eff_diff_solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                    geom, ba_original, dm_original, mf_phase_input,
                    phase_id, dir_k, solver_type, resultsdir,
                    verbose, (write_plotfile != 0)
                );

                current_solve_converged = eff_diff_solver->solve();

                if (current_solve_converged) {
                    if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "  Solver for chi_" << dir_k_str << " CONVERGED.\n";
                    }
                    if (dir_k == OpenImpala::Direction::X) eff_diff_solver->getChiSolution(mf_chi_x);
                    else if (dir_k == OpenImpala::Direction::Y) eff_diff_solver->getChiSolution(mf_chi_y);
                    else if (dir_k == OpenImpala::Direction::Z) eff_diff_solver->getChiSolution(mf_chi_z);
                } else {
                    if (verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) { // Print even if not verbose=1
                        amrex::Print() << "  WARNING: Solver for chi_" << dir_k_str << " DID NOT CONVERGE.\n";
                    }
                    all_solves_converged = false;
                    // Set corresponding mf_chi to 0 or handle as error
                    if (dir_k == OpenImpala::Direction::X) mf_chi_x.setVal(0.0);
                    else if (dir_k == OpenImpala::Direction::Y) mf_chi_y.setVal(0.0);
                    else if (dir_k == OpenImpala::Direction::Z) mf_chi_z.setVal(0.0);
                }

            } catch (const std::exception& e) {
                amrex::Print() << "  ERROR during EffectiveDiffusivityHypre construction or solve for chi_"
                               << dir_k_str << ": " << e.what() << std::endl;
                all_solves_converged = false;
            }
        } // End loop over directions

        // --- Basic Test Verdict ---
        // For this initial test, PASS if all solves attempted and either converged or handled non-convergence gracefully.
        // A more advanced test would check the D_eff values.

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Effective Diffusivity Test Summary ---\n";
            amrex::Print() << "  Total Run Time: " << stop_time << " sec\n";
            if (all_solves_converged) {
                amrex::Print() << "  All chi_k solver instances reported convergence (or no active phase).\n";
                amrex::Print() << "  TEST RESULT: PASS (Basic execution and solver convergence)\n";
                amrex::Print() << "  Further validation of D_eff tensor values requires a separate step.\n";
            } else {
                amrex::Print() << "  One or more chi_k solver instances FAILED to converge or encountered an error.\n";
                amrex::Print() << "  TEST RESULT: FAIL (Check logs for solver issues)\n";
            }
            amrex::Print() << "-----------------------------------------\n";
        }

        if (!all_solves_converged) {
            amrex::Abort("EffectiveDiffusivity Test FAILED due to solver non-convergence or errors.");
        }

    } // End of AMReX scope
    amrex::Finalize();

    // Finalize HYPRE
    hypre_ierr = HYPRE_Finalize(); // ADD THIS LINE
    if (hypre_ierr != 0) {
        fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
        return 1;
    }

    return 0;
}
