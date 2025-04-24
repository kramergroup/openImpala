#include "../io/TiffReader.H" // Assuming TiffReader is in OpenImpala namespace
#include "TortuosityHypre.H"  // Assuming TortuosityHypre is in OpenImpala namespace
#include "VolumeFraction.H"   // Assuming VolumeFraction is in OpenImpala namespace

#include <AMReX.H>
#include <AMReX_ParmParse.H>        // For reading parameters
#include <AMReX_Utility.H>          // For amrex::UtilCreateDirectory
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>     // Include for plotfile writing if enabled
#include <AMReX_MultiFabUtil.H>     // For amrex::Copy

#include <cstdlib>   // For getenv
#include <string>    // For std::string
#include <stdexcept> // For std::runtime_error (optional error handling)
#include <cmath>     // For std::abs, std::isnan, std::isinf
#include <limits>    // For numeric_limits
#include <memory>    // For std::unique_ptr
#include <iomanip>   // For std::setprecision
#include <stdio.h>   // For fprintf

#include <HYPRE.h>   // Include main HYPRE header
#include <mpi.h>     // Include MPI header for MPI_Finalize


// Helper function to convert string to Direction enum
OpenImpala::Direction stringToDirection(const std::string& dir_str) {
    if (dir_str == "X" || dir_str == "x") {
        return OpenImpala::Direction::X;
    } else if (dir_str == "Y" || dir_str == "y") {
        return OpenImpala::Direction::Y;
    } else if (dir_str == "Z" || dir_str == "z") {
        return OpenImpala::Direction::Z;
    } else {
        amrex::Abort("Invalid direction string: " + dir_str + ". Use X, Y, or Z.");
        return OpenImpala::Direction::X; // Avoid compiler warning
    }
}

// Helper function to convert string to SolverType enum
OpenImpala::TortuosityHypre::SolverType stringToSolverType(const std::string& solver_str) {
    if (solver_str == "Jacobi") {
        return OpenImpala::TortuosityHypre::SolverType::Jacobi;
    } else if (solver_str == "GMRES") {
        return OpenImpala::TortuosityHypre::SolverType::GMRES;
    } else if (solver_str == "FlexGMRES") {
        return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
    } else if (solver_str == "PCG") {
        return OpenImpala::TortuosityHypre::SolverType::PCG;
    }
    else {
        amrex::Abort("Invalid solver string: " + solver_str + ". Supported: Jacobi, GMRES, FlexGMRES, PCG, ...");
        return OpenImpala::TortuosityHypre::SolverType::GMRES; // Avoid compiler warning
    }
}


int main (int argc, char* argv[])
{
    // Initialize HYPRE First (as per previous debugging step)
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
         fprintf(stderr, "FATAL ERROR: HYPRE_Init() failed with code %d\n", hypre_ierr);
         return 1;
    }

    amrex::Initialize(argc, argv);
    { // Start AMReX scope
        amrex::Real strt_time = amrex::second();

        // --- Configuration via ParmParse ---
        std::string tifffile;
        std::string resultsdir;
        int phase_id = 1;
        std::string direction_str = "X";
        std::string solver_str = "GMRES";
        int box_size = 32;
        int verbose = 1;
        int write_plotfile = 0;
        amrex::Real expected_vf = -1.0;
        amrex::Real expected_tau = -1.0;
        amrex::Real tolerance = 1e-9;
        amrex::Real threshold_val = 0.5;
        amrex::Real v_lo = 0.0;
        amrex::Real v_hi = 1.0;

        {
            amrex::ParmParse pp;
            pp.get("tifffile", tifffile);
            if (!pp.query("resultsdir", resultsdir)) {
                const char* homeDir_cstr = getenv("HOME");
                if (!homeDir_cstr) {
                    amrex::Warning("Cannot determine default results directory: 'resultsdir' not in inputs and $HOME not set. Using './tortuosity_results'");
                    resultsdir = "./tortuosity_results";
                } else {
                    std::string homeDir = homeDir_cstr;
                    resultsdir = homeDir + "/openimpalaresults";
                }
                 if(amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << " Parameter 'resultsdir' not specified, using default: " << resultsdir << "\n";
            }
            pp.query("phase_id", phase_id);
            pp.query("direction", direction_str);
            pp.query("solver", solver_str);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("write_plotfile", write_plotfile);
            pp.query("expected_vf", expected_vf);
            pp.query("expected_tau", expected_tau);
            pp.query("tolerance", tolerance);
            pp.query("threshold_val", threshold_val);
            pp.query("v_lo", v_lo);
            pp.query("v_hi", v_hi);
        }

        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str);

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Tortuosity Test Configuration ---\n";
            // ... (print config details) ...
            amrex::Print() << "------------------------------------\n\n";
        }

        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_with_ghost;

        try {
            // ... (TIFF reading and grid/phase setup as before) ...
             if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile);
            if (!reader.isRead()) { throw std::runtime_error("Reader failed to read metadata."); }
            const amrex::Box domain_box = reader.box();
            if (domain_box.isEmpty()) { throw std::runtime_error("Reader returned empty domain box."); }
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(amrex::Real(domain_box.length(0)), amrex::Real(domain_box.length(1)), amrex::Real(domain_box.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
            geom.define(domain_box, &rb, 0, is_periodic.data());
            ba_original.define(domain_box);
            ba_original.maxSize(box_size);
            dm_original.define(ba_original);
            amrex::iMultiFab mf_phase_no_ghost(ba_original, dm_original, 1, 0);
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << " Thresholding data...\n"; }
            reader.threshold(threshold_val, 1, 0, mf_phase_no_ghost);
            int min_phase_tmp = mf_phase_no_ghost.min(0);
            int max_phase_tmp = mf_phase_no_ghost.max(0);
             if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "    Temporary phase field min/max: " << min_phase_tmp << " / " << max_phase_tmp << "\n"; }
             if (min_phase_tmp == max_phase_tmp && ba_original.numPts() > 0) { amrex::Abort("FAIL: Phase field uniform."); }
            const int required_ghost_cells = 1;
            mf_phase_with_ghost.define(ba_original, dm_original, 1, required_ghost_cells);
            amrex::Copy(mf_phase_with_ghost, mf_phase_no_ghost, 0, 0, 1, 0);
            mf_phase_with_ghost.FillBoundary(geom.periodicity());
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";
        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader/grid setup: " + std::string(e.what()));
        }

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Volume Fraction...\n";
        OpenImpala::VolumeFraction vf(mf_phase_with_ghost, phase_id);
        amrex::Real actual_vf = vf.value(false);
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Volume Fraction: " << actual_vf << "\n";
        // ... (VF check as before) ...

        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) { /* Create dir */ }
        amrex::ParallelDescriptor::Barrier();

        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        bool test_passed = true; // Assume pass initially

        if (actual_vf > 0.0)
        {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Tortuosity...\n";
            OpenImpala::TortuosityHypre tortuosity( geom, ba_original, dm_original, mf_phase_with_ghost, actual_vf, phase_id, direction, solver_type, resultsdir, v_lo, v_hi, verbose, (write_plotfile != 0) );
            actual_tau = tortuosity.value(); // Calculate tortuosity

            // <<< --- ADDED CHECK for NaN/Inf result --- >>>
            if (std::isnan(actual_tau) || std::isinf(actual_tau)) {
                 amrex::Print() << "FAIL: Calculated tortuosity is NaN or Inf!\n";
                 test_passed = false; // Mark test as failed
            }
            // <<< --- END ADDED CHECK --- >>>

        } else {
             if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Skipping Tortuosity calculation because Volume Fraction is zero.\n";
             // If VF is zero, expected tortuosity should likely also be handled differently or skipped
             if (expected_tau >= 0.0) {
                 amrex::Print() << "WARNING: VF is zero, but expected_tau was provided. Check test logic.\n";
                 // Decide if this should be a failure
                 // test_passed = false;
             }
        }

         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Tortuosity:    " << actual_tau << "\n";
        if (expected_tau >= 0.0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Expected Tortuosity:    " << expected_tau << "\n";
            bool actual_is_invalid = std::isnan(actual_tau) || std::isinf(actual_tau);
            bool expected_is_invalid = std::isnan(expected_tau) || std::isinf(expected_tau); // Should not happen if input is valid

            if (actual_is_invalid) { // Already checked above, but be explicit
                 amrex::Print() << "FAIL: Tortuosity mismatch. Calculated value is NaN/Inf.\n";
                 test_passed = false;
            } else if (std::abs(actual_tau - expected_tau) > tolerance) {
                 amrex::Print() << "FAIL: Tortuosity mismatch. Diff: " << std::abs(actual_tau - expected_tau) << "\n";
                 test_passed = false;
            }
            // Only print PASS if the check was performed and no failure occurred yet
             if(test_passed && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Tortuosity Check:         PASS\n";
        } else {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Tortuosity Check:         SKIPPED (no expected value provided)\n";
        }

        // --- Final Verdict ---
        if (test_passed) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "\n Test Completed Successfully.\n";
        } else {
             // Use amrex::Abort to ensure the test run fails in CI
             amrex::Abort("Tortuosity Test FAILED (NaN/Inf result or mismatch with expected value).");
        }

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End of AMReX scope
    amrex::Finalize();

    // Finalize HYPRE
    hypre_ierr = HYPRE_Finalize();
     if (hypre_ierr != 0) {
         fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
         return 1; // Return error code if finalize fails
     }

    return 0; // Indicate success only if no Abort happened
}
