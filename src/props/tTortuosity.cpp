#include "../io/TiffReader.H" // Assuming TiffReader is in OpenImpala namespace
#include "TortuosityHypre.H"  // Assuming TortuosityHypre is in OpenImpala namespace
#include "VolumeFraction.H"   // Assuming VolumeFraction is in OpenImpala namespace

#include <AMReX.H>
#include <AMReX_ParmParse.H>         // For reading parameters
#include <AMReX_Utility.H>           // For amrex::UtilCreateDirectory
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>      // Include for plotfile writing if enabled
#include <AMReX_MultiFabUtil.H>      // For amrex::Copy
#include <AMReX_Exception.H>

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
    // Allow case-insensitive matching
    std::string lower_dir_str = dir_str;
    std::transform(lower_dir_str.begin(), lower_dir_str.end(), lower_dir_str.begin(), ::tolower);

    if (lower_dir_str == "x") {
        return OpenImpala::Direction::X;
    } else if (lower_dir_str == "y") {
        return OpenImpala::Direction::Y;
    } else if (lower_dir_str == "z") {
        return OpenImpala::Direction::Z;
    } else {
        amrex::Abort("Invalid direction string: " + dir_str + ". Use X, Y, or Z.");
        return OpenImpala::Direction::X; // Avoid compiler warning
    }
}

// Helper function to convert string to SolverType enum
// <<< UPDATED to include SMG >>>
OpenImpala::TortuosityHypre::SolverType stringToSolverType(const std::string& solver_str) {
    // Allow case-insensitive matching
    std::string lower_solver_str = solver_str;
    std::transform(lower_solver_str.begin(), lower_solver_str.end(), lower_solver_str.begin(), ::tolower);

    if (lower_solver_str == "jacobi") {
        return OpenImpala::TortuosityHypre::SolverType::Jacobi;
    } else if (lower_solver_str == "gmres") {
        return OpenImpala::TortuosityHypre::SolverType::GMRES;
    } else if (lower_solver_str == "flexgmres") {
        return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
    } else if (lower_solver_str == "pcg") {
        return OpenImpala::TortuosityHypre::SolverType::PCG;
    } else if (lower_solver_str == "bicgstab") {
        return OpenImpala::TortuosityHypre::SolverType::BiCGSTAB;
    } else if (lower_solver_str == "smg") { // <<< ADDED SMG CHECK HERE
        return OpenImpala::TortuosityHypre::SolverType::SMG;
    }
    else {
        // <<< UPDATED Error Message >>>
        amrex::Abort("Invalid solver string: " + solver_str + ". Supported: Jacobi, GMRES, FlexGMRES, PCG, BiCGSTAB, SMG");
        // Return a default to avoid compiler warnings, although Abort stops execution
        return OpenImpala::TortuosityHypre::SolverType::GMRES;
    }
}


int main (int argc, char* argv[])
{
    // Initialize HYPRE First (as per previous debugging step)
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL ERROR: HYPRE_Init() failed with code %d\n", hypre_ierr);
        // No MPI_Abort here as MPI might not be initialized yet. Standard return.
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
        std::string solver_str = "GMRES"; // Default if not specified in inputs
        int box_size = 32;
        int verbose = 1;
        int write_plotfile = 0;
        amrex::Real expected_vf = -1.0;  // Use negative to indicate not set
        amrex::Real expected_tau = -1.0; // Use negative to indicate not set
        amrex::Real tolerance = 1e-9;    // Default tolerance for value check if expected_tau is set
        amrex::Real threshold_val = 0.5; // Default threshold for TiffReader
        amrex::Real v_lo = 0.0;          // Default boundary value low
        amrex::Real v_hi = 1.0;          // Default boundary value high

        {
            amrex::ParmParse pp; // Default ParmParse prefix (no argument)
            // Use get() for required parameters, query() for optional
            pp.get("tifffile", tifffile);
            // Handle resultsdir default using getenv for HOME
            if (!pp.query("resultsdir", resultsdir)) {
                const char* homeDir_cstr = getenv("HOME");
                if (!homeDir_cstr) {
                    amrex::Warning("Cannot determine default results directory: 'resultsdir' not in inputs and $HOME not set. Using './tortuosity_results'");
                    resultsdir = "./tortuosity_results";
                } else {
                    std::string homeDir = homeDir_cstr;
                    // Ensure results dir ends with separator if needed, depends on UtilCreateDirectory handling
                    resultsdir = homeDir + "/openimpalaresults";
                }
                 if(amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << " Parameter 'resultsdir' not specified, using default: " << resultsdir << "\n";
            }
            // Query optional parameters, keeping defaults if not found
            pp.query("phase_id", phase_id);
            pp.query("direction", direction_str);
            pp.query("solver", solver_str);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("write_plotfile", write_plotfile);
            pp.query("expected_vf", expected_vf);
            pp.query("expected_tau", expected_tau);
            pp.query("tolerance", tolerance); // Tolerance for comparing tau_actual vs tau_expected
            pp.query("threshold_val", threshold_val);
            pp.query("v_lo", v_lo);
            pp.query("v_hi", v_hi);
        }

        // Convert string parameters to enums
        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str); // Use updated function

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Tortuosity Test Configuration ---\n";
            amrex::Print() << "  Input TIFF File:    " << tifffile << "\n";
            amrex::Print() << "  Results Dir:        " << resultsdir << "\n";
            amrex::Print() << "  Phase ID:           " << phase_id << "\n";
            amrex::Print() << "  Direction:          " << direction_str << "\n";
            amrex::Print() << "  Solver:             " << solver_str << "\n"; // Print the string read from input
            amrex::Print() << "  Box Size:           " << box_size << "\n";
            amrex::Print() << "  Threshold Value:    " << threshold_val << "\n";
            amrex::Print() << "  Boundary Values:    " << v_lo << " (lo), " << v_hi << " (hi)\n";
            amrex::Print() << "  Write Plotfile:     " << (write_plotfile != 0 ? "Yes" : "No") << "\n";
            amrex::Print() << "  Verbose Level:      " << verbose << "\n";
            if (expected_tau >= 0.0) {
                 amrex::Print() << "  Expected Tau:       " << expected_tau << "\n";
                 amrex::Print() << "  Check Tolerance:    " << tolerance << "\n";
            }
            amrex::Print() << "------------------------------------\n\n";
        }

        // --- AMReX Grid and Data Setup ---
        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_with_ghost; // Phase data with required ghost cells

        try {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile); // Create reader for the TIFF file
            if (!reader.isRead()) { throw std::runtime_error("Reader failed to read metadata."); }

            const amrex::Box domain_box = reader.box(); // Get domain box from reader
            if (domain_box.isEmpty()) { throw std::runtime_error("Reader returned empty domain box."); }

            // Define geometry based on image dimensions
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(amrex::Real(domain_box.length(0)), amrex::Real(domain_box.length(1)), amrex::Real(domain_box.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)}; // Assuming non-periodic domain
            geom.define(domain_box, &rb, 0, is_periodic.data());

            // Create BoxArray and DistributionMapping
            ba_original.define(domain_box);
            ba_original.maxSize(box_size); // Chop into boxes of max size
            dm_original.define(ba_original); // Create distribution mapping

            // Create iMultiFab for phase data (no ghost cells initially)
            amrex::iMultiFab mf_phase_no_ghost(ba_original, dm_original, 1, 0);

            // Threshold the data directly into the iMultiFab
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << " Thresholding data...\n"; }
            // Assign phase '1' if > threshold, '0' otherwise
            reader.threshold(threshold_val, 1, 0, mf_phase_no_ghost);

            // Basic check on thresholded data
            int min_phase_tmp = mf_phase_no_ghost.min(0);
            int max_phase_tmp = mf_phase_no_ghost.max(0);
              if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   Temporary phase field min/max: " << min_phase_tmp << " / " << max_phase_tmp << "\n"; }
              // Check if the entire domain is a single phase after thresholding
              if (min_phase_tmp == max_phase_tmp && ba_original.numPts() > 0) {
                   amrex::Abort("FAIL: Phase field uniform after thresholding. Check threshold value or input image.");
              }

            // Create the final phase iMultiFab with ghost cells and copy data
            const int required_ghost_cells = 1; // Need 1 ghost cell for TortuosityHypre
            mf_phase_with_ghost.define(ba_original, dm_original, 1, required_ghost_cells);
            amrex::Copy(mf_phase_with_ghost, mf_phase_no_ghost, 0, 0, 1, 0); // Copy valid data

            // Fill ghost cells (important for phase checks in TortuosityHypre)
            mf_phase_with_ghost.FillBoundary(geom.periodicity());

            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader/grid setup: " + std::string(e.what()));
        }

        // --- Calculate Volume Fraction ---
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Volume Fraction...\n";
        // Use the phase iMultiFab with ghost cells, VolumeFraction class should handle valid regions
        OpenImpala::VolumeFraction vf(mf_phase_with_ghost, phase_id);
        amrex::Real actual_vf = vf.value(false); // Calculate VF for the specified phase_id
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Volume Fraction: " << actual_vf << "\n";

        // Optional check against expected VF
        if (expected_vf >= 0.0) {
              // Allow small tolerance for floating point comparison
              if (std::abs(actual_vf - expected_vf) > 1e-9) {
                   amrex::Abort("FAIL: Volume fraction mismatch. Expected: " + std::to_string(expected_vf) + ", Calculated: " + std::to_string(actual_vf));
              } else {
                   if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:    PASS\n";
              }
        } else {
              if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:    SKIPPED (no expected value provided)\n";
        }


        // --- Calculate Tortuosity ---
        if (!resultsdir.empty()) {
            amrex::UtilCreateDirectory(resultsdir, 0755); // Create results directory if specified
        }
        amrex::ParallelDescriptor::Barrier(); // Ensure directory exists on all ranks if needed by HYPRE/Plotfiles

        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        bool test_passed = true; // Assume pass initially

        // Only calculate tortuosity if the phase exists
        if (actual_vf > std::numeric_limits<amrex::Real>::epsilon()) // Use epsilon check for > 0
        {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Tortuosity...\n";
            try {
                // Create and run the Tortuosity calculation
                OpenImpala::TortuosityHypre tortuosity(
                    geom, ba_original, dm_original, mf_phase_with_ghost,
                    actual_vf, phase_id, direction, solver_type, resultsdir,
                    v_lo, v_hi, verbose, (write_plotfile != 0)
                );
                actual_tau = tortuosity.value(); // Calculate tortuosity (this calls solve() internally)

                // Check if the calculation itself resulted in NaN/Inf (e.g., due to solver failure)
                if (std::isnan(actual_tau) || std::isinf(actual_tau)) {
                    amrex::Print() << "FAIL: Calculated tortuosity is NaN or Inf!\n";
                    // NOTE: The TortuosityHypre::value() method should have already printed a warning
                    // if the solver failed. This catches cases where flux calculation might fail too.
                    test_passed = false; // Mark test as failed
                }
            } catch (const std::exception& stdExc) {
                 amrex::Print() << "FAIL: Caught std::exception during Tortuosity calculation: " << stdExc.what() << "\n";
                 test_passed = false;
                 actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
            } catch (...) {
                 amrex::Print() << "FAIL: Caught unknown exception during Tortuosity calculation.\n";
                 test_passed = false;
                 actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
            }
        } else {
             if (verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) // Print even if verbose=0
                 amrex::Print() << " Skipping Tortuosity calculation because Volume Fraction is effectively zero (" << actual_vf << ").\n";
             // Decide how to handle expected_tau if VF is zero
             if (expected_tau >= 0.0) {
                 // If an expected tau was given but VF=0, this is likely a test setup mismatch
                 amrex::Print() << "WARNING: VF is zero, but expected_tau > 0 was provided. Check test logic.\n";
                 test_passed = false; // Consider this a failure
             } else {
                 // If VF=0 and no tau expected, this is okay, test logic might rely on skipping.
                 // test_passed remains true by default.
             }
             actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN(); // Ensure tau is NaN if skipped
        }

        // --- Check Tortuosity Value (if calculation was attempted and didn't fail immediately) ---
         if(amrex::ParallelDescriptor::IOProcessor()) {
             // Print calculated value regardless of whether expected value was provided
             amrex::Print() << " Calculated Tortuosity:    " << std::fixed << std::setprecision(8) << actual_tau << "\n";
         }

        if (expected_tau >= 0.0) { // Check only if an expected value is provided
             if(amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Expected Tortuosity:    " << std::fixed << std::setprecision(8) << expected_tau << "\n";
             }
             bool actual_is_invalid = std::isnan(actual_tau) || std::isinf(actual_tau);

             if (actual_is_invalid) {
                 // Failure message already printed above if tau is NaN/Inf
                 test_passed = false; // Ensure test_passed is false
             } else {
                 // Compare valid numbers using the specified tolerance
                 if (std::abs(actual_tau - expected_tau) > tolerance) {
                     if(amrex::ParallelDescriptor::IOProcessor()) {
                         amrex::Print() << "FAIL: Tortuosity mismatch. Diff: "
                                        << std::scientific << std::setprecision(6)
                                        << std::abs(actual_tau - expected_tau)
                                        << " > Tolerance: " << tolerance << "\n";
                     }
                     test_passed = false;
                 }
             }
             // Only print PASS if the check was performed and no failure occurred
             if(test_passed && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Tortuosity Check:         PASS\n";
             }
        } else { // No expected value provided
             if(amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Tortuosity Check:         SKIPPED (no expected value provided)\n";
             }
             // If tau is NaN/Inf here, it means calculation failed, test_passed should already be false.
             // If tau is a valid number, test passes by default if no expected value given.
        }

        // --- Final Verdict ---
        if (test_passed) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "\n Test Completed Successfully.\n";
        } else {
             // Use amrex::Abort to ensure the test run fails with non-zero exit code in CI
             amrex::Abort("Tortuosity Test FAILED.");
        }

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End of AMReX scope
    amrex::Finalize();

    // Finalize HYPRE
    hypre_ierr = HYPRE_Finalize();
     if (hypre_ierr != 0) {
         // Use fprintf as amrex::Print might not be available after amrex::Finalize
         fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
         return 1; // Return error code if finalize fails
     }

    // If amrex::Abort was called, this return statement might not be reached,
    // but it's good practice for cases where Abort might be caught or disabled.
    return 0; // Indicate success only if no Abort happened
}
