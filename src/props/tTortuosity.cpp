#include "../io/TiffReader.H" // Assuming TiffReader is in OpenImpala namespace
#include "TortuosityHypre.H"  // Assuming TortuosityHypre is in OpenImpala namespace
#include "VolumeFraction.H"   // Assuming VolumeFraction is in OpenImpala namespace

#include <AMReX.H>
#include <AMReX_ParmParse.H>       // For reading parameters
#include <AMReX_Utility.H>         // For amrex::UtilCreateDirectory
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>    // Include for plotfile writing if enabled
#include <AMReX_MultiFabUtil.H>    // For amrex::Copy

#include <cstdlib>   // For getenv
#include <string>    // For std::string
#include <stdexcept> // For std::runtime_error (optional error handling)
#include <cmath>     // For std::abs
#include <limits>    // For numeric_limits
#include <memory>    // For std::unique_ptr
#include <iomanip>   // For std::setprecision

// Helper function to convert string to Direction enum
// (Assumes Direction enum exists in OpenImpala namespace)
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
// (Assumes SolverType enum exists in OpenImpala::TortuosityHypre)
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
    // Add other supported solvers here
    else {
        amrex::Abort("Invalid solver string: " + solver_str + ". Supported: Jacobi, GMRES, FlexGMRES, PCG, ...");
        return OpenImpala::TortuosityHypre::SolverType::GMRES; // Avoid compiler warning
    }
}


int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Real strt_time = amrex::second();

        // --- Configuration via ParmParse ---
        std::string tifffile;
        std::string resultsdir;
        int phase_id = 1; // Default phase ID to calculate tortuosity for (e.g., the conducting phase)
        std::string direction_str = "X";
        std::string solver_str = "GMRES";
        int box_size = 32;
        int verbose = 1; // Print more details by default
        int write_plotfile = 0; // Default to no plotfile writing in test
        amrex::Real expected_vf = -1.0; // Use -1 to indicate not set
        amrex::Real expected_tau = -1.0;
        amrex::Real tolerance = 1e-9; // Default tolerance for comparisons
        amrex::Real threshold_val = 0.5; // Default threshold for segmenting 0/1 data
        amrex::Real v_lo = 0.0; // Default low boundary potential
        amrex::Real v_hi = 1.0; // Default high boundary potential

        {
            amrex::ParmParse pp; // Default scope
            // Use get! Abort if tifffile is not provided in the inputs file.
            pp.get("tifffile", tifffile);

            // Results directory: Try ParmParse first, then default
            if (!pp.query("resultsdir", resultsdir)) {
                const char* homeDir_cstr = getenv("HOME");
                if (!homeDir_cstr) {
                    amrex::Warning("Cannot determine default results directory: 'resultsdir' not in inputs and $HOME not set. Using './tortuosity_results'");
                    resultsdir = "./tortuosity_results";
                } else {
                    std::string homeDir = homeDir_cstr;
                    resultsdir = homeDir + "/openimpalaresults"; // Assumes Unix '/' separator
                }
                 if(amrex::ParallelDescriptor::IOProcessor()) // Print only once
                    amrex::Print() << " Parameter 'resultsdir' not specified, using default: " << resultsdir << "\n";
            }

            pp.query("phase_id", phase_id); // Which phase ID to analyze (e.g., 1 after thresholding)
            pp.query("direction", direction_str);
            pp.query("solver", solver_str);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("write_plotfile", write_plotfile); // Control plotfile generation
            pp.query("expected_vf", expected_vf);
            pp.query("expected_tau", expected_tau);
            pp.query("tolerance", tolerance);
            pp.query("threshold_val", threshold_val); // Allow overriding threshold value
            pp.query("v_lo", v_lo); // Allow overriding BC potential
            pp.query("v_hi", v_hi); // Allow overriding BC potential
        }

        // Convert string parameters to enums
        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str);

        // Print configuration (only Rank 0)
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Tortuosity Test Configuration ---\n";
            amrex::Print() << " TIF File:              " << tifffile << "\n";
            amrex::Print() << " Results Directory:     " << resultsdir << "\n";
            amrex::Print() << " Phase ID to Analyze:   " << phase_id << "\n";
            amrex::Print() << " Threshold Value:       " << threshold_val << "\n";
            amrex::Print() << " Direction:             " << direction_str << "\n";
            amrex::Print() << " Solver:                " << solver_str << "\n";
            amrex::Print() << " Box Size:              " << box_size << "\n";
            amrex::Print() << " Verbose:               " << verbose << "\n";
            amrex::Print() << " Write Plotfile:        " << write_plotfile << "\n"; // Print the int value read
            amrex::Print() << " Comparison Tolerance:  " << tolerance << "\n";
            amrex::Print() << " Boundary Potential Lo: " << v_lo << "\n";
            amrex::Print() << " Boundary Potential Hi: " << v_hi << "\n";
            if (expected_vf >= 0.0) amrex::Print() << " Expected VF:           " << expected_vf << "\n";
            if (expected_tau >= 0.0) amrex::Print() << " Expected Tortuosity:   " << expected_tau << "\n";
            amrex::Print() << "------------------------------------\n\n";
        }

        // Define AMReX objects
        amrex::Geometry geom;
        amrex::BoxArray ba_original; // Keep original name for clarity
        amrex::DistributionMapping dm_original; // Keep original name
        // Declare the final iMultiFab (with ghost cells) needed by TortuosityHypre
        amrex::iMultiFab mf_phase_with_ghost;

        // --- Read TIFF Metadata and Setup Grids/Geometry ---
        try {
            // Read metadata only first
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile); // Reads metadata in constructor
            if (!reader.isRead()) {
                throw std::runtime_error("Reader failed to read metadata (isRead() is false).");
            }

            const amrex::Box domain_box = reader.box();
            if (domain_box.isEmpty()) {
                throw std::runtime_error("Reader returned an empty domain box after metadata read.");
            }

            // Define physical domain size and geometry
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)}; // Non-periodic
            geom.define(domain_box, &rb, 0, is_periodic.data());

            // *** Build ORIGINAL BoxArray and DistributionMapping based on box_size ***
            ba_original.define(domain_box);
            ba_original.maxSize(box_size);
            dm_original.define(ba_original);

            // *** Step 1: Create temporary iMultiFab with 0 ghost cells using ORIGINAL BA/DM ***
            amrex::iMultiFab mf_phase_no_ghost(ba_original, dm_original, 1, 0); // Nghost=0
            mf_phase_no_ghost.setVal(-1); // Initialize (optional)

            // *** Step 2: Call reader threshold into the 0-ghost cell MultiFab ***
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Thresholding data (Phase=1 if > " << threshold_val << ", else 0) into temporary MF...\n";
            }
            reader.threshold(threshold_val, 1, 0, mf_phase_no_ghost); // Output 1 if > thresh, else 0

            // *** Verification Step (Optional but Recommended) ***
            int min_phase_tmp = mf_phase_no_ghost.min(0);
            int max_phase_tmp = mf_phase_no_ghost.max(0);
             if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "   Temporary phase field min/max: " << min_phase_tmp << " / " << max_phase_tmp << "\n";
             }
             if (min_phase_tmp == max_phase_tmp && ba_original.numPts() > 0) { // Check numPts > 0
                 amrex::Abort("FAIL: Phase field is uniform after thresholding! Check threshold/data.");
             }

            // *** Step 3: Define the final phase MultiFab with 1 ghost cell using ORIGINAL BA/DM ***
            const int required_ghost_cells = 1;
            mf_phase_with_ghost.define(ba_original, dm_original, 1, required_ghost_cells);

            // *** Step 4: Copy data from 0-ghost MF to ghosted MF (valid regions) ***
            amrex::Copy(mf_phase_with_ghost, mf_phase_no_ghost, 0, 0, 1, 0); // Comp 0 to Comp 0, 1 component, 0 src ghost

            // *** Step 5: Fill ghost cells of the final MultiFab ***
            mf_phase_with_ghost.FillBoundary(geom.periodicity());

            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader processing or grid setup: " + std::string(e.what()));
        }

        // --- Calculate and Verify Volume Fraction ---
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Volume Fraction for phase " << phase_id << "...\n";
        OpenImpala::VolumeFraction vf(mf_phase_with_ghost, phase_id);
        amrex::Real actual_vf = vf.value(false); // Get global value

         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Volume Fraction: " << actual_vf << "\n";
        if (expected_vf >= 0.0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Expected Volume Fraction:   " << expected_vf << "\n";
            if (std::abs(actual_vf - expected_vf) > tolerance) {
                amrex::Abort("FAIL: Volume Fraction mismatch. Diff: " + std::to_string(std::abs(actual_vf - expected_vf)));
            }
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:    PASS\n";
        } else {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:    SKIPPED (no expected value provided)\n";
        }
         if (actual_vf <= 0.0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Warning: Volume fraction for target phase " << phase_id << " is zero or negative. Tortuosity is ill-defined." << std::endl;
         }


        // --- Calculate and Verify Tortuosity ---

        // Ensure results directory exists (only IOProcessor needs to create)
        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            if (!amrex::UtilCreateDirectory(resultsdir, 0755)) {
                amrex::Warning("Could not create results directory: " + resultsdir);
            }
        }
        amrex::ParallelDescriptor::Barrier(); // Ensure directory exists before others proceed

        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        if (actual_vf > 0.0) // Only calculate if volume fraction is non-zero
        {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Tortuosity for phase " << phase_id << " in direction " << direction_str << " using " << solver_str << "...\n";

            // *** DEBUGGING STEP: Create simplified BA/DM for TortuosityHypre ***
            amrex::Box domain_box = geom.Domain();
            amrex::BoxArray simplified_ba(domain_box); // BoxArray containing only the domain
            // Make a default mapping (assigns all boxes - just one here - to rank 0)
            // This assumes the test runs with a single MPI rank (-np 1)
            amrex::DistributionMapping simplified_dm(simplified_ba, amrex::ParallelDescriptor::NProcs());

            if (verbose > 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "   DEBUG: Using simplified BoxArray/DM for TortuosityHypre constructor.\n";
                amrex::Print() << "   Simplified BA: " << simplified_ba << "\n";
            }
            // *** END DEBUGGING STEP ***


            // *** Step 6: Pass the GHOSTED MultiFab and SIMPLIFIED BA/DM to TortuosityHypre constructor ***
            // <<< UPDATED CONSTRUCTOR CALL >>>
            OpenImpala::TortuosityHypre tortuosity(
                geom,
                simplified_ba,   // <--- Use simplified BoxArray
                simplified_dm,   // <--- Use simplified DistributionMapping
                mf_phase_with_ghost, // Use the original ghosted phase data
                actual_vf, phase_id, direction,
                solver_type, resultsdir,
                v_lo, v_hi, // Use potentials read from ParmParse or defaults
                verbose,
                (write_plotfile != 0) // Pass boolean plotfile flag
            );

            actual_tau = tortuosity.value(); // Calculate tortuosity
        } else {
             if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Skipping Tortuosity calculation because Volume Fraction is zero.\n";
        }


         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Tortuosity:    " << actual_tau << "\n";
        if (expected_tau >= 0.0) {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Expected Tortuosity:    " << expected_tau << "\n";
            bool actual_is_invalid = std::isnan(actual_tau) || std::isinf(actual_tau);
            bool expected_is_invalid = std::isnan(expected_tau) || std::isinf(expected_tau);

            if (actual_is_invalid != expected_is_invalid) {
                 amrex::Abort("FAIL: Tortuosity mismatch. One value is finite, the other is NaN/Inf.");
            } else if (!actual_is_invalid && std::abs(actual_tau - expected_tau) > tolerance) {
                 amrex::Abort("FAIL: Tortuosity mismatch. Diff: " + std::to_string(std::abs(actual_tau - expected_tau)));
            }
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Tortuosity Check:         PASS\n";
        } else {
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Tortuosity Check:         SKIPPED (no expected value provided)\n";
        }

        // --- Success & Timing ---
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "\n Test Completed Successfully.\n";

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End of scope for AMReX objects
    amrex::Finalize();
    return 0; // Indicate success
}
