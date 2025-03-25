#include "../io/TiffReader.H" // Assuming TiffReader is in OpenImpala namespace
#include "TortuosityHypre.H"  // Assuming TortuosityHypre is in OpenImpala namespace
#include "VolumeFraction.H"   // Assuming VolumeFraction is in OpenImpala namespace

#include <AMReX.H>
#include <AMReX_ParmParse.H>      // For reading parameters
#include <AMReX_Utility.H>        // For amrex::UtilCreateDirectory
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>

#include <cstdlib>     // For getenv
#include <string>      // For std::string
#include <stdexcept>   // For std::runtime_error (optional error handling)
#include <cmath>       // For std::abs
#include <limits>      // For numeric_limits

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
        int phase_id = 0;
        std::string direction_str = "X";
        std::string solver_str = "GMRES";
        int box_size = 32;
        int verbose = 1; // Print more details by default
        int write_plotfile = 0; // Default to no plotfile writing in test
        amrex::Real expected_vf = -1.0; // Use -1 to indicate not set
        amrex::Real expected_tau = -1.0;
        amrex::Real tolerance = 1e-9; // Default tolerance for comparisons

        {
            amrex::ParmParse pp; // Default scope
            pp.get("tifffile", tifffile); // Mandatory: Test file path

            // Results directory: Try ParmParse first, then default to $HOME/openimpalaresults
            if (!pp.query("resultsdir", resultsdir)) {
                const char* homeDir_cstr = getenv("HOME");
                if (!homeDir_cstr) {
                    amrex::Abort("Cannot determine results directory: 'resultsdir' not in inputs and $HOME not set.");
                }
                std::string homeDir = homeDir_cstr;
                resultsdir = homeDir + "/openimpalaresults"; // Assumes Unix '/' separator
                amrex::Print() << " Parameter 'resultsdir' not specified, using default: " << resultsdir << "\n";
            }

            pp.query("phase", phase_id);
            pp.query("direction", direction_str);
            pp.query("solver", solver_str);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("write_plotfile", write_plotfile); // Control plotfile generation
            pp.query("expected_vf", expected_vf);
            pp.query("expected_tau", expected_tau);
            pp.query("tolerance", tolerance);
        }

        // Convert string parameters to enums
        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str);

        if (verbose > 0) {
            amrex::Print() << "\n--- Tortuosity Test Configuration ---\n";
            amrex::Print() << " TIF File:           " << tifffile << "\n";
            amrex::Print() << " Results Directory:  " << resultsdir << "\n";
            amrex::Print() << " Phase ID:           " << phase_id << "\n";
            amrex::Print() << " Direction:          " << direction_str << "\n";
            amrex::Print() << " Solver:             " << solver_str << "\n";
            amrex::Print() << " Box Size:           " << box_size << "\n";
            amrex::Print() << " Verbose:            " << verbose << "\n";
            amrex::Print() << " Write Plotfile:     " << write_plotfile << "\n";
            amrex::Print() << " Tolerance:          " << tolerance << "\n";
            if (expected_vf >= 0.0) amrex::Print() << " Expected VF:        " << expected_vf << "\n";
            if (expected_tau >= 0.0) amrex::Print() << " Expected Tortuosity:" << expected_tau << "\n";
            amrex::Print() << "------------------------------------\n\n";
        }

        // Define AMReX objects
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::iMultiFab mf_phase; // Phase data MultiFab

        // --- Read TIFF and Setup Grids/Geometry ---
        try {
            // Limit TiffReader scope to release memory early
            if (verbose > 0) amrex::Print() << " Reading file " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile);

            const amrex::Box domain_box = reader.box();

            // Define physical domain size - using simple {0,0,0} to {Lx,Ly,Lz}
            // Assumes voxel size is implicitly 1 unit. Adjust if voxel size is known.
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});

            amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)}; // Non-periodic
            geom.define(domain_box, &rb, 0, is_periodic.data());

            // Build BoxArray and DistributionMapping
            ba.define(domain_box);
            ba.maxSize(box_size);
            dm.define(ba);

            // Define the phase MultiFab with 1 component and 1 ghost cell layer
            mf_phase.define(ba, dm, 1, 1);

            // Read/threshold phase data into the valid regions of mf_phase
            if (verbose > 0) amrex::Print() << " Thresholding phase data for phase " << phase_id << "...\n";
            reader.threshold(phase_id, mf_phase); // Assuming this puts 1 for phase_id, 0 otherwise? Or phase IDs directly? Adjust VF/Tortuosity phase param accordingly.
                                                  // Let's assume it fills with actual phase IDs from file.

            // Fill ghost cells based on neighboring valid data
            mf_phase.FillBoundary(geom.periodicity());

            if (verbose > 0) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader processing or grid setup: " + std::string(e.what()));
        }

        // --- Calculate and Verify Volume Fraction ---
        if (verbose > 0) amrex::Print() << " Calculating Volume Fraction for phase " << phase_id << "...\n";
        OpenImpala::VolumeFraction vf(mf_phase, phase_id); // Assumes VF calculates for phase `phase_id` vs others. Check component index if needed.
        amrex::Real actual_vf = vf.value(false); // Get global value

        amrex::Print() << " Calculated Volume Fraction: " << actual_vf << "\n";
        if (expected_vf >= 0.0) { // Check only if an expected value was provided
            amrex::Print() << " Expected Volume Fraction:   " << expected_vf << "\n";
            if (std::abs(actual_vf - expected_vf) > tolerance) {
                amrex::Abort("FAIL: Volume Fraction mismatch. Diff: " + std::to_string(std::abs(actual_vf - expected_vf)));
            }
             amrex::Print() << " Volume Fraction Check:      PASS\n";
        } else {
             amrex::Print() << " Volume Fraction Check:      SKIPPED (no expected value provided)\n";
        }


        // --- Calculate and Verify Tortuosity ---

        // Ensure results directory exists
        if (!amrex::UtilCreateDirectory(resultsdir, 0755)) {
            amrex::Warning("Could not create results directory: " + resultsdir);
            // Depending on TortuosityHypre implementation, this might be non-fatal if plotfiles are disabled
        }

        if (verbose > 0) amrex::Print() << " Calculating Tortuosity for phase " << phase_id << " in direction " << direction_str << " using " << solver_str << "...\n";

        // Create Tortuosity object
        // Note: Assumes TortuosityHypre constructor takes: geom, ba, dm, phase_mf, vf_value, phase_id, direction, solver_type, results_dir, [optional: plot_flag]
        // Adjust parameters based on the actual TortuosityHypre constructor signature.
        OpenImpala::TortuosityHypre tortuosity(geom, ba, dm, mf_phase, actual_vf, phase_id, direction, solver_type, resultsdir /*, maybe write_plotfile flag? */);

        amrex::Real actual_tau = tortuosity.value(); // Calculate tortuosity

        amrex::Print() << " Calculated Tortuosity:      " << actual_tau << "\n";
        if (expected_tau >= 0.0) { // Check only if an expected value was provided
             amrex::Print() << " Expected Tortuosity:      " << expected_tau << "\n";
            if (!std::isnan(actual_tau) && !std::isinf(actual_tau) && // Check for valid number before comparison
                 std::abs(actual_tau - expected_tau) > tolerance) {
                 amrex::Abort("FAIL: Tortuosity mismatch. Diff: " + std::to_string(std::abs(actual_tau - expected_tau)));
             }
             // Handle expected NaN/Inf cases if necessary for specific tests
             else if ((std::isnan(actual_tau) || std::isinf(actual_tau)) && !(std::isnan(expected_tau) || std::isinf(expected_tau))) {
                 amrex::Abort("FAIL: Tortuosity mismatch. Calculated NaN/Inf, expected finite.");
             }
             else if (!(std::isnan(actual_tau) || std::isinf(actual_tau)) && (std::isnan(expected_tau) || std::isinf(expected_tau))) {
                 amrex::Abort("FAIL: Tortuosity mismatch. Calculated finite, expected NaN/Inf.");
             }
             amrex::Print() << " Tortuosity Check:         PASS\n";
        } else {
             amrex::Print() << " Tortuosity Check:         SKIPPED (no expected value provided)\n";
        }

        // --- Success & Timing ---
        amrex::Print() << "\n Test Completed Successfully.\n";

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End of scope for AMReX objects
    amrex::Finalize();
    return 0; // Indicate success
}
