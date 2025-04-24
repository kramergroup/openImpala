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
#include <cmath>     // For std::abs
#include <limits>    // For numeric_limits
#include <memory>    // For std::unique_ptr
#include <iomanip>   // For std::setprecision
#include <stdio.h>   // For fprintf in HYPRE_Init check

#include <HYPRE.h>   // <<< ADDED: Include main HYPRE header >>>
#include <mpi.h>     // <<< ADDED: Include MPI header for MPI_Finalize >>>


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
    // --- Try Initializing HYPRE First ---
    // Note: AMReX_Initialize will handle MPI_Init if not already called,
    // but HYPRE_Init might need MPI to be initialized first depending on its build.
    // It's generally safer to let AMReX handle MPI_Init first.
    // However, for debugging this specific issue, we try HYPRE_Init first.
    // If this causes MPI issues, revert this change.
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
         // Basic error print if HYPRE_Init fails
         // Note: Cannot use amrex::Print before amrex::Initialize
         fprintf(stderr, "FATAL ERROR: HYPRE_Init() failed with code %d\n", hypre_ierr);
         // Attempt to finalize MPI if it might have been implicitly initialized by HYPRE
         // This is speculative and might not be necessary or correct.
         // MPI_Finalize();
         return 1;
    }
    // --- End HYPRE Init ---


    amrex::Initialize(argc, argv);
    { // Start AMReX scope
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
            amrex::Print() << " TIF File:               " << tifffile << "\n";
            amrex::Print() << " Results Directory:      " << resultsdir << "\n";
            amrex::Print() << " Phase ID to Analyze:    " << phase_id << "\n";
            amrex::Print() << " Threshold Value:        " << threshold_val << "\n";
            amrex::Print() << " Direction:              " << direction_str << "\n";
            amrex::Print() << " Solver:                 " << solver_str << "\n";
            amrex::Print() << " Box Size:               " << box_size << "\n";
            amrex::Print() << " Verbose:                " << verbose << "\n";
            amrex::Print() << " Write Plotfile:         " << write_plotfile << "\n";
            amrex::Print() << " Comparison Tolerance:   " << tolerance << "\n";
            amrex::Print() << " Boundary Potential Lo:  " << v_lo << "\n";
            amrex::Print() << " Boundary Potential Hi:  " << v_hi << "\n";
            if (expected_vf >= 0.0) amrex::Print() << " Expected VF:            " << expected_vf << "\n";
            if (expected_tau >= 0.0) amrex::Print() << " Expected Tortuosity:    " << expected_tau << "\n";
            amrex::Print() << "------------------------------------\n\n";
        }

        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_with_ghost;

        try {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile);
            if (!reader.isRead()) {
                throw std::runtime_error("Reader failed to read metadata (isRead() is false).");
            }

            const amrex::Box domain_box = reader.box();
            if (domain_box.isEmpty()) {
                throw std::runtime_error("Reader returned an empty domain box after metadata read.");
            }

            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
            geom.define(domain_box, &rb, 0, is_periodic.data());

            ba_original.define(domain_box);
            ba_original.maxSize(box_size);
            dm_original.define(ba_original);

            amrex::iMultiFab mf_phase_no_ghost(ba_original, dm_original, 1, 0);
            mf_phase_no_ghost.setVal(-1);

            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Thresholding data (Phase=1 if > " << threshold_val << ", else 0) into temporary MF...\n";
            }
            reader.threshold(threshold_val, 1, 0, mf_phase_no_ghost);

            int min_phase_tmp = mf_phase_no_ghost.min(0);
            int max_phase_tmp = mf_phase_no_ghost.max(0);
             if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                  amrex::Print() << "    Temporary phase field min/max: " << min_phase_tmp << " / " << max_phase_tmp << "\n";
             }
             if (min_phase_tmp == max_phase_tmp && ba_original.numPts() > 0) {
                  amrex::Abort("FAIL: Phase field is uniform after thresholding! Check threshold/data.");
             }

            const int required_ghost_cells = 1;
            mf_phase_with_ghost.define(ba_original, dm_original, 1, required_ghost_cells);

            amrex::Copy(mf_phase_with_ghost, mf_phase_no_ghost, 0, 0, 1, 0);

            mf_phase_with_ghost.FillBoundary(geom.periodicity());

            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader processing or grid setup: " + std::string(e.what()));
        }

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Volume Fraction for phase " << phase_id << "...\n";
        OpenImpala::VolumeFraction vf(mf_phase_with_ghost, phase_id);
        amrex::Real actual_vf = vf.value(false);

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

        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            if (!amrex::UtilCreateDirectory(resultsdir, 0755)) {
                amrex::Warning("Could not create results directory: " + resultsdir);
            }
        }
        amrex::ParallelDescriptor::Barrier();

        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        if (actual_vf > 0.0)
        {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Tortuosity for phase " << phase_id << " in direction " << direction_str << " using " << solver_str << "...\n";

            OpenImpala::TortuosityHypre tortuosity(
                geom,
                ba_original,
                dm_original,
                mf_phase_with_ghost,
                actual_vf, phase_id, direction,
                solver_type, resultsdir,
                v_lo, v_hi,
                verbose,
                (write_plotfile != 0)
            );

            actual_tau = tortuosity.value();
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

         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "\n Test Completed Successfully.\n";

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End of AMReX scope
    amrex::Finalize();

    // --- Finalize HYPRE Last ---
    hypre_ierr = HYPRE_Finalize();
     if (hypre_ierr != 0) {
         // Use fprintf as amrex::Print might not be available after amrex::Finalize
         fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
         // Return non-zero to indicate potential issue during cleanup
         return 1;
     }
    // --- End HYPRE Finalize ---

    return 0; // Indicate success
}
