#include "../io/TiffReader.H" // Assuming TiffReader is in OpenImpala namespace
#include "TortuosityHypre.H"  // Assuming TortuosityHypre is in OpenImpala namespace
#include "VolumeFraction.H"   // Assuming VolumeFraction is in OpenImpala namespace
#include "Tortuosity.H"       // Include base class for OpenImpala::Direction enum

#include <AMReX.H>
#include <AMReX_ParmParse.H>           // For reading parameters
#include <AMReX_Utility.H>             // For amrex::UtilCreateDirectory
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>        // Include for plotfile writing if enabled
#include <AMReX_MultiFabUtil.H>        // For amrex::Copy, Convert
#include <AMReX_Exception.H>           // For std::exception, AbortException (though catch might be removed)

#include <cstdlib>   // For getenv
#include <string>    // For std::string
#include <stdexcept> // For std::runtime_error (optional error handling)
#include <cmath>     // For std::abs, std::isnan, std::isinf
#include <limits>    // For numeric_limits
#include <memory>    // For std::unique_ptr
#include <iomanip>   // For std::setprecision
#include <stdio.h>   // For fprintf
#include <algorithm> // For std::transform

#include <HYPRE.h>   // Include main HYPRE header
#include <mpi.h>     // Include MPI header for MPI_Finalize


// Helper function to convert string to Direction enum
OpenImpala::Direction stringToDirection(const std::string& dir_str) {
    // Allow case-insensitive matching
    std::string lower_dir_str = dir_str;
    std::transform(lower_dir_str.begin(), lower_dir_str.end(), lower_dir_str.begin(),
                   [](unsigned char c){ return std::tolower(c); }); // Use lambda for safety

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
OpenImpala::TortuosityHypre::SolverType stringToSolverType(const std::string& solver_str) {
    // Allow case-insensitive matching
    std::string lower_solver_str = solver_str;
     std::transform(lower_solver_str.begin(), lower_solver_str.end(), lower_solver_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });

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
    } else if (lower_solver_str == "smg") {
        return OpenImpala::TortuosityHypre::SolverType::SMG;
    } else if (lower_solver_str == "pfmg") {
        return OpenImpala::TortuosityHypre::SolverType::PFMG;
    }
    else {
        amrex::Abort("Invalid solver string: '" + solver_str + "'. Supported: Jacobi, GMRES, FlexGMRES, PCG, BiCGSTAB, SMG, PFMG");
        return OpenImpala::TortuosityHypre::SolverType::GMRES;
    }
}


int main (int argc, char* argv[])
{
    // Initialize HYPRE First
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
                 if(amrex::ParallelDescriptor::IOProcessor()) // <<< CORRECTED NAMESPACE
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

        // Convert string parameters to enums
        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str);

        // --- CORRECTED NAMESPACE for Print ---
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Tortuosity Test Configuration ---\n";
            amrex::Print() << "  Input TIFF File:    " << tifffile << "\n";
            amrex::Print() << "  Results Dir:        " << resultsdir << "\n";
            amrex::Print() << "  Phase ID:           " << phase_id << "\n";
            amrex::Print() << "  Direction:          " << direction_str << "\n";
            amrex::Print() << "  Solver:             " << solver_str << "\n";
            amrex::Print() << "  Box Size:           " << box_size << "\n";
            amrex::Print() << "  Threshold Value:    " << threshold_val << "\n";
            amrex::Print() << "  Boundary Values:    " << v_lo << " (lo), " << v_hi << " (hi)\n";
            amrex::Print() << "  Write Plotfile:     " << (write_plotfile != 0 ? "Yes" : "No") << "\n";
            amrex::Print() << "  Verbose Level:      " << verbose << "\n";
            if (expected_tau >= 0.0) {
                 amrex::Print() << "  Expected Tau:       " << expected_tau << "\n";
                 amrex::Print() << "  Check Tolerance:    " << tolerance << "\n";
            }
            if (expected_vf >= 0.0) {
                  amrex::Print() << "  Expected VF:        " << expected_vf << "\n";
            }
            amrex::Print() << "------------------------------------\n\n";
        }

        // --- AMReX Grid and Data Setup ---
        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_with_ghost;

        try {
             // --- CORRECTED NAMESPACE for Print ---
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

            // --- CORRECTED NAMESPACE for Print ---
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << " Thresholding data...\n"; }
            reader.threshold(threshold_val, 1, 0, mf_phase_no_ghost);

            int min_phase_tmp = mf_phase_no_ghost.min(0);
            int max_phase_tmp = mf_phase_no_ghost.max(0);
             // --- CORRECTED NAMESPACE for Print ---
               if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   Temporary phase field min/max: " << min_phase_tmp << " / " << max_phase_tmp << "\n"; }
               if (min_phase_tmp == max_phase_tmp && ba_original.numPts() > 0) {
                    amrex::Abort("FAIL: Phase field uniform after thresholding. Check threshold value or input image.");
               }

            const int required_ghost_cells = 1;
            mf_phase_with_ghost.define(ba_original, dm_original, 1, required_ghost_cells);
            amrex::Copy(mf_phase_with_ghost, mf_phase_no_ghost, 0, 0, 1, 0);
            mf_phase_with_ghost.FillBoundary(geom.periodicity());

             // --- CORRECTED NAMESPACE for Print ---
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader/grid setup: " + std::string(e.what()));
        }

        // --- Calculate Volume Fraction ---
         // --- CORRECTED NAMESPACE for Print ---
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Volume Fraction...\n";
        OpenImpala::VolumeFraction vf(mf_phase_with_ghost, phase_id);
        amrex::Real actual_vf = vf.value(false);
         // --- CORRECTED NAMESPACE for Print ---
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Volume Fraction: " << actual_vf << "\n";

        if (expected_vf >= 0.0) {
              if (std::abs(actual_vf - expected_vf) > 1e-9) {
                   amrex::Abort("FAIL: Volume fraction mismatch. Expected: " + std::to_string(expected_vf) + ", Calculated: " + std::to_string(actual_vf));
              } else {
                   // --- CORRECTED NAMESPACE for Print ---
                   if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:    PASS\n";
              }
        } else {
             // --- CORRECTED NAMESPACE for Print ---
               if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:    SKIPPED (no expected value provided)\n";
        }


        // --- Calculate Tortuosity ---
        if (!resultsdir.empty()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        bool test_passed = true;

        if (actual_vf > std::numeric_limits<amrex::Real>::epsilon())
        {
             // --- CORRECTED NAMESPACE for Print ---
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Constructing TortuosityHypre object...\n";
            std::unique_ptr<OpenImpala::TortuosityHypre> tortuosity_ptr;
            try {
                 tortuosity_ptr = std::make_unique<OpenImpala::TortuosityHypre>(
                     geom, ba_original, dm_original, mf_phase_with_ghost,
                     actual_vf, phase_id, direction, solver_type, resultsdir,
                     v_lo, v_hi, verbose, (write_plotfile != 0)
                 );
            } catch (const std::exception& stdExc) {
                amrex::Print() << "FAIL: Caught std::exception during TortuosityHypre construction: " << stdExc.what() << "\n";
                test_passed = false;
                amrex::Abort("TortuosityHypre construction failed.");
            } catch (...) {
                amrex::Print() << "FAIL: Caught unknown exception during TortuosityHypre construction.\n";
                 test_passed = false;
                 amrex::Abort("TortuosityHypre construction failed.");
            }

            if (test_passed) { // Only proceed if construction succeeded
                 // --- Perform Matrix Checks ---
                 // --- CORRECTED NAMESPACE ---
                 if (amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << " Performing mathematical checks on the assembled matrix...\n";
                 }
                 bool matrix_checks_ok = tortuosity_ptr->checkMatrixProperties();

                 if (!matrix_checks_ok) {
                    amrex::Abort("FATAL: Assembled matrix/vector failed property checks in tTortuosity.");
                 } else {
                      // --- CORRECTED NAMESPACE ---
                      if (amrex::ParallelDescriptor::IOProcessor()) {
                          amrex::Print() << " Matrix property checks passed.\n";
                      }
                 }

                 // --- Calculate Tortuosity (calls solve internally) ---
                 if (matrix_checks_ok) { // Only calculate if checks passed
                      // --- CORRECTED NAMESPACE ---
                      if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Tortuosity value...\n";
                      try {
                          actual_tau = tortuosity_ptr->value();

                          if (std::isnan(actual_tau) || std::isinf(actual_tau)) {
                              amrex::Print() << "FAIL: Calculated tortuosity is NaN or Inf!\n";
                              test_passed = false;
                          }
                      // --- REMOVED catch for AbortException ---
                      } catch (const std::exception& stdExc) {
                           amrex::Print() << "FAIL: Caught std::exception during Tortuosity calculation: " << stdExc.what() << "\n";
                           test_passed = false;
                           actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
                      } catch (...) {
                           amrex::Print() << "FAIL: Caught unknown exception during Tortuosity calculation.\n";
                           test_passed = false;
                           actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
                      }
                 } else { // matrix checks failed
                    test_passed = false; // Ensure test fails if matrix checks fail
                 }
            } // end if test_passed (after construction)

        } else { // VF is zero
             // --- CORRECTED NAMESPACE for Print ---
             if (verbose >= 0 && amrex::ParallelDescriptor::IOProcessor())
                 amrex::Print() << " Skipping Tortuosity calculation because Volume Fraction is effectively zero (" << actual_vf << ").\n";
             if (expected_tau >= 0.0) {
                 // --- CORRECTED NAMESPACE for Print ---
                 amrex::Print() << "WARNING: VF is zero, but expected_tau > 0 was provided. Check test logic.\n";
                 test_passed = false;
             }
             actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        }

        // --- Check Tortuosity Value ---
         // --- CORRECTED NAMESPACE for Print ---
         if(amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << " Calculated Tortuosity:    " << std::fixed << std::setprecision(8) << actual_tau << "\n";
         }

        if (expected_tau >= 0.0) {
             // --- CORRECTED NAMESPACE for Print ---
              if(amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Expected Tortuosity:    " << std::fixed << std::setprecision(8) << expected_tau << "\n";
              }
              bool actual_is_invalid = std::isnan(actual_tau) || std::isinf(actual_tau);

              if (actual_is_invalid) {
                  test_passed = false;
              } else {
                  if (std::abs(actual_tau - expected_tau) > tolerance) {
                       // --- CORRECTED NAMESPACE for Print ---
                       if(amrex::ParallelDescriptor::IOProcessor()) {
                           amrex::Print() << "FAIL: Tortuosity mismatch. Diff: "
                                          << std::scientific << std::setprecision(6)
                                          << std::abs(actual_tau - expected_tau)
                                          << " > Tolerance: " << tolerance << "\n";
                       }
                       test_passed = false;
                  }
              }
             // --- CORRECTED NAMESPACE for Print ---
              if(test_passed && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Tortuosity Check:         PASS\n";
              }
        } else {
             // --- CORRECTED NAMESPACE for Print ---
              if(amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " Tortuosity Check:         SKIPPED (no expected value provided)\n";
              }
        }

        // --- Final Verdict ---
        if (test_passed) {
             // --- CORRECTED NAMESPACE for Print ---
             if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "\n Test Completed Successfully.\n";
        } else {
             amrex::Abort("Tortuosity Test FAILED.");
        }

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
         // --- CORRECTED NAMESPACE for Print ---
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End of AMReX scope
    amrex::Finalize();

    // Finalize HYPRE
    hypre_ierr = HYPRE_Finalize();
     if (hypre_ierr != 0) {
         fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
         return 1;
     }

    return 0;
}
