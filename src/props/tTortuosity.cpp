#include "../io/TiffReader.H" // Assuming TiffReader is in OpenImpala namespace
#include "TortuosityHypre.H"  // Assuming TortuosityHypre is in OpenImpala namespace
#include "VolumeFraction.H"   // Assuming VolumeFraction is in OpenImpala namespace
#include "Tortuosity.H"       // Include base class for OpenImpala::Direction enum

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Exception.H>

#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <memory>
#include <iomanip>
#include <stdio.h>
#include <algorithm>
#include <vector>

#include <HYPRE.h>
#include <mpi.h>

// Helper function definitions (stringToDirection, stringToSolverType) remain the same...
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

OpenImpala::TortuosityHypre::SolverType stringToSolverType(const std::string& solver_str) {
    std::string lower_solver_str = solver_str;
    std::transform(lower_solver_str.begin(), lower_solver_str.end(), lower_solver_str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (lower_solver_str == "jacobi") return OpenImpala::TortuosityHypre::SolverType::Jacobi;
    if (lower_solver_str == "gmres") return OpenImpala::TortuosityHypre::SolverType::GMRES;
    if (lower_solver_str == "flexgmres") return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
    if (lower_solver_str == "pcg") return OpenImpala::TortuosityHypre::SolverType::PCG;
    if (lower_solver_str == "bicgstab") return OpenImpala::TortuosityHypre::SolverType::BiCGSTAB;
    if (lower_solver_str == "smg") return OpenImpala::TortuosityHypre::SolverType::SMG;
    if (lower_solver_str == "pfmg") return OpenImpala::TortuosityHypre::SolverType::PFMG;
    amrex::Abort("Invalid solver string: '" + solver_str + "'. Supported: Jacobi, GMRES, FlexGMRES, PCG, BiCGSTAB, SMG, PFMG");
    return OpenImpala::TortuosityHypre::SolverType::GMRES; // Should not reach here
}


// --- Test Status Helper ---
struct TestStatus {
    bool passed = true;
    std::vector<std::string> fail_reasons;
    bool manual_checks_required = false;

    void recordFail(const std::string& reason) {
        passed = false;
        fail_reasons.push_back(reason);
    }

    void requireManualChecks() {
        manual_checks_required = true;
    }

    void printSummary() {
        if (passed) {
            if(amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n----------------------------------------\n";
                amrex::Print() << "--- TEST RESULT: PASS (Programmatic Checks) ---\n";
                if (manual_checks_required) {
                     amrex::Print() << "*** IMPORTANT: Manual checks required! ***\n";
                     amrex::Print() << "  Please verify solver convergence details and\n";
                     amrex::Print() << "  'Flux conservation check' messages in the log output above.\n";
                }
                amrex::Print() << "----------------------------------------\n";
            }
        } else {
            if(amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n-------------------------\n";
                amrex::Print() << "--- TEST RESULT: FAIL ---\n";
                for (const auto& reason : fail_reasons) {
                    amrex::Print() << "  Reason: " << reason << "\n";
                }
                 amrex::Print() << "-------------------------\n";
            }
            // Ensure abort happens after printing reasons
            amrex::Abort("Tortuosity Test FAILED programmatic checks.");
        }
    }
};


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
        TestStatus test_status; // Use helper struct

        // --- Configuration via ParmParse ---
        // (Keep parameter reading block as before)
        std::string tifffile;
        std::string resultsdir;
        int phase_id = 1;
        std::string direction_str = "X";
        std::string solver_str = "GMRES";
        int box_size = 32;
        int verbose = 1;
        int write_plotfile = 0;
        amrex::Real expected_vf = -1.0;   // Use negative as flag for "not set"
        amrex::Real expected_tau = -1.0;  // Use negative as flag for "not set"
        amrex::Real vf_tolerance = 1e-9;  // Tolerance for VF check
        amrex::Real tau_tolerance = 1e-5; // Default tolerance for Tau check
        // Note: Cannot check flux tolerance programmatically without modifying TortuosityHypre
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
            pp.query("vf_tolerance", vf_tolerance);
            pp.query("tau_tolerance", tau_tolerance);
            // pp.query("flux_tolerance", flux_tolerance); // Cannot use programmatically here
            pp.query("threshold_val", threshold_val);
            pp.query("v_lo", v_lo);
            pp.query("v_hi", v_hi);
        }

        OpenImpala::Direction direction = stringToDirection(direction_str);
        OpenImpala::TortuosityHypre::SolverType solver_type = stringToSolverType(solver_str);

        // Print configuration (Keep as before)
        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Tortuosity Test Configuration ---\n";
            amrex::Print() << "  Input TIFF File:   " << tifffile << "\n";
            amrex::Print() << "  Results Dir:       " << resultsdir << "\n";
            amrex::Print() << "  Phase ID:          " << phase_id << "\n";
            amrex::Print() << "  Direction:         " << direction_str << "\n";
            amrex::Print() << "  Solver:            " << solver_str << "\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Threshold Value:   " << threshold_val << "\n";
            amrex::Print() << "  Boundary Values:   " << v_lo << " (lo), " << v_hi << " (hi)\n";
            amrex::Print() << "  Write Plotfile:    " << (write_plotfile != 0 ? "Yes" : "No") << "\n";
            amrex::Print() << "  Verbose Level:     " << verbose << "\n";
            if (expected_tau >= 0.0) {
                amrex::Print() << "  Expected Tau:      " << expected_tau << "\n";
                amrex::Print() << "  Tau Tolerance:     " << tau_tolerance << "\n";
            }
            if (expected_vf >= 0.0) {
                amrex::Print() << "  Expected VF:       " << expected_vf << "\n";
                amrex::Print() << "  VF Tolerance:      " << vf_tolerance << "\n";
            }
            amrex::Print() << "  Flux Tolerance:    (Manual Check Needed)\n"; // Acknowledge limitation
            amrex::Print() << "------------------------------------\n\n";
        }

        // --- AMReX Grid and Data Setup ---
        // (Keep setup block as before)
        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_with_ghost;

        try {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
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
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << " Thresholding data...\n"; }
            reader.threshold(threshold_val, 1, 0, mf_phase_no_ghost);
            int min_phase_tmp = mf_phase_no_ghost.min(0);
            int max_phase_tmp = mf_phase_no_ghost.max(0);
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << "   Temporary phase field min/max: " << min_phase_tmp << " / " << max_phase_tmp << "\n"; }
            if (min_phase_tmp == max_phase_tmp && ba_original.numPts() > 0) {
                 test_status.recordFail("Phase field uniform after thresholding.");
            }
            const int required_ghost_cells = 1;
            mf_phase_with_ghost.define(ba_original, dm_original, 1, required_ghost_cells);
            amrex::Copy(mf_phase_with_ghost, mf_phase_no_ghost, 0, 0, 1, 0);
            mf_phase_with_ghost.FillBoundary(geom.periodicity());
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";
        } catch (const std::exception& e) {
            test_status.recordFail("Error during TiffReader/grid setup: " + std::string(e.what()));
            test_status.printSummary(); // Print summary and exit
            amrex::Finalize(); HYPRE_Finalize(); return 1;
        }

        // --- Calculate and Check Volume Fraction ---
        amrex::Real actual_vf = 0.0;
        if (test_status.passed) {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Volume Fraction...\n";
            try {
                OpenImpala::VolumeFraction vf_calc(mf_phase_with_ghost, phase_id);
                actual_vf = vf_calc.value(false);
                if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculated Volume Fraction (Phase " << phase_id << "): " << actual_vf << "\n";
                if (expected_vf >= 0.0) {
                     if (std::abs(actual_vf - expected_vf) > vf_tolerance) {
                        test_status.recordFail("Volume fraction mismatch. Expected: " + std::to_string(expected_vf) +
                                               ", Calculated: " + std::to_string(actual_vf) +
                                               ", Tolerance: " + std::to_string(vf_tolerance));
                     } else {
                         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:      PASS\n";
                     }
                } else {
                     if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Volume Fraction Check:      SKIPPED\n";
                }
            } catch (const std::exception& e) {
                test_status.recordFail("Error during Volume Fraction calculation: " + std::string(e.what()));
            }
        }

        // --- Calculate Tortuosity ---
        amrex::Real actual_tau = std::numeric_limits<amrex::Real>::quiet_NaN();
        std::unique_ptr<OpenImpala::TortuosityHypre> tortuosity_ptr;

        // Only proceed if previous steps passed and VF is non-zero
        if (test_status.passed && actual_vf > std::numeric_limits<amrex::Real>::epsilon())
        {
            if (!resultsdir.empty()) { amrex::UtilCreateDirectory(resultsdir, 0755); }
            amrex::ParallelDescriptor::Barrier();

            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Constructing TortuosityHypre object...\n";
            try {
                 tortuosity_ptr = std::make_unique<OpenImpala::TortuosityHypre>(
                     geom, ba_original, dm_original, mf_phase_with_ghost,
                     actual_vf, phase_id, direction, solver_type, resultsdir,
                     v_lo, v_hi, verbose, (write_plotfile != 0)
                 );
            } catch (const std::exception& stdExc) {
                 test_status.recordFail("TortuosityHypre construction failed: " + std::string(stdExc.what()));
                 tortuosity_ptr.reset();
            } catch (...) {
                 test_status.recordFail("Unknown exception during TortuosityHypre construction.");
                 tortuosity_ptr.reset();
            }

            // Check Matrix Properties (already aborts on fail internally via HYPRE_CHECK/amrex::Abort)
            if (test_status.passed && tortuosity_ptr) {
                if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Performing mathematical checks on the assembled matrix...\n";
                // This call might abort internally if it fails, otherwise it prints success/fail
                bool matrix_checks_ok = tortuosity_ptr->checkMatrixProperties();
                if (!matrix_checks_ok) {
                     // The function likely aborted already, but record failure just in case.
                     test_status.recordFail("Assembled matrix/vector failed property checks (check log).");
                } else {
                    if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Matrix property checks:     (Completed by TortuosityHypre)\n";
                }
            }

            // Calculate Tortuosity Value (calls solve and global_fluxes internally)
            if (test_status.passed && tortuosity_ptr) {
                 if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Calculating Tortuosity value...\n";
                 try {
                     actual_tau = tortuosity_ptr->value(); // Call the main method

                     // --- Check if calculation resulted in NaN/Inf ---
                     if (std::isnan(actual_tau) || std::isinf(actual_tau)) {
                         test_status.recordFail("Calculated tortuosity is NaN or Inf! Indicates potential solver or calculation failure.");
                     } else {
                         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Value Calculation Check:    PASS (Result is finite)\n";
                     }

                     // --- Add note about manual checks ---
                     test_status.requireManualChecks();
                     if (amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << " Solver Conv./Flux Check:  REQUIRES MANUAL LOG INSPECTION\n";
                        amrex::Print() << "   (Check 'HYPRE Final Relative Residual Norm' & 'Flux conservation check' messages above)\n";
                     }

                 } catch (const std::exception& stdExc) {
                    test_status.recordFail("Exception during Tortuosity calculation: " + std::string(stdExc.what()));
                 } catch (...) {
                    test_status.recordFail("Unknown exception during Tortuosity calculation.");
                 }
            }

            // --- Check Final Tortuosity Value against Expected ---
            // Only if previous steps passed AND the calculated value is valid
            if (test_status.passed && !(std::isnan(actual_tau) || std::isinf(actual_tau))) {
                 if(amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << " Final Calculated Tortuosity: " << std::fixed << std::setprecision(8) << actual_tau << "\n";
                 }
                 if (expected_tau >= 0.0) {
                     if (amrex::ParallelDescriptor::IOProcessor()) {
                         amrex::Print() << " Expected Tortuosity:       " << std::fixed << std::setprecision(8) << expected_tau << "\n";
                     }
                     if (std::abs(actual_tau - expected_tau) > tau_tolerance) {
                         test_status.recordFail("Tortuosity mismatch. Diff: " +
                                                std::to_string(std::abs(actual_tau - expected_tau)) +
                                                " > Tolerance: " + std::to_string(tau_tolerance));
                     } else {
                         if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Tortuosity Value Check:    PASS\n";
                     }
                 } else {
                      if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Tortuosity Value Check:    SKIPPED\n";
                 }
            } // End check tortuosity value

        } else if (test_status.passed) { // VF is zero, but setup was okay
             if (verbose >= 0 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << " Skipping Tortuosity calculation because Volume Fraction is effectively zero (" << actual_vf << ").\n";
             if (expected_tau >= 0.0) {
                 test_status.recordFail("Volume Fraction is zero, but an expected_tau > 0 was provided.");
             }
        }

        // --- Final Verdict ---
        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
         if(amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "\n Run time = " << stop_time << " sec\n";

        test_status.printSummary(); // Prints PASS/FAIL details and aborts on FAIL

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
