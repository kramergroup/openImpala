// Test driver for the OpenImpala::VolumeFraction class.
// Reads phase data (typically from a TIFF file), calculates the volume
// fraction of specified phases using the VolumeFraction class, compares
// the result against expected values (if provided) and/or a direct
// summation using AMReX tools, and reports PASS/FAIL.
// Configuration is handled via ParmParse inputs.

#include "../io/TiffReader.H"  // Assuming defines OpenImpala::TiffReader
#include "VolumeFraction.H"    // Assuming defines OpenImpala::VolumeFraction

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>      // For std::runtime_error
#include <fstream>        // For std::ifstream check
#include <iomanip>        // For std::setprecision, std::setw, std::setfill
#include <cmath>          // For std::abs
#include <limits>         // For std::numeric_limits
#include <memory>         // For std::unique_ptr
#include <iostream>       // <<< ADDED for std::flush >>>

#include <AMReX.H>
#include <AMReX_ParmParse.H>         // For reading parameters
#include <AMReX_Utility.H>           // For amrex::UtilCreateDirectory (optional for output)
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFabUtil.H>    // For amrex::Loop, Array4
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Reduce*
#include <AMReX_ParallelReduce.H>    // For ParallelAllReduce
// #include <AMReX_Reduce.H>            // <<< REMOVED: Not using Reduce::Sum anymore >>>
#include <AMReX_Random.H>            // For potential synthetic data later
#include <AMReX_MFIter.H>            // Include MFIter explicitly
#include <AMReX_GpuQualifiers.H>   // <<< ADDED for AMREX_GPU_DEVICE >>>


// --- Default Test Parameters ---

// Default relative path to the sample TIFF file
// IMPORTANT: Assumes this file exists relative to execution path
const std::string default_tiff_filename = "data/SampleData_2Phase_stack_3d_1bit.tif";
// Output directory relative to executable location
const std::string test_output_dir = "tVolumeFraction_output"; // Renamed output dir

// IMPORTANT: These expected values MUST match the properties of the default sample file
// and the effect of the default threshold. Assuming 8-bit data 0-255.
constexpr int default_width = 100;
constexpr int default_height = 100;
constexpr int default_depth = 100;
// Default threshold value (e.g., halfway for UINT8 [0-255])
constexpr double default_threshold_value = 0.5;
constexpr int default_phase0_id = 0; // ID assigned to values <= threshold
constexpr int default_phase1_id = 1; // ID assigned to values > threshold
constexpr int default_comp = 0;      // Component index in iMultiFab for phase data
constexpr amrex::Real default_tolerance = 1e-9; // Tolerance for checking sums/results
constexpr int default_box_size = 32; // Max grid size for BoxArray

namespace // Anonymous namespace for internal helpers
{
    // Helper function to calculate VF directly using AMReX loops and reductions
    // --- MODIFIED to return counts ---
    void calculate_counts_direct(const amrex::iMultiFab& mf, int phase_id, int comp,
                                 long long& phase_count, long long& total_count)
    {
        long long local_phase_count = 0;
        long long local_total_count = 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:local_phase_count, local_total_count)
#endif
        // Use tiling iterator (true argument)
        for (amrex::MFIter mfi(mf, true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            const auto& fab = mf.const_array(mfi, comp); // Use specified component

            amrex::Loop(bx, [&] (int i, int j, int k) // Capture locals by reference for OMP reduction
            {
                if (fab(i, j, k) == phase_id) {
                    local_phase_count += 1; // Directly increment thread-local sum (OMP handles reduction)
                }
            });

            local_total_count += bx.numPts();
        }

        // Reduce across MPI ranks
        amrex::ParallelAllReduce::Sum(local_phase_count, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());

        // Assign to output parameters
        phase_count = local_phase_count;
        total_count = local_total_count;
    }
} // namespace


int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        amrex::Real strt_time = amrex::second();

        // --- Configuration via ParmParse ---
        std::string tifffile = default_tiff_filename;
        int phase0_id = default_phase0_id;
        int phase1_id = default_phase1_id;
        int phase_comp = default_comp; // Component to store/analyze phase data
        double threshold_val = default_threshold_value;
        int box_size = default_box_size;
        int verbose = 1;
        amrex::Real expected_vf0 = -1.0; // Use negative to indicate not checking
        amrex::Real expected_vf1 = -1.0;
        amrex::Real tolerance = default_tolerance;
        bool check_boundary_voxels = true; // Add flag to enable/disable boundary check

        {
            amrex::ParmParse pp;
            pp.query("tifffile", tifffile);
            pp.query("phase0_id", phase0_id);
            pp.query("phase1_id", phase1_id);
            pp.query("comp", phase_comp);
            pp.query("threshold", threshold_val);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("expected_vf0", expected_vf0);
            pp.query("expected_vf1", expected_vf1);
            pp.query("tolerance", tolerance);
            pp.query("check_boundary_voxels", check_boundary_voxels); // Query the new flag
        }

        amrex::SetVerbose(verbose); // <<<====== ADD THIS LINE

        // Check if input file exists
        {
            std::ifstream test_ifs(tifffile);
            if (!test_ifs) {
                 amrex::Abort("Error: Cannot open input tifffile: " + tifffile);
            }
        }

        if (verbose > 0) {
            amrex::Print() << "\n--- VolumeFraction Test Configuration ---\n";
            amrex::Print() << " TIF File:              " << tifffile << "\n";
            amrex::Print() << " Phase IDs:             " << phase0_id << " (<= T), " << phase1_id << " (> T)\n";
            amrex::Print() << " Phase Component:       " << phase_comp << "\n";
            amrex::Print() << " Threshold Value:       " << threshold_val << "\n";
            amrex::Print() << " Box Size:              " << box_size << "\n";
            amrex::Print() << " Verbose:               " << verbose << "\n";
            amrex::Print() << " Tolerance:             " << tolerance << "\n";
            amrex::Print() << " Check Boundary Voxels: " << (check_boundary_voxels ? "Yes" : "No") << "\n";
            if (expected_vf0 >= 0.0) amrex::Print() << " Expected VF[" << phase0_id << "]:        " << expected_vf0 << "\n";
            if (expected_vf1 >= 0.0) amrex::Print() << " Expected VF[" << phase1_id << "]:        " << expected_vf1 << "\n";
            amrex::Print() << "--------------------------------------\n\n";
            amrex::OutStream() << std::flush; // Flush config output
        }

        // Define AMReX objects
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::iMultiFab mf_phase; // Phase data MultiFab
        amrex::Box domain_box; // Store domain box for later use

        // --- Read TIFF and Setup Grids/Geometry ---
        try {
            std::unique_ptr<OpenImpala::TiffReader> reader_ptr;
            if (verbose > 0) {
                amrex::Print() << " Reading file " << tifffile << "...\n";
                amrex::OutStream() << std::flush;
            }
            reader_ptr = std::make_unique<OpenImpala::TiffReader>(tifffile); // Assumes constructor throws on fail

            // Basic check on reader state
             if (!reader_ptr->isRead()) {
                 throw std::runtime_error("TiffReader::isRead() returned false after construction.");
             }
            // *** ADDED: Print FillOrder ***
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " TiffReader FillOrder used: " << reader_ptr->getFillOrder() << " (1=MSB2LSB, 2=LSB2MSB)\n";
                 amrex::OutStream() << std::flush;
            }
            // *** END ADDED ***


            domain_box = reader_ptr->box(); // Get domain box from reader
            if (domain_box.isEmpty()) { amrex::Abort("FAIL: TiffReader returned an empty box."); }

            { // Setup geometry
                 amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                   {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                                 amrex::Real(domain_box.length(1)),
                                                 amrex::Real(domain_box.length(2)))});
                 amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0};
                 amrex::Geometry::Setup(&rb, 0, is_periodic.data());
                 geom.define(domain_box);
            }

            ba.define(domain_box);
            ba.maxSize(box_size);
            dm.define(ba);

            // Define the phase MultiFab with 1 component, 0 ghost cells
            mf_phase.define(ba, dm, 1, 0);
            mf_phase.setVal(-1); // Initialize to dummy value

            // Threshold image data into mf_phase using phase1_id and phase0_id
            if (verbose > 0) {
                amrex::Print() << " Thresholding data (Phase " << phase1_id << " if > " << threshold_val << ")...\n";
                amrex::OutStream() << std::flush;
            }
            reader_ptr->threshold(threshold_val, phase1_id, phase0_id, mf_phase); // Use flexible overload

            // Check threshold result basic validity
            int min_val = mf_phase.min(phase_comp);
            int max_val = mf_phase.max(phase_comp);
            if (min_val != phase0_id || max_val != phase1_id) {
                 amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                                << ") not the expected {" << phase0_id << ", " << phase1_id
                                << "}. Check threshold value or sample data.\n";
                 amrex::OutStream() << std::flush;
            } else {
                 if (verbose > 0) {
                     amrex::Print() << "  Threshold output range {" << min_val << ", " << max_val << "} looks plausible.\n";
                     amrex::OutStream() << std::flush;
                 }
            }

            // No FillBoundary needed for mf_phase with 0 ghost cells

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader processing or grid setup: " + std::string(e.what()));
        }

        // --- *** MODIFIED: Check Boundary Voxel Values with Debug Prints *** ---
        if (amrex::ParallelDescriptor::IOProcessor()) { // Print debug status only on IO rank
            amrex::Print() << "DEBUG: Before boundary check block. check_boundary_voxels=" << check_boundary_voxels
                           << ", IOProcessor=" << amrex::ParallelDescriptor::IOProcessor() << "\n";
            amrex::OutStream() << std::flush;
        }

        if (check_boundary_voxels && amrex::ParallelDescriptor::IOProcessor()) // Run full check only on Rank 0
        {
            amrex::Print() << "DEBUG: ENTERING boundary check block.\n";
            amrex::OutStream() << std::flush;

            amrex::Print() << " Checking values of boundary voxels (corners):\n";
            amrex::OutStream() << std::flush; // Force flush

            const amrex::IntVect dom_lo = domain_box.smallEnd();
            const amrex::IntVect dom_hi = domain_box.bigEnd();
            amrex::Vector<amrex::IntVect> corners = {
                dom_lo,                                               // (0,0,0)
                amrex::IntVect(dom_hi[0], dom_lo[1], dom_lo[2]),      // (Xmax, 0, 0)
                amrex::IntVect(dom_lo[0], dom_hi[1], dom_lo[2]),      // (0, Ymax, 0)
                amrex::IntVect(dom_lo[0], dom_lo[1], dom_hi[2]),      // (0, 0, Zmax)
                amrex::IntVect(dom_hi[0], dom_hi[1], dom_lo[2]),      // (Xmax, Ymax, 0)
                amrex::IntVect(dom_hi[0], dom_lo[1], dom_hi[2]),      // (Xmax, 0, Zmax)
                amrex::IntVect(dom_lo[0], dom_hi[1], dom_hi[2]),      // (0, Ymax, Zmax)
                dom_hi                                                // (Xmax, Ymax, Zmax)
            };

            // Need to iterate MFIter to access data, but only check points if they are corners
            // This is inefficient but simple for a debug check.
            for (amrex::MFIter mfi(mf_phase); mfi.isValid(); ++mfi) {
                 const amrex::Box& bx = mfi.validbox(); // Use validbox (no ghost cells)
                 const auto& fab_arr = mf_phase.const_array(mfi, phase_comp);

                 for(const auto& corner : corners) {
                      if (bx.contains(corner)) { // Check if this Fab owns the corner point
                           amrex::Print() << "  Corner " << corner << " value: " << fab_arr(corner) << "\n";
                           amrex::OutStream() << std::flush; // Force flush
                      }
                 }
            }
            amrex::Print() << "------------------------------------------\n";
            amrex::OutStream() << std::flush; // Force flush
        } else if (amrex::ParallelDescriptor::IOProcessor()) { // Print skip message only on IO rank
             amrex::Print() << "DEBUG: SKIPPING boundary check block (check_boundary_voxels="
                            << check_boundary_voxels << " or not IOProcessor).\n";
             amrex::OutStream() << std::flush;
        }

        // Ensure all ranks wait if check was done (or skipped) before proceeding
        amrex::ParallelDescriptor::Barrier("BoundaryCheck");

        if (amrex::ParallelDescriptor::IOProcessor()) { // Print debug status only on IO rank
            amrex::Print() << "DEBUG: After boundary check block structure.\n";
            amrex::OutStream() << std::flush;
        }
        // --- *** END MODIFIED BOUNDARY CHECK *** ---


        // --- Test VolumeFraction Class ---
        if (verbose > 0) {
            amrex::Print() << " Calculating Volume Fractions using VolumeFraction class...\n";
            amrex::OutStream() << std::flush;
        }

        // Declare variables to hold counts
        long long phase0_count_global = 0, total_count0_global = 0;
        long long phase0_count_local = 0, total_count0_local = 0;

        // Test Phase 0
        OpenImpala::VolumeFraction vf0(mf_phase, phase0_id, phase_comp); // Pass component
        vf0.value(phase0_count_global, total_count0_global, false); // Get global counts
        vf0.value(phase0_count_local, total_count0_local, true);   // Get local counts

        // Calculate VF from counts
        amrex::Real actual_vf0_global = (total_count0_global > 0)
                                      ? static_cast<amrex::Real>(phase0_count_global) / total_count0_global
                                      : 0.0;
        amrex::Real actual_vf0_local = (total_count0_local > 0)
                                     ? static_cast<amrex::Real>(phase0_count_local) / total_count0_local
                                     : 0.0;

        amrex::Print() << "  VF[" << phase0_id << "] (Global): " << actual_vf0_global << "\n";
        // *** ADDED: Print global counts ***
        amrex::Print() << "   Phase[" << phase0_id << "] Count (Global): " << phase0_count_global << "\n";
        amrex::Print() << "   Total Count (Global):      " << total_count0_global << "\n";
        amrex::OutStream() << std::flush; // Flush
        // *** END ADDED ***
        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { // Print local only on Rank 0 for clarity
            amrex::Print() << "  VF[" << phase0_id << "] (Local Rank 0): " << actual_vf0_local << "\n";
            amrex::OutStream() << std::flush;
        }


        if (expected_vf0 >= 0.0) { // Check global against expected
            if (std::abs(actual_vf0_global - expected_vf0) > tolerance) {
                 amrex::Abort("FAIL: Global Volume Fraction mismatch for phase " + std::to_string(phase0_id));
            }
             if (verbose > 0) {
                 amrex::Print() << "  VF[" << phase0_id << "] Global Check: PASS\n";
                 amrex::OutStream() << std::flush;
             }
        }
        if (actual_vf0_local < 0.0 - tolerance || actual_vf0_local > 1.0 + tolerance) { // Check local range with tolerance
            // This check might fail spuriously on ranks with no cells, do it carefully
            if (total_count0_local > 0) { // Only check if this rank actually has cells
                amrex::Abort("FAIL: Local Volume Fraction for phase " + std::to_string(phase0_id) + " out of range [0, 1] on Rank " + std::to_string(amrex::ParallelDescriptor::MyProc()) + ".");
            }
        }


        // Declare variables for phase 1 counts
        long long phase1_count_global = 0, total_count1_global = 0;
        long long phase1_count_local = 0, total_count1_local = 0;

        // Test Phase 1
        OpenImpala::VolumeFraction vf1(mf_phase, phase1_id, phase_comp); // Pass component
        vf1.value(phase1_count_global, total_count1_global, false); // Global
        vf1.value(phase1_count_local, total_count1_local, true);   // Local

        // Calculate VF
        amrex::Real actual_vf1_global = (total_count1_global > 0)
                                      ? static_cast<amrex::Real>(phase1_count_global) / total_count1_global
                                      : 0.0;
        amrex::Real actual_vf1_local = (total_count1_local > 0)
                                     ? static_cast<amrex::Real>(phase1_count_local) / total_count1_local
                                     : 0.0;

        amrex::Print() << "  VF[" << phase1_id << "] (Global): " << actual_vf1_global << "\n";
        // *** ADDED: Print global counts ***
        amrex::Print() << "   Phase[" << phase1_id << "] Count (Global): " << phase1_count_global << "\n";
        amrex::Print() << "   Total Count (Global):      " << total_count1_global << " (Should match total above)\n";
        amrex::OutStream() << std::flush; // Flush
        // *** END ADDED ***

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { // Print local only on Rank 0
            amrex::Print() << "  VF[" << phase1_id << "] (Local Rank 0): " << actual_vf1_local << "\n";
            amrex::OutStream() << std::flush;
        }


        if (expected_vf1 >= 0.0) { // Check global against expected
             if (std::abs(actual_vf1_global - expected_vf1) > tolerance) {
                 amrex::Abort("FAIL: Global Volume Fraction mismatch for phase " + std::to_string(phase1_id));
             }
              if (verbose > 0) {
                  amrex::Print() << "  VF[" << phase1_id << "] Global Check: PASS\n";
                  amrex::OutStream() << std::flush;
              }
        }
         if (actual_vf1_local < 0.0 - tolerance || actual_vf1_local > 1.0 + tolerance) { // Check local range with tolerance
             if(total_count1_local > 0) { // Only check if this rank actually has cells
                amrex::Abort("FAIL: Local Volume Fraction for phase " + std::to_string(phase1_id) + " out of range [0, 1] on Rank " + std::to_string(amrex::ParallelDescriptor::MyProc()) + ".");
             }
        }

        // --- Check Sums ---
        if (verbose > 0) {
            amrex::Print() << " Checking if volume fractions sum to 1.0...\n";
            amrex::OutStream() << std::flush;
        }
        amrex::Real vf_sum_global = actual_vf0_global + actual_vf1_global;
        amrex::Real vf_sum_local = actual_vf0_local + actual_vf1_local; // Note: This is Rank 0's local sum

        amrex::Print() << "  Global VF Sum = " << vf_sum_global << "\n";
        amrex::OutStream() << std::flush;

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) { // Print local sum only on Rank 0
             amrex::Print() << "  Local VF Sum (Rank 0) = " << vf_sum_local << "\n";
             amrex::OutStream() << std::flush;
        }


        if (std::abs(1.0 - vf_sum_global) > tolerance) {
            amrex::Warning("Global Volume Fractions do not sum to 1.0 within tolerance.");
        } else {
             if (verbose > 0) {
                 amrex::Print() << "  Global Sum Check:       PASS\n";
                 amrex::OutStream() << std::flush;
             }
        }
        // Local sum check requires care in parallel if mf_phase has zero cells on some ranks
        if (mf_phase.boxArray().numPts() > 0 && std::abs(1.0 - vf_sum_local) > tolerance && amrex::ParallelDescriptor::IOProcessor()) {
             // Only warn if Rank 0 (which we printed) doesn't sum, and it has points.
             amrex::Warning("Local Volume Fractions do not sum to 1.0 on Rank 0.");
        }


        // --- Compare with Direct AMReX Summation ---
        if (verbose > 0) {
            amrex::Print() << " Comparing VolumeFraction class against direct AMReX summation...\n";
            amrex::OutStream() << std::flush;
        }
        long long direct_phase0_count=0, direct_total0_count=0;
        long long direct_phase1_count=0, direct_total1_count=0;

        calculate_counts_direct(mf_phase, phase0_id, phase_comp, direct_phase0_count, direct_total0_count);
        calculate_counts_direct(mf_phase, phase1_id, phase_comp, direct_phase1_count, direct_total1_count);

        // Compare counts directly
        if (direct_phase0_count != phase0_count_global || direct_total0_count != total_count0_global) {
            amrex::Abort("FAIL: VolumeFraction::value() counts differ from direct AMReX Sum counts for phase " + std::to_string(phase0_id));
        } else {
             if (verbose > 0) {
                 amrex::Print() << "  Direct Sum Check[" << phase0_id << "]: PASS\n";
                 amrex::OutStream() << std::flush;
             }
        }
         if (direct_phase1_count != phase1_count_global || direct_total1_count != total_count1_global) {
              amrex::Abort("FAIL: VolumeFraction::value() counts differ from direct AMReX Sum counts for phase " + std::to_string(phase1_id));
         } else {
               if (verbose > 0) {
                   amrex::Print() << "  Direct Sum Check[" << phase1_id << "]: PASS\n";
                   amrex::OutStream() << std::flush;
               }
         }


        // --- Optional Synthetic Test Case ---
        /* ... remains the same ... */


        // --- Final Success ---
        amrex::Print() << "\n tVolumeFraction Test Completed Successfully.\n";
        amrex::OutStream() << std::flush;

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << " Run time = " << stop_time << " sec\n";
            amrex::OutStream() << std::flush;
        }

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Indicate success
}
