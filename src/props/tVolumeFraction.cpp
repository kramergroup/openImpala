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
#include <stdexcept>    // For std::runtime_error
#include <fstream>      // For std::ifstream check
#include <iomanip>      // For std::setprecision, std::setw, std::setfill
#include <cmath>        // For std::abs
#include <limits>       // For std::numeric_limits
#include <memory>       // For std::unique_ptr

#include <AMReX.H>
#include <AMReX_ParmParse.H>        // For reading parameters
#include <AMReX_Utility.H>          // For amrex::UtilCreateDirectory (optional for output)
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFabUtil.H>   // For amrex::Loop, Array4
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Reduce*
#include <AMReX_ParallelReduce.H>   // For ParallelAllReduce
// #include <AMReX_Reduce.H>           // <<< REMOVED: Not using Reduce::Sum anymore >>>
#include <AMReX_Random.H>         // For potential synthetic data later
#include <AMReX_MFIter.H>         // Include MFIter explicitly
#include <AMReX_GpuQualifiers.H>  // <<< ADDED for AMREX_GPU_DEVICE >>>


// --- Default Test Parameters ---

// Default relative path to the sample TIFF file
// IMPORTANT: Assumes this file exists relative to execution path
const std::string default_tiff_filename = "data/SampleData_2Phase.tif";
// Output directory relative to executable location
const std::string test_output_dir = "tVolumeFraction_output"; // Renamed output dir

// IMPORTANT: These expected values MUST match the properties of the default sample file
// and the effect of the default threshold. Assuming 8-bit data 0-255.
constexpr int default_width = 100;
constexpr int default_height = 100;
constexpr int default_depth = 100;
// Default threshold value (e.g., halfway for UINT8 [0-255])
constexpr double default_threshold_value = 127.5;
constexpr int default_phase0_id = 0; // ID assigned to values <= threshold
constexpr int default_phase1_id = 1; // ID assigned to values > threshold
constexpr int default_comp = 0;      // Component index in iMultiFab for phase data
constexpr amrex::Real default_tolerance = 1e-9; // Tolerance for checking sums/results
constexpr int default_box_size = 32; // Max grid size for BoxArray

namespace // Anonymous namespace for internal helpers
{
    // Helper function to calculate VF directly using AMReX loops and reductions
    amrex::Real calculate_vf_direct(const amrex::iMultiFab& mf, int phase_id, int comp)
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

            // FIX: Replace Reduce::Sum with amrex::Loop
            amrex::Loop(bx, [&] (int i, int j, int k) // Capture locals by reference for OMP reduction
            {
                if (fab(i, j, k) == phase_id) {
                    local_phase_count += 1; // Directly increment thread-local sum (OMP handles reduction)
                }
            });
            // End of replacement

            local_total_count += bx.numPts();
        }

        // Reduce across MPI ranks
        amrex::ParallelAllReduce::Sum(local_phase_count, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());

        return (local_total_count > 0)
               ? static_cast<amrex::Real>(local_phase_count) / local_total_count
               : 0.0;
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
        }

        // Check if input file exists
        {
            std::ifstream test_ifs(tifffile);
            if (!test_ifs) {
                 amrex::Abort("Error: Cannot open input tifffile: " + tifffile);
            }
        }

        if (verbose > 0) {
            amrex::Print() << "\n--- VolumeFraction Test Configuration ---\n";
            amrex::Print() << " TIF File:            " << tifffile << "\n";
            amrex::Print() << " Phase IDs:           " << phase0_id << " (<= T), " << phase1_id << " (> T)\n";
            amrex::Print() << " Phase Component:     " << phase_comp << "\n";
            amrex::Print() << " Threshold Value:     " << threshold_val << "\n";
            amrex::Print() << " Box Size:            " << box_size << "\n";
            amrex::Print() << " Verbose:             " << verbose << "\n";
            amrex::Print() << " Tolerance:           " << tolerance << "\n";
            if (expected_vf0 >= 0.0) amrex::Print() << " Expected VF[" << phase0_id << "]:      " << expected_vf0 << "\n";
            if (expected_vf1 >= 0.0) amrex::Print() << " Expected VF[" << phase1_id << "]:      " << expected_vf1 << "\n";
            amrex::Print() << "--------------------------------------\n\n";
        }

        // Define AMReX objects
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::iMultiFab mf_phase; // Phase data MultiFab

        // --- Read TIFF and Setup Grids/Geometry ---
        try {
            std::unique_ptr<OpenImpala::TiffReader> reader_ptr;
            if (verbose > 0) amrex::Print() << " Reading file " << tifffile << "...\n";
            reader_ptr = std::make_unique<OpenImpala::TiffReader>(tifffile); // Assumes constructor throws on fail

            // Basic check on reader state
             if (!reader_ptr->isRead()) {
                 throw std::runtime_error("TiffReader::isRead() returned false after construction.");
             }
            // TODO: Add checks for expected metadata (BPS, Format, SPP) if desired

            const amrex::Box domain_box = reader_ptr->box();
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
            if (verbose > 0) amrex::Print() << " Thresholding data (Phase " << phase1_id << " if > " << threshold_val << ")...\n";
            reader_ptr->threshold(threshold_val, phase1_id, phase0_id, mf_phase); // Use flexible overload

            // Check threshold result basic validity
            int min_val = mf_phase.min(phase_comp);
            int max_val = mf_phase.max(phase_comp);
            if (min_val != phase0_id || max_val != phase1_id) {
                 amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                                << ") not the expected {" << phase0_id << ", " << phase1_id
                                << "}. Check threshold value or sample data.\n";
            } else {
                 if (verbose > 0) amrex::Print() << "  Threshold output range {" << min_val << ", " << max_val << "} looks plausible.\n";
            }

            // No FillBoundary needed for mf_phase with 0 ghost cells

        } catch (const std::exception& e) {
            amrex::Abort("Error during TiffReader processing or grid setup: " + std::string(e.what()));
        }


        // --- Test VolumeFraction Class ---
        if (verbose > 0) amrex::Print() << " Calculating Volume Fractions using VolumeFraction class...\n";

        // Test Phase 0
        OpenImpala::VolumeFraction vf0(mf_phase, phase0_id, phase_comp); // Pass component
        amrex::Real actual_vf0_global = vf0.value(false); // Global
        amrex::Real actual_vf0_local = vf0.value(true);  // Local

        amrex::Print() << "  VF[" << phase0_id << "] (Global): " << actual_vf0_global << "\n";
        if (verbose > 0) amrex::Print() << "  VF[" << phase0_id << "] (Local Rank 0): " << actual_vf0_local << "\n";

        if (expected_vf0 >= 0.0) { // Check global against expected
            if (std::abs(actual_vf0_global - expected_vf0) > tolerance) {
                 amrex::Abort("FAIL: Global Volume Fraction mismatch for phase " + std::to_string(phase0_id));
            }
             if (verbose > 0) amrex::Print() << "  VF[" << phase0_id << "] Global Check: PASS\n";
        }
        if (actual_vf0_local < 0.0 - tolerance || actual_vf0_local > 1.0 + tolerance) { // Check local range with tolerance
            amrex::Abort("FAIL: Local Volume Fraction for phase " + std::to_string(phase0_id) + " out of range [0, 1].");
        }


        // Test Phase 1
        OpenImpala::VolumeFraction vf1(mf_phase, phase1_id, phase_comp); // Pass component
        amrex::Real actual_vf1_global = vf1.value(false); // Global
        amrex::Real actual_vf1_local = vf1.value(true);  // Local

        amrex::Print() << "  VF[" << phase1_id << "] (Global): " << actual_vf1_global << "\n";
         if (verbose > 0) amrex::Print() << "  VF[" << phase1_id << "] (Local Rank 0): " << actual_vf1_local << "\n";

        if (expected_vf1 >= 0.0) { // Check global against expected
             if (std::abs(actual_vf1_global - expected_vf1) > tolerance) {
                 amrex::Abort("FAIL: Global Volume Fraction mismatch for phase " + std::to_string(phase1_id));
             }
              if (verbose > 0) amrex::Print() << "  VF[" << phase1_id << "] Global Check: PASS\n";
        }
        if (actual_vf1_local < 0.0 - tolerance || actual_vf1_local > 1.0 + tolerance) { // Check local range with tolerance
            amrex::Abort("FAIL: Local Volume Fraction for phase " + std::to_string(phase1_id) + " out of range [0, 1].");
        }

        // --- Check Sums ---
        if (verbose > 0) amrex::Print() << " Checking if volume fractions sum to 1.0...\n";
        amrex::Real vf_sum_global = actual_vf0_global + actual_vf1_global;
        amrex::Real vf_sum_local = actual_vf0_local + actual_vf1_local;

        amrex::Print() << "  Global VF Sum = " << vf_sum_global << "\n";
         if (verbose > 0) amrex::Print() << "  Local VF Sum (Rank " << amrex::ParallelDescriptor::MyProc() << ") = " << vf_sum_local << "\n";

        if (std::abs(1.0 - vf_sum_global) > tolerance) {
            amrex::Warning("Global Volume Fractions do not sum to 1.0 within tolerance.");
        } else {
             if (verbose > 0) amrex::Print() << "  Global Sum Check:    PASS\n";
        }
        // Local sum check requires care in parallel if mf_phase has zero cells on some ranks
        if (mf_phase.boxArray().numPts() > 0 && std::abs(1.0 - vf_sum_local) > tolerance) {
             amrex::Warning("Local Volume Fractions do not sum to 1.0 on Rank " + std::to_string(amrex::ParallelDescriptor::MyProc()) );
        }


        // --- Compare with Direct AMReX Summation ---
        if (verbose > 0) amrex::Print() << " Comparing VolumeFraction class against direct AMReX summation...\n";
        double direct_vf0 = calculate_vf_direct(mf_phase, phase0_id, phase_comp);
        double direct_vf1 = calculate_vf_direct(mf_phase, phase1_id, phase_comp);

        if (std::abs(direct_vf0 - actual_vf0_global) > tolerance) {
             amrex::Abort("FAIL: VolumeFraction::value() result differs from direct AMReX Sum for phase " + std::to_string(phase0_id));
        } else {
             if (verbose > 0) amrex::Print() << "  Direct Sum Check[" << phase0_id << "]: PASS\n";
        }
         if (std::abs(direct_vf1 - actual_vf1_global) > tolerance) {
             amrex::Abort("FAIL: VolumeFraction::value() result differs from direct AMReX Sum for phase " + std::to_string(phase1_id));
         } else {
              if (verbose > 0) amrex::Print() << "  Direct Sum Check[" << phase1_id << "]: PASS\n";
         }


        // --- Optional Synthetic Test Case ---
        /*
        if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Running synthetic test case...\n";
        // Define small box, geom, ba, dm, imf_synth
        // Fill imf_synth: e.g., loop and set lower half to phase0_id, upper half to phase1_id
        // OpenImpala::VolumeFraction vf0_synth(imf_synth, phase0_id, 0);
        // OpenImpala::VolumeFraction vf1_synth(imf_synth, phase1_id, 0);
        // amrex::Real synth_vf0 = vf0_synth.value();
        // amrex::Real synth_vf1 = vf1_synth.value();
        // if (std::abs(synth_vf0 - 0.5) > tolerance || std::abs(synth_vf1 - 0.5) > tolerance) {
        //     amrex::Abort("FAIL: Synthetic test case failed.");
        // } else {
        //     amrex::Print() << "  Synthetic Test:      PASS\n";
        // }
        */


        // --- Final Success ---
        amrex::Print() << "\n tVolumeFraction Test Completed Successfully.\n";

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << " Run time = " << stop_time << " sec\n";

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Indicate success
}
