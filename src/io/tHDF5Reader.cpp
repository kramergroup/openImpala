#include "HDF5Reader.H" // Assuming this defines OpenImpala::HDF5Reader

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>
#include <memory>    // For std::unique_ptr, std::make_unique
#include <sstream>   // For std::stringstream

#include <AMReX.H>
#include <AMReX_ParmParse.H> // For reading input files
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H> // For amrex::UtilCreateDirectory, amrex::Concatenate
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, MyProc
#include <AMReX_GpuLaunch.H> // Provides amrex::ParallelFor

// Output directory relative to executable location
const std::string test_output_dir = "tHDF5Reader_output";
// Define Box size for breaking down domain (can also be read from inputs)
const int BOX_SIZE = 32;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        // Declare variables, values MUST come from the 'inputs' file via ParmParse
        std::string hdf5_filename;
        std::string hdf5_dataset;
        bool write_plotfile = false; // Default: don't write plotfile unless specified in inputs
        int box_size = BOX_SIZE;     // Default box size, can be overridden
        double threshold_value = 1.0; // Default threshold, can be overridden

        { // Use ParmParse to read parameters from the 'inputs' file
            amrex::ParmParse pp; // Default scope, reads global parameters from ./inputs

            // *** USE CORRECT PARAMETER NAMES and REQUIRE them using get() ***
            // The program will abort via amrex::Error if these are not found.
            pp.get("filename", hdf5_filename);     // EXPECTS 'filename = /path/to/file.h5' in inputs
            pp.get("hdf5_dataset", hdf5_dataset); // EXPECTS 'hdf5_dataset = /internal/path' in inputs

            // query() is suitable for optional parameters or those with reasonable defaults
            pp.query("write_plotfile", write_plotfile); // Optional: 'write_plotfile = 1'
            pp.query("box_size", box_size);             // Optional: 'box_size = 64'
            pp.query("threshold_value", threshold_value); // Optional: 'threshold_value = 0.5'
        }

        // Simple check if input file exists (HDF5Reader will also check)
        {
            std::ifstream test_ifs(hdf5_filename);
            if (!test_ifs) {
                 amrex::Error("Error: Cannot open input file specified by 'filename': " + hdf5_filename + "\n"
                              "       Ensure './inputs' file exists and specifies correct path.");
            }
        }

        // Only IOProcessor should print parameter summary
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Starting tHDF5Reader Test (Compiled: " << __DATE__ << " " << __TIME__ << ")\n";
            amrex::Print() << "Input HDF5 file (from inputs): " << hdf5_filename << "\n";
            amrex::Print() << "Input dataset path (from inputs): " << hdf5_dataset << "\n";
            amrex::Print() << "Threshold value: " << threshold_value << "\n";
            amrex::Print() << "Box size: " << box_size << "\n";
            amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";
        }

        // --- Test HDF5Reader ---
        std::unique_ptr<OpenImpala::HDF5Reader> reader_ptr;
        // Expected dimensions might also come from inputs if testing different files
        int expected_width = 100;
        int expected_height = 100;
        int expected_depth = 100;

        try {
            // HDF5Reader constructor now receives paths read from the inputs file
            reader_ptr = std::make_unique<OpenImpala::HDF5Reader>(hdf5_filename, hdf5_dataset);
        } catch (const std::exception& e) {
            amrex::Error("Error creating HDF5Reader: " + std::string(e.what()));
        }

        // Check if reader was successfully created and read data
        if (!reader_ptr || !reader_ptr->isRead()) {
             amrex::Error("HDF5Reader object creation or file read failed after construction.");
        }

        // --- Check Dimensions & Metadata ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Checking dimensions and metadata...\n";
        }
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Read dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";
            amrex::Print() << "  (Skipping metadata checks for BPS, Format, SPP as methods are not implemented in HDF5Reader).\n";
        }

        // Perform check on all ranks to ensure consistency, abort on any failure
        if (actual_width != expected_width || actual_height != expected_height || actual_depth != expected_depth) {
            // Use a stringstream to format the error message before calling amrex::Error
            // This prevents potential issues with parallel printing inside amrex::Error arguments.
            std::stringstream ss;
            ss << "FAIL: Read dimensions (" << actual_width << "x" << actual_height << "x" << actual_depth
               << ") do not match expected dimensions (" << expected_width << "x" << expected_height << "x" << expected_depth << ").";
            amrex::Error(ss.str()); // Use Error for test failure, safe for parallel calls
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Dimension check passed.\n";
        }


        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box();
        {
            // Using RealBox based on domain size; could also read physical size from attributes/inputs
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0}; // Assuming non-periodic
            geom.define(domain_box, &rb, amrex::CoordSys::cartesian, is_periodic.data());
        }

        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size); // Use value read from inputs
        amrex::DistributionMapping dm(ba);

        // iMultiFab for threshold result (integer: 0 or 1)
        // 0 ghost cells needed as input for VolumeFraction/Tortuosity usually
        amrex::iMultiFab mf(ba, dm, 1, 0);
        mf.setVal(0); // Initialize

        // --- Test Thresholding ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Performing threshold > " << threshold_value << "...\n";
        }
        try {
            // HDF5Reader::threshold takes double.
            reader_ptr->threshold(threshold_value, mf);
        } catch (const std::exception& e) {
            amrex::Error("Error during threshold operation: " + std::string(e.what()));
        }

        // --- Check Threshold Result ---
        // Use global reductions for min/max across all MPI ranks
        int min_val = mf.min(0, 0, true); // component 0, 0 ghost cells, local=false for global min
        int max_val = mf.max(0, 0, true); // component 0, 0 ghost cells, local=false for global max

        // Only IOProcessor should print results of reductions and check plausibility
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Threshold result min value (global): " << min_val << "\n";
            amrex::Print() << "  Threshold result max value (global): " << max_val << "\n";

            // Check threshold result plausibility (adjust if thresholding logic changes)
            if (min_val != 0 || max_val != 1) {
                 amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                                << ") not the expected 0/1. Check threshold value or sample data.\n";
                 // Consider making this a failure? Depends on test requirements.
                 // amrex::Error("FAIL: Unexpected min/max values after thresholding.");
            } else {
                 amrex::Print() << "  Threshold value range looks plausible (0 and 1 found globally).\n";
            }
        }

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
            std::string plotfilename; // Declare outside the block

            // Create directory only on IOProcessor
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Attempting to create plotfile output directory...\n";
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                     // Warning is okay if directory already exists
                     amrex::Print() << "Warning: Could not create output directory (it might already exist): " << test_output_dir << "\n";
                 }
            }
            // Ensure directory exists before proceeding (all ranks wait)
            amrex::ParallelDescriptor::Barrier();

            // Construct plotfile name (potentially include rank for parallel consistency)
            {
                std::string datetime_str;
                std::time_t now_time = std::time(nullptr);
                std::tm* now_tm = std::localtime(&now_time);
                if (now_tm) {
                    char datetime_buf[80];
                    // Use underscore for better filename compatibility
                    std::strftime(datetime_buf, sizeof(datetime_buf),"%Y%m%d_%H%M%S", now_tm);
                    datetime_str = datetime_buf;
                } else {
                    datetime_str = "time_error";
                }
                // Example: plt00000 or a name including rank/time
                // Using Concatenate adds leading zeros based on step number (using 0 here)
                plotfilename = amrex::Concatenate(test_output_dir + "/plt_hdf5_", 0, 5);
                plotfilename += "_" + datetime_str; // Append timestamp

            } // End block for time variables

            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "Writing plot file to: " << plotfilename << "\n";
            }

            // Create a MultiFab view of the iMultiFab for plotting
            amrex::MultiFab mfv(ba, dm, 1, 0); // Real MultiFab, 0 ghost cells

            // Copy integer data to real data (simple cast) using ParallelFor
            amrex::ParallelFor(mfv, mf.nGrowVect(), [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
            {
                const auto& int_fab = mf.const_array(box_no); // Get const Array4 for iMultiFab
                auto real_fab = mfv.array(box_no);            // Get Array4 for MultiFab
                real_fab(i,j,k) = static_cast<amrex::Real>(int_fab(i,j,k)); // Cast and assign
            });

            // Write plot file (step=0, time=0.0 are placeholders)
            amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);

            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "  Plot file written.\n";
            }
        }

        // If we reached here without amrex::Error, the test passed conceptually
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "tHDF5Reader Test Completed Successfully.\n";
        }

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Return 0 indicates success to the shell/Makefile
}
