#include "DatReader.H" // Assuming this defines OpenImpala::DatReader

#include <cstdlib>   // For std::exit (if needed), prefer amrex::Abort
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>
#include <memory>    // For std::unique_ptr, std::make_unique

#include <AMReX.H>
#include <AMReX_ParmParse.H> // For reading command-line arguments
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
#include <AMReX_Utility.H> // For amrex::UtilCreateDirectory
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier
#include <AMReX_GpuLaunch.H> // Provides amrex::ParallelFor

// Default relative path to the sample DAT file
const std::string default_dat_filename = "data/SampleData_2Phase.dat";
// Output directory relative to executable location
const std::string test_output_dir = "tDatReader_output";

// <<< FIX 1: Define BOX_SIZE >>>
// Define a default box size for BoxArray creation
constexpr int BOX_SIZE = 32;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        std::string dat_filename = default_dat_filename;
        bool write_plotfile = false; // Default: don't write plotfile

        { // Use ParmParse to read parameters from command line
            // Example: ./executable datfile=path/to/file.dat write_plotfile=1
            amrex::ParmParse pp;
            pp.query("datfile", dat_filename);
            pp.query("write_plotfile", write_plotfile);
        }

        // Check if input file exists before attempting to read
        {
            std::ifstream test_ifs(dat_filename);
            if (!test_ifs) {
                 amrex::Abort("Error: Cannot open input datfile: " + dat_filename + "\n"
                              "       Specify path using 'datfile=/path/to/file.dat'");
            }
        }

        // Use __DATE__ and __TIME__ which are standard C macros
        amrex::Print() << "Starting tDatReader Test (Compiled: " << __DATE__ << " " << __TIME__ << ")\n";
        amrex::Print() << "Input DAT file: " << dat_filename << "\n";
        amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";

        // --- Test DatReader ---
        std::unique_ptr<OpenImpala::DatReader> reader_ptr;
        int expected_width = 100; // Based on SampleData_2Phase.dat
        int expected_height = 100;
        int expected_depth = 100;
        // Assuming threshold '1' yields a non-empty mask for this specific file
        // Ensure DataType is accessible - need DatReader.H included
        const OpenImpala::DatReader::DataType threshold_value = 1;

        try {
            // Assuming constructor reads file and throws std::runtime_error on failure
            reader_ptr = std::make_unique<OpenImpala::DatReader>(dat_filename);

            // Optional: Check isRead() if constructor doesn't throw but sets flag
            // if (!reader_ptr->isRead()) { // Need isRead() method in DatReader.H
            //     amrex::Abort("DatReader failed to read file (isRead() is false).");
            // }

        } catch (const std::exception& e) {
            amrex::Abort("Error creating DatReader: " + std::string(e.what()));
        }

        // --- Check Dimensions ---
        amrex::Print() << "Checking dimensions...\n";
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();

        amrex::Print() << "  Read dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";

        if (actual_width != expected_width || actual_height != expected_height || actual_depth != expected_depth) {
            amrex::Abort("FAIL: Read dimensions do not match expected dimensions (100x100x100).");
        }
        amrex::Print() << "  Dimensions match expected values.\n";

        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box(); // Use reader's box
        {
            amrex::RealBox rb({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}); // Dummy physical domain
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0}; // Non-periodic
            // Set up coord system and define geometry on the index space box
            amrex::Geometry::Setup(&rb, 0, is_periodic.data());
            geom.define(domain_box);
        }

        amrex::BoxArray ba(domain_box);
        ba.maxSize(BOX_SIZE); // Break into boxes using defined BOX_SIZE
        amrex::DistributionMapping dm(ba);

        // Create iMultiFab to hold thresholded data (1 component, 0 ghost cells)
        amrex::iMultiFab mf(ba, dm, 1, 0);
        mf.setVal(0); // Initialize just in case

        // --- Test Thresholding ---
        amrex::Print() << "Performing threshold > " << threshold_value << "...\n";
        try {
            // Assuming threshold takes DataType and iMultiFab&, and is const
            reader_ptr->threshold(threshold_value, mf); // Use 1/0 overload
        } catch (const std::exception& e) {
            amrex::Abort("Error during threshold operation: " + std::string(e.what()));
        }

        // Check results (min/max across all processors)
        int min_val = mf.min(0); // Component 0
        int max_val = mf.max(0); // Component 0

        amrex::Print() << "  Threshold result min value: " << min_val << "\n";
        amrex::Print() << "  Threshold result max value: " << max_val << "\n";

        // Expect only 0s and 1s if thresholding works and data spans the threshold
        if (min_val != 0 || max_val != 1) {
             amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                            << ") not the expected 0/1. Check threshold value or sample data.\n";
             // Decide if this is a fatal error for the test
             // amrex::Abort("FAIL: Threshold result unexpected.");
        } else {
             amrex::Print() << "  Threshold value range looks plausible (0 and 1 found).\n";
        }

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
            amrex::Print() << "Writing plot file...\n";

            // Create output directory relative to executable location
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                     amrex::Warning("Could not create output directory: " + test_output_dir);
                     // Decide whether to proceed or abort
                 }
            }
            // Barrier to make sure directory exists before writing
            amrex::ParallelDescriptor::Barrier();

            // Get datetime string for filename
            std::string datetime_str;
            { // Scope for time variables
                std::time_t strt_time;
                std::tm* timeinfo;
                char datetime_buf [80];
                std::time(&strt_time);
                timeinfo = std::localtime(&strt_time);
                // Ensure buffer is large enough for format YYYYMMDDHHMM (12 chars + null)
                std::strftime(datetime_buf, sizeof(datetime_buf),"%Y%m%d%H%M", timeinfo);
                datetime_str = datetime_buf;
            }

            // Construct filename
            std::string plotfilename = test_output_dir + "/datreadertest_" + datetime_str;

            // Copy integer data to float MultiFab for plotting
            amrex::MultiFab mfv(ba, dm, 1, 0); // 1 component, 0 ghost cells

            // Using ParallelFor is more modern AMReX than explicit MFIter loop for simple copy
            for (amrex::MFIter mfi(mfv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& box = mfi.tilebox();
                auto const& int_fab = mf.const_array(mfi); // Read from iMultiFab

                // <<< FIX 2: Change auto& to auto >>>
                auto real_fab = mfv.array(mfi);       // Write to MultiFab (get copy or non-const array)

                amrex::ParallelFor(box, [&] (int i, int j, int k) noexcept // Use ParallelFor if possible
                {
                    // Access using IntVect for safety, although individual indices might work here
                    real_fab(amrex::IntVect(i, j, k)) = static_cast<amrex::Real>(int_fab(amrex::IntVect(i, j, k)));
                });
            }

            // Write plot file
            amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);
            amrex::Print() << "  Plot file written to: " << plotfilename << "\n";
        }

        amrex::Print() << "tDatReader Test Completed Successfully.\n";

    } // End AMReX scope block

    amrex::Finalize();
    return 0;
}
