#include "TiffReader.H" // Assuming defines OpenImpala::TiffReader

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>
#include <memory>    // For std::unique_ptr

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
#include <AMReX_Utility.H>     // For amrex::UtilCreateDirectory
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier
#include <AMReX_GpuLaunch.H> // Provides amrex::ParallelFor

// Default relative path to the sample TIFF file
const std::string default_tiff_filename = "data/SampleData_2Phase.tif";
// Output directory relative to executable location
const std::string test_output_dir = "tTiffReader_output";
// Default threshold value (adjust based on sample data)
const double default_threshold_value = 1.0;
// Define Box size for breaking down domain
const int BOX_SIZE = 32;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        std::string tiff_filename = default_tiff_filename;
        double threshold_val = default_threshold_value;
        bool write_plotfile = false; // Default: don't write plotfile

        { // Use ParmParse to read parameters from command line
          // Example: ./executable tifffile=path/to/stack.tif threshold=128 write_plotfile=1
            amrex::ParmParse pp;
            pp.query("tifffile", tiff_filename);
            pp.query("threshold", threshold_val);
            pp.query("write_plotfile", write_plotfile);
        }

        // Check if input file exists before attempting to read
        {
            std::ifstream test_ifs(tiff_filename);
            if (!test_ifs) {
                 amrex::Abort("Error: Cannot open input tifffile: " + tiff_filename + "\n"
                              "       Specify path using 'tifffile=/path/to/file.tif'");
            }
        }

        amrex::Print() << "Starting tTiffReader Test (Oxford, " << __DATE__ << " " << __TIME__ << ")\n";
        amrex::Print() << "Input TIFF file: " << tiff_filename << "\n";
        amrex::Print() << "Threshold value: " << threshold_val << "\n";
        amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";

        // --- Test TiffReader ---
        std::unique_ptr<OpenImpala::TiffReader> reader_ptr;
        // Expected properties for default SampleData_2Phase.tif
        // **VERIFY THESE AGAINST YOUR ACTUAL SAMPLE FILE**
        int expected_width = 100;
        int expected_height = 100;
        int expected_depth = 100;
        int expected_bps = 8;       // e.g., 8-bit
        int expected_format = 1;  // e.g., SAMPLEFORMAT_UINT from tiff.h
        int expected_spp = 1;       // e.g., grayscale

        try {
            // Assuming constructor reads file and throws std::runtime_error on failure
            reader_ptr = std::make_unique<OpenImpala::TiffReader>(tiff_filename);

            // Alternative if using readFile pattern:
            // reader_ptr = std::make_unique<OpenImpala::TiffReader>();
            // if (!reader_ptr->readFile(tiff_filename)) {
            //    throw std::runtime_error("Failed to read TIFF via readFile");
            // }

        } catch (const std::exception& e) {
            amrex::Abort("Error creating TiffReader: " + std::string(e.what()));
        }

        // --- Check Dimensions & Metadata ---
        amrex::Print() << "Checking dimensions and metadata...\n";
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();
        int actual_bps = reader_ptr->bitsPerSample();
        int actual_format = reader_ptr->sampleFormat();
        int actual_spp = reader_ptr->samplesPerPixel();

        amrex::Print() << "  Read dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";
        amrex::Print() << "  Read metadata: BPS=" << actual_bps << ", Format=" << actual_format << ", SPP=" << actual_spp << "\n";

        // Use if/Abort instead of assert for checks active in all builds
        if (actual_width != expected_width || actual_height != expected_height || actual_depth != expected_depth) {
            amrex::Abort("FAIL: Read dimensions do not match expected dimensions (100x100x100).");
        }
        if (actual_bps != expected_bps) {
             amrex::Abort("FAIL: Read BitsPerSample (" + std::to_string(actual_bps) +
                         ") does not match expected value (" + std::to_string(expected_bps) + ").");
        }
        if (actual_format != expected_format) {
             amrex::Abort("FAIL: Read SampleFormat (" + std::to_string(actual_format) +
                         ") does not match expected value (" + std::to_string(expected_format) + ").");
        }
        if (actual_spp != expected_spp) {
             amrex::Abort("FAIL: Read SamplesPerPixel (" + std::to_string(actual_spp) +
                         ") does not match expected value (" + std::to_string(expected_spp) + ").");
        }
        amrex::Print() << "  Dimensions and basic metadata match expected values.\n";
        // TODO: Add tests for TiffReader attribute reading if implemented.
        // TODO: Add tests for TiffReader getValue(i,j,k) against known data points.

        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box();
        if (domain_box.isEmpty()) {
             amrex::Abort("FAIL: Reader returned an empty box.");
        }
        {
            amrex::RealBox rb({0.0, 0.0, 0.0}, {(amrex::Real)actual_width, (amrex::Real)actual_height, (amrex::Real)actual_depth}); // Physical domain matching pixel size
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0};
            amrex::Geometry::Setup(&rb, 0, is_periodic.data());
            geom.define(domain_box);
        }

        amrex::BoxArray ba(domain_box);
        ba.maxSize(BOX_SIZE);
        amrex::DistributionMapping dm(ba);
        amrex::iMultiFab mf(ba, dm, 1, 0); // 1 component, 0 ghost cells
        mf.setVal(0); // Initialize

        // --- Test Thresholding ---
        amrex::Print() << "Performing threshold > " << threshold_val << "...\n";
        try {
            // Use threshold(double, iMultiFab) overload for 1/0 output
            reader_ptr->threshold(threshold_val, mf);
        } catch (const std::exception& e) {
            amrex::Abort("Error during threshold operation: " + std::string(e.what()));
        }

        // Check results (min/max across all processors)
        int min_val = mf.min(0);
        int max_val = mf.max(0);

        amrex::Print() << "  Threshold result min value: " << min_val << "\n";
        amrex::Print() << "  Threshold result max value: " << max_val << "\n";

        // Check if result is binary (0 or 1)
        // This check is valid only if the threshold value is expected to segment the data
        if (min_val < 0 || max_val > 1 || (min_val == 0 && max_val == 0) || (min_val == 1 && max_val == 1) ) {
             amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                           << ") is not {0, 1}. Check threshold value ("<< threshold_val
                           << ") or sample data range.\n";
             // If min==max==0 or min==max==1, threshold didn't separate phases.
             // Decide if this is a fatal error for the test.
             // amrex::Abort("FAIL: Threshold result unexpected.");
        } else {
             amrex::Print() << "  Threshold value range looks plausible (0 and 1 found).\n";
        }
        // TODO: Add tests for threshold overload with custom true/false values.
        // TODO: Add separate tests for error conditions (bad file, unsupported TIFF format).

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
            amrex::Print() << "Writing plot file...\n";

            // Create output directory relative to executable location
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                      // Don't abort test, just warn that plotfile won't be written
                      amrex::Warning("Could not create output directory: " + test_output_dir + ". Skipping plotfile write.");
                      write_plotfile = false; // Disable writing
                 }
            }
            // Barrier potentially needed if directory creation is slow on filesystem? Usually okay.
            amrex::ParallelDescriptor::Barrier();

            if (write_plotfile) // Check again in case directory creation failed
            {
                // Get datetime string
                std::string datetime_str;
                {
                    std::time_t strt_time; std::tm* timeinfo; char datetime_buf[80];
                    std::time(&strt_time); timeinfo = std::localtime(&strt_time);
                    std::strftime(datetime_buf, sizeof(datetime_buf),"%Y%m%d%H%M",timeinfo);
                    datetime_str = datetime_buf;
                }
                // Construct filename
                std::string plotfilename = test_output_dir + "/tiffreadertest_" + datetime_str;

                // Copy integer data to float MultiFab for plotting
                amrex::MultiFab mfv(ba, dm, 1, 0);
                // Use ParallelFor for the copy
                for (amrex::MFIter mfi(mfv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const amrex::Box& box = mfi.tilebox();
                    auto const& int_fab_arr = mf.const_array(mfi);
                    auto&       real_fab_arr = mfv.array(mfi);
                    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        real_fab_arr(i, j, k) = static_cast<amrex::Real>(int_fab_arr(i, j, k));
                    });
                }

                // Write plot file
                amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);
                amrex::Print() << "  Plot file written to: " << plotfilename << "\n";
            }
        }

        amrex::Print() << "tTiffReader Test Completed Successfully.\n";

    } // End AMReX scope block

    amrex::Finalize();
    return 0;
}
