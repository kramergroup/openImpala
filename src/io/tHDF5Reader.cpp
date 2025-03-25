#include "HDF5Reader.H" // Assuming this defines OpenImpala::HDF5Reader

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>

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

// Default relative path to the sample HDF5 file
const std::string default_hdf5_filename = "data/SampleData_2Phase.h5";
// Default dataset path within the HDF5 file
const std::string default_hdf5_dataset = "exchange/data";
// Output directory relative to executable location
const std::string test_output_dir = "tHDF5Reader_output";
// Define Box size for breaking down domain
const int BOX_SIZE = 32;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        std::string hdf5_filename = default_hdf5_filename;
        std::string hdf5_dataset  = default_hdf5_dataset;
        bool write_plotfile = false; // Default: don't write plotfile

        { // Use ParmParse to read parameters from command line
          // Example: ./executable hdf5file=path/to/file.h5 dataset=path/in/h5 write_plotfile=1
            amrex::ParmParse pp;
            pp.query("hdf5file", hdf5_filename);
            pp.query("dataset", hdf5_dataset);
            pp.query("write_plotfile", write_plotfile);
        }

        // Check if input file exists before attempting to read
        {
            std::ifstream test_ifs(hdf5_filename);
            if (!test_ifs) {
                 amrex::Abort("Error: Cannot open input hdf5file: " + hdf5_filename + "\n"
                              "       Specify path using 'hdf5file=/path/to/file.h5'");
            }
        }

        amrex::Print() << "Starting tHDF5Reader Test (Oxford, " << __DATE__ << " " << __TIME__ << ")\n";
        amrex::Print() << "Input HDF5 file: " << hdf5_filename << "\n";
        amrex::Print() << "Input dataset path: " << hdf5_dataset << "\n";
        amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";

        // --- Test HDF5Reader ---
        std::unique_ptr<OpenImpala::HDF5Reader> reader_ptr;
        // Expected properties for default SampleData_2Phase.h5 at exchange/data
        int expected_width = 100;
        int expected_height = 100;
        int expected_depth = 100;
        int expected_bps = 8;       // Assuming 8-bit
        int expected_format = 1;  // Assuming uint (check HDF5Reader definition: 1=uint, 2=int, 3=float?)
        int expected_spp = 1;       // Assuming grayscale
        // Assuming threshold '1' yields a non-empty mask for this specific file
        const OpenImpala::HDF5Reader::DataType threshold_value = 1; // Assuming DataType compatible with int/double

        try {
            // Assuming constructor reads file/dataset and throws std::runtime_error on failure
            reader_ptr = std::make_unique<OpenImpala::HDF5Reader>(hdf5_filename, hdf5_dataset);

        } catch (const std::exception& e) {
            amrex::Abort("Error creating HDF5Reader: " + std::string(e.what()));
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
        // TODO: Add tests for HDF5Reader attribute reading if implemented.
        // TODO: Add tests for HDF5Reader getValue(i,j,k) against known data points.

        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box(); // Use reader's box
        {
            amrex::RealBox rb({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}); // Dummy physical domain
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0}; // Non-periodic
            amrex::Geometry::Setup(&rb, 0, is_periodic.data());
            geom.define(domain_box);
        }

        amrex::BoxArray ba(domain_box);
        ba.maxSize(BOX_SIZE);
        amrex::DistributionMapping dm(ba);

        // Create iMultiFab to hold thresholded data
        amrex::iMultiFab mf(ba, dm, 1, 0); // 1 component, 0 ghost cells
        mf.setVal(0); // Initialize

        // --- Test Thresholding ---
        amrex::Print() << "Performing threshold > " << threshold_value << "...\n";
        try {
            // Assuming threshold takes DataType and iMultiFab&, and is const
            reader_ptr->threshold(threshold_value, mf);
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
        // TODO: Add tests for threshold overload with custom true/false values.
        // TODO: Add separate tests for error conditions (bad file, bad dataset path).

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
            amrex::Print() << "Writing plot file...\n";

            // Create output directory relative to executable location
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                      amrex::Warning("Could not create output directory: " + test_output_dir);
                 }
            }
            amrex::ParallelDescriptor::Barrier();

            // Get datetime string for filename
            std::string datetime_str;
            {
                std::time_t strt_time;
                std::tm* timeinfo;
                char datetime_buf [80];
                std::time(&strt_time);
                timeinfo = std::localtime(&strt_time);
                std::strftime(datetime_buf, sizeof(datetime_buf),"%Y%m%d%H%M",timeinfo);
                datetime_str = datetime_buf;
            }

            // Construct filename
            std::string plotfilename = test_output_dir + "/hdf5readertest_" + datetime_str;

            // Copy integer data to float MultiFab for plotting
            amrex::MultiFab mfv(ba, dm, 1, 0);
            for (amrex::MFIter mfi(mfv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& box = mfi.tilebox();
                auto const& int_fab = mf.const_array(mfi);
                auto&       real_fab = mfv.array(mfi);

                amrex::ParallelFor(box, [&] (int i, int j, int k) noexcept
                {
                    real_fab(i, j, k) = static_cast<amrex::Real>(int_fab(i, j, k));
                });
            }

            // Write plot file
            amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);
            amrex::Print() << "  Plot file written to: " << plotfilename << "\n";
        }

        amrex::Print() << "tHDF5Reader Test Completed Successfully.\n";

    } // End AMReX scope block

    amrex::Finalize();
    return 0;
}
