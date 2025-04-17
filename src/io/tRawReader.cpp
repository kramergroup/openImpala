#include "RawReader.H" // Assuming defines OpenImpala::RawReader and OpenImpala::RawDataType

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>
#include <memory>    // For std::unique_ptr
#include <map>       // For mapping string to enum

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

// --- Default Test Parameters ---

// Default relative path to the sample RAW file
// IMPORTANT: Adjust this to your actual sample file location and name
const std::string default_raw_filename = "data/SampleData_2Phase.raw";
// Output directory relative to executable location
const std::string test_output_dir = "tRawReader_output";

// IMPORTANT: These MUST match the actual properties of your default sample file
const int default_width = 100;
const int default_height = 100;
const int default_depth = 100;
const std::string default_datatype_str = "UINT8"; // Example: Assume 8-bit unsigned

// Default threshold value (adjust based on sample data type and range)
// e.g., halfway for UINT8 [0-255]
const double default_threshold_value = 127.5;

// Define Box size for breaking down domain
const int BOX_SIZE = 32;

// Helper to convert string to RawDataType enum
OpenImpala::RawDataType StringToRawDataType(const std::string& type_str) {
    // Create a map for easy lookup (case-insensitive might be better)
    static const std::map<std::string, OpenImpala::RawDataType> type_map = {
        {"UINT8", OpenImpala::RawDataType::UINT8},
        {"INT8", OpenImpala::RawDataType::INT8},
        {"UINT16_LE", OpenImpala::RawDataType::UINT16_LE},
        {"UINT16_BE", OpenImpala::RawDataType::UINT16_BE},
        {"INT16_LE", OpenImpala::RawDataType::INT16_LE},
        {"INT16_BE", OpenImpala::RawDataType::INT16_BE},
        {"UINT32_LE", OpenImpala::RawDataType::UINT32_LE},
        {"UINT32_BE", OpenImpala::RawDataType::UINT32_BE},
        {"INT32_LE", OpenImpala::RawDataType::INT32_LE},
        {"INT32_BE", OpenImpala::RawDataType::INT32_BE},
        {"FLOAT32_LE", OpenImpala::RawDataType::FLOAT32_LE},
        {"FLOAT32_BE", OpenImpala::RawDataType::FLOAT32_BE},
        {"FLOAT64_LE", OpenImpala::RawDataType::FLOAT64_LE},
        {"FLOAT64_BE", OpenImpala::RawDataType::FLOAT64_BE}
        // Add more if needed
    };

    auto it = type_map.find(type_str);
    if (it != type_map.end()) {
        return it->second;
    } else {
        amrex::Warning("Unrecognized RawDataType string: " + type_str + ". Using UNKNOWN.");
        return OpenImpala::RawDataType::UNKNOWN;
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        std::string raw_filename = default_raw_filename;
        int width  = default_width;
        int height = default_height;
        int depth  = default_depth;
        std::string datatype_str = default_datatype_str;
        double threshold_val = default_threshold_value;
        bool write_plotfile = false;

        { // Use ParmParse to read parameters from command line
          // Example: ./executable rawfile=path/data.raw width=100 height=100 depth=100 datatype=UINT8 threshold=127.5 write_plotfile=1
            amrex::ParmParse pp;
            pp.query("rawfile", raw_filename);
            pp.query("width", width);
            pp.query("height", height);
            pp.query("depth", depth);
            pp.query("datatype", datatype_str); // Read type as string
            pp.query("threshold", threshold_val);
            pp.query("write_plotfile", write_plotfile);
        }

        // Convert datatype string to enum
        OpenImpala::RawDataType data_type_enum = StringToRawDataType(datatype_str);
        if (data_type_enum == OpenImpala::RawDataType::UNKNOWN) {
             amrex::Abort("Error: Invalid datatype string specified: " + datatype_str + "\n"
                         "       Supported types: UINT8, INT8, UINT16_LE, UINT16_BE, INT16_LE, ..., FLOAT64_BE");
        }
        if (width <= 0 || height <= 0 || depth <= 0) {
             amrex::Abort("Error: Invalid dimensions specified (W=" + std::to_string(width)
                          + ", H=" + std::to_string(height) + ", D=" + std::to_string(depth) + "). Must be positive.");
        }


        // Check if input file exists before attempting to read
        {
            std::ifstream test_ifs(raw_filename);
            if (!test_ifs) {
                 amrex::Abort("Error: Cannot open input rawfile: " + raw_filename + "\n"
                              "       Specify path using 'rawfile=/path/to/file.raw'");
            }
        }

        amrex::Print() << "Starting tRawReader Test (Oxford, " << __DATE__ << " " << __TIME__ << ")\n";
        amrex::Print() << "Input RAW file: " << raw_filename << "\n";
        amrex::Print() << "Input Dimensions: " << width << "x" << height << "x" << depth << "\n";
        amrex::Print() << "Input DataType: " << datatype_str << "\n";
        amrex::Print() << "Threshold value: " << threshold_val << "\n";
        amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";

        // --- Test RawReader ---
        std::unique_ptr<OpenImpala::RawReader> reader_ptr;
        try {
            // Use the constructor that takes metadata and throws on failure
            reader_ptr = std::make_unique<OpenImpala::RawReader>(raw_filename, width, height, depth, data_type_enum);

        } catch (const std::exception& e) {
            amrex::Abort("Error creating RawReader: " + std::string(e.what()));
        }

        // --- Check Dimensions & Type (should match input if successful) ---
        amrex::Print() << "Checking dimensions and data type...\n";
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();
        OpenImpala::RawDataType actual_type = reader_ptr->getDataType();

        amrex::Print() << "  Reader dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";
        amrex::Print() << "  Reader DataType: " << static_cast<int>(actual_type) << "\n"; // Simple print

        if (actual_width != width || actual_height != height || actual_depth != depth) {
            amrex::Abort("FAIL: Reader dimensions do not match input dimensions.");
        }
         if (actual_type != data_type_enum) {
             amrex::Abort("FAIL: Reader data type does not match input data type.");
         }
        amrex::Print() << "  Dimensions and data type confirmed.\n";
        // TODO: Add tests for RawReader getValue(i,j,k) against known data points if available.
        // TODO: Add separate tests for error conditions (bad file path, size mismatch, unsupported type).

        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box();
        if (domain_box.isEmpty()) {
             amrex::Abort("FAIL: Reader returned an empty box.");
        }
        {
            // Use physical domain size equal to index space for simplicity here
            amrex::RealBox rb({0.0, 0.0, 0.0}, {(amrex::Real)actual_width, (amrex::Real)actual_height, (amrex::Real)actual_depth});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0};
            amrex::Geometry::Setup(&rb, 0, is_periodic.data());
            geom.define(domain_box);
        }

        amrex::BoxArray ba(domain_box);
        ba.maxSize(BOX_SIZE);
        amrex::DistributionMapping dm(ba);
        amrex::iMultiFab mf(ba, dm, 1, 0); // 1 component, 0 ghost cells
        mf.setVal(-1); // Initialize to -1 to check if all cells get set by threshold

        // --- Test Thresholding ---
        amrex::Print() << "Performing threshold > " << threshold_val << "...\n";
        try {
            // Use threshold(double, iMultiFab) overload for 1/0 output
            reader_ptr->threshold(threshold_val, mf);
            // TODO: Test threshold overload with custom true/false values
        } catch (const std::exception& e) {
            // Catch potential errors from getValue if threshold calls it and it throws
            // Or internal Abort from threshold itself if data wasn't read.
            amrex::Abort("Error during threshold operation: " + std::string(e.what()));
        }

        // Check results (min/max across all processors)
        int min_val = mf.min(0);
        int max_val = mf.max(0);

        amrex::Print() << "  Threshold result min value: " << min_val << "\n";
        amrex::Print() << "  Threshold result max value: " << max_val << "\n";

        // Check if result is binary (0 or 1)
        if (min_val < 0 || max_val > 1 || (min_val == 0 && max_val == 0) || (min_val == 1 && max_val == 1) ) {
             amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                           << ") not {0, 1}. Check threshold value ("<< threshold_val
                           << ") or sample data range / content.\n";
             // If min==max==0 or min==max==1, threshold didn't segment phases or data is uniform.
             // Decide if this is a fatal error for the test.
        } else {
             amrex::Print() << "  Threshold value range looks plausible (0 and 1 found).\n";
        }

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
            amrex::Print() << "Writing plot file...\n";

            // Create output directory relative to executable location
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                      amrex::Warning("Could not create output directory: " + test_output_dir + ". Skipping plotfile write.");
                      write_plotfile = false; // Disable writing
                 }
            }
            amrex::ParallelDescriptor::Barrier();

            if (write_plotfile)
            {
                // Get datetime string
                std::string datetime_str;
                {
                    std::time_t strt_time; std::tm* timeinfo; char datetime_buf [80];
                    std::time(&strt_time); timeinfo = std::localtime(&strt_time);
                    std::strftime(datetime_buf, sizeof(datetime_buf),"%Y%m%d%H%M",timeinfo);
                    datetime_str = datetime_buf;
                }
                std::string plotfilename = test_output_dir + "/rawreadertest_" + datetime_str;

                // Copy integer data to float MultiFab for plotting
                amrex::MultiFab mfv(ba, dm, 1, 0);
                for (amrex::MFIter mfi(mfv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const amrex::Box& box = mfi.tilebox();
                    auto const& int_fab_arr = mf.const_array(mfi);
                    auto       real_fab_arr = mfv.array(mfi);
                    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        real_fab_arr(i, j, k) = static_cast<amrex::Real>(int_fab_arr(i, j, k));
                    });
                }

                amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);
                amrex::Print() << "  Plot file written to: " << plotfilename << "\n";
            }
        }

        amrex::Print() << "tRawReader Test Completed Successfully.\n";

    } // End AMReX scope block

    amrex::Finalize();
    return 0;
}
